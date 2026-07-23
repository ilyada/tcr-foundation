"""
TCRFoundation single-stage JOINT pretrain entry point.

Trains ONE model (per-chain BertForMaskedLM + a single mixed pooler) jointly on BOTH:
  - paired data (OTS + pan_disease + Tanno) -- MLM(both chains) + paired InfoNCE + chain-drop;
  - Emerson beta-only -- MLM (the publicness signal for B1) + beta autocontrast.
Both modalities share the SAME encoder + pooler -> beta-only and paired live in ONE coordinate
system. Negatives are modality-separated (two loaders, zipped 50/50). This SUPERSEDES the
two-stage (beta-first backbone + paired LoRA) design.

See Knowledge_Base/project_TCRFoundation/context/experiments (joint single-stage) for the spec.

Usage:
    python tcr_foundation_pretrain_joint.py --config config/foundation/foundation_joint_tiny.yaml \\
        --save-path ../models/foundation/tcr-foundation-joint-tiny \\
        --experiment-name "TCR-Foundation | MLM(b+pair)+InfoNCE+chain-drop | OTS+pan+Tanno+Emerson | tiny | bf16 lr3e-4"
"""

import comet_ml  # noqa: F401 - Must come before torch for auto-logging
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # safe with DataLoader num_workers>0
import sys
import json
import argparse
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ root for `from utils`
from utils.data_utils import (TCRJointDataset, TCRBetaOnlyMLMDataset,
                              load_paired_sequences, load_emerson_beta_pretrain)
from utils.model_utils import TCRFoundationJoint, build_perchain_mlm, load_perchain_mlm
from utils.training_utils import train_joint_epoch
from utils.config_utils import load_config, setup_comet_experiment


def _seed_worker(worker_id):
    """Reseed numpy per DataLoader worker (dataset uses np.random for MLM masking + CDR3
    censoring; torch reseeds its own RNG per worker but NOT numpy). Without this every worker
    would repeat the same augmentation, hurting the two-view contrastive."""
    import numpy as _np
    _np.random.seed(torch.initial_seed() % (2 ** 32))


def parse_args():
    p = argparse.ArgumentParser(description="TCRFoundation joint single-stage pretrain")
    p.add_argument("--config",            required=True)
    p.add_argument("--paired-data-path")
    p.add_argument("--emerson-data-path")
    p.add_argument("--save-path")
    p.add_argument("--epochs",            type=int)
    p.add_argument("--batch-size",        type=int)
    p.add_argument("--lr",                type=float)
    p.add_argument("--lambda-mlm",        type=float)
    p.add_argument("--lambda-pair",       type=float)
    p.add_argument("--lambda-drop",       type=float)
    p.add_argument("--lambda-betaonly",   type=float)
    p.add_argument("--lambda-drop-beta",  type=float,
                   help="V-token/alpha-detach: beta->pair chain-drop weight (set -> split mode)")
    p.add_argument("--lambda-drop-alpha", type=float, help="alpha->pair chain-drop weight (0 = detached)")
    p.add_argument("--lambda-alpha-simcse", type=float, help="alpha's own SimCSE (two-view) weight")
    p.add_argument("--temperature",       type=float)
    p.add_argument("--emerson-per-epoch", type=int,
                   help="Emerson beta samples per epoch (default: match paired count for ~50/50)")
    p.add_argument("--num-workers",       type=int,
                   help="DataLoader workers PER loader (>0 on Linux cluster; 0 on Windows)")
    p.add_argument("--hidden-size",       type=int)   # arch overrides -> run light on the tiny config
    p.add_argument("--num-layers",        type=int)
    p.add_argument("--num-heads",         type=int)
    p.add_argument("--intermediate-size", type=int)
    p.add_argument("--experiment-name")
    p.add_argument("--comet-project")
    p.add_argument("--no-comet",          action="store_true")
    p.add_argument("--smoke",             action="store_true",
                   help="Fast mechanics check: 1 epoch, batch=16, ~4 steps, no comet, datasets capped")
    p.add_argument("--resume",            action="store_true",
                   help="Resume from save_path/joint_config.json:last_completed_epoch")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    for k_cli, k_cfg in [("paired_data_path", "paired_data_path"),
                         ("emerson_data_path", "emerson_data_path"),
                         ("save_path", "save_path"), ("epochs", "epochs"),
                         ("batch_size", "batch_size"), ("lr", "lr"),
                         ("lambda_mlm", "lambda_mlm"), ("lambda_pair", "lambda_pair"),
                         ("lambda_drop", "lambda_drop"), ("lambda_betaonly", "lambda_betaonly"),
                         ("lambda_drop_beta", "lambda_drop_beta"),
                         ("lambda_drop_alpha", "lambda_drop_alpha"),
                         ("lambda_alpha_simcse", "lambda_alpha_simcse"),
                         ("temperature", "temperature"), ("emerson_per_epoch", "emerson_per_epoch"),
                         ("num_workers", "num_workers"),
                         ("hidden_size", "hidden_size"), ("num_layers", "num_hidden_layers"),
                         ("num_heads", "num_attention_heads"),
                         ("intermediate_size", "intermediate_size")]:
        v = getattr(args, k_cli)
        if v is not None:
            config[k_cfg] = v

    # ---- smoke mode: cap everything for a fast wiring check (set BEFORE comet + dataset build) ----
    if args.smoke:
        config["epochs"]      = 1
        config["batch_size"]  = min(config.get("batch_size", 512), 16)
        config["num_workers"] = 0
        args.no_comet = True
        print(">>> SMOKE MODE: epochs=1, batch=16, ~4 steps, no comet, datasets capped <<<")

    # Print the RESOLVED config (YAML + CLI overrides) -- exact values used this run.
    print("==================== RESOLVED JOINT CONFIG ====================")
    print(f"  arch                : {config['hidden_size']}/{config['num_hidden_layers']}/"
          f"{config['num_attention_heads']}/{config['intermediate_size']}  "
          f"max_pos={config['max_position_embeddings']}  attention=per-chain")
    print(f"  paired_data_path    : {config['paired_data_path']}")
    print(f"  dataset_sources     : {config.get('dataset_sources')}  (None = OTS+pan+Tanno)")
    print(f"  emerson_data_path   : {config['emerson_data_path']}")
    print(f"  emerson_per_epoch   : {config.get('emerson_per_epoch')}  (None = match paired count)")
    print(f"  save_path           : {config['save_path']}")
    print(f"  epochs              : {config.get('epochs', 5)}")
    print(f"  batch_size          : {config.get('batch_size', 512)}  (per modality -> ~2x/step)")
    print(f"  lr                  : {config.get('lr', 3e-4)}")
    print(f"  warmup_frac         : {config.get('warmup_frac', 0.05)}")
    print(f"  weight_decay        : {config.get('weight_decay', 0.01)}")
    print(f"  lambda_mlm/pair/drop/betaonly : {config.get('lambda_mlm', 1.0)}/"
          f"{config.get('lambda_pair', 1.0)}/{config.get('lambda_drop', 1.0)}/"
          f"{config.get('lambda_betaonly', 1.0)}")
    print(f"  input_format        : {config.get('input_format', 'cdr123')}  "
          f"(tokenizer={config.get('tokenizer_path', 'wukevin/tcr-bert')})")
    if config.get("input_format") == "vtoken":
        print(f"  lambda drop_beta/drop_alpha/alpha_simcse : {config.get('lambda_drop_beta')}/"
              f"{config.get('lambda_drop_alpha', 0.0)}/{config.get('lambda_alpha_simcse', 0.0)}  "
              f"(alpha DETACHED: drop_alpha=0 + own SimCSE)")
    print(f"  temperature         : {config.get('temperature', 0.05)}")
    print(f"  mlm_prob            : {config.get('mlm_prob', 0.15)}")
    print(f"  censor apply/res    : {config.get('censor_apply_prob', 0.5)}/{config.get('censor_res_prob', 0.20)}")
    print(f"  max_len paired/beta : {config.get('max_len_paired', 96)}/{config.get('max_len_beta', 64)}")
    print(f"  seed                : {config.get('seed', 42)}")
    print(f"  use_amp             : {config.get('use_amp', True)}")
    print(f"  num_workers/loader  : {config.get('num_workers', 0)}")
    print("===============================================================")

    exp = setup_comet_experiment(config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Tokenizer: default wukevin (legacy CDR1|CDR2|CDR3), or a custom dir (e.g. the atomic V-token
    # tokenizer models/tokenizers/tcr-vtoken) when tokenizer_path is set in the config.
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path", "wukevin/tcr-bert"))

    # V-token mode: input_format="vtoken" builds "[V:gene] | CDR3" (V replaces CDR1/CDR2) and requires
    # a vgene_map (raw v_gene -> atomic V token). None -> legacy cdr123 grammar (backward-compatible).
    input_format = config.get("input_format", "cdr123")
    vgene_map = None
    if input_format == "vtoken":
        with open(config["vgene_map_path"]) as f:
            vgene_map = json.load(f)
        print(f"V-token mode: {len(vgene_map)} raw V-genes -> atomic tokens "
              f"(from {config['vgene_map_path']})")

    # ---- resume or fresh per-chain backbone ----
    save_path    = config["save_path"]
    backbone_dir = os.path.join(save_path, "backbone")
    pooler_path  = os.path.join(save_path, "poolers", "joint.pt")
    config_path  = os.path.join(save_path, "joint_config.json")

    resume_start_epoch = 0
    resume = args.resume and os.path.isfile(config_path)
    if args.resume and not resume:
        print(f"--resume requested but no joint_config.json at {save_path}; fresh init.")

    if resume:
        print(f"RESUME: loading per-chain backbone from {backbone_dir}")
        bert = load_perchain_mlm(backbone_dir)
        with open(config_path) as f:
            resume_start_epoch = int(json.load(f).get("last_completed_epoch", 0))
        print(f"  continuing from epoch {resume_start_epoch + 1}")
    else:
        print("Building fresh per-chain BertForMaskedLM (tiny + per-chain RPE) ...")
        arch = {
            "hidden_size":             config["hidden_size"],
            "num_hidden_layers":       config["num_hidden_layers"],
            "num_attention_heads":     config["num_attention_heads"],
            "intermediate_size":       config["intermediate_size"],
            "max_position_embeddings": config["max_position_embeddings"],
            "type_vocab_size":         config.get("type_vocab_size", 2),
            "vocab_size":              config.get("vocab_size", 26),
            "hidden_act":              config.get("hidden_act", "gelu"),
            "hidden_dropout_prob":     config.get("hidden_dropout_prob", 0.1),
            "attention_probs_dropout_prob": config.get("attention_probs_dropout_prob", 0.1),
            "pad_token_id":            21,   # wukevin tokenizer PAD
        }
        bert = build_perchain_mlm(arch)

    hidden_size = bert.config.hidden_size
    num_layers  = bert.config.num_hidden_layers
    model = TCRFoundationJoint(
        bert=bert, hidden_size=hidden_size, num_layers=num_layers,
        pooling=config.get("pooling", "mixed"), dropout=config.get("dropout", 0.1),
    ).to(device)
    if resume and os.path.isfile(pooler_path):
        model.pooler.load_state_dict(torch.load(pooler_path, map_location=device))
        print(f"RESUME: joint pooler loaded from {pooler_path}")

    # ---- data: paired (all 3 sources) + Emerson beta (full 10M; per-epoch subsample) ----
    print("Loading paired sequences (OTS+pan+Tanno) ...")
    pseq = load_paired_sequences(config["paired_data_path"],
                                 dataset_source=config.get("dataset_sources"),  # None = all
                                 vgene_map=vgene_map)                            # None -> no V tokens
    is_vtoken = input_format == "vtoken"
    paired_dataset = TCRJointDataset(
        alpha_cdr1=pseq["alpha_cdr1"], alpha_cdr2=pseq["alpha_cdr2"], alpha_cdr3=pseq["alpha_cdr3"],
        beta_cdr1=pseq["beta_cdr1"],   beta_cdr2=pseq["beta_cdr2"],   beta_cdr3=pseq["beta_cdr3"],
        tokenizer=tokenizer, max_len=config.get("max_len_paired", 96),
        mlm_prob=config.get("mlm_prob", 0.15), view_strategy="exp9_chain_drop_loss",
        censor_apply_prob=config.get("censor_apply_prob", 0.5),
        censor_residue_prob=config.get("censor_res_prob", 0.20),
        input_format=input_format,
        alpha_vtoken=pseq.get("alpha_vtoken"), beta_vtoken=pseq.get("beta_vtoken"),
        alpha_simcse=is_vtoken,   # vtoken run -> alpha detached (own SimCSE, two views)
    )
    print(f"Paired clonotypes: {len(paired_dataset):,}  (input_format={input_format}, "
          f"alpha_simcse={is_vtoken})")
    if args.smoke:
        # Cap paired to ~4 batches; Emerson is then capped via emer_per_epoch = len(paired) below.
        n_smoke = min(config["batch_size"] * 4, len(paired_dataset))
        paired_dataset = Subset(paired_dataset, list(range(n_smoke)))
        print(f"SMOKE: capped paired to {len(paired_dataset)} clonotypes")

    print("Loading Emerson beta corpus ...")
    eseq = load_emerson_beta_pretrain(config["emerson_data_path"])
    # V-token mode: map Emerson raw V-genes -> atomic beta V-tokens (unmapped -> [V_UNK]) so the beta-only
    # corpus uses the SAME grammar as the paired encoder (one input space).
    emer_beta_vtoken = None
    if is_vtoken:
        emer_beta_vtoken = [vgene_map.get(str(v), "[V_UNK]") for v in eseq["v_gene"]]
        n_unk = sum(t == "[V_UNK]" for t in emer_beta_vtoken)
        print(f"  Emerson V-token coverage: {len(emer_beta_vtoken) - n_unk:,}/{len(emer_beta_vtoken):,} "
              f"mapped ({n_unk:,} -> [V_UNK])")
    emerson_dataset = TCRBetaOnlyMLMDataset(
        cdr1=eseq["cdr1"], cdr2=eseq["cdr2"], cdr3=eseq["cdr3"],
        tokenizer=tokenizer, max_len=config.get("max_len_beta", 64),
        mlm_prob=config.get("mlm_prob", 0.15),
        censor_apply_prob=config.get("censor_apply_prob", 0.5),
        censor_res_prob=config.get("censor_res_prob", 0.20),
        input_format=input_format, beta_vtoken=emer_beta_vtoken,
    )
    print(f"Emerson beta rows (full pool): {len(emerson_dataset):,}")

    batch_size  = config.get("batch_size", 512)
    num_workers = config.get("num_workers", 0)
    use_pin     = platform.system() != "Windows"
    # Emerson per epoch: default match paired count -> ~50/50 and zip-aligned batch counts.
    emer_per_epoch = config.get("emerson_per_epoch") or len(paired_dataset)
    emer_per_epoch = min(emer_per_epoch, len(emerson_dataset))

    # Paired loader: built once, full shuffled pass, persistent workers (reused every epoch).
    paired_loader = DataLoader(
        paired_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_pin, drop_last=True, worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )

    # Fixed shuffled permutation of the full Emerson pool (deterministic from seed). Each epoch
    # takes the NEXT contiguous emer_per_epoch slice (wrapping) -> DISJOINT chunks that together
    # cover the WHOLE pool over ~ceil(total/per_epoch) epochs (not overlapping random draws).
    # Resume-safe: the epoch index alone determines the slice.
    emerson_perm = np.random.default_rng(seed).permutation(len(emerson_dataset))

    def make_emerson_loader(epoch: int) -> DataLoader:
        """Next disjoint ~emer_per_epoch slice of `emerson_perm` (wraps around) so the model sees
        all of Emerson across epochs. Rebuilt per epoch, so persistent_workers=False."""
        total = len(emerson_dataset)
        take  = np.arange(epoch * emer_per_epoch, (epoch + 1) * emer_per_epoch) % total
        idx   = emerson_perm[take]
        return DataLoader(
            emerson_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx.tolist()),
            num_workers=num_workers, pin_memory=use_pin, drop_last=True,
            worker_init_fn=_seed_worker, persistent_workers=False,
        )

    n_steps = min(len(paired_loader), emer_per_epoch // batch_size)   # zip stops at the shorter
    print(f"Steps/epoch: {n_steps}  (paired batches={len(paired_loader)}, "
          f"emerson batches={emer_per_epoch // batch_size}, ~50/50)")

    # ---- optimizer / scheduler ----
    optimizer    = AdamW(model.parameters(), lr=config.get("lr", 3e-4),
                         weight_decay=config.get("weight_decay", 0.01))
    total_steps  = config.get("epochs", 5) * n_steps
    warmup_steps = int(config.get("warmup_frac", 0.05) * total_steps)
    scheduler    = LambdaLR(optimizer,
                            lr_lambda=lambda s: s / max(1, warmup_steps) if s < warmup_steps else 1.0)
    global_step  = resume_start_epoch * n_steps
    for _ in range(global_step):
        scheduler.step()

    lambda_mlm      = config.get("lambda_mlm", 1.0)
    lambda_pair     = config.get("lambda_pair", 1.0)
    lambda_drop     = config.get("lambda_drop", 1.0)
    lambda_betaonly = config.get("lambda_betaonly", 1.0)
    temperature     = config.get("temperature", 0.05)
    # Split chain-drop (V-token / alpha-detach). lambda_drop_beta=None -> legacy symmetric chain-drop
    # (both chains pulled to the pair via lambda_drop). Set -> beta kept, alpha detached (+ own SimCSE).
    lambda_drop_beta    = config.get("lambda_drop_beta")
    lambda_drop_alpha   = config.get("lambda_drop_alpha", 0.0)
    lambda_alpha_simcse = config.get("lambda_alpha_simcse", 0.0)

    # ---- mixed precision: prefer bf16 (fp16 overflows low-tau InfoNCE in the forward -> spikes);
    #      bf16 needs Ampere+, else fp16 + GradScaler. (Same convention as Stage 1/2.) ----
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype, scaler = torch.bfloat16, None
    elif use_amp:
        amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")
    else:
        amp_dtype, scaler = None, None
    print(f"Mixed precision (amp): {use_amp} "
          f"(dtype={amp_dtype}, gradscaler={'yes' if scaler is not None else 'no'})")

    os.makedirs(backbone_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pooler_path), exist_ok=True)

    def save_checkpoint(epoch_idx: int) -> None:
        """Save backbone (encoder + the single MLM head) + the single pooler + tokenizer + config.
        In V-token mode also copy the vgene_map into the checkpoint so eval uses the SAME map."""
        model.bert.save_pretrained(backbone_dir)
        torch.save(model.pooler.state_dict(), pooler_path)
        tok_dir = os.path.join(save_path, "tokenizer")
        if not os.path.isdir(tok_dir):
            tokenizer.save_pretrained(tok_dir)
        # V-token eval needs the exact training vgene_map -> co-locate it with the checkpoint.
        if is_vtoken and vgene_map is not None:
            with open(os.path.join(save_path, "vgene_map.json"), "w") as vf:
                json.dump(vgene_map, vf, indent=2)
        with open(config_path, "w") as f:
            json.dump({
                "framework":               "tcr_foundation_joint",
                "attention":               "perchain",
                "hidden_size":             hidden_size,
                "num_hidden_layers":       num_layers,
                "num_attention_heads":     bert.config.num_attention_heads,
                "intermediate_size":       bert.config.intermediate_size,
                "max_position_embeddings": bert.config.max_position_embeddings,
                "position_embedding_type": bert.config.position_embedding_type,
                "type_vocab_size":         bert.config.type_vocab_size,
                "vocab_size":              bert.config.vocab_size,
                "pooling":                 config.get("pooling", "mixed"),
                "dropout":                 config.get("dropout", 0.1),
                "max_len_paired":          config.get("max_len_paired", 96),
                "max_len_beta":            config.get("max_len_beta", 64),
                # V-token descriptors: eval reads input_format + the tokenizer/ dir + vgene_map.json.
                "input_format":            input_format,
                "vgene_map":               "vgene_map.json" if is_vtoken else None,
                "paired_data_path":        config["paired_data_path"],
                "dataset_sources":         config.get("dataset_sources"),
                "emerson_data_path":       config["emerson_data_path"],
                "emerson_per_epoch":       emer_per_epoch,
                "lambda_mlm":              lambda_mlm, "lambda_pair": lambda_pair,
                "lambda_drop":             lambda_drop, "lambda_betaonly": lambda_betaonly,
                "lambda_drop_beta":        lambda_drop_beta, "lambda_drop_alpha": lambda_drop_alpha,
                "lambda_alpha_simcse":     lambda_alpha_simcse,
                "temperature":             temperature, "mlm_prob": config.get("mlm_prob", 0.15),
                "censor_apply_prob":       config.get("censor_apply_prob", 0.5),
                "censor_res_prob":         config.get("censor_res_prob", 0.20),
                "seed":                    seed, "epochs": config.get("epochs", 5),
                "batch_size":              batch_size, "lr": config.get("lr", 3e-4),
                "last_completed_epoch":    epoch_idx,
            }, f, indent=2)

    for epoch in range(resume_start_epoch, config.get("epochs", 5)):
        print(f"\nEpoch {epoch + 1}/{config.get('epochs', 5)}")
        emerson_loader = make_emerson_loader(epoch)   # fresh Emerson subsample each epoch
        epoch_loss, global_step = train_joint_epoch(
            model, paired_loader, emerson_loader, optimizer, scheduler, device, exp, global_step,
            lambda_mlm=lambda_mlm, lambda_pair=lambda_pair, lambda_drop=lambda_drop,
            lambda_betaonly=lambda_betaonly, temperature=temperature,
            scaler=scaler, amp_dtype=amp_dtype, max_grad_norm=config.get("max_grad_norm", 1.0),
            lambda_drop_beta=lambda_drop_beta, lambda_drop_alpha=lambda_drop_alpha,
            lambda_alpha_simcse=lambda_alpha_simcse,
        )
        print(f"Epoch {epoch + 1} -- loss={epoch_loss:.4f}")
        if exp:
            exp.log_metric("epoch_loss", epoch_loss, epoch=epoch)
        save_checkpoint(epoch + 1)
        print(f"  checkpoint saved to {save_path} (last_completed_epoch={epoch + 1})")

    print(f"\nFinal save -> {save_path}")
    save_checkpoint(config.get("epochs", 5))
    print("Done.")


if __name__ == "__main__":
    main()
