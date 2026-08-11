"""
benchmark_utils.py -- Utilities for TCR-BERT benchmark evaluation.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

from utils.model_utils import BertPooler, load_perchain_mlm, TCRFoundationJoint


_TOKENIZERS = {}


def _get_tokenizer(path=None):
    """Cached tokenizer, keyed by path. Default = wukevin/tcr-bert (legacy CDR1|CDR2|CDR3 vocab).
    Pass a checkpoint's tokenizer/ dir to load a custom vocab (e.g. the atomic V-token tokenizer)."""
    key = path or "wukevin/tcr-bert"
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = AutoTokenizer.from_pretrained(key)
    return _TOKENIZERS[key]


def extract_beta_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                             chain_token="[unused1]", batch_size=512, device=None,
                             pooler_file="pooler_beta.pt"):
    """
    Extract single-chain embeddings from a pretrained TCR-BERT model.

    Defaults to the beta chain (chain_token "[unused1]", pooler_beta.pt). For alpha-only
    inference pass chain_token="[unused0]" and pooler_file="pooler_alpha.pt". If the
    named pooler is absent, falls back to a shared pooler.pt, then to the CLS token.

    Args:
        model_path:  Path to AutoModelForMaskedLM checkpoint
        cdr1_list:   List of CDR1 strings (e.g. "MNIHD")
        cdr2_list:   List of CDR2 strings
        cdr3_list:   List of CDR3 strings
        chain_token: Chain prefix token inserted before CDR1 (default "[unused1]" for beta)
        batch_size:  Inference batch size
        device:      torch.device (default: cuda if available)
        pooler_file: Trained pooler filename in the checkpoint (default "pooler_beta.pt";
                     use "pooler_alpha.pt" for the alpha chain)

    Returns:
        np.ndarray [N, hidden_size]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()
    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    # Load trained BertPooler if saved alongside the checkpoint
    pooler = None
    pooler_path = os.path.join(model_path, pooler_file)
    shared_pooler_path = os.path.join(model_path, "pooler.pt")
    config_path = os.path.join(model_path, "pairing_config.json")

    if os.path.exists(pooler_path) and os.path.exists(config_path):
        with open(config_path) as f:
            pcfg = json.load(f)
        pooler = BertPooler(
            pcfg.get("pooling", "mixed"),
            encoder.config.hidden_size,
            encoder.config.num_hidden_layers,
            pcfg.get("dropout", 0.1),
        )
        pooler.load_state_dict(torch.load(pooler_path, map_location=device))
        pooler.to(device).eval()
        print(f"Loaded {pooler_file[:-3]} ({pcfg.get('pooling','mixed')} pooling) from {model_path}")
    elif os.path.exists(shared_pooler_path) and os.path.exists(config_path):
        with open(config_path) as f:
            pcfg = json.load(f)
        pooler = BertPooler(
            pcfg.get("pooling", "mixed"),
            encoder.config.hidden_size,
            encoder.config.num_hidden_layers,
            pcfg.get("dropout", 0.1),
        )
        pooler.load_state_dict(torch.load(shared_pooler_path, map_location=device))
        pooler.to(device).eval()
        print(f"Loaded shared pooler ({pcfg.get('pooling','mixed')} pooling) from {model_path}")
    else:
        print(f"No pooler found in {model_path} -- using CLS token")

    prefix = f"{chain_token} " if chain_token else ""
    n = len(cdr1_list)
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Extracting embeddings"):
            end = min(start + batch_size, n)
            texts = []
            for i in range(start, end):
                s1 = " ".join(cdr1_list[i])
                s2 = " ".join(cdr2_list[i])
                s3 = " ".join(cdr3_list[i])
                texts.append(
                    f"{prefix}{s1} {tokenizer.sep_token} {s2} {tokenizer.sep_token} {s3}"
                )
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc)
            if pooler is not None:
                emb = pooler(out.last_hidden_state, enc["attention_mask"])
            else:
                emb = out.last_hidden_state[:, 0, :]
            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_align_beta_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                  chain_token="[unused1]", batch_size=512, device=None):
    """
    Experiment 6 (CLIP) β-only inference: pooler_β -> align_β MLP.

    Encodes β-chain, pools, then applies the align_β projector to lift the embedding
    into the shared CLIP alignment subspace. Used for β-only Benchmark A under the
    Exp 6 framework.

    Args:
        model_path: TCRBertPairing checkpoint with align_beta.pt
        cdr1/2/3_list: β-chain CDR strings
        chain_token: β chain prefix (default "[unused1]")
        batch_size, device: standard

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()

    with open(os.path.join(model_path, "pairing_config.json")) as f:
        pcfg = json.load(f)

    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers
    pooling     = pcfg.get("pooling", "mixed")
    dropout     = pcfg.get("dropout", 0.1)

    pooler_beta = BertPooler(pooling, hidden_size, num_layers, dropout)
    shared_path = os.path.join(model_path, "pooler.pt")
    beta_path   = os.path.join(model_path, "pooler_beta.pt")
    if pcfg.get("shared_pooler", False) and os.path.exists(shared_path):
        pooler_beta.load_state_dict(torch.load(shared_path, map_location=device))
    else:
        pooler_beta.load_state_dict(torch.load(beta_path, map_location=device))
    pooler_beta.to(device).eval()

    align_beta = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    align_beta.load_state_dict(torch.load(
        os.path.join(model_path, "align_beta.pt"), map_location=device
    ))
    align_beta.to(device).eval()

    prefix = f"{chain_token} " if chain_token else ""
    need_hidden = pooler_beta.needs_all_hidden_states
    n = len(cdr1_list)
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="CLIP align_β embeddings"):
            end = min(start + batch_size, n)
            texts = [
                f"{prefix}{' '.join(cdr1_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr2_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr3_list[i])}"
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc, output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            emb = pooler_beta(out.last_hidden_state, enc["attention_mask"], all_hid)
            z = align_beta(emb)
            all_embs.append(z.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_chaintoken_beta_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                       chain_token="[unused1]", batch_size=512, device=None):
    """
    Experiment 7 (ChainToken) β-only inference via pair_projector with proxy_alpha:
        z_beta_only = pair_projector(cat[proxy_alpha, proxy_alpha - emb_beta, emb_beta])

    Loads pooler_β, pair_projector, and the learnable proxy_alpha (stand-in for missing α).
    Result lives in the same pair-space as paired z and the matching α-only embedding.

    Args:
        model_path: TCRBertPairing checkpoint with proxy_alpha.pt + pair_projector.pt

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()

    with open(os.path.join(model_path, "pairing_config.json")) as f:
        pcfg = json.load(f)

    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers
    pooling     = pcfg.get("pooling", "mixed")
    dropout     = pcfg.get("dropout", 0.1)

    pooler_beta = BertPooler(pooling, hidden_size, num_layers, dropout)
    shared_path = os.path.join(model_path, "pooler.pt")
    beta_path   = os.path.join(model_path, "pooler_beta.pt")
    if pcfg.get("shared_pooler", False) and os.path.exists(shared_path):
        pooler_beta.load_state_dict(torch.load(shared_path, map_location=device))
    else:
        pooler_beta.load_state_dict(torch.load(beta_path, map_location=device))
    pooler_beta.to(device).eval()

    pair_proj = nn.Sequential(
        nn.Linear(hidden_size * 3, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    pair_proj.load_state_dict(torch.load(
        os.path.join(model_path, "pair_projector.pt"), map_location=device
    ))
    pair_proj.to(device).eval()

    proxy_alpha = torch.load(
        os.path.join(model_path, "proxy_alpha.pt"), map_location=device
    )
    if isinstance(proxy_alpha, dict):
        # Backward compat: state_dict around a Parameter
        proxy_alpha = next(iter(proxy_alpha.values()))
    proxy_alpha = proxy_alpha.to(device).view(1, hidden_size)

    prefix = f"{chain_token} " if chain_token else ""
    need_hidden = pooler_beta.needs_all_hidden_states
    n = len(cdr1_list)
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="ChainToken β embeddings"):
            end = min(start + batch_size, n)
            texts = [
                f"{prefix}{' '.join(cdr1_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr2_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr3_list[i])}"
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc, output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            emb = pooler_beta(out.last_hidden_state, enc["attention_mask"], all_hid)
            B = emb.size(0)
            proxy = proxy_alpha.expand(B, -1)
            z = pair_proj(torch.cat([proxy, proxy - emb, emb], dim=-1))
            all_embs.append(z.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_bridge_beta_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                   chain_token="[unused1]", batch_size=512, device=None):
    """
    Experiment 8 (DoubleDist) β-only inference: pooler_β -> bridge_β MLP.

    Bridge projector h_β maps β-space directly into pair-space. Used at inference
    when only β is available.

    Args:
        model_path: TCRBertPairing checkpoint with bridge_beta.pt

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()

    with open(os.path.join(model_path, "pairing_config.json")) as f:
        pcfg = json.load(f)

    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers
    pooling     = pcfg.get("pooling", "mixed")
    dropout     = pcfg.get("dropout", 0.1)

    pooler_beta = BertPooler(pooling, hidden_size, num_layers, dropout)
    shared_path = os.path.join(model_path, "pooler.pt")
    beta_path   = os.path.join(model_path, "pooler_beta.pt")
    if pcfg.get("shared_pooler", False) and os.path.exists(shared_path):
        pooler_beta.load_state_dict(torch.load(shared_path, map_location=device))
    else:
        pooler_beta.load_state_dict(torch.load(beta_path, map_location=device))
    pooler_beta.to(device).eval()

    bridge_beta = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    bridge_beta.load_state_dict(torch.load(
        os.path.join(model_path, "bridge_beta.pt"), map_location=device
    ))
    bridge_beta.to(device).eval()

    prefix = f"{chain_token} " if chain_token else ""
    need_hidden = pooler_beta.needs_all_hidden_states
    n = len(cdr1_list)
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="DoubleDist bridge_β embeddings"):
            end = min(start + batch_size, n)
            texts = [
                f"{prefix}{' '.join(cdr1_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr2_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr3_list[i])}"
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc, output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            emb = pooler_beta(out.last_hidden_state, enc["attention_mask"], all_hid)
            z = bridge_beta(emb)
            all_embs.append(z.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_bridge_alpha_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                    chain_token="[unused0]", batch_size=512, device=None):
    """
    Experiment 8 (DoubleDist) alpha-only projected into pair-space:
        pooler_alpha -> bridge_alpha MLP.

    Mirror of extract_bridge_beta_embeddings but for the alpha chain. Used for the
    compositional test (h_alpha(alpha) + h_beta(beta) approx z_paired).

    Args:
        model_path: TCRBertPairing checkpoint with bridge_alpha.pt
        chain_token: alpha chain prefix (default "[unused0]" -- matches training)

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()

    with open(os.path.join(model_path, "pairing_config.json")) as f:
        pcfg = json.load(f)

    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers
    pooling     = pcfg.get("pooling", "mixed")
    dropout     = pcfg.get("dropout", 0.1)

    pooler_alpha = BertPooler(pooling, hidden_size, num_layers, dropout)
    shared_path = os.path.join(model_path, "pooler.pt")
    alpha_path  = os.path.join(model_path, "pooler_alpha.pt")
    if pcfg.get("shared_pooler", False) and os.path.exists(shared_path):
        pooler_alpha.load_state_dict(torch.load(shared_path, map_location=device))
    else:
        pooler_alpha.load_state_dict(torch.load(alpha_path, map_location=device))
    pooler_alpha.to(device).eval()

    bridge_alpha = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    bridge_alpha.load_state_dict(torch.load(
        os.path.join(model_path, "bridge_alpha.pt"), map_location=device
    ))
    bridge_alpha.to(device).eval()

    prefix = f"{chain_token} " if chain_token else ""
    need_hidden = pooler_alpha.needs_all_hidden_states
    n = len(cdr1_list)
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="DoubleDist bridge_alpha embeddings"):
            end = min(start + batch_size, n)
            texts = [
                f"{prefix}{' '.join(cdr1_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr2_list[i])} {tokenizer.sep_token} "
                f"{' '.join(cdr3_list[i])}"
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc, output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            emb = pooler_alpha(out.last_hidden_state, enc["attention_mask"], all_hid)
            z = bridge_alpha(emb)
            all_embs.append(z.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def _joint_pair_text(tokenizer, a1, a2, a3, b1, b2, b3):
    """Joint pair format with the real sep token between CDRs and between chains.
    Chain identity is carried by token_type_ids, not by string markers."""
    sep = tokenizer.sep_token
    a = f"{' '.join(a1)} {sep} {' '.join(a2)} {sep} {' '.join(a3)}"
    b = f"{' '.join(b1)} {sep} {' '.join(b2)} {sep} {' '.join(b3)}"
    return f"{a} {sep} {b}"


def _joint_single_text(tokenizer, c1, c2, c3):
    """Single-chain format (no chain marker; chain identity via token_type_ids)."""
    sep = tokenizer.sep_token
    return f"{' '.join(c1)} {sep} {' '.join(c2)} {sep} {' '.join(c3)}"


def _joint_single_vtoken_text(tokenizer, vtok, c3):
    """V-token single-chain format "[V:gene] | CDR3" (atomic V token in place of CDR1/CDR2).
    Mirrors TCRJointDataset._chain_str in vtoken mode."""
    sep = tokenizer.sep_token
    return f"{vtok} {sep} {' '.join(c3)}"


def _joint_pair_token_type_ids(tokenizer, a1, a2, a3, b1, b2, b3, max_len):
    """Build token_type_ids for the joint pair tokenisation matching _joint_pair_text:
    0s across [CLS, alpha-end], 1s from the middle real-SEP through beta + trailing [SEP],
    0 in padding tail."""
    sep = tokenizer.sep_token
    alpha_text = f"{' '.join(a1)} {sep} {' '.join(a2)} {sep} {' '.join(a3)}"
    beta_text  = f"{' '.join(b1)} {sep} {' '.join(b2)} {sep} {' '.join(b3)}"
    n_alpha = len(tokenizer.tokenize(alpha_text))
    n_beta  = len(tokenizer.tokenize(beta_text))
    type_ids = torch.zeros(max_len, dtype=torch.long)
    start_b = 1 + n_alpha
    end_b   = min(max_len, 1 + n_alpha + 1 + n_beta + 1)
    type_ids[start_b:end_b] = 1
    return type_ids


def _load_joint_model(model_path, device):
    """Load a TCRBertJoint checkpoint -> (encoder, pooler, max_len)."""
    with open(os.path.join(model_path, "joint_config.json")) as f:
        jcfg = json.load(f)
    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()
    pooler = BertPooler(
        jcfg.get("pooling", "mixed"),
        encoder.config.hidden_size,
        encoder.config.num_hidden_layers,
        jcfg.get("dropout", 0.1),
    )
    pooler.load_state_dict(torch.load(os.path.join(model_path, "pooler.pt"), map_location=device))
    pooler.to(device).eval()
    return encoder, pooler, jcfg.get("max_len", 128)


def extract_joint_paired_embeddings(model_path,
                                    alpha_cdr1, alpha_cdr2, alpha_cdr3,
                                    beta_cdr1, beta_cdr2, beta_cdr3,
                                    batch_size=256, device=None):
    """
    Exp 9 joint encoder: encode the full paired TCR as ONE sequence, mixed-pool -> z.

    Args:
        model_path: TCRBertJoint checkpoint (bert + pooler.pt + joint_config.json)

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _get_tokenizer()
    encoder, pooler, max_len = _load_joint_model(model_path, device)
    need_hidden = pooler.needs_all_hidden_states

    n = len(alpha_cdr1)
    all_z = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Joint paired embeddings"):
            end = min(start + batch_size, n)
            texts = [
                _joint_pair_text(tokenizer,
                                 alpha_cdr1[i], alpha_cdr2[i], alpha_cdr3[i],
                                 beta_cdr1[i],  beta_cdr2[i],  beta_cdr3[i])
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt")
            # token_type_ids: 0 across [CLS, alpha-end], 1 from middle real-SEP through beta + trailing [SEP].
            type_ids = torch.stack([
                _joint_pair_token_type_ids(tokenizer,
                                           alpha_cdr1[i], alpha_cdr2[i], alpha_cdr3[i],
                                           beta_cdr1[i],  beta_cdr2[i],  beta_cdr3[i],
                                           max_len)
                for i in range(start, end)
            ])
            enc = {k: v.to(device) for k, v in enc.items()}
            type_ids = type_ids.to(device)
            out = encoder(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"],
                          token_type_ids=type_ids,
                          output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            z = pooler(out.last_hidden_state, enc["attention_mask"], all_hid)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


def extract_joint_single_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                    chain="beta", batch_size=256, device=None):
    """
    Exp 9 joint encoder, single-chain inference: encode one chain (no chain marker;
    chain identity carried by token_type_ids) in the joint format -> mixed-pool -> z.
    Relies on chain-drop training keeping single-chain inputs in distribution.

    Args:
        model_path: TCRBertJoint checkpoint
        chain:      "alpha" (token_type_id 0) or "beta" (token_type_id 1)

    Returns:
        np.ndarray [N, hidden]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _get_tokenizer()
    encoder, pooler, max_len = _load_joint_model(model_path, device)
    need_hidden = pooler.needs_all_hidden_states
    type_id_value = 0 if chain == "alpha" else 1

    n = len(cdr1_list)
    all_z = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Joint {chain}-only embeddings"):
            end = min(start + batch_size, n)
            texts = [
                _joint_single_text(tokenizer, cdr1_list[i], cdr2_list[i], cdr3_list[i])
                for i in range(start, end)
            ]
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt")
            type_ids = torch.full_like(enc["input_ids"], type_id_value, dtype=torch.long)
            enc = {k: v.to(device) for k, v in enc.items()}
            type_ids = type_ids.to(device)
            out = encoder(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"],
                          token_type_ids=type_ids,
                          output_hidden_states=need_hidden)
            all_hid = out.hidden_states if need_hidden else None
            z = pooler(out.last_hidden_state, enc["attention_mask"], all_hid)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


# ================= PER-CHAIN JOINT FOUNDATION (per-chain RPE) =================
# The single-stage joint foundation uses per-chain relative positions, so it CANNOT go through the
# vanilla _load_joint_model / extract_joint_* path (a plain forward gives wrong attention). These
# helpers load it correctly (load_perchain_mlm on backbone/ + poolers/joint.pt) and reuse the model's
# own _encode, which threads apply_rel_idx + per-chain position_ids exactly as in training.

def _load_perchain_joint_model(model_path, device):
    """Load the per-chain joint foundation -> (TCRFoundationJoint in eval, joint_config dict).
    Layout: model_path/backbone (BertForMaskedLM perchain) + model_path/poolers/joint.pt + joint_config.json."""
    with open(os.path.join(model_path, "joint_config.json")) as f:
        jcfg = json.load(f)
    mlm = load_perchain_mlm(os.path.join(model_path, "backbone"))
    model = TCRFoundationJoint(
        bert=mlm,
        hidden_size=mlm.config.hidden_size,
        num_layers=mlm.config.num_hidden_layers,
        pooling=jcfg.get("pooling", "mixed"),
        dropout=jcfg.get("dropout", 0.1),
    )
    model.pooler.load_state_dict(
        torch.load(os.path.join(model_path, "poolers", "joint.pt"), map_location=device))
    model.to(device).eval()
    return model, jcfg


def _joint_pair_position_ids(tokenizer, a1, a2, a3, b1, b2, b3, max_len):
    """Per-chain RESET positions for the joint pair (mirrors TCRJointDataset._joint_position_ids):
    alpha positions 0..n_alpha (CLS + alpha tokens), beta RESET to 0.. at the chain boundary; pad 0.
    Uses the same whole-chain token counts that _joint_pair_token_type_ids uses."""
    sep = tokenizer.sep_token
    alpha_text = f"{' '.join(a1)} {sep} {' '.join(a2)} {sep} {' '.join(a3)}"
    beta_text  = f"{' '.join(b1)} {sep} {' '.join(b2)} {sep} {' '.join(b3)}"
    n_alpha = len(tokenizer.tokenize(alpha_text))
    n_beta  = len(tokenizer.tokenize(beta_text))
    pos = torch.zeros(max_len, dtype=torch.long)
    a_end = min(1 + n_alpha, max_len)
    pos[0:a_end] = torch.arange(a_end)
    b_start = 1 + n_alpha
    b_end   = min(max_len, 1 + n_alpha + 1 + n_beta + 1)
    if b_start < max_len:
        pos[b_start:b_end] = torch.arange(b_end - b_start)
    return pos


def _single_chain_position_ids(attention_mask):
    """Contiguous positions 0..span over a single chain's active tokens; pad stays 0
    (mirrors TCRJointDataset._single_chain_position_ids)."""
    span = int(attention_mask.sum())
    pos = torch.zeros_like(attention_mask, dtype=torch.long)
    pos[:span] = torch.arange(span)
    return pos


def extract_perchain_joint_paired_embeddings(model_path,
                                             alpha_cdr1, alpha_cdr2, alpha_cdr3,
                                             beta_cdr1, beta_cdr2, beta_cdr3,
                                             batch_size=256, device=None):
    """Per-chain joint foundation: encode the full paired TCR (per-chain positions) -> mixed-pool z.
    Returns np.ndarray [N, hidden]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _get_tokenizer()
    model, jcfg = _load_perchain_joint_model(model_path, device)
    max_len = jcfg.get("max_len_paired", 96)

    n = len(alpha_cdr1)
    all_z = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Perchain joint paired embeddings"):
            end = min(start + batch_size, n)
            texts, type_ids, pos_ids = [], [], []
            for i in range(start, end):
                args = (alpha_cdr1[i], alpha_cdr2[i], alpha_cdr3[i],
                        beta_cdr1[i],  beta_cdr2[i],  beta_cdr3[i])
                texts.append(_joint_pair_text(tokenizer, *args))
                type_ids.append(_joint_pair_token_type_ids(tokenizer, *args, max_len))
                pos_ids.append(_joint_pair_position_ids(tokenizer, *args, max_len))
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            ttid = torch.stack(type_ids).to(device)
            pos  = torch.stack(pos_ids).to(device)
            _, z = model._encode(ids, mask, None, ttid, pos)   # _encode threads apply_rel_idx + pooling
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


def extract_perchain_joint_single_embeddings(model_path, cdr1_list, cdr2_list, cdr3_list,
                                             chain="beta", batch_size=256, device=None):
    """Per-chain joint foundation, single-chain inference (per-chain positions, token_type 0/1) -> z.
    chain="alpha" (type 0) or "beta" (type 1). Returns np.ndarray [N, hidden]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _get_tokenizer()
    model, jcfg = _load_perchain_joint_model(model_path, device)
    max_len = jcfg.get("max_len_beta", 64)
    type_id_value = 0 if chain == "alpha" else 1

    n = len(cdr1_list)
    all_z = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Perchain joint {chain}-only embeddings"):
            end = min(start + batch_size, n)
            texts = [_joint_single_text(tokenizer, cdr1_list[i], cdr2_list[i], cdr3_list[i])
                     for i in range(start, end)]
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            ttid = torch.full_like(ids, type_id_value)
            pos  = torch.stack([_single_chain_position_ids(m) for m in enc["attention_mask"]]).to(device)
            _, z = model._encode(ids, mask, None, ttid, pos)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


def extract_wukevin_embeddings(cdr3_list, batch_size=512, device=None):
    """
    Extract wukevin/tcr-bert embeddings using the protocol from Nagano et al. 2024:
      - CDR3 sequence only (space-joined amino acids)
      - Average pool of amino acid token embeddings from layer 8 (1-indexed)

    Args:
        cdr3_list:  List of CDR3 strings
        batch_size: Inference batch size
        device:     torch.device (default: cuda if available)

    Returns:
        np.ndarray [N, hidden_size]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()
    mlm = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert")
    encoder = mlm.bert
    encoder.to(device).eval()

    n = len(cdr3_list)
    all_embs = []
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="wukevin embeddings"):
            end = min(start + batch_size, n)
            texts = [" ".join(cdr3_list[i]) for i in range(start, end)]
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            # hidden_states[0] = embedding layer, hidden_states[8] = layer 8 output
            out = encoder(**enc, output_hidden_states=True)
            hidden = out.hidden_states[8]  # [B, L, D]

            # Average pool over amino acid tokens only (exclude CLS, SEP, PAD)
            ids = enc["input_ids"]
            mask = enc["attention_mask"].float()
            mask[(ids == cls_id) | (ids == sep_id)] = 0.0
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            avg = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # [B, D]
            all_embs.append(avg.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_paired_embeddings(model_path,
                               alpha_cdr1, alpha_cdr2, alpha_cdr3,
                               beta_cdr1, beta_cdr2, beta_cdr3,
                               alpha_chain_token="[unused0]",
                               beta_chain_token="[unused1]",
                               batch_size=256, device=None):
    """
    Extract pair_projector(z) embeddings from a TCRBertPairing checkpoint.

    Encodes both alpha and beta chains through the shared BERT encoder, pools
    each with its dedicated pooler, then computes:
        z = pair_projector(cat([ea, ea-eb, eb]))  -- shape [N, hidden_size]

    Requires: pooler_alpha.pt + pooler_beta.pt (or pooler.pt), pair_projector.pt,
              pairing_config.json alongside the AutoModelForMaskedLM checkpoint.

    Args:
        model_path:          Path to the TCRBertPairing checkpoint directory
        alpha_cdr1/2/3:      Lists of CDR1/CDR2/CDR3 strings for the alpha chain
        beta_cdr1/2/3:       Lists of CDR1/CDR2/CDR3 strings for the beta chain
        alpha_chain_token:   Chain prefix token for alpha (default "[unused0]" -- matches training)
        beta_chain_token:    Chain prefix token for beta  (default "[unused1]")
        batch_size:          Inference batch size
        device:              torch.device (default: cuda if available)

    Returns:
        np.ndarray [N, hidden_size]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _get_tokenizer()

    config_path = os.path.join(model_path, "pairing_config.json")
    with open(config_path) as f:
        pcfg = json.load(f)

    mlm = AutoModelForMaskedLM.from_pretrained(model_path)
    encoder = mlm.bert
    encoder.to(device).eval()

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers
    pooling     = pcfg.get("pooling", "mixed")
    dropout     = pcfg.get("dropout", 0.1)

    pooler_alpha = BertPooler(pooling, hidden_size, num_layers, dropout)
    pooler_beta  = BertPooler(pooling, hidden_size, num_layers, dropout)

    shared_pooler_path = os.path.join(model_path, "pooler.pt")
    pooler_alpha_path  = os.path.join(model_path, "pooler_alpha.pt")
    pooler_beta_path   = os.path.join(model_path, "pooler_beta.pt")

    if pcfg.get("shared_pooler", False) and os.path.exists(shared_pooler_path):
        pooler_alpha.load_state_dict(torch.load(shared_pooler_path, map_location=device))
        pooler_beta = pooler_alpha
        print(f"Loaded shared pooler from {model_path}")
    else:
        pooler_alpha.load_state_dict(torch.load(pooler_alpha_path, map_location=device))
        pooler_beta.load_state_dict(torch.load(pooler_beta_path, map_location=device))
        print(f"Loaded pooler_alpha + pooler_beta from {model_path}")

    pooler_alpha.to(device).eval()
    if pooler_beta is not pooler_alpha:
        pooler_beta.to(device).eval()

    pair_proj = nn.Sequential(
        nn.Linear(hidden_size * 3, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    pair_proj.load_state_dict(torch.load(
        os.path.join(model_path, "pair_projector.pt"), map_location=device
    ))
    pair_proj.to(device).eval()

    need_hidden = pooler_alpha.needs_all_hidden_states

    prefix_a = f"{alpha_chain_token} " if alpha_chain_token else ""
    prefix_b = f"{beta_chain_token} " if beta_chain_token else ""

    n = len(alpha_cdr1)
    all_z = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Extracting paired embeddings"):
            end = min(start + batch_size, n)

            texts_a, texts_b = [], []
            for i in range(start, end):
                s1a = " ".join(alpha_cdr1[i])
                s2a = " ".join(alpha_cdr2[i])
                s3a = " ".join(alpha_cdr3[i])
                s1b = " ".join(beta_cdr1[i])
                s2b = " ".join(beta_cdr2[i])
                s3b = " ".join(beta_cdr3[i])
                texts_a.append(
                    f"{prefix_a}{s1a} {tokenizer.sep_token} {s2a} {tokenizer.sep_token} {s3a}"
                )
                texts_b.append(
                    f"{prefix_b}{s1b} {tokenizer.sep_token} {s2b} {tokenizer.sep_token} {s3b}"
                )

            enc_a = tokenizer(texts_a, padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            enc_b = tokenizer(texts_b, padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}

            out_a = encoder(**enc_a, output_hidden_states=need_hidden)
            out_b = encoder(**enc_b, output_hidden_states=need_hidden)

            all_hid_a = out_a.hidden_states if need_hidden else None
            all_hid_b = out_b.hidden_states if need_hidden else None

            ea = pooler_alpha(out_a.last_hidden_state, enc_a["attention_mask"], all_hid_a)
            eb = pooler_beta(out_b.last_hidden_state,  enc_b["attention_mask"], all_hid_b)

            z = pair_proj(torch.cat([ea, ea - eb, eb], dim=-1))
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0)


def sample_ref_indices(pmc_labels, max_ref_size, n_trials=100, seed=42):
    """
    Pre-sample shared reference indices for each trial x pMHC.
    Used for ref_size > 1; ref_size=k uses ref_indices[trial][pmc][:k].

    Args:
        pmc_labels:   array-like of N pMHC strings
        max_ref_size: maximum reference set size (e.g. 200)
        n_trials:     number of random splits
        seed:         base random seed (trial t uses seed+t)

    Returns:
        dict: {trial: {pmc: np.ndarray of shape [max_ref_size]}}
              Only includes pMHCs with len(pos_idx) > max_ref_size.
    """
    pmc_labels = np.asarray(pmc_labels)
    unique_pmcs = np.unique(pmc_labels)
    pmc_idx = {p: np.where(pmc_labels == p)[0] for p in unique_pmcs}

    ref_indices = {}
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        ref_indices[trial] = {}
        for p in unique_pmcs:
            pos_idx = pmc_idx[p]
            if len(pos_idx) > max_ref_size:
                ref_indices[trial][p] = rng.choice(pos_idx, size=max_ref_size, replace=False)
    return ref_indices


def knn_auroc(embs, pmc_labels, ref_sizes, n_trials=100, seed=42, ref_indices=None):
    """
    Nearest-neighbor AUROC for TCR pMHC specificity (SCEPTR / Nagano et al. 2024 protocol).

    For ref_size == 1: exhaustive -- every TCR in each pMHC serves as reference once.
    For ref_size > 1:  100 random splits (shared across models via ref_indices).

    Nested sampling: ref_indices[trial][pmc] holds max_ref_size indices; ref_size=k
    uses the first k of those, so splits are consistent across ref_sizes.

    Args:
        embs:        np.ndarray [N, D] -- sequence embeddings (need not be normalized)
        pmc_labels:  array-like of N pMHC strings
        ref_sizes:   list of int -- reference set sizes, e.g. [1, 2, 5, 10, 20, 50, 100, 200]
        n_trials:    number of random splits for ref_size > 1 (default 100)
        seed:        base random seed
        ref_indices: pre-sampled indices from sample_ref_indices(); generated internally if None

    Returns:
        dict: {ref_size: {"mean": float, "std": float, "per_pmc": {pmc: float}}}
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.clip(norms, 1e-8, None)

    pmc_labels = np.asarray(pmc_labels)
    unique_pmcs = np.unique(pmc_labels)
    pmc_idx = {p: np.where(pmc_labels == p)[0] for p in unique_pmcs}

    non_one_sizes = [s for s in ref_sizes if s > 1]
    max_ref = max(non_one_sizes) if non_one_sizes else 1

    if ref_indices is None and non_one_sizes:
        ref_indices = sample_ref_indices(pmc_labels, max_ref, n_trials, seed)

    results = {}

    for ref_size in tqdm(ref_sizes, desc="ref sizes"):
        trial_means = []
        pmc_auroc_acc = {p: [] for p in unique_pmcs}

        if ref_size == 1:
            # Exhaustive: every TCR in each pMHC serves as reference once
            neg_idx_global = {
                p: np.concatenate([pmc_idx[q] for q in unique_pmcs if q != p])
                for p in unique_pmcs
            }
            pmc_aurocs = []
            for p in unique_pmcs:
                pos_idx = pmc_idx[p]
                neg_idx = neg_idx_global[p]
                if len(pos_idx) < 2:
                    continue

                # Vectorized: compute similarities from every pos to all sequences
                sims_pos_pos = embs_norm[pos_idx] @ embs_norm[pos_idx].T  # [n_p, n_p]
                sims_pos_neg = embs_norm[pos_idx] @ embs_norm[neg_idx].T  # [n_p, n_neg]

                aurocs_p = []
                for r in range(len(pos_idx)):
                    pos_scores = np.delete(sims_pos_pos[r], r)  # remaining positives
                    neg_scores = sims_pos_neg[r]
                    scores = np.concatenate([pos_scores, neg_scores])
                    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int8)
                    if labels.sum() == 0 or (1 - labels).sum() == 0:
                        continue
                    aurocs_p.append(float(roc_auc_score(labels, scores)))

                if aurocs_p:
                    mean_p = float(np.mean(aurocs_p))
                    pmc_aurocs.append(mean_p)
                    pmc_auroc_acc[p].append(mean_p)

            # k=1 exhaustive has no trial variance; report as single "trial"
            if pmc_aurocs:
                trial_means.append(float(np.mean(pmc_aurocs)))

        else:
            for trial in range(n_trials):
                pmc_aurocs = []
                for p in unique_pmcs:
                    pos_idx = pmc_idx[p]
                    if p not in ref_indices[trial]:
                        continue

                    ref_idx = ref_indices[trial][p][:ref_size]
                    ref_embs = embs_norm[ref_idx]  # [ref_size, D]

                    remaining_pos = np.setdiff1d(pos_idx, ref_idx)
                    neg_idx = np.concatenate([pmc_idx[q] for q in unique_pmcs if q != p])
                    test_idx = np.concatenate([remaining_pos, neg_idx])
                    is_pos = np.array(
                        [1] * len(remaining_pos) + [0] * len(neg_idx), dtype=np.int8
                    )

                    test_embs = embs_norm[test_idx]       # [n_test, D]
                    sims = test_embs @ ref_embs.T         # [n_test, ref_size]
                    scores = sims.max(axis=1)             # [n_test]

                    if is_pos.sum() == 0 or (1 - is_pos).sum() == 0:
                        continue

                    auroc = float(roc_auc_score(is_pos, scores))
                    pmc_aurocs.append(auroc)
                    pmc_auroc_acc[p].append(auroc)

                if pmc_aurocs:
                    trial_means.append(float(np.mean(pmc_aurocs)))

        results[ref_size] = {
            "mean":    float(np.mean(trial_means)) if trial_means else float("nan"),
            "std":     float(np.std(trial_means))  if trial_means else float("nan"),
            "per_pmc": {p: float(np.mean(v)) for p, v in pmc_auroc_acc.items() if v},
        }

    return results


def _build_td_cell_df(tcr_df, chains):
    """Build a tcrdist3-compatible cell_df with allele-normalized V genes."""
    def _norm(g):
        return g.str.replace(r"\*\d+$", "", regex=True) + "*01"

    cols = {
        "cdr3_b_aa": tcr_df["cdr3b"].values,
        "v_b_gene":  _norm(tcr_df["vb"]).values,
        "count":     1,
    }
    if "alpha" in chains:
        cols["cdr3_a_aa"] = tcr_df["cdr3a"].values
        cols["v_a_gene"]  = _norm(tcr_df["va"]).values
    return pd.DataFrame(cols)


def compute_tcrdist_matrix(tcr_df, chains=("beta",), organism="human"):
    """
    Compute the full [N, N] TCRdist matrix via a single TCRrep initialization.

    A single init lets tcrdist3 infer cdr1/cdr2/pmhc columns from the V gene for both
    the rows and columns of the matrix (fixes the rect-distance KeyError where df2 was
    never initialized). deduplicate=False preserves row order so matrix indices match
    tcr_df row indices.

    Args:
        tcr_df:   DataFrame with cdr3b, vb (+ cdr3a, va if "alpha" in chains)
        chains:   tuple/list, e.g. ("beta",) or ("alpha", "beta")
        organism: "human" or "mouse"

    Returns:
        np.ndarray [N, N] -- combined TCRdist (pw_alpha + pw_beta for paired)
    """
    try:
        from tcrdist.repertoire import TCRrep
    except ImportError:
        raise ImportError("tcrdist3 is required: pip install tcrdist3")

    chains = list(chains)
    cell_df = _build_td_cell_df(tcr_df, chains)
    # For N > 10000, tcrdist3 skips the automatic compute at init (size guard) and
    # pw_* attributes are never created. Build with compute_distances=False and call
    # compute_distances() explicitly to force the full dense [N, N] matrices.
    tr = TCRrep(cell_df=cell_df, organism=organism, chains=chains,
                compute_distances=False, deduplicate=False)
    tr.cpus = 1  # Windows-safe (avoid multiprocessing issues)
    tr.compute_distances()

    if "alpha" in chains and "beta" in chains:
        return np.asarray(tr.pw_alpha) + np.asarray(tr.pw_beta)
    elif "beta" in chains:
        return np.asarray(tr.pw_beta)
    elif "alpha" in chains:
        return np.asarray(tr.pw_alpha)
    raise ValueError(f"Unsupported chains: {chains}")


def tcrdist_knn_auroc(tcr_df, pmc_labels, ref_sizes, n_trials=100, seed=42,
                      ref_indices=None, organism="human", chains=("beta",)):
    """
    TCRdist-based kNN AUROC for TCR pMHC specificity (Nagano et al. 2024 protocol).

    Computes the full [N, N] TCRdist matrix once (via compute_tcrdist_matrix), then runs
    the same kNN protocol as knn_auroc with score = -min(distance to reference set).

    Args:
        tcr_df:      DataFrame with cdr3b, vb (+ cdr3a, va if paired)
        pmc_labels:  array-like of N pMHC strings
        ref_sizes:   list of int
        n_trials:    number of random splits for ref_size > 1
        seed:        base random seed
        ref_indices: pre-sampled from sample_ref_indices(); generated internally if None
        organism:    "human" or "mouse"
        chains:      ("beta",) or ("alpha", "beta")

    Returns:
        dict: {ref_size: {"mean": float, "std": float, "per_pmc": {pmc: float}}}
    """
    pmc_labels = np.asarray(pmc_labels)
    unique_pmcs = np.unique(pmc_labels)
    pmc_idx = {p: np.where(pmc_labels == p)[0] for p in unique_pmcs}

    non_one_sizes = [s for s in ref_sizes if s > 1]
    max_ref = max(non_one_sizes) if non_one_sizes else 1
    if ref_indices is None and non_one_sizes:
        ref_indices = sample_ref_indices(pmc_labels, max_ref, n_trials, seed)

    print(f"Computing TCRdist matrix (chains={tuple(chains)}, N={len(tcr_df)})...")
    dist = compute_tcrdist_matrix(tcr_df, chains=chains, organism=organism)

    results = {}
    for ref_size in tqdm(ref_sizes, desc="ref sizes"):
        trial_means = []
        pmc_auroc_acc = {p: [] for p in unique_pmcs}

        if ref_size == 1:
            # Exhaustive: every TCR of a pMHC serves as reference once
            for p in unique_pmcs:
                pos_idx = pmc_idx[p]
                neg_idx = np.concatenate([pmc_idx[q] for q in unique_pmcs if q != p])
                if len(pos_idx) < 2:
                    continue
                d_pos_pos = dist[np.ix_(pos_idx, pos_idx)]   # [n_p, n_p]
                d_pos_neg = dist[np.ix_(pos_idx, neg_idx)]   # [n_p, n_neg]
                aurocs_p = []
                for r in range(len(pos_idx)):
                    pos_scores = -np.delete(d_pos_pos[r], r)
                    neg_scores = -d_pos_neg[r]
                    scores = np.concatenate([pos_scores, neg_scores])
                    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int8)
                    if labels.sum() == 0 or (1 - labels).sum() == 0:
                        continue
                    aurocs_p.append(float(roc_auc_score(labels, scores)))
                if aurocs_p:
                    mean_p = float(np.mean(aurocs_p))
                    pmc_auroc_acc[p].append(mean_p)
            pmc_means = [np.mean(v) for v in pmc_auroc_acc.values() if v]
            if pmc_means:
                trial_means.append(float(np.mean(pmc_means)))

        else:
            for trial in range(n_trials):
                pmc_aurocs = []
                for p in unique_pmcs:
                    pos_idx = pmc_idx[p]
                    if p not in ref_indices[trial]:
                        continue
                    ref_idx = ref_indices[trial][p][:ref_size]
                    remaining_pos = np.setdiff1d(pos_idx, ref_idx)
                    neg_idx = np.concatenate([pmc_idx[q] for q in unique_pmcs if q != p])
                    test_idx = np.concatenate([remaining_pos, neg_idx])
                    is_pos = np.array(
                        [1] * len(remaining_pos) + [0] * len(neg_idx), dtype=np.int8
                    )
                    scores = -dist[np.ix_(test_idx, ref_idx)].min(axis=1)
                    if is_pos.sum() == 0 or (1 - is_pos).sum() == 0:
                        continue
                    auroc = float(roc_auc_score(is_pos, scores))
                    pmc_aurocs.append(auroc)
                    pmc_auroc_acc[p].append(auroc)
                if pmc_aurocs:
                    trial_means.append(float(np.mean(pmc_aurocs)))

        results[ref_size] = {
            "mean":    float(np.mean(trial_means)) if trial_means else float("nan"),
            "std":     float(np.std(trial_means))  if trial_means else float("nan"),
            "per_pmc": {p: float(np.mean(v)) for p, v in pmc_auroc_acc.items() if v},
        }

    return results
