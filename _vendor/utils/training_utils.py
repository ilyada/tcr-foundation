"""
Training loops and evaluation utilities for TCR-BERT training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm


# ================= TRAINING FUNCTIONS =================

def train_mlm_epoch(model, loader, optimizer, scheduler, device, exp, global_step: int,
                    property_head=None, property_alpha: float = 0.5,
                    prototype_head=None) -> tuple:
    """
    Train one epoch of Masked Language Modeling with optional auxiliary objectives.

    Three objectives can run simultaneously on the CLS token:
        1. MLM          — always active; reconstructs masked amino acids
        2. Property     — optional; predicts physicochemical properties (MSE loss)
        3. V gene proto — optional; cosine prototype loss for V gene identity

    Loss combination (property_alpha = MLM weight when auxiliaries are present):
        If property only:   loss = property_alpha * MLM + (1-property_alpha) * prop
        If prototype only:  loss = property_alpha * MLM + (1-property_alpha) * proto
        If both:            loss = property_alpha * MLM + (1-property_alpha)/2 * prop
                                                        + (1-property_alpha)/2 * proto
        If neither:         loss = MLM

    Args:
        model:          MLM model (AutoModelForMaskedLM)
        loader:         DataLoader for training data
        optimizer:      Optimizer instance
        scheduler:      Learning rate scheduler
        device:         Training device
        exp:            Comet ML experiment (or None)
        global_step:    Current global step count
        property_head:  PropertyPredictionHead module (or None to disable)
        property_alpha: Weight for MLM when auxiliary objectives are present (default 0.5)
        prototype_head: VGenePrototypes module (or None to disable)

    Returns:
        Tuple of (average_epoch_loss, updated_global_step)
    """
    model.train()
    if property_head  is not None: property_head.train()
    if prototype_head is not None: prototype_head.train()

    has_aux = (property_head is not None) or (prototype_head is not None)
    epoch_loss       = 0.0
    epoch_mlm_loss   = 0.0
    epoch_prop_loss  = 0.0
    epoch_proto_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training MLM")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Pop auxiliary targets before forwarding through MLM model
        property_targets = batch.pop("property_targets", None)
        v_gene_ids       = batch.pop("v_gene_ids", None)
        n_donors_batch       = batch.pop("n_donors", None)
        n_convergence_batch  = batch.pop("n_convergence", None)

        outputs  = model(**batch, output_hidden_states=has_aux)
        mlm_loss = outputs.loss

        prop_loss  = None
        proto_loss = None

        if has_aux:
            cls_emb = outputs.hidden_states[-1][:, 0, :]  # [B, hidden_size]

            if property_head is not None and property_targets is not None:
                prop_preds = property_head(cls_emb)
                mask = ~torch.isnan(property_targets)
                prop_loss = F.mse_loss(prop_preds[mask], property_targets[mask]) \
                            if mask.any() else torch.tensor(0.0, device=device)

            if prototype_head is not None and v_gene_ids is not None:
                proto_loss = prototype_head(cls_emb, v_gene_ids)

        # Combine losses: MLM at property_alpha weight, auxiliaries split equally
        n_aux = (prop_loss is not None) + (proto_loss is not None)
        if n_aux == 0:
            loss = mlm_loss
        else:
            aux_weight = (1 - property_alpha) / n_aux
            loss = property_alpha * mlm_loss
            if prop_loss  is not None: loss = loss + aux_weight * prop_loss
            if proto_loss is not None: loss = loss + aux_weight * proto_loss

        optimizer.zero_grad()
        loss.backward()

        params = list(model.parameters())
        if property_head  is not None: params += list(property_head.parameters())
        if prototype_head is not None: params += list(prototype_head.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss       += loss.item()
        epoch_mlm_loss   += mlm_loss.item()
        if prop_loss  is not None: epoch_prop_loss  += prop_loss.item()
        if proto_loss is not None: epoch_proto_loss += proto_loss.item()
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}", "mlm": f"{mlm_loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
        if prop_loss  is not None: postfix["prop"]  = f"{prop_loss.item():.4f}"
        if proto_loss is not None: postfix["proto"] = f"{proto_loss.item():.4f}"
        pbar.set_postfix(postfix)

        if exp:
            exp.log_metric("mlm_loss_step", mlm_loss.item(), step=global_step)
            exp.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)
            if prop_loss  is not None:
                exp.log_metric("property_loss_step",  prop_loss.item(),  step=global_step)
            if proto_loss is not None:
                exp.log_metric("prototype_loss_step", proto_loss.item(), step=global_step)
            if n_aux > 0:
                exp.log_metric("total_pretrain_loss_step", loss.item(), step=global_step)

            if n_donors_batch is not None:
                with torch.no_grad():
                    logits  = outputs.logits                         # [B, L, vocab]
                    lbl     = batch["labels"]                        # [B, L]
                    per_tok = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        lbl.reshape(-1),
                        reduction="none", ignore_index=-100
                    ).reshape(lbl.shape)                             # [B, L]
                    valid       = (lbl != -100)
                    per_seq_loss = (per_tok * valid).sum(1) / valid.sum(1).clamp(min=1)
                psl = -per_seq_loss.cpu().numpy()  # pseudo-ll proxy: higher = more confident
                log_nd = torch.log1p(n_donors_batch).cpu().numpy()
                rho_nd, _ = spearmanr(psl, log_nd)
                if not np.isnan(rho_nd):
                    exp.log_metric("rho_mlm_log_donors_step", float(rho_nd), step=global_step)
                if n_convergence_batch is not None:
                    log_nc = torch.log1p(n_convergence_batch).cpu().numpy()
                    rho_nc, _ = spearmanr(psl, log_nc)
                    if not np.isnan(rho_nc):
                        exp.log_metric("rho_mlm_log_convg_step", float(rho_nc), step=global_step)

        global_step += 1

    n = max(num_batches, 1)
    avg_loss       = epoch_loss       / n
    avg_mlm_loss   = epoch_mlm_loss   / n
    avg_prop_loss  = epoch_prop_loss  / n if property_head  is not None else None
    avg_proto_loss = epoch_proto_loss / n if prototype_head is not None else None
    return avg_loss, avg_mlm_loss, avg_prop_loss, avg_proto_loss, global_step


def evaluate_pseudo_likelihood(model, sequences: dict, tokenizer, device,
                                max_len: int = 64, inner_batch: int = 256,
                                chain_token: str = "",
                                force_token_type_id: int = None,
                                perchain_attns=None, max_pos: int = None) -> dict:
    """
    Compute masked pseudo-log-likelihood for each sequence in hold-out and
    return Spearman rho with log(n_donors) and log(n_multiplicity).

    For each sequence, every non-special token position is masked once;
    log P(correct_token | masked context) is averaged over positions to give
    the per-sequence pseudo-log-likelihood (higher = model is more confident
    = sequence is more "typical" = higher Ppost).

    Args:
        model:        MLM model (AutoModelForMaskedLM), must be in eval mode.
        sequences:    Dict with keys cdr1, cdr2, cdr3, n_donors (required),
                      and optionally n_multiplicity. All lists/arrays of equal length.
        tokenizer:    HuggingFace tokenizer.
        device:       Torch device.
        max_len:      Tokenizer max length (must match training).
        inner_batch:  GPU batch size for masked-variant forward passes.
        chain_token:  Optional chain prefix token (e.g. "[unused1]" for beta).
                      Must match the token used during pretraining.
        force_token_type_id: If set (0 or 1), fill token_type_ids with this value
                      over all attention-active positions (used for joint-encoder
                      single-chain PLL: 0 for alpha, 1 for beta). When None,
                      fall back to the legacy heuristic of marking positions
                      after the 2nd sep as type_id=1 (the original two-chain
                      string-marker layout).

    Returns:
        Dict with keys:
            rho_nd          Spearman rho(pseudo_ll, log(n_donors))
            rho_nc          Spearman rho(pseudo_ll, log(n_convergence)), or NaN
            mean_pseudo_ll  Mean pseudo-log-likelihood across sequences
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    n_donors      = sequences["n_donors"]
    n_convergence = sequences.get("n_convergence", None)

    all_pseudo_ll = []
    prefix = f"{chain_token} " if chain_token else ""

    with torch.no_grad():
        for i in range(len(sequences["cdr3"])):
            s1 = " ".join(sequences["cdr1"][i])
            s2 = " ".join(sequences["cdr2"][i])
            s3 = " ".join(sequences["cdr3"][i])
            text = f"{prefix}{s1} {tokenizer.sep_token} {s2} {tokenizer.sep_token} {s3}"

            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            base_ids = enc["input_ids"].squeeze(0)          # [L]
            attn     = enc["attention_mask"].squeeze(0)     # [L]
            ttids    = torch.zeros_like(base_ids)

            if force_token_type_id is not None:
                ttids[attn.bool()] = int(force_token_type_id)
            else:
                sep_pos = (base_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_pos) >= 2:
                    ttids[sep_pos[1] + 1:] = 1

            # Per-chain models need explicit single-chain position_ids (0..span over active tokens)
            # + apply_rel_idx before each forward (the absolute position table is unused; the
            # per-chain relative signal lives in the attention). None -> vanilla path (unchanged).
            pos_ids = None
            if perchain_attns is not None:
                span = int(attn.sum())
                pos_ids = torch.zeros_like(base_ids)
                pos_ids[:span] = torch.arange(span)

            special = torch.tensor(
                tokenizer.get_special_tokens_mask(base_ids.tolist(),
                                                  already_has_special_tokens=True),
                dtype=torch.bool
            )
            real_pos = (~special & attn.bool()).nonzero(as_tuple=True)[0]  # positions to mask
            L_real = len(real_pos)
            if L_real == 0:
                all_pseudo_ll.append(0.0)
                continue

            # Build L_real masked copies: copy i has position real_pos[i] set to [MASK]
            ids_expanded  = base_ids.unsqueeze(0).expand(L_real, -1).clone()   # [L_real, L]
            attn_expanded = attn.unsqueeze(0).expand(L_real, -1)
            ttids_expanded = ttids.unsqueeze(0).expand(L_real, -1)
            pos_expanded = (pos_ids.unsqueeze(0).expand(L_real, -1)
                            if pos_ids is not None else None)

            for k, pos in enumerate(real_pos):
                ids_expanded[k, pos] = mask_id

            # Forward in inner_batch chunks
            log_probs = []
            for start in range(0, L_real, inner_batch):
                end = min(start + inner_batch, L_real)
                ti = ids_expanded[start:end].to(device)
                am = attn_expanded[start:end].to(device)
                tt = ttids_expanded[start:end].to(device)
                if perchain_attns is not None:
                    from utils.perchain_attention import apply_rel_idx
                    pi = pos_expanded[start:end].to(device)
                    apply_rel_idx(perchain_attns, pi, tt, max_pos)   # set rel_idx for this chunk
                    out = model(input_ids=ti, attention_mask=am, token_type_ids=tt, position_ids=pi)
                else:
                    out = model(input_ids=ti, attention_mask=am, token_type_ids=tt)
                logits_chunk = out.logits  # [chunk, L, vocab]
                lp = torch.log_softmax(logits_chunk, dim=-1)
                for j, pos in enumerate(real_pos[start:end]):
                    correct = base_ids[pos].item()
                    log_probs.append(lp[j, pos, correct].item())

            all_pseudo_ll.append(float(np.mean(log_probs)))

    pseudo_ll = np.array(all_pseudo_ll)
    log_nd = np.log1p(np.array(n_donors, dtype=float))
    rho_nd, _ = spearmanr(pseudo_ll, log_nd)
    r_nd,   _ = pearsonr(pseudo_ll, log_nd)

    rho_nc = float("nan")
    if n_convergence is not None:
        log_nc = np.log1p(np.array(n_convergence, dtype=float))
        rho_nc, _ = spearmanr(pseudo_ll, log_nc)

    rho_pd = float("nan")
    p_data = sequences.get("p_data", None)
    if p_data is not None:
        log_pd = np.log(np.array(p_data, dtype=float) + 1e-10)
        rho_pd, _ = spearmanr(pseudo_ll, log_pd)

    return {
        "r_nd":           float(r_nd)   if not np.isnan(r_nd)   else 0.0,
        "rho_nd":         float(rho_nd) if not np.isnan(rho_nd) else 0.0,
        "rho_nc":         float(rho_nc) if not np.isnan(rho_nc) else 0.0,
        "rho_pd":         float(rho_pd) if not np.isnan(rho_pd) else 0.0,
        "mean_pseudo_ll": float(pseudo_ll.mean()),
        "pseudo_ll":      pseudo_ll,
    }


def train_causal_epoch(model, loader, optimizer, scheduler, device, exp,
                       global_step: int) -> tuple:
    """
    Train one epoch of autoregressive (causal) language modeling.

    Args:
        model:       GPT-style causal LM (AutoModelForCausalLM)
        loader:      DataLoader yielding input_ids, attention_mask, labels
        optimizer:   Optimizer instance
        scheduler:   Learning rate scheduler
        device:      Training device
        exp:         Comet ML experiment (or None)
        global_step: Current global step count

    Returns:
        Tuple of (average_epoch_loss, updated_global_step)
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training AR")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("n_donors", None)
        batch.pop("n_convergence", None)

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if exp:
            exp.log_metric("ar_loss_step", loss.item(), step=global_step)
            exp.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)

        global_step += 1

    return epoch_loss / max(num_batches, 1), global_step


def evaluate_ar_likelihood(model, sequences: dict, tokenizer, device,
                            max_len: int = 64, batch_size: int = 256) -> dict:
    """
    Compute autoregressive log-likelihood for each sequence in hold-out and
    return Spearman rho with log(n_donors) and log(n_multiplicity).

    Single forward pass per batch; all positions computed in parallel via causal
    mask. LL per sequence = mean log P(xᵢ | x_{<i}) over amino acid positions
    (special tokens excluded to match BERT pseudo-likelihood convention).

    Args:
        model:      Causal LM (AutoModelForCausalLM), must be in eval mode.
        sequences:  Dict with keys cdr1, cdr2, cdr3, n_donors (required),
                    and optionally n_multiplicity, p_data.
        tokenizer:  HuggingFace tokenizer.
        device:     Torch device.
        max_len:    Tokenizer max length (must match training).
        batch_size: Number of sequences per GPU forward pass.

    Returns:
        Dict with the same keys as evaluate_pseudo_likelihood:
            r_nd        Pearson r(ar_ll, log(n_donors))
            rho_nd      Spearman rho(ar_ll, log(n_donors))
            rho_nc      Spearman rho(ar_ll, log(n_convergence)), or NaN
            rho_pd      Spearman rho(ar_ll, log(p_data)), or NaN
            mean_ar_ll  Mean AR log-likelihood across sequences
            ar_ll       numpy array of per-sequence scores
    """
    model.eval()
    n_donors      = sequences["n_donors"]
    n_convergence = sequences.get("n_convergence", None)
    p_data = sequences.get("p_data", None)
    special_ids = set(tokenizer.all_special_ids)
    all_ll = []

    with torch.no_grad():
        n = len(sequences["cdr3"])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids_list = []
            batch_masks_list = []

            for i in range(start, end):
                s1 = " ".join(sequences["cdr1"][i])
                s2 = " ".join(sequences["cdr2"][i])
                s3 = " ".join(sequences["cdr3"][i])
                text = f"{s1} {tokenizer.sep_token} {s2} {tokenizer.sep_token} {s3}"
                enc = tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt"
                )
                batch_ids_list.append(enc["input_ids"].squeeze(0))
                batch_masks_list.append(enc["attention_mask"].squeeze(0))

            ids = torch.stack(batch_ids_list).to(device)    # [B, L]
            masks = torch.stack(batch_masks_list).to(device) # [B, L]

            logits = model(input_ids=ids, attention_mask=masks).logits  # [B, L, V]
            log_probs = torch.log_softmax(logits, dim=-1)                # [B, L, V]

            # log P(ids[:, k] | ids[:, <k]) = log_probs[:, k-1, ids[:, k]]
            # targets[b, j] = ids[b, j+1] for j in 0..L-2
            targets = ids[:, 1:]                                         # [B, L-1]
            gathered = torch.gather(
                log_probs[:, :-1, :], dim=2, index=targets.unsqueeze(-1)
            ).squeeze(-1)                                                # [B, L-1]

            special_mask = torch.zeros_like(targets, dtype=torch.bool)
            for sp_id in special_ids:
                special_mask |= (targets == sp_id)

            for b_idx in range(ids.shape[0]):
                valid = ~special_mask[b_idx]
                if valid.any():
                    all_ll.append(gathered[b_idx][valid].mean().item())
                else:
                    all_ll.append(0.0)

    ar_ll = np.array(all_ll)
    log_nd = np.log1p(np.array(n_donors, dtype=float))
    rho_nd, _ = spearmanr(ar_ll, log_nd)
    r_nd,   _ = pearsonr(ar_ll, log_nd)

    rho_nc = float("nan")
    if n_convergence is not None:
        log_nc = np.log1p(np.array(n_convergence, dtype=float))
        rho_nc, _ = spearmanr(ar_ll, log_nc)

    rho_pd = float("nan")
    if p_data is not None:
        log_pd = np.log(np.array(p_data, dtype=float) + 1e-10)
        rho_pd, _ = spearmanr(ar_ll, log_pd)

    return {
        "r_nd":       float(r_nd)   if not np.isnan(r_nd)   else 0.0,
        "rho_nd":     float(rho_nd) if not np.isnan(rho_nd) else 0.0,
        "rho_nc":     float(rho_nc) if not np.isnan(rho_nc) else 0.0,
        "rho_pd":     float(rho_pd) if not np.isnan(rho_pd) else 0.0,
        "mean_ar_ll": float(ar_ll.mean()),
        "ar_ll":      ar_ll,
    }


def train_classifier_epoch(model, loader, optimizer, scheduler, device, exp,
                          global_step: int, phase: str = "finetune") -> tuple:
    """
    Train one classification epoch.

    Args:
        model: TCRBertClassifier
        loader: DataLoader for training data
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler (or None for warmup)
        device: Training device
        exp: Comet ML experiment (or None)
        global_step: Current global step count
        phase: "warmup" or "finetune" (for logging)

    Returns:
        Tuple of (average_epoch_loss, updated_global_step)

    Side Effects:
        - Updates model parameters
        - Logs metrics to Comet ML
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Training {phase}")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss  += loss.item()
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}"}
        if scheduler:
            lrs = scheduler.get_last_lr()
            postfix["lr_bert"] = f"{lrs[0]:.2e}"
            postfix["lr_head"] = f"{lrs[1]:.2e}" if len(lrs) == 2 else None
        else:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        pbar.set_postfix({k: v for k, v in postfix.items() if v is not None})

        if exp:
            exp.log_metric("train_loss_step", loss.item(), step=global_step)

            # Log learning rates
            if scheduler:
                lrs = scheduler.get_last_lr()
                if len(lrs) == 2:  # Dual learning rates
                    exp.log_metric("lr_bert", lrs[0], step=global_step)
                    exp.log_metric("lr_head", lrs[1], step=global_step)
                else:
                    exp.log_metric("lr_head", lrs[0], step=global_step)
            else:
                # Warmup phase - single LR
                exp.log_metric("lr_head", optimizer.param_groups[0]["lr"], step=global_step)

        global_step += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


# ================= EVALUATION FUNCTIONS =================

def predict_probs(model, loader, device) -> tuple:
    """
    Generate prediction probabilities.

    Args:
        model: TCRBertClassifier
        loader: DataLoader
        device: Device

    Returns:
        Tuple of (probabilities, true_labels)
        - probabilities: numpy array of shape (N,) with class 1 probabilities
        - true_labels: numpy array of shape (N,) with ground truth labels
    """
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            p = torch.softmax(outputs["logits"], dim=1)[:, 1]  # Probability of class 1
            probs.extend(p.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    return np.array(probs), np.array(labels)


def evaluate_classifier(model, loader, device) -> dict:
    """
    Evaluate classifier and return metrics.

    Args:
        model: TCRBertClassifier
        loader: DataLoader for evaluation data
        device: Evaluation device

    Returns:
        Dictionary with keys:
            - bal_acc: Balanced accuracy
            - auc: ROC AUC score

    Note:
        Uses labels from the loader (not external labels) to ensure
        predictions and labels are in the same order even if loader is shuffled.
    """
    model.eval()

    # Generate predictions and get labels from loader (same order)
    probs, true_labels = predict_probs(model, loader, device)
    preds = (probs >= 0.5).astype(int)

    # Calculate metrics
    bal_acc = balanced_accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs)

    return {
        "bal_acc": bal_acc,
        "auc": auc
    }


# ================= SONIA LOSS FUNCTIONS =================

def sonia_loss(scores_data: torch.Tensor, scores_generated: torch.Tensor,
               gamma: float = 0.0) -> torch.Tensor:
    """
    soNNia maximum likelihood selection loss (Isacchini et al. 2021).

    The soNNia model factorises P_post(σ) = Q(σ) × P_gen(σ) / Z.
    Maximising the log-likelihood over productive sequences gives:

        L = E_data[log Q(σ)]  −  log E_gen[Q(σ)]
          = mean(log Q over productive)  −  log(mean(Q over generated))

    We MINIMISE the negation of this, with an optional gauge-fixing term
    matching the authors' implementation (Isacchini et al. 2021, SI §5):

        loss = −mean(log Q_data) + log Z + γ · (Z − 1)²

    The γ · (Z − 1)² term penalises deviations of Z from 1, pinning the
    partition function and enabling absolute Q score interpretation.
    Note: (Z − 1)² ≠ (log Z)² — the former is the correct paper formula.

    logsumexp provides numerical stability:
        log Z = log(mean(Q)) = logsumexp(scores_gen) − log(N_G)

    Args:
        scores_data:      [N_D] log Q scores for productive (sorted) sequences
        scores_generated: [N_G] log Q scores for NCE negative sequences
        gamma:            Gauge-fixing weight (default 0.0 = disabled).
                          Recommended range: 0.5–2.0 when enabled.

    Returns:
        Scalar loss to minimise
    """
    assert scores_data.numel() > 0, "scores_data is empty"
    assert scores_generated.numel() > 0, "scores_generated is empty"

    term1 = scores_data.mean()
    N_G = scores_generated.shape[0]
    # log Z = log(mean(Q)) via numerically stable logsumexp
    log_z = torch.logsumexp(scores_generated, dim=0) - torch.log(
        torch.tensor(N_G, dtype=torch.float, device=scores_generated.device)
    )
    loss = -(term1 - log_z)
    if gamma > 0.0:
        loss = loss + gamma * (log_z.exp() - 1) ** 2
    return loss


def bce_selection_loss(scores_data: torch.Tensor,
                       scores_generated: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy selection loss (NCE approximation).

    Treats selection as binary classification: productive sequences are
    positive (label 1), NCE negative are negative (label 0).  The
    Bayes-optimal classifier recovers f*(σ) = log P_data(σ)/P_gen(σ),
    which is exactly the selection factor log Q* — so both losses converge
    to the same optimal solution.  BCE is faster to optimise but does not
    estimate the partition function Z globally; it is a biased estimator
    that converges to MLE as N_generated → ∞.

    Args:
        scores_data:      [N_D] log Q scores for productive (sorted) sequences
        scores_generated: [N_G] log Q scores for NCE negative sequences

    Returns:
        Scalar loss to minimise
    """
    scores = torch.cat([scores_data, scores_generated])
    labels = torch.cat([
        torch.ones(len(scores_data),      device=scores_data.device),
        torch.zeros(len(scores_generated), device=scores_generated.device)
    ])
    return F.binary_cross_entropy_with_logits(scores, labels)


def discriminative_delta_loss(score_cd4: torch.Tensor, score_cd8: torch.Tensor,
                               cd4_mask: torch.Tensor,
                               cd8_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence CD4/CD8 separation loss on delta = log Q^CD4 - log Q^CD8.

    Treats delta as a binary classifier: CD4 sequences should have delta > 0,
    CD8 sequences delta < 0. Implemented as BCE with logits so the sigmoid is
    numerically stable. Operates only on sorted sequences (CD4 + CD8); Emerson
    negatives are ignored.

        L_disc = -E_CD4[log sigma(delta)] - E_CD8[log sigma(-delta)]
               = BCE(delta[sorted], labels)   labels=1 for CD4, 0 for CD8

    At the true NCE optimum delta > 0 for CD4 by definition, so the gradient
    of L_disc goes to zero there — it reshapes the path without moving the optimum.

    Returns 0.0 if no sorted sequences are present in the batch.
    """
    sorted_mask = cd4_mask | cd8_mask
    if not sorted_mask.any():
        return torch.tensor(0.0, device=score_cd4.device)
    delta  = score_cd4[sorted_mask] - score_cd8[sorted_mask]
    labels = cd4_mask[sorted_mask].float()
    return F.binary_cross_entropy_with_logits(delta, labels)


class SelectionLoss:
    """
    Configurable wrapper that dispatches to soNNia or BCE selection loss.

    For "sonia" loss, the gauge-fixing term γ·(log Z)² is baked in here,
    matching the authors' implementation exactly (Isacchini et al. 2021).
    For "bce" loss, gamma is ignored (BCE has no natural log Z term).

    Usage:
        loss_fn = SelectionLoss("sonia", gamma=1.0)
        loss = loss_fn(score_cd4[cd4_mask], score_cd4[neg_mask])
    """

    def __init__(self, loss_type: str = "sonia", gamma: float = 0.0):
        """
        Args:
            loss_type: "sonia" (exact MLE, logsumexp-stabilised) or
                       "bce"   (NCE approximation, simpler to optimise)
            gamma:     Gauge-fixing weight for soNNia loss (default 0.0 = disabled).
                       Ignored when loss_type="bce".
        """
        assert loss_type in ("sonia", "bce"), \
            f"Unknown loss_type '{loss_type}'. Choose 'sonia' or 'bce'."
        self.loss_type = loss_type
        self.gamma     = gamma

    def __call__(self, scores_data: torch.Tensor,
                 scores_generated: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "sonia":
            return sonia_loss(scores_data, scores_generated, gamma=self.gamma)
        return bce_selection_loss(scores_data, scores_generated)


# ================= SONIA TRAINING FUNCTIONS =================

def train_selection_epoch(model, loader, optimizer, scheduler, device, exp,
                          global_step: int, selection_loss: SelectionLoss,
                          lambda_sel_cd4: float = 1.0,
                          lambda_sel_cd8: float = 1.0,
                          lambda_disc: float = 0.0) -> tuple:
    """
    Train one epoch of stage 3a — selection head training.

    The BERT encoder must already be frozen (model.freeze_bert() called before
    this function).  Only pooler_selection and Q heads receive gradient updates.

    Batch format (from SoNNiaDataset):
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        token_type_ids: [B, seq_len]  (optional)
        sequence_type:  [B] int — 0 = CD4-sorted, 1 = CD8-sorted, 2 = NEG (NCE negatives)

    Loss structure:
        Q^CD4 main:   selection_loss(score_cd4[cd4_mask], score_cd4[neg_mask])
                      — standard NCE: CD4 as positives, Emerson as reference (Q ≈ 0)
        Q^CD8 main:   selection_loss(score_cd8[cd8_mask], score_cd8[neg_mask])
                      — standard NCE: CD8 as positives, Emerson as reference (Q ≈ 0)
    Args:
        model:          TCRBertSoNNia
        loader:         DataLoader for stage 3a data
        optimizer:      Optimizer (should cover pooler_cd4, pooler_cd8 + Q heads only)
        scheduler:      LR scheduler (or None)
        device:         Training device
        exp:            Comet ML experiment (or None)
        global_step:    Current global step counter
        selection_loss: SelectionLoss instance — carries loss_type and gamma.
                        Gauge-fixing (γ·(log Z)²) is baked into SelectionLoss
                        when loss_type="sonia", matching the authors' implementation.
        lambda_sel_cd4: Weight for Q^CD4 NCE loss term
        lambda_sel_cd8: Weight for Q^CD8 NCE loss term

    Returns:
        Tuple of (avg_total_loss, updated_global_step)
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Stage 3a: selection heads")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        sequence_type  = batch["sequence_type"].to(device)

        cd4_mask = (sequence_type == 0)
        cd8_mask = (sequence_type == 1)
        neg_mask = (sequence_type == 2)

        # Forward — only score_cd4 and score_cd8 enter the loss
        score_cd4, score_cd8, _ = model(input_ids, attention_mask, token_type_ids)

        # Q^CD4: CD4 productive vs NCE negatives; CD8 sequences are ignored
        if cd4_mask.any() and neg_mask.any():
            L_cd4 = selection_loss(score_cd4[cd4_mask], score_cd4[neg_mask])
        else:
            L_cd4 = torch.tensor(0.0, device=device)

        # Q^CD8: CD8 productive vs NCE negatives; CD4 sequences are ignored
        if cd8_mask.any() and neg_mask.any():
            L_cd8 = selection_loss(score_cd8[cd8_mask], score_cd8[neg_mask])
        else:
            L_cd8 = torch.tensor(0.0, device=device)

        L_disc = torch.tensor(0.0, device=device)
        if lambda_disc > 0.0 and cd4_mask.any() and cd8_mask.any():
            L_disc = discriminative_delta_loss(score_cd4, score_cd8, cd4_mask, cd8_mask)

        loss = lambda_sel_cd4 * L_cd4 + lambda_sel_cd8 * L_cd8 + lambda_disc * L_disc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss  += loss.item()
        num_batches += 1

        postfix = {
            "loss":  f"{loss.item():.4f}",
            "L_cd4": f"{L_cd4.item():.4f}",
            "L_cd8": f"{L_cd8.item():.4f}",
            "L_disc": f"{L_disc.item():.4f}",
        }
        if scheduler:
            postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
        pbar.set_postfix(postfix)

        if exp:
            exp.log_metric("sel_loss_step",     loss.item(),   step=global_step)
            exp.log_metric("sel_cd4_loss_step", L_cd4.item(),  step=global_step)
            exp.log_metric("sel_cd8_loss_step", L_cd8.item(),  step=global_step)
            exp.log_metric("l_disc_step",       L_disc.item(), step=global_step)
            if scheduler:
                exp.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)
            # Log mean Q scores and delta per group
            with torch.no_grad():
                delta = score_cd4 - score_cd8
                if cd4_mask.any():
                    exp.log_metric("mean_delta_cd4", delta[cd4_mask].mean().item(), step=global_step)
                if cd8_mask.any():
                    exp.log_metric("mean_delta_cd8", delta[cd8_mask].mean().item(), step=global_step)
                if cd4_mask.any():
                    exp.log_metric("mean_q_cd4_data",
                                   score_cd4[cd4_mask].mean().item(), step=global_step)
                    exp.log_metric("mean_q_cd8_cd4",
                                   score_cd8[cd4_mask].mean().item(), step=global_step)
                if cd8_mask.any():
                    exp.log_metric("mean_q_cd8_data",
                                   score_cd8[cd8_mask].mean().item(), step=global_step)
                    exp.log_metric("mean_q_cd4_cd8",
                                   score_cd4[cd8_mask].mean().item(), step=global_step)
                if neg_mask.any():
                    exp.log_metric("mean_q_cd4_neg",
                                   score_cd4[neg_mask].mean().item(), step=global_step)
                    exp.log_metric("mean_q_cd8_neg",
                                   score_cd8[neg_mask].mean().item(), step=global_step)

        global_step += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def train_sonia_classifier_epoch(model, loader, optimizer, scheduler,
                                  device, exp, global_step: int) -> tuple:
    """
    Train one epoch of stage 3b — classifier training.

    pooler_selection, Q heads, and BERT encoder must already be frozen before
    calling this (model.freeze_selection_branch() and model.freeze_bert()).
    Only pooler_classifier and the classifier head receive gradient updates.

    Batch contains only CD4/CD8 sorted sequences — no NEG sequences.
    Labels: CD4-sorted = 1.0, CD8-sorted = 0.0

    Batch format (SoNNiaDataset with only CD4/CD8 — no NEG):
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        token_type_ids: [B, seq_len]  (optional)
        sequence_type:  [B] int — 0 = CD4, 1 = CD8

    Args:
        model:       TCRBertSoNNia (with selection branch frozen)
        loader:      DataLoader for stage 3b data (sorted sequences only)
        optimizer:   Optimizer covering pooler_classifier + classifier head
        scheduler:   LR scheduler (or None)
        device:      Training device
        exp:         Comet ML experiment (or None)
        global_step: Current global step counter

    Returns:
        Tuple of (avg_loss, updated_global_step)
    """
    model.train()
    epoch_loss  = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Stage 3b: classifier")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        sequence_type  = batch["sequence_type"].to(device)

        score_cd4 = batch["score_cd4"].to(device) if "score_cd4" in batch else None
        score_cd8 = batch["score_cd8"].to(device) if "score_cd8" in batch else None

        labels = (sequence_type == 0).float()   # CD4 = 1.0, CD8 = 0.0

        _, _, logit = model(input_ids, attention_mask, token_type_ids,
                            precomputed_score_cd4=score_cd4,
                            precomputed_score_cd8=score_cd8)
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss  += loss.item()
        num_batches += 1

        postfix = {"loss": f"{loss.item():.4f}"}
        if scheduler:
            lrs = scheduler.get_last_lr()
            postfix["bert_lr"] = f"{lrs[0]:.2e}"
            postfix["head_lr"] = f"{lrs[1]:.2e}" if len(lrs) > 1 else "—"
        pbar.set_postfix(postfix)

        if exp:
            exp.log_metric("cls_loss_step", loss.item(), step=global_step)
            if scheduler:
                lrs = scheduler.get_last_lr()
                exp.log_metric("lr_bert", lrs[0], step=global_step)
                if len(lrs) > 1:
                    exp.log_metric("lr_head", lrs[1], step=global_step)

        global_step += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


# ================= SONIA EVALUATION =================

def evaluate_sonia_classifier(model, loader, device) -> dict:
    """
    Evaluate the SoNNia classifier on sorted sequences.

    The logit from SoNNiaClassifierHead is CD4-positive, so sigmoid(logit)
    gives P(CD4).  AUC and balanced accuracy are computed against true labels.

    Args:
        model:  TCRBertSoNNia (selection branch should be frozen for evaluation)
        loader: DataLoader from SoNNiaDataset with sequence_type 0=CD4, 1=CD8 (no NEG)
        device: Evaluation device

    Returns:
        dict with keys:
            bal_acc: Balanced accuracy at threshold 0.5
            auc:     ROC AUC score
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            sequence_type  = batch["sequence_type"].to(device)

            score_cd4 = batch["score_cd4"].to(device) if "score_cd4" in batch else None
            score_cd8 = batch["score_cd8"].to(device) if "score_cd8" in batch else None

            labels = (sequence_type == 0).float()   # CD4 = 1, CD8 = 0

            _, _, logit = model(input_ids, attention_mask, token_type_ids,
                                precomputed_score_cd4=score_cd4,
                                precomputed_score_cd8=score_cd8)
            probs = torch.sigmoid(logit)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)

    return {
        "bal_acc": balanced_accuracy_score(all_labels, preds),
        "auc":     roc_auc_score(all_labels, all_probs),
    }


def log_split_metrics(exp, split_name: str, metrics: dict, epoch: int = None) -> None:
    """
    Log metrics for a data split to Comet ML.

    Args:
        exp: Comet ML experiment
        split_name: Name of the split (e.g., "hd", "t1d", "ms")
        metrics: Dictionary with keys: bal_acc, auc
        epoch: Epoch number (optional)
    """
    if exp is None:
        return

    if epoch is not None:
        exp.log_metric(f"{split_name}_bal_acc", metrics["bal_acc"], epoch=epoch)
        exp.log_metric(f"{split_name}_auc", metrics["auc"], epoch=epoch)
    else:
        exp.log_metric(f"{split_name}_bal_acc", metrics["bal_acc"])
        exp.log_metric(f"{split_name}_auc", metrics["auc"])


# ================= PAIRED CHAIN TRAINING =================



def infonce_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Bidirectional InfoNCE (NT-Xent) loss for in-batch paired sequences.

    Args:
        emb_a:       [B, hidden] embeddings for chain A (alpha)
        emb_b:       [B, hidden] embeddings for chain B (beta)
        temperature: softmax temperature tau

    Returns:
        Scalar loss: mean of (alpha->beta CE + beta->alpha CE) / 2
    """
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    sim = torch.matmul(emb_a, emb_b.T) / temperature   # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def train_pairing_epoch(model, loader, optimizer, scheduler, device, exp,
                        global_step: int,
                        lambda_alpha:  float = 1.0,
                        lambda_beta:   float = 1.0,
                        lambda_pair:   float = 1.0,
                        lambda_bridge: float = 1.0,
                        temperature:   float = 0.05) -> tuple:
    """
    Train one epoch of paired alpha-beta pretraining (PMI framework, Exp 6+).

    Common loss terms (all variants):
        L_mlm          MLM on both chains, view 1
        L_alpha_simcse InfoNCE(emb_alpha_v1, emb_alpha_v2) -- alpha-space PMI
        L_beta_simcse  InfoNCE(emb_beta_v1,  emb_beta_v2)  -- beta-space PMI
        L_pair_simcse  InfoNCE(z_v1, z_v2)                  -- pair-space PMI
                       where z = pair_projector(cat[alpha, alpha-beta, beta])

    Variant-specific L_bridge (selected by model.pmi_variant):
        "clip"       -- InfoNCE(align_alpha(emb_alpha), align_beta(emb_beta))
                        (Exp 6: shared subspace via separate projectors)
        "chaintoken" -- sum of three InfoNCE terms over pair_projector with proxy tokens
                        (Exp 7: paired/alpha-only/beta-only modes anchored together)

    Args:
        model:         TCRBertPairing instance (with align_* or proxy_* depending on variant)
        loader:        DataLoader over TCRPairedDataset (two independently masked views)
        optimizer:     AdamW optimizer
        scheduler:     LambdaLR scheduler
        device:        Training device
        exp:           Comet ML experiment (or None)
        global_step:   Current global step
        lambda_alpha:  Weight for alpha SimCSE
        lambda_beta:   Weight for beta SimCSE
        lambda_pair:   Weight for pair SimCSE
        lambda_bridge: Weight for variant-specific cross-chain alignment
        temperature:   InfoNCE temperature (shared across all InfoNCE terms)

    Returns:
        Tuple of (average_epoch_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        alpha_mask = batch["alpha_attention_mask"]
        beta_mask  = batch["beta_attention_mask"]

        # --- view 1 ---
        mlm_a_v1, emb_a_v1 = model._encode(
            batch["alpha_input_ids_v1"], alpha_mask, batch["alpha_labels_v1"], model.pooler_alpha,
        )
        mlm_b_v1, emb_b_v1 = model._encode(
            batch["beta_input_ids_v1"], beta_mask, batch["beta_labels_v1"], model.pooler_beta,
        )

        # --- view 2: independent masking ---
        mlm_a_v2, emb_a_v2 = model._encode(
            batch["alpha_input_ids_v2"], alpha_mask, batch["alpha_labels_v2"], model.pooler_alpha,
        )
        mlm_b_v2, emb_b_v2 = model._encode(
            batch["beta_input_ids_v2"], beta_mask, batch["beta_labels_v2"], model.pooler_beta,
        )

        # --- common losses ---
        L_mlm          = (mlm_a_v1 + mlm_b_v1) / 2
        L_alpha_simcse = infonce_loss(emb_a_v1, emb_a_v2, temperature)
        L_beta_simcse  = infonce_loss(emb_b_v1, emb_b_v2, temperature)

        z_v1 = model.pair_projector(torch.cat([emb_a_v1, emb_a_v1 - emb_b_v1, emb_b_v1], dim=-1))
        z_v2 = model.pair_projector(torch.cat([emb_a_v2, emb_a_v2 - emb_b_v2, emb_b_v2], dim=-1))
        L_pair_simcse = infonce_loss(z_v1, z_v2, temperature)

        # --- variant-specific bridge ---
        if model.pmi_variant == "clip":
            z_align_a = model.align_alpha(emb_a_v1)
            z_align_b = model.align_beta(emb_b_v1)
            L_bridge = infonce_loss(z_align_a, z_align_b, temperature)
            bridge_parts = {"cross_clip": L_bridge}

        elif model.pmi_variant == "chaintoken":
            proxy_a = model.proxy_alpha.expand_as(emb_a_v1)
            proxy_b = model.proxy_beta.expand_as(emb_b_v1)
            z_alpha_only = model.pair_projector(torch.cat(
                [emb_a_v1, emb_a_v1 - proxy_b, proxy_b], dim=-1
            ))
            z_beta_only = model.pair_projector(torch.cat(
                [proxy_a, proxy_a - emb_b_v1, emb_b_v1], dim=-1
            ))
            L_bridge_ab  = infonce_loss(z_alpha_only, z_beta_only,           temperature)
            L_bridge_b_p = infonce_loss(z_beta_only,  z_v1.detach(),          temperature)
            L_bridge_a_p = infonce_loss(z_alpha_only, z_v1.detach(),          temperature)
            L_bridge = L_bridge_ab + L_bridge_b_p + L_bridge_a_p
            bridge_parts = {
                "bridge_ab":  L_bridge_ab,
                "bridge_b_p": L_bridge_b_p,
                "bridge_a_p": L_bridge_a_p,
                "bridge":     L_bridge,
            }

        elif model.pmi_variant == "doubledist":
            h_alpha = model.bridge_alpha(emb_a_v1)
            h_beta  = model.bridge_beta(emb_b_v1)
            L_bridge_a = infonce_loss(h_alpha, z_v1.detach(), temperature)
            L_bridge_b = infonce_loss(h_beta,  z_v1.detach(), temperature)
            L_bridge = L_bridge_a + L_bridge_b
            bridge_parts = {
                "bridge_alpha": L_bridge_a,
                "bridge_beta":  L_bridge_b,
                "bridge":       L_bridge,
            }

        else:
            raise ValueError(f"Unknown model.pmi_variant: {model.pmi_variant!r}")

        loss = (L_mlm
                + lambda_alpha  * L_alpha_simcse
                + lambda_beta   * L_beta_simcse
                + lambda_pair   * L_pair_simcse
                + lambda_bridge * L_bridge)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        global_step += 1

        if exp:
            exp.log_metric("loss",         loss.item(),           step=global_step)
            exp.log_metric("mlm_loss",     L_mlm.item(),          step=global_step)
            exp.log_metric("alpha_simcse", L_alpha_simcse.item(), step=global_step)
            exp.log_metric("beta_simcse",  L_beta_simcse.item(),  step=global_step)
            exp.log_metric("pair_simcse",  L_pair_simcse.item(),  step=global_step)
            for k, v in bridge_parts.items():
                exp.log_metric(k, v.item(), step=global_step)

    return total_loss / len(loader), global_step


def train_joint_epoch(model, loader, optimizer, scheduler, device, exp,
                      global_step: int,
                      lambda_mlm:  float = 1.0,
                      lambda_pair: float = 1.0,
                      lambda_drop: float = 1.0,
                      temperature: float = 0.05,
                      scaler=None) -> tuple:
    """
    Train one epoch of joint paired pretraining (Exp 9, SCEPTR-style + chain-drop).

    Four forward passes through the shared joint encoder per batch:
        z_v1, mlm_v1 = enc(full, mask A)   -- full pair, masking view 1 (MLM loss here)
        z_v2         = enc(full, mask B)   -- full pair, masking view 2
        z_alpha      = enc(alpha-only, masked)  -- beta chain dropped
        z_beta       = enc(beta-only,  masked)  -- alpha chain dropped

    Loss:
        L = lambda_mlm  * mlm_v1
          + lambda_pair * InfoNCE(z_v1, z_v2)                              -- autocontrastive
          + lambda_drop * [ InfoNCE(z_alpha, sg(z_v1)) + InfoNCE(z_beta, sg(z_v1)) ]
                                                                           -- chain-drop consistency

    sg = stop-gradient: z_v1 is the (full-pair) teacher for the chain-dropped students.

    Args:
        model:       TCRBertJoint instance
        loader:      DataLoader over TCRJointDataset
        optimizer:   AdamW
        scheduler:   LambdaLR
        device:      training device
        exp:         Comet experiment (or None)
        global_step: current global step
        lambda_mlm:  weight for MLM loss (view 1)
        lambda_pair: weight for autocontrastive pair SimCSE
        lambda_drop: weight for chain-drop consistency (sum of alpha+beta)
        temperature: InfoNCE temperature
        scaler:      torch.cuda.amp.GradScaler for mixed precision (or None to disable)

    Returns:
        Tuple of (average_epoch_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Detect mode by what the dataset emitted. SCEPTR-replica path has per-view masks
        # and no single-chain views; Exp 9 path has a shared mask and alpha/beta views.
        has_drop = "alpha_input_ids" in batch

        with torch.autocast(device_type="cuda", enabled=use_amp):
            if has_drop:
                # Exp 9: full v1 + full v2 + alpha-only + beta-only, shared attention mask
                full_mask     = batch["full_attention_mask"]
                alpha_mask    = batch["alpha_attention_mask"]
                beta_mask     = batch["beta_attention_mask"]
                full_type_ids  = batch.get("full_token_type_ids")
                alpha_type_ids = batch.get("alpha_token_type_ids")
                beta_type_ids  = batch.get("beta_token_type_ids")

                mlm_v1, z_v1 = model._encode(batch["full_input_ids_v1"], full_mask, batch["full_labels_v1"], full_type_ids)
                _,      z_v2 = model._encode(batch["full_input_ids_v2"], full_mask, batch["full_labels_v2"], full_type_ids)
                _,      z_a  = model._encode(batch["alpha_input_ids"],   alpha_mask, None, alpha_type_ids)
                _,      z_b  = model._encode(batch["beta_input_ids"],    beta_mask,  None, beta_type_ids)

                L_mlm    = mlm_v1
                L_pair   = infonce_loss(z_v1, z_v2, temperature)
                L_drop_a = infonce_loss(z_a, z_v1.detach(), temperature)
                L_drop_b = infonce_loss(z_b, z_v1.detach(), temperature)
                L_drop   = L_drop_a + L_drop_b

                loss = lambda_mlm * L_mlm + lambda_pair * L_pair + lambda_drop * L_drop
            else:
                # SCEPTR replica: two independently censored views, per-view masks,
                # no chain-drop loss. View 2 carries no MLM labels (no -100 target tensor).
                v1_mask = batch["full_attention_mask_v1"]
                v2_mask = batch["full_attention_mask_v2"]
                v1_type_ids = batch.get("full_token_type_ids_v1")
                v2_type_ids = batch.get("full_token_type_ids_v2")

                mlm_v1, z_v1 = model._encode(batch["full_input_ids_v1"], v1_mask, batch["full_labels_v1"], v1_type_ids)
                _,      z_v2 = model._encode(batch["full_input_ids_v2"], v2_mask, None, v2_type_ids)

                L_mlm  = mlm_v1
                L_pair = infonce_loss(z_v1, z_v2, temperature)

                loss = lambda_mlm * L_mlm + lambda_pair * L_pair

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        global_step += 1

        if exp:
            exp.log_metric("loss",        loss.item(),   step=global_step)
            exp.log_metric("mlm_loss",    L_mlm.item(),  step=global_step)
            exp.log_metric("pair_simcse", L_pair.item(), step=global_step)
            if has_drop:
                exp.log_metric("drop_alpha", L_drop_a.item(), step=global_step)
                exp.log_metric("drop_beta",  L_drop_b.item(), step=global_step)

    return total_loss / len(loader), global_step


def train_stage1_epoch(model, loader, optimizer, scheduler, device, exp,
                       global_step: int,
                       lambda_mlm:  float = 1.0,
                       lambda_pair: float = 1.0,
                       temperature: float = 0.05,
                       scaler=None, amp_dtype=None,
                       max_grad_norm: float = 1.0) -> tuple:
    """
    TCRFoundation Stage 1 epoch: beta-only MLM + autocontrastive InfoNCE.

    AMP: pass `amp_dtype=torch.bfloat16` (preferred, no scaler) or `amp_dtype=torch.float16`
    WITH a GradScaler. bf16 is much more stable for low-temperature InfoNCE (fp16's narrow
    exponent range overflows on exp(cos/tau) -> intermittent loss spikes).

    Two forwards per batch through the same shared TCRBetaOnlyEncoder:
        mlm_v1, z_v1 = enc(view_1, labels=v1_labels)   # MLM loss on view 1 only
        _,       z_v2 = enc(view_2, labels=None)         # contrastive partner

    Loss:
        L = lambda_mlm  * mlm_v1
          + lambda_pair * InfoNCE(z_v1, z_v2; tau)

    Args:
        model:       TCRBetaOnlyEncoder instance.
        loader:      DataLoader over TCRBetaOnlyMLMDataset (two-view items).
        optimizer:   AdamW.
        scheduler:   LambdaLR.
        device:      Training device.
        exp:         Comet experiment (or None).
        global_step: Current global step (carried across epochs).
        lambda_mlm:  Weight for MLM term.
        lambda_pair: Weight for autocontrastive InfoNCE term.
        temperature: InfoNCE temperature tau.
        scaler:      torch.cuda.amp.GradScaler for mixed precision (or None).

    Returns:
        Tuple of (average_epoch_loss, updated_global_step).
    """
    model.train()
    total_loss = 0.0
    use_amp = amp_dtype is not None

    pbar = tqdm(loader, desc="Stage 1")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            mlm_v1, z_v1 = model._encode(
                batch["input_ids_v1"], batch["attention_mask_v1"],
                batch["labels_v1"], batch["token_type_ids_v1"],
                batch.get("position_ids_v1"),
            )
            _, z_v2 = model._encode(
                batch["input_ids_v2"], batch["attention_mask_v2"],
                None, batch["token_type_ids_v2"],
                batch.get("position_ids_v2"),
            )
            L_mlm  = mlm_v1
            L_pair = infonce_loss(z_v1, z_v2, temperature)
            loss = lambda_mlm * L_mlm + lambda_pair * L_pair

        if scaler is not None:          # fp16 path (loss scaling)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                                    # unscale before clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:                            # bf16 or fp32: plain backward/step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # spike insurance
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        global_step += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mlm":  f"{L_mlm.item():.4f}",
            "pair": f"{L_pair.item():.4f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
        })

        if exp:
            exp.log_metric("loss",      loss.item(),   step=global_step)
            exp.log_metric("mlm_loss",  L_mlm.item(),  step=global_step)
            exp.log_metric("pair_loss", L_pair.item(), step=global_step)

    return total_loss / len(loader), global_step


def train_stage2_epoch(model, loader, optimizer, scheduler, device, exp,
                       global_step: int,
                       lambda_pair: float = 1.0,
                       lambda_drop: float = 1.0,
                       lambda_mlm:  float = 1.0,
                       temperature: float = 0.05,
                       scaler=None, amp_dtype=None,
                       max_grad_norm: float = 1.0) -> tuple:
    """
    TCRFoundation Stage 2 epoch: full-sequence MLM + paired autocontrastive InfoNCE +
    chain-drop consistency -- the same SCEPTR recipe Stage 1 gave beta, now on the pair.

    The MLM term runs through the FROZEN head into the trainable adapters (LoRA + alpha
    type-row): it teaches alpha grammar and re-anchors beta under cross-chain attention
    WITHOUT touching the head -> Benchmark B1 (disable_adapter, beta-only) stays bit-for-bit
    Stage 1. MLM is computed on full view v1 only (both chains are masked there).

    AMP: same convention as train_stage1_epoch -- pass `amp_dtype=torch.bfloat16` (no scaler)
    or `amp_dtype=torch.float16` WITH a GradScaler. bf16 avoids the fp16 InfoNCE overflow.

    Per batch (TCRJointDataset exp9 path, per-chain CDR3 censoring + per-chain position_ids):
        z_v1, logits_v1 = enc_mlm(full view 1)   z_v2 = enc(full view 2)
        z_a  = enc(alpha-only)    z_b  = enc(beta-only)
        L = lambda_mlm  * MLM(logits_v1, full_labels_v1)
          + lambda_pair * InfoNCE(z_v1, z_v2)
          + lambda_drop * [ InfoNCE(z_a, sg z_v1) + InfoNCE(z_b, sg z_v1) ]

    `model` is a TCRFoundationStage2 exposing encode(...) -> z and encode_mlm(...) -> (z, logits).
    """
    model.train()
    total_loss = 0.0
    use_amp = amp_dtype is not None

    pbar = tqdm(loader, desc="Stage 2")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            # Full v1 also yields MLM logits (frozen head); v1/v2 use their own CENSORED masks.
            z_v1, logits_v1 = model.encode_mlm(
                batch["full_input_ids_v1"], batch["full_attention_mask_v1"],
                batch["full_token_type_ids"], batch["full_position_ids"])
            z_v2 = model.encode(batch["full_input_ids_v2"], batch["full_attention_mask_v2"],
                                batch["full_token_type_ids"], batch["full_position_ids"])
            z_a  = model.encode(batch["alpha_input_ids"], batch["alpha_attention_mask"],
                                batch["alpha_token_type_ids"], batch["alpha_position_ids"])
            z_b  = model.encode(batch["beta_input_ids"],  batch["beta_attention_mask"],
                                batch["beta_token_type_ids"],  batch["beta_position_ids"])

            # MLM on full v1 (both chains): cross-entropy over masked positions (-100 ignored).
            L_mlm    = F.cross_entropy(logits_v1.reshape(-1, logits_v1.size(-1)),
                                       batch["full_labels_v1"].reshape(-1), ignore_index=-100)
            L_pair   = infonce_loss(z_v1, z_v2, temperature)
            L_drop_a = infonce_loss(z_a, z_v1.detach(), temperature)
            L_drop_b = infonce_loss(z_b, z_v1.detach(), temperature)
            L_drop   = L_drop_a + L_drop_b
            loss = lambda_mlm * L_mlm + lambda_pair * L_pair + lambda_drop * L_drop

        if scaler is not None:          # fp16 path (loss scaling)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                                    # unscale before clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:                            # bf16 or fp32: plain backward/step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # spike insurance
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        global_step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mlm": f"{L_mlm.item():.4f}",
                          "pair": f"{L_pair.item():.4f}", "drop": f"{L_drop.item():.4f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        if exp:
            exp.log_metric("loss",      loss.item(),   step=global_step)
            exp.log_metric("mlm_loss",  L_mlm.item(),  step=global_step)
            exp.log_metric("pair_loss", L_pair.item(), step=global_step)
            exp.log_metric("drop_loss", L_drop.item(), step=global_step)

    return total_loss / len(loader), global_step


def train_joint_epoch(model, paired_loader, emerson_loader, optimizer, scheduler, device, exp,
                      global_step: int,
                      lambda_mlm:      float = 1.0,
                      lambda_pair:     float = 1.0,
                      lambda_drop:     float = 1.0,
                      lambda_betaonly: float = 1.0,
                      temperature:     float = 0.05,
                      scaler=None, amp_dtype=None,
                      max_grad_norm:   float = 1.0,
                      lambda_drop_beta:    float = None,
                      lambda_drop_alpha:   float = 0.0,
                      lambda_alpha_simcse: float = 0.0) -> tuple:
    """
    Single-stage joint TCRFoundation epoch: ONE model (one MLM head + one mixed pooler) trained on
    BOTH paired data and Emerson beta-only, zipped ~50/50 per step. Gives one coordinate system for
    beta-only and paired (shared encoder + pooler), and preserves publicness (the MLM term on
    Emerson beta calibrates the head -> B1 PLL).

    Chain-drop has TWO modes, selected by what the paired dataset emits:
      * LEGACY (lambda_drop_beta is None, dataset emits a single `alpha_input_ids`):
          L_drop = lambda_drop * [InfoNCE(z_pa, sg z_pv1) + InfoNCE(z_pb, sg z_pv1)]
        i.e. BOTH single-chain students are pulled to the pair (symmetric chain-drop).
      * SPLIT / alpha-detach (lambda_drop_beta set, dataset emits `alpha_input_ids_v1/v2`):
          beta kept:  lambda_drop_beta  * InfoNCE(z_pb, sg z_pv1)
          alpha->pair: lambda_drop_alpha * InfoNCE(z_pa, sg z_pv1)   (default 0 -> alpha DETACHED)
          alpha SimCSE: lambda_alpha_simcse * InfoNCE(z_pa1, z_pa2)  (alpha learns its OWN space)
        so the V-token / alpha-detach experiment removes the alpha->pair pull and gives alpha its
        own two-view contrast while beta stays unified with the pair.

    Per step (pair_batch, emer_batch):
        mlm_p, z_pv1 = enc(full_v1, labels)   z_pv2 = enc(full_v2)
        z_pb = enc(beta) ; z_pa / (z_pa1, z_pa2) depending on mode
        mlm_e, z_ev1 = enc(beta_v1, labels)   z_ev2 = enc(beta_v2)          # Emerson beta-only
        L = lambda_mlm  * (mlm_p + mlm_e) + lambda_pair * InfoNCE(z_pv1, z_pv2)
          + chain-drop (see modes) + lambda_betaonly * InfoNCE(z_ev1, z_ev2)

    Negatives are MODALITY-SEPARATED by construction: paired InfoNCE / chain-drop use only the
    paired batch, beta autocontrast only the Emerson batch -> pairs and beta are never mutual
    negatives (which would segregate the space). beta<->paired alignment comes from the shared
    per-chain encoder + chain-drop, not from cross-modality positives/negatives.

    AMP: bf16 (amp_dtype, no scaler) preferred; fp16 path uses the GradScaler. Same as Stage 2.
    """
    model.train()
    total_loss = 0.0
    use_amp = amp_dtype is not None
    n_steps = min(len(paired_loader), len(emerson_loader))   # zip stops at the shorter loader

    pbar = tqdm(zip(paired_loader, emerson_loader), total=n_steps, desc="Joint")
    for pair_batch, emer_batch in pbar:
        pair_batch = {k: v.to(device) for k, v in pair_batch.items()}
        emer_batch = {k: v.to(device) for k, v in emer_batch.items()}

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            # ---- paired modality: MLM on full v1 (both chains) + pair InfoNCE + chain-drop ----
            mlm_p, z_pv1 = model._encode(
                pair_batch["full_input_ids_v1"], pair_batch["full_attention_mask_v1"],
                pair_batch["full_labels_v1"], pair_batch["full_token_type_ids"],
                pair_batch["full_position_ids"])
            _, z_pv2 = model._encode(
                pair_batch["full_input_ids_v2"], pair_batch["full_attention_mask_v2"],
                None, pair_batch["full_token_type_ids"], pair_batch["full_position_ids"])
            L_pair = infonce_loss(z_pv1, z_pv2, temperature)

            # beta single-chain student -> pulled to the pair (KEEP: beta/paired unification).
            _, z_pb = model._encode(
                pair_batch["beta_input_ids"], pair_batch["beta_attention_mask"],
                None, pair_batch["beta_token_type_ids"], pair_batch["beta_position_ids"])
            L_drop_beta = infonce_loss(z_pb, z_pv1.detach(), temperature)

            if "alpha_input_ids_v1" in pair_batch:
                # SPLIT / alpha-detach: no alpha->pair pull; alpha gets its OWN SimCSE (two views).
                _, z_pa1 = model._encode(
                    pair_batch["alpha_input_ids_v1"], pair_batch["alpha_attention_mask_v1"],
                    None, pair_batch["alpha_token_type_ids"], pair_batch["alpha_position_ids"])
                _, z_pa2 = model._encode(
                    pair_batch["alpha_input_ids_v2"], pair_batch["alpha_attention_mask_v2"],
                    None, pair_batch["alpha_token_type_ids"], pair_batch["alpha_position_ids"])
                L_alpha_simcse = infonce_loss(z_pa1, z_pa2, temperature)
                L_drop_alpha   = z_pv1.new_zeros(())          # alpha->pair OFF (detached)
                lam_db = lambda_drop if lambda_drop_beta is None else lambda_drop_beta
                lam_da = lambda_drop_alpha
                lam_as = lambda_alpha_simcse
            else:
                # LEGACY: single alpha view ALSO pulled to the pair (symmetric chain-drop).
                _, z_pa = model._encode(
                    pair_batch["alpha_input_ids"], pair_batch["alpha_attention_mask"],
                    None, pair_batch["alpha_token_type_ids"], pair_batch["alpha_position_ids"])
                L_drop_alpha   = infonce_loss(z_pa, z_pv1.detach(), temperature)
                L_alpha_simcse = z_pv1.new_zeros(())
                lam_db = lam_da = lambda_drop
                lam_as = 0.0
            L_drop = lam_db * L_drop_beta + lam_da * L_drop_alpha

            # ---- Emerson beta-only: MLM (publicness) + autocontrast ----
            mlm_e, z_ev1 = model._encode(
                emer_batch["input_ids_v1"], emer_batch["attention_mask_v1"],
                emer_batch["labels_v1"], emer_batch["token_type_ids_v1"],
                emer_batch["position_ids_v1"])
            _, z_ev2 = model._encode(
                emer_batch["input_ids_v2"], emer_batch["attention_mask_v2"],
                None, emer_batch["token_type_ids_v2"], emer_batch["position_ids_v2"])
            L_beta = infonce_loss(z_ev1, z_ev2, temperature)

            # L_drop already carries its per-chain weights (lam_db/lam_da); add alpha SimCSE (lam_as, 0
            # in legacy mode). This reduces EXACTLY to the old lambda_drop*(drop_a+drop_b) when legacy.
            loss = (lambda_mlm * (mlm_p + mlm_e)
                    + lambda_pair * L_pair
                    + L_drop
                    + lam_as * L_alpha_simcse
                    + lambda_betaonly * L_beta)

        if scaler is not None:          # fp16 path (loss scaling)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:                            # bf16 or fp32: plain backward/step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        global_step += 1
        pbar.set_postfix({
            "loss":  f"{loss.item():.4f}", "mlm_p": f"{mlm_p.item():.3f}", "mlm_e": f"{mlm_e.item():.3f}",
            "pair":  f"{L_pair.item():.3f}", "drop_b": f"{L_drop_beta.item():.3f}",
            "a_sim": f"{L_alpha_simcse.item():.3f}", "beta": f"{L_beta.item():.3f}",
            "lr":    f"{scheduler.get_last_lr()[0]:.2e}",
        })
        if exp:
            exp.log_metric("loss",          loss.item(),          step=global_step)
            exp.log_metric("mlm_pair_loss", mlm_p.item(),         step=global_step)
            exp.log_metric("mlm_emer_loss", mlm_e.item(),         step=global_step)
            exp.log_metric("pair_loss",     L_pair.item(),        step=global_step)
            exp.log_metric("drop_loss",     L_drop.item(),        step=global_step)
            exp.log_metric("drop_beta",     L_drop_beta.item(),   step=global_step)
            exp.log_metric("drop_alpha",    L_drop_alpha.item(),  step=global_step)
            exp.log_metric("alpha_simcse",  L_alpha_simcse.item(),step=global_step)
            exp.log_metric("betaonly_loss", L_beta.item(),        step=global_step)

    return total_loss / max(1, n_steps), global_step
