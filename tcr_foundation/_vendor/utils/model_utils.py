"""
Model initialization and management utilities for TCR-BERT training.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import BertConfig, BertForMaskedLM


# ================= AUXILIARY HEADS =================

class PropertyPredictionHead(nn.Module):
    """
    Predicts physicochemical properties from CLS embedding during pretraining.

    Simple linear projection from hidden_size to num_properties.
    Used alongside MLM to enrich the CLS token representation.
    """

    def __init__(self, hidden_size: int, num_properties: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_properties)

    def forward(self, cls_embedding):
        """
        Args:
            cls_embedding: [B, hidden_size] CLS token embedding
        Returns:
            [B, num_properties] predicted property values
        """
        return self.head(cls_embedding)


class VGenePrototypes(nn.Module):
    """
    Learnable V gene prototype embeddings for auxiliary pretraining loss.

    Operates on the CLS token from the last BERT hidden layer.
    V gene identity is fully germline-encoded — it's a structural property
    of the sequence, not related to CD4/CD8 biology — so it belongs in
    pretraining alongside MLM and physicochemical property prediction.

    Teaching the CLS token to encode V gene structure during pretraining
    enriches the representation for all downstream tasks (soNNia selection,
    classification) without polluting the finetuning objective with a
    non-task-relevant auxiliary.
    """

    def __init__(self, hidden_size: int, num_v_genes: int, temperature: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_v_genes, hidden_size))
        nn.init.xavier_uniform_(self.prototypes)
        self.temperature = temperature

    def forward(self, pooled, v_gene_ids):
        """
        Args:
            pooled: [B, hidden_size] pooled representation (after pooling strategy)
            v_gene_ids: [B] integer V gene labels
        Returns:
            prototype_loss: scalar cross-entropy loss
        """
        emb_norm = F.normalize(pooled, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)

        # Cosine similarity: [B, num_v_genes]
        sim = torch.mm(emb_norm, proto_norm.t()) / self.temperature

        return F.cross_entropy(sim, v_gene_ids)


# ================= BERT POOLER =================

class BertPooler(nn.Module):
    """
    Configurable BERT pooling module.

    Extracts a fixed-dimensional representation from BERT's token outputs,
    supporting three strategies:

        cls:   [CLS] token directly. Simple and fast; CLS was enriched during
               pretraining to summarise sequence-level properties.
        mixed: Inception-style [max + avg + attention-weighted] pooling over
               sequence tokens (excluding CLS), projected back to hidden_size,
               then added to CLS via a skip connection. The skip guarantees
               performance >= CLS-only: even if max/avg/attn paths add no value,
               the network can zero the projection weights and use CLS alone
               (same principle as ResNet residual connections).
        wk:    Weighted sum of all BERT layer outputs (learned per-layer scalar
               weights, softmax-normalised), then [CLS] of that weighted sum.
               Different layers capture different linguistic levels; WK lets the
               model learn which layers are most informative for this task.

    All strategies produce output_dim = hidden_size.

    Shared across TCRBertClassifier and TCRBertSoNNia so all downstream heads
    operate on an identical representation computed once per forward pass.
    """

    def __init__(self, pooling: str, hidden_size: int, num_layers: int,
                 dropout: float = 0.0):
        """
        Args:
            pooling:     Pooling strategy — "cls", "mixed", or "wk"
            hidden_size: BERT hidden dimension
            num_layers:  Number of BERT encoder layers (needed by WK for
                         weight vector length = num_layers + 1 embedding layer)
            dropout:     Dropout applied inside mixed pooling's projection layer
        """
        super().__init__()
        self.pooling = pooling

        self.attention = None
        self.concat_projection = None
        self.layer_weights = None

        if pooling == "mixed":
            self.attention = nn.Linear(hidden_size, 1)
            self.concat_projection = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif pooling == "wk":
            # +1 for embedding layer (index 0 in hidden_states tuple)
            self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))

        self._output_dim = hidden_size  # All strategies output hidden_size

    @property
    def output_dim(self) -> int:
        """Output dimension of the pooled representation."""
        return self._output_dim

    @property
    def needs_all_hidden_states(self) -> bool:
        """Whether BERT must be called with output_hidden_states=True."""
        return self.pooling == "wk"

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor = None,
                all_hidden_states=None) -> torch.Tensor:
        """
        Args:
            hidden:            [B, seq_len, hidden_size] last hidden state from BERT
            attention_mask:    [B, seq_len] mask (0 for padding, 1 for real tokens)
            all_hidden_states: tuple of [B, seq_len, hidden_size] from all BERT layers;
                               required only for pooling="wk" (pass out.hidden_states)
        Returns:
            [B, hidden_size] pooled representation
        """
        if self.pooling == "mixed":
            cls = hidden[:, 0, :]       # [B, hidden] — skip connection
            seq = hidden[:, 1:, :]      # [B, seq_len-1, hidden] — non-CLS tokens

            if attention_mask is not None:
                seq_mask = attention_mask[:, 1:]
                seq_mask_exp = seq_mask.unsqueeze(-1).float()
                seq_for_max = seq.masked_fill(seq_mask_exp == 0, float('-inf'))
                seq_for_avg = seq * seq_mask_exp
                seq_lengths = seq_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            else:
                seq_for_max = seq
                seq_for_avg = seq
                seq_lengths = torch.tensor(seq.size(1), device=seq.device).float()

            max_pooled, _ = seq_for_max.max(dim=1)
            avg_pooled = seq_for_avg.sum(dim=1) / seq_lengths

            attn_scores = self.attention(seq).squeeze(-1)
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(seq_mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            attn_pooled = (seq * attn_weights).sum(dim=1)

            concat = torch.cat([max_pooled, avg_pooled, attn_pooled], dim=-1)
            return self.concat_projection(concat) + cls             # [B, hidden]

        elif self.pooling == "wk":
            weights = torch.softmax(self.layer_weights, dim=0)
            stacked = torch.stack(all_hidden_states, dim=0)         # [L+1, B, seq, h]
            weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
            return weighted[:, 0, :]                                # [B, hidden]

        else:  # CLS (default)
            return hidden[:, 0, :]                                  # [B, hidden]


# ================= MODEL CLASSES =================

class TCRBertClassifier(nn.Module):
    """
    Binary classifier wrapping BERT encoder for CD4 vs CD8 prediction.

    Pooling strategies:
        - "cls": [CLS] only
        - "mixed": Inception-style parallel pooling with CLS skip connection:
                   [MaxPool, AvgPool, AttentionPool] → concat → project + CLS (skip)
                   Captures: strongest signals (max), overall character (avg),
                   learned position importance (attn, useful for CDR1/CDR2)
                   CLS skip ensures "at least as good as CLS-only" performance
        - "wk": Weighted sum of all BERT layers

    Head architecture (configurable):
        - head_layers=1: input → 2
        - head_layers=2: input → hidden → GELU → Dropout → 2
        - head_layers=3: input → h1 → GELU → Dropout → h2 → GELU → Dropout → 2

    V gene prototype learning has been moved to pretraining (see VGenePrototypes).
    The classifier focuses purely on CD4/CD8 discrimination.
    """

    def __init__(self, model_name: str, dropout: float = 0.25, label_smoothing: float = 0.1,
                 pooling: str = "cls", head_layers: int = 1, head_hidden: list = None):
        """
        Args:
            model_name: HuggingFace model identifier or local path
            dropout: Dropout probability for classifier head
            label_smoothing: Label smoothing factor for cross-entropy loss
            pooling: Pooling strategy - "cls", "mixed", or "wk"
            head_layers: Number of linear layers in classifier head (1, 2, or 3)
            head_hidden: List of intermediate dimensions (length = head_layers - 1)
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.bert.config.hidden_size
        num_layers = self.bert.config.num_hidden_layers

        self.pooler = BertPooler(pooling, hidden_size, num_layers, dropout)
        input_dim = self.pooler.output_dim

        # Determine intermediate dimensions for classification head
        if head_hidden is None:
            if head_layers == 1:
                head_hidden = []
            elif head_layers == 2:
                head_hidden = [min(input_dim, 256)]
            else:  # 3+ layers
                head_hidden = [input_dim, input_dim // 2]

        # Store head config for save/load
        self.head_layers = head_layers
        self.head_hidden = list(head_hidden) if head_hidden else []

        # Build classification head dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.head_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))  # Final output layer

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Segment IDs (optional)
            labels: Binary labels for loss calculation (optional)

        Returns:
            Dictionary with keys: loss (if labels provided), logits
        """
        out = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            output_hidden_states=self.pooler.needs_all_hidden_states
        )
        hidden = out.last_hidden_state  # [B, seq_len, hidden]

        pooled = self.pooler(
            hidden, attention_mask,
            all_hidden_states=out.hidden_states if self.pooler.needs_all_hidden_states else None
        )

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ================= MODEL INITIALIZATION =================

def init_mlm_model(model_path: str, device: torch.device):
    """
    Initialize Masked Language Model for pretraining.

    Args:
        model_path: Path to pretrained model or HuggingFace identifier
        device: Device to load model on (CPU or CUDA)

    Returns:
        Initialized model in training mode
    """
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.train()
    return model


def init_gpt_model(model_path: str, device: torch.device):
    """
    Initialize a GPT-style causal language model for autoregressive pretraining.

    Args:
        model_path: Path to model config directory (untrained) or checkpoint
        device: Device to load model on

    Returns:
        Initialized model in training mode
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    model.train()
    return model


def init_classifier(model_name: str, dropout: float, label_smoothing: float,
                   device: torch.device, pooling: str = "cls",
                   head_layers: int = 1, head_hidden: list = None) -> TCRBertClassifier:
    """
    Initialize binary classifier.

    Args:
        model_name: Pretrained BERT model path or HuggingFace identifier
        dropout: Dropout probability
        label_smoothing: Label smoothing factor
        device: Device to load model on
        pooling: Pooling strategy - "cls", "mixed", or "wk"
        head_layers: Number of linear layers in classifier head (1, 2, or 3)
        head_hidden: List of intermediate dimensions (length = head_layers - 1)

    Returns:
        Initialized TCRBertClassifier in training mode
    """
    model = TCRBertClassifier(model_name, dropout, label_smoothing, pooling,
                              head_layers, head_hidden)
    model.to(device)
    model.train()
    return model


# ================= MODEL MANIPULATION =================

def freeze_bert_layers(model: nn.Module, freeze_all: bool = True) -> None:
    """
    Freeze BERT encoder parameters for warmup phase.

    Args:
        model: TCRBertClassifier instance
        freeze_all: If True, freeze entire BERT; if False, freeze only embeddings

    Side Effects:
        Sets requires_grad=False for frozen parameters
    """
    if freeze_all:
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        # Freeze only embeddings
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False


def unfreeze_last_n_layers(model: nn.Module, n_layers: int) -> None:
    """
    Unfreeze last N encoder layers for fine-tuning.

    Args:
        model: TCRBertClassifier instance
        n_layers: Number of layers to unfreeze from the end

    Example:
        If n_layers=6 and model has 12 layers, unfreeze layers 6-11
    """
    # First, ensure all BERT is frozen
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze last N layers
    num_layers = len(model.bert.encoder.layer)
    start_layer = max(0, num_layers - n_layers)

    for layer in model.bert.encoder.layer[start_layer:]:
        for param in layer.parameters():
            param.requires_grad = True


# ================= MODEL SAVING/LOADING =================

def save_model_checkpoint(model: nn.Module, save_path: str) -> None:
    """
    Save model to disk.

    The tokenizer is the standard TCR-BERT tokenizer and is not saved here —
    load it from the original base model path when needed.

    Args:
        model: Model to save (MLM or TCRBertClassifier)
        save_path: Directory to save checkpoint

    Saves:
        - Model weights and config
        - Classifier head (if TCRBertClassifier)
    """
    os.makedirs(save_path, exist_ok=True)

    # Save model
    if hasattr(model, "bert"):
        # TCRBertClassifier - save BERT and classifier separately
        model.bert.save_pretrained(save_path)
        torch.save(
            model.classifier.state_dict(),
            os.path.join(save_path, "classifier_head.pt")
        )
        config = {
            "dropout": model.dropout.p,
            "label_smoothing": model.loss_fn.label_smoothing,
            "pooling": model.pooling,
            "head_layers": model.head_layers,
            "head_hidden": model.head_hidden,
        }
        torch.save(config, os.path.join(save_path, "classifier_config.pt"))

        # Save BertPooler weights
        torch.save(
            model.pooler.state_dict(),
            os.path.join(save_path, "bert_pooler.pt")
        )
    else:
        # MLM model - direct save
        model.save_pretrained(save_path)

    print(f"Model saved to {save_path}")


def load_classifier_checkpoint(save_path: str, device: torch.device) -> "TCRBertClassifier":
    """
    Load a saved TCRBertClassifier checkpoint.

    The tokenizer is not stored in the checkpoint — load it separately from
    the original base model path if needed.

    Args:
        save_path: Directory containing the checkpoint
        device: Device to load model on

    Returns:
        Loaded TCRBertClassifier in eval mode
    """
    # Load config
    config = torch.load(os.path.join(save_path, "classifier_config.pt"))

    # Initialize model
    model = TCRBertClassifier(
        save_path,  # BERT saved in same directory
        dropout=config["dropout"],
        label_smoothing=config["label_smoothing"],
        pooling=config.get("pooling", "cls"),
        head_layers=config.get("head_layers", 1),
        head_hidden=config.get("head_hidden", None),
    )

    # Load classifier head weights
    classifier_state = torch.load(os.path.join(save_path, "classifier_head.pt"))
    if "weight" in classifier_state and "0.weight" not in classifier_state:
        print("WARNING: Old checkpoint format (single Linear). Weights NOT loaded for head.")
    else:
        model.classifier.load_state_dict(classifier_state)

    # Load BertPooler weights
    pooler_path = os.path.join(save_path, "bert_pooler.pt")
    if os.path.exists(pooler_path):
        model.pooler.load_state_dict(torch.load(pooler_path))
    else:
        print("WARNING: bert_pooler.pt not found — pooler weights not loaded.")

    model.to(device)
    model.eval()

    return model


# ================= SONIA MODEL COMPONENTS =================

class SelectionHead(nn.Module):
    """
    Scalar selection factor head: pooled representation → log Q score.

    One instance per cell type (Q^CD4 and Q^CD8). Takes the output of
    pooler_selection and maps it to a single scalar log-odds score that
    measures how much more likely the sequence is under cell-type-specific
    thymic selection compared to random VDJ recombination (OLGA baseline).

    Architecture: Linear(input_dim → input_dim//2) → GELU → Dropout → Linear(→ 1)
    GELU matches BERT's internal FFN activation and avoids the dead-neuron
    problem of ReLU on BERT's typically small-magnitude representations.
    No activation on the output — the raw scalar is the log Q score used
    directly by the soNNia loss or treated as a logit by the BCE loss.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1, linear: bool = False):
        """
        Args:
            input_dim: Dimension of the pooled representation (= hidden_size)
            dropout:   Dropout probability applied after GELU (ignored when linear=True)
            linear:    If True, use a single linear layer (matches soNNia architecture).
                       If False (default), use a 2-layer MLP with GELU.
        """
        super().__init__()
        if linear:
            self.mlp = nn.Linear(input_dim, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1)
            )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [B, input_dim] pooled sequence representation
        Returns:
            [B] scalar log Q score per sequence
        """
        return self.mlp(pooled).squeeze(-1)


class SoNNiaClassifierHead(nn.Module):
    """
    CD4/CD8 classifier that combines the pooled representation with
    selection scores from the frozen Q heads.

    Input: [pooled_cls || detach(score_cd4) || detach(score_cd8)]

    pooled_cls comes from pooler_classifier (trainable in stage 3b) and
    is free to specialise for classification. The two score scalars carry
    the biologically-grounded selection signal from the frozen stage-3a
    branch. Detaching the scores is belt-and-suspenders: even if something
    were accidentally left unfrozen, classifier gradients cannot reach the
    Q heads via the score pathway.

    Architecture: Linear(input_dim+2 → input_dim//2) → GELU → Dropout → Linear(→ 1)
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of pooled_cls (= hidden_size).
                       The actual first Linear layer receives input_dim + 2
                       because two score scalars are appended.
            dropout:   Dropout probability applied after GELU
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, pooled_cls: torch.Tensor,
                score_cd4: torch.Tensor,
                score_cd8: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_cls: [B, input_dim] from pooler_classifier
            score_cd4:  [B] log Q^CD4 score (detached internally)
            score_cd8:  [B] log Q^CD8 score (detached internally)
        Returns:
            [B] logit for CD4 (positive = 1) vs CD8 (negative = 0)
        """
        x = torch.cat([
            pooled_cls,
            score_cd4.detach().unsqueeze(-1),
            score_cd8.detach().unsqueeze(-1)
        ], dim=-1)
        return self.mlp(x).squeeze(-1)


class TCRBertSoNNia(nn.Module):
    """
    TCR-BERT with soNNia-style selection modelling.

    Two-branch twin-pooler architecture:

        BERT encoder (always frozen in stages 3a and 3b)
                |
        ┌───────┴────────────────────────────────┐
        │                                        │
     pooler_selection                   pooler_classifier
     (trained in 3a, frozen in 3b)     (warm-start from selection
        │                               after 3a, trainable in 3b)
        ├── Q^CD4 head  (trained 3a)            │
        ├── Q^CD8 head  (trained 3a)         pooled_cls
        │                                       │
     score_cd4, score_cd8   [pooled_cls || detach(score_cd4) || detach(score_cd8)]
                                                ↓
                                      Classifier head (trained 3b)
                                                ↓
                                        CD4 / CD8 prediction

    The two poolers share the same architecture (same pooling strategy) but
    have separate parameters. This lets the classifier branch adapt its
    representation in 3b without drifting the Q head inputs away from what
    those heads were trained on.

    Stage transitions:
        Before 3a : call freeze_bert()
        After  3a : call freeze_selection_branch(), then copy_pooler_to_classifier()
        During 3b : only pooler_classifier and classifier are trainable
    """

    def __init__(self, model_name: str, pooling: str = "cls", dropout: float = 0.1,
                 linear_heads: bool = False):
        """
        Args:
            model_name:    HuggingFace model identifier or local path to pretrained BERT
            pooling:       Pooling strategy for both poolers — "cls", "mixed", or "wk"
            dropout:       Dropout probability used in mixed pooling projections and
                           the hidden layers of SelectionHead and SoNNiaClassifierHead
            linear_heads:  If True, Q heads are single linear layers (soNNia-style).
                           If False (default), Q heads are 2-layer MLPs.
        """
        super().__init__()
        self.pooling = pooling

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        num_layers = self.bert.config.num_hidden_layers

        # Three poolers: two independent selection poolers (one per head) + classifier pooler
        # pooler_cd4 receives gradients only from L_cd4, pooler_cd8 only from L_cd8 —
        # preventing the shared-representation head collapse seen with a single pooler_selection.
        self.pooler_cd4 = BertPooler(pooling, hidden_size, num_layers, dropout)
        self.pooler_cd8 = BertPooler(pooling, hidden_size, num_layers, dropout)
        self.pooler_classifier = BertPooler(pooling, hidden_size, num_layers, dropout)

        # Selection heads — one per cell type
        self.q_cd4 = SelectionHead(hidden_size, dropout, linear=linear_heads)
        self.q_cd8 = SelectionHead(hidden_size, dropout, linear=linear_heads)

        # Classifier head — takes pooled_cls + two detached scores
        self.classifier = SoNNiaClassifierHead(hidden_size, dropout)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                precomputed_score_cd4: torch.Tensor = None,
                precomputed_score_cd8: torch.Tensor = None) -> tuple:
        """
        Full forward pass through both branches.

        In stage 3a, both branches share the same frozen BERT hidden states.
        In stage 3b, pre-computed Q scores are passed in directly — the selection
        branch is bypassed entirely and only the BERT + classifier branch runs.
        This allows BERT to fine-tune for classification without corrupting the
        Q head scores (which were trained on the original frozen representations).

        Args:
            input_ids:              [B, seq_len] token IDs
            attention_mask:         [B, seq_len] 1 = real token, 0 = padding
            token_type_ids:         [B, seq_len] optional segment IDs
            precomputed_score_cd4:  [B] cached Q^CD4 scores from stage 3a (stage 3b only)
            precomputed_score_cd8:  [B] cached Q^CD8 scores from stage 3a (stage 3b only)

        Returns:
            Tuple (score_cd4, score_cd8, logit):
                score_cd4: [B] log Q^CD4 per sequence
                score_cd8: [B] log Q^CD8 per sequence
                logit:     [B] CD4 vs CD8 classification logit
        """
        needs_all = (self.pooler_cd4.needs_all_hidden_states or
                     self.pooler_cd8.needs_all_hidden_states or
                     self.pooler_classifier.needs_all_hidden_states)

        out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=needs_all
        )
        hidden = out.last_hidden_state
        all_hidden = out.hidden_states if needs_all else None

        if precomputed_score_cd4 is not None:
            # Stage 3b: skip frozen selection branch, use cached scores
            score_cd4 = precomputed_score_cd4
            score_cd8 = precomputed_score_cd8
        else:
            # Stage 3a: independent poolers — gradients cannot mix between heads
            score_cd4 = self.q_cd4(self.pooler_cd4(hidden, attention_mask, all_hidden))
            score_cd8 = self.q_cd8(self.pooler_cd8(hidden, attention_mask, all_hidden))

        pooled_cls = self.pooler_classifier(hidden, attention_mask, all_hidden)
        logit = self.classifier(pooled_cls, score_cd4, score_cd8)

        return score_cd4, score_cd8, logit

    # ---- Stage transition helpers ----

    def freeze_bert(self) -> None:
        """
        Freeze the BERT encoder.
        Call this before stage 3a begins — the encoder stays frozen through 3b.
        """
        for param in self.bert.parameters():
            param.requires_grad = False

    def freeze_selection_branch(self) -> None:
        """
        Freeze both selection poolers and both Q heads.
        Call this after stage 3a converges, before starting stage 3b.
        """
        for param in self.pooler_cd4.parameters():
            param.requires_grad = False
        for param in self.pooler_cd8.parameters():
            param.requires_grad = False
        for param in self.q_cd4.parameters():
            param.requires_grad = False
        for param in self.q_cd8.parameters():
            param.requires_grad = False

    def copy_pooler_to_classifier(self) -> None:
        """
        Warm-start pooler_classifier from the average of pooler_cd4 and pooler_cd8 weights.

        Call this immediately after freeze_selection_branch(), before 3b.
        Averaging gives the classifier a balanced starting point that captures
        both CD4 and CD8 selection representations.
        """
        sd4 = self.pooler_cd4.state_dict()
        sd8 = self.pooler_cd8.state_dict()
        avg = {k: (sd4[k] + sd8[k]) / 2 for k in sd4}
        self.pooler_classifier.load_state_dict(avg)


# ================= SONIA SAVE/LOAD =================

def save_sonia_checkpoint(model: TCRBertSoNNia, save_path: str, stage: str) -> None:
    """
    Save a TCRBertSoNNia checkpoint after completing a training stage.

    The tokenizer is the standard TCR-BERT tokenizer and is not saved here —
    load it from the original pretrained BERT path when needed.

    Args:
        model:     TCRBertSoNNia instance
        save_path: Directory to save checkpoint
        stage:     "3a" or "3b" — recorded in config so load knows what was trained

    Saves:
        bert config + weights  — BERT encoder
        pooler_selection.pt    — selection branch pooler
        pooler_classifier.pt   — classifier branch pooler
        q_cd4_head.pt          — Q^CD4 head weights
        q_cd8_head.pt          — Q^CD8 head weights
        classifier_head.pt     — SoNNiaClassifierHead weights
        sonia_config.pt        — pooling strategy and completed stage
    """
    os.makedirs(save_path, exist_ok=True)

    model.bert.save_pretrained(save_path)

    torch.save(model.pooler_cd4.state_dict(),
               os.path.join(save_path, "pooler_cd4.pt"))
    torch.save(model.pooler_cd8.state_dict(),
               os.path.join(save_path, "pooler_cd8.pt"))
    torch.save(model.pooler_classifier.state_dict(),
               os.path.join(save_path, "pooler_classifier.pt"))
    torch.save(model.q_cd4.state_dict(),
               os.path.join(save_path, "q_cd4_head.pt"))
    torch.save(model.q_cd8.state_dict(),
               os.path.join(save_path, "q_cd8_head.pt"))
    torch.save(model.classifier.state_dict(),
               os.path.join(save_path, "classifier_head.pt"))
    torch.save({"pooling": model.pooling, "stage": stage,
                "linear_heads": isinstance(model.q_cd4.mlp, nn.Linear)},
               os.path.join(save_path, "sonia_config.pt"))

    print(f"SoNNia checkpoint (stage {stage}) saved to {save_path}")


def load_sonia_checkpoint(save_path: str, device: torch.device) -> tuple:
    """
    Load a saved TCRBertSoNNia checkpoint.

    The tokenizer is not stored in the checkpoint — load it separately from
    the original pretrained BERT path if needed.

    Args:
        save_path: Directory containing the checkpoint
        device:    Device to load model on

    Returns:
        Tuple of (model, stage) where stage is "3a" or "3b"
    """
    config = torch.load(os.path.join(save_path, "sonia_config.pt"))

    # Determine linear_heads: prefer saved flag, fall back to key inspection for
    # old checkpoints that were saved before the flag was added.
    if "linear_heads" in config:
        linear_heads = config["linear_heads"]
    else:
        q_head_path = os.path.join(save_path, "q_cd4_head.pt")
        q_keys = torch.load(q_head_path, map_location="cpu").keys()
        linear_heads = "mlp.weight" in q_keys  # linear: "mlp.weight"; MLP: "mlp.0.weight"

    model = TCRBertSoNNia(
        model_name=save_path,
        pooling=config["pooling"],
        linear_heads=linear_heads,
    )

    def _load(filename: str, target: nn.Module) -> None:
        path = os.path.join(save_path, filename)
        if os.path.exists(path):
            target.load_state_dict(torch.load(path))
        else:
            print(f"WARNING: {filename} not found — weights not loaded.")

    _load("pooler_cd4.pt",        model.pooler_cd4)
    _load("pooler_cd8.pt",        model.pooler_cd8)
    _load("pooler_classifier.pt", model.pooler_classifier)
    _load("q_cd4_head.pt",        model.q_cd4)
    _load("q_cd8_head.pt",        model.q_cd8)
    _load("classifier_head.pt",   model.classifier)

    model.to(device)
    model.eval()
    return model, config.get("stage", "unknown")


# ================= PAIRED CHAIN MODEL =================

class TCRBertPairing(nn.Module):
    """
    Siamese TCR-BERT for paired alpha-beta chain pretraining (PMI framework, Exp 6+).

    Builds PMI-geometry at three levels through InfoNCE:
        - alpha-space  via SimCSE on emb_alpha (two masked views)
        - beta-space   via SimCSE on emb_beta  (two masked views)
        - pair-space   via SimCSE on z = pair_projector(cat[alpha, alpha-beta, beta])

    Cross-chain alignment (Exp 6, CLIP-style): align_alpha / align_beta project each
    chain into a shared subspace where InfoNCE(align_alpha(emb_alpha), align_beta(emb_beta))
    encodes pairing PMI without merging the chain-specific PMI-geometries.

    Theoretical basis: [[infonce_geometry]] (Tehenan EMNLP 2025).
    """

    def __init__(self, bert, hidden_size: int, num_layers: int,
                 pooling: str = "mixed", dropout: float = 0.1,
                 shared_pooler: bool = False,
                 pmi_variant: str = "clip",
                 proxy_alpha_init: torch.Tensor = None,
                 proxy_beta_init:  torch.Tensor = None):
        super().__init__()
        self.bert          = bert
        self.shared_pooler = shared_pooler
        self.pmi_variant   = pmi_variant
        self.pooler_alpha  = BertPooler(pooling, hidden_size, num_layers, dropout)
        self.pooler_beta   = self.pooler_alpha if shared_pooler else BertPooler(pooling, hidden_size, num_layers, dropout)
        self.pair_projector = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        if pmi_variant == "clip":
            # Exp 6: separate projectors map each chain into a shared subspace.
            self.align_alpha = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.align_beta = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif pmi_variant == "chaintoken":
            # Exp 7: learnable proxy tokens stand in for the missing chain.
            # Init from chain-type word embeddings ([unused0]=alpha, [unused1]=beta)
            # which carry "alpha-ness" / "beta-ness" from MLM pretraining.
            if proxy_alpha_init is None:
                proxy_alpha_init = torch.randn(hidden_size) * 0.02
            if proxy_beta_init is None:
                proxy_beta_init  = torch.randn(hidden_size) * 0.02
            self.proxy_alpha = nn.Parameter(proxy_alpha_init.detach().clone())
            self.proxy_beta  = nn.Parameter(proxy_beta_init.detach().clone())
        elif pmi_variant == "doubledist":
            # Exp 8: two bridge projectors map each chain directly into pair-space.
            # No explicit alpha<->beta term -- alignment emerges from shared z_paired target.
            self.bridge_alpha = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.bridge_beta = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown pmi_variant: {pmi_variant!r} (expected 'clip', 'chaintoken', or 'doubledist')")

    def _encode(self, input_ids, attention_mask, labels, pooler,
                return_logits=False, return_hidden=False):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        all_hidden = out.hidden_states if pooler.needs_all_hidden_states else None
        emb = pooler(out.hidden_states[-1], attention_mask, all_hidden)
        if return_logits:
            return out.loss, emb, out.logits
        if return_hidden:
            return out.loss, emb, out.hidden_states
        return out.loss, emb

    def forward(self, alpha_input_ids, alpha_attention_mask, alpha_labels,
                beta_input_ids, beta_attention_mask, beta_labels):
        mlm_a, emb_a = self._encode(alpha_input_ids, alpha_attention_mask, alpha_labels, self.pooler_alpha)
        mlm_b, emb_b = self._encode(beta_input_ids,  beta_attention_mask,  beta_labels,  self.pooler_beta)
        return mlm_a, mlm_b, emb_a, emb_b


# ================= JOINT PAIRED MODEL (Exp 9) =================

class TCRBertJoint(nn.Module):
    """
    Joint (single-sequence) paired TCR encoder (Exp 9, SCEPTR-style).

    The full paired TCR is encoded as ONE sequence so BERT self-attention models
    inter-chain (alpha<->beta) interactions directly -- unlike the siamese TCRBertPairing
    where chains are encoded separately and combined only by a late linear pair_projector.

    A single mixed pooler produces the sequence-level embedding z. Pre-training combines
    MLM, autocontrastive (SimCSE on two masked views), and chain-drop consistency.

    Theoretical basis: [[infonce_geometry]] / [[embedding_geometry]].
    """

    def __init__(self, bert, hidden_size: int, num_layers: int,
                 pooling: str = "mixed", dropout: float = 0.1):
        super().__init__()
        self.bert   = bert
        self.pooler = BertPooler(pooling, hidden_size, num_layers, dropout)

    def _encode(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
        )
        all_hidden = out.hidden_states if self.pooler.needs_all_hidden_states else None
        z = self.pooler(out.hidden_states[-1], attention_mask, all_hidden)
        return out.loss, z

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        return self._encode(input_ids, attention_mask, labels, token_type_ids)


# ================= TCR FOUNDATION STAGE 1 (beta-only backbone) =================

class TCRBetaOnlyEncoder(nn.Module):
    """
    Stage 1 backbone for TCRFoundation: BertForMaskedLM + a single mixed pooler
    (`pooler_beta`). Trained with MLM + InfoNCE (autocontrastive) on the deduplicated
    Emerson beta-only corpus.

    Once Stage 1 completes, the encoder + MLM head become the foundation's frozen
    P(beta) prior; the pooler_beta is the frozen beta-only embedding head. Both are
    consumed by Stage 2 (paired LoRA) and every downstream task.

    Forward signature mirrors TCRBertJoint._encode for training-loop reuse:
      _encode(input_ids, attention_mask, labels, token_type_ids, position_ids) -> (mlm_loss, z)
    """

    def __init__(self, bert, hidden_size: int, num_layers: int,
                 pooling: str = "mixed", dropout: float = 0.1):
        super().__init__()
        self.bert   = bert
        self.pooler = BertPooler(pooling, hidden_size, num_layers, dropout)
        # Per-chain attention support: if the backbone was built with build_perchain_mlm it
        # carries `_perchain_attns`; we set rel_idx on those before every forward.
        self.perchain = getattr(bert, "_perchain_attns", None)
        self.max_pos  = bert.config.max_position_embeddings

    def _encode(self, input_ids, attention_mask, labels=None, token_type_ids=None,
                position_ids=None):
        if self.perchain is not None:
            from utils.perchain_attention import apply_rel_idx
            apply_rel_idx(self.perchain, position_ids, token_type_ids, self.max_pos)
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        all_hidden = out.hidden_states if self.pooler.needs_all_hidden_states else None
        z = self.pooler(out.hidden_states[-1], attention_mask, all_hidden)
        return out.loss, z

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None,
                position_ids=None):
        return self._encode(input_ids, attention_mask, labels, token_type_ids, position_ids)


# Single-stage joint TCRFoundation reuses this exact module: one MLM head + one mixed pooler, and
# an input-agnostic _encode that handles BOTH paired and beta-only views (chain identity rides on
# token_type_ids / per-chain position_ids, not on separate heads/poolers). Alias for clarity.
TCRFoundationJoint = TCRBetaOnlyEncoder


# ================= PER-CHAIN RPE BACKBONE BUILDERS =================

def build_perchain_mlm(arch: dict) -> BertForMaskedLM:
    """
    Build a fresh BertForMaskedLM whose self-attention is replaced by
    PerChainRelativeSelfAttention (per-chain relative positions + a cross-chain bucket).

    `position_embedding_type` is set to "relative_key_query" so BertEmbeddings does NOT add
    absolute position embeddings -- the (per-chain) relative signal lives entirely in the
    swapped attention. The list of inserted attention modules is stashed on the model as
    `model._perchain_attns` so the training loop can set rel_idx via
    perchain_attention.apply_rel_idx before each forward.

    Args:
        arch: dict with hidden_size, num_hidden_layers, num_attention_heads,
              intermediate_size, max_position_embeddings, type_vocab_size, vocab_size,
              and optional hidden_act / dropout / pad_token_id.

    Returns:
        BertForMaskedLM (fresh random init) with per-chain attention.
    """
    from utils.perchain_attention import swap_to_perchain

    cfg = BertConfig(
        hidden_size              = arch["hidden_size"],
        num_hidden_layers        = arch["num_hidden_layers"],
        num_attention_heads      = arch["num_attention_heads"],
        intermediate_size        = arch["intermediate_size"],
        max_position_embeddings  = arch["max_position_embeddings"],
        position_embedding_type  = "relative_key_query",   # -> no absolute PE in embeddings
        type_vocab_size          = arch.get("type_vocab_size", 2),
        vocab_size               = arch.get("vocab_size", 26),
        hidden_act               = arch.get("hidden_act", "gelu"),
        hidden_dropout_prob      = arch.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob = arch.get("attention_probs_dropout_prob", 0.1),
        layer_norm_eps           = arch.get("layer_norm_eps", 1e-12),
        pad_token_id             = arch.get("pad_token_id", 21),
    )
    cfg._attn_implementation = "eager"   # we replace attention.self regardless
    model = BertForMaskedLM(cfg)
    model._perchain_attns = swap_to_perchain(model.bert)
    return model


def load_perchain_mlm(path: str) -> BertForMaskedLM:
    """
    Reload a per-chain BertForMaskedLM saved by build_perchain_mlm + save_pretrained.

    Plain from_pretrained cannot be used: the swapped attention is not in the config, and our
    distance_embedding has 2*max rows (HF's BertSelfAttention has 2*max-1). So we rebuild the
    per-chain architecture from the saved config, then load the state dict (keys + shapes
    match the swapped modules).

    Args:
        path: directory written by BertForMaskedLM.save_pretrained (config.json + safetensors).

    Returns:
        BertForMaskedLM with per-chain attention and the loaded weights; `_perchain_attns` set.
    """
    cfg = BertConfig.from_pretrained(path)
    arch = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "type_vocab_size": cfg.type_vocab_size,
        "vocab_size": cfg.vocab_size,
        "hidden_act": cfg.hidden_act,
        "pad_token_id": cfg.pad_token_id,
    }
    model = build_perchain_mlm(arch)
    # Load the saved weights (safetensors preferred, else pytorch bin).
    state = None
    st = os.path.join(path, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        state = load_file(st)
    else:
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()   # MLM decoder is tied to word embeddings (not stored separately)
    # Tolerate (a) the unused absolute position_ids buffer and (b) tied MLM decoder keys.
    benign = ("position_ids", "cls.predictions.decoder")
    hard_missing = [k for k in missing if not any(b in k for b in benign)]
    if hard_missing or unexpected:
        print(f"load_perchain_mlm: missing={hard_missing}  unexpected={unexpected}")
    return model


# ================= TCR FOUNDATION STAGE 2 (paired LoRA) =================

class TCRFoundationStage2(nn.Module):
    """
    Stage 2 paired adapter over a frozen Stage 1 (per-chain) backbone.

    Frozen: the Stage 1 BertForMaskedLM encoder (incl. MLM head, unused here) and its
    distance_embedding WITHIN-chain rows + token_type beta row.
    Trainable (the paired delta):
      - LoRA A/B on bert attention query/value (peft),
      - a fresh `pooler_paired` (mixed pooling),
      - the CROSS-CHAIN row of every layer's distance_embedding (index 2*max-1) -- the only
        new positional signal vs Stage 1; never trained in Stage 1,
      - the ALPHA row of token_type_embeddings (row 0) -- alpha identity, untrained in Stage 1.
    Per-row training is enforced with gradient masks (within-chain / beta rows get zero grad).
    The backbone on disk is NOT overwritten, so the disable-adapter (B1) path always sees the
    pristine Stage 1 weights.

    `encode(input_ids, attention_mask, token_type_ids, position_ids) -> z` (pooled; no MLM).
    """

    def __init__(self, foundation_path: str,
                 lora_rank: int = 16, lora_alpha: int = 32,
                 lora_targets=("query", "value"), lora_dropout: float = 0.0,
                 pooling: str = "mixed", dropout: float = 0.1):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from utils.perchain_attention import apply_rel_idx
        self._apply_rel_idx = apply_rel_idx

        mlm = load_perchain_mlm(os.path.join(foundation_path, "backbone"))
        bert = mlm.bert
        self.perchain = mlm._perchain_attns          # list of PerChainRelativeSelfAttention
        self.max_pos  = bert.config.max_position_embeddings
        hidden_size   = bert.config.hidden_size
        num_layers    = bert.config.num_hidden_layers

        # Freeze the whole backbone first.
        for p in bert.parameters():
            p.requires_grad = False

        # Inject LoRA (peft re-marks LoRA A/B trainable on query/value inside our attention).
        lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                              target_modules=list(lora_targets), lora_dropout=lora_dropout,
                              bias="none", task_type="FEATURE_EXTRACTION", inference_mode=False)
        self.bert = get_peft_model(bert, lora_cfg)

        # Frozen MLM head, re-attached from the Stage 1 backbone (BertForMaskedLM.cls, whose
        # decoder is tied to the word-embeddings). Stage 2 routes an MLM loss THROUGH this
        # frozen head into the trainable adapters -- the head's own weights never update, so
        # the Benchmark B1 publicness PLL (disable_adapter path) stays bit-for-bit Stage 1.
        # NB: the `for p in bert.parameters()` freeze above did NOT cover mlm.cls (it is not
        # part of bert), so we freeze it explicitly here.
        self.mlm_head = mlm.cls
        for p in self.mlm_head.parameters():
            p.requires_grad = False

        # Fresh paired pooler (trainable).
        self.pooler = BertPooler(pooling, hidden_size, num_layers, dropout)

        # Trainable extras with per-row gradient masks.
        self._extra_params = []
        # (a) cross-chain row of every layer's distance_embedding
        for attn in self.perchain:
            w = attn.distance_embedding.weight        # [2*max, head_dim]
            w.requires_grad = True
            mask = torch.zeros_like(w); mask[2 * self.max_pos - 1] = 1.0   # only cross-chain row
            # mask is built on CPU at __init__ (before .to(device)); move it onto the gradient's
            # device at call time so g(cuda) * m stays on one device after the model moves to GPU.
            w.register_hook(lambda g, m=mask: g * m.to(g.device))
            self._extra_params.append(w)
        # (b) alpha row of token_type_embeddings
        tte = bert.embeddings.token_type_embeddings.weight   # [2, hidden]
        tte.requires_grad = True
        tmask = torch.zeros_like(tte); tmask[0] = 1.0          # only alpha row
        tte.register_hook(lambda g, m=tmask: g * m.to(g.device))   # same CPU->grad-device fix
        self._extra_params.append(tte)
        self._token_type_weight = tte

    def _encode_forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        """Shared forward: thread per-chain rel_idx, run the (LoRA-adapted) backbone, pool z.
        Returns (z, last_hidden_state). last_hidden is reused by encode_mlm for the MLM head."""
        self._apply_rel_idx(self.perchain, position_ids, token_type_ids, self.max_pos)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, position_ids=position_ids,
                        output_hidden_states=True)
        all_hidden = out.hidden_states if self.pooler.needs_all_hidden_states else None
        z = self.pooler(out.hidden_states[-1], attention_mask, all_hidden)
        return z, out.hidden_states[-1]

    def encode(self, input_ids, attention_mask, token_type_ids, position_ids):
        """Pooled paired/single-chain embedding (no MLM)."""
        z, _ = self._encode_forward(input_ids, attention_mask, token_type_ids, position_ids)
        return z

    def encode_mlm(self, input_ids, attention_mask, token_type_ids, position_ids):
        """Like encode, but also returns MLM logits from the FROZEN head (same single forward).
        Used for the Stage 2 MLM term: gradients flow through the frozen head into the adapters
        (LoRA + alpha type-row), teaching alpha grammar + re-anchoring beta in the paired regime.
        Returns (z, mlm_logits) where mlm_logits is [B, L, vocab]."""
        z, last_hidden = self._encode_forward(input_ids, attention_mask, token_type_ids, position_ids)
        mlm_logits = self.mlm_head(last_hidden)
        return z, mlm_logits

    def extra_parameters(self):
        """Trainable cross-chain / alpha-type parameters (use a no-weight-decay optimizer group
        so the masked-out rows do not decay)."""
        return list(self._extra_params)

    def count_trainable(self):
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in self.parameters())
        return n_tr, n_tot


# ================= LoRA-BASED DOWNSTREAM WRAPPER =================

class TCRBertJointDownstream(nn.Module):
    """
    Frozen joint-encoder backbone + LoRA adapter + task head, for downstream supervised
    tasks (CD4/CD8 classification, supervised pMHC, pairing predictor, etc.).

    Components:
      - bert (frozen) + pooler (frozen) -- the joint-encoder backbone, loaded from a
        checkpoint dir containing bert/, pooler.pt, joint_config.json.
      - LoRA adapter injected into bert's attention (q, v) via HuggingFace peft. Only
        the LoRA A/B matrices are trainable inside bert.
      - Task head on top of the pooled embedding z. "binary" = Linear(hidden, 1) for
        BCE-with-logits; "mlp" = Linear -> ReLU -> Dropout -> Linear.

    Forward signature: (input_ids, attention_mask, token_type_ids) -> logits.
    Loss is applied outside (BCEWithLogitsLoss for binary).
    """

    def __init__(self, backbone_path: str,
                 head_type: str = "binary",
                 head_hidden: int = None,
                 head_dropout: float = 0.1,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_targets=("query", "value"),
                 lora_dropout: float = 0.0):
        super().__init__()
        import json
        from peft import LoraConfig, get_peft_model

        # Load joint config (records pooling, dropout, etc. used at pretrain time)
        with open(os.path.join(backbone_path, "joint_config.json")) as f:
            jcfg = json.load(f)
        self.joint_config = jcfg

        # Load backbone BERT + pooler from the joint checkpoint
        mlm = AutoModelForMaskedLM.from_pretrained(backbone_path)
        bert = mlm.bert
        hidden_size = bert.config.hidden_size
        num_layers = bert.config.num_hidden_layers
        pooler = BertPooler(
            jcfg.get("pooling", "mixed"),
            hidden_size,
            num_layers,
            jcfg.get("dropout", 0.1),
        )
        pooler.load_state_dict(
            torch.load(os.path.join(backbone_path, "pooler.pt"), map_location="cpu")
        )

        # Freeze backbone entirely; only LoRA + head will be trainable
        for p in bert.parameters():
            p.requires_grad = False
        for p in pooler.parameters():
            p.requires_grad = False

        # Inject LoRA into bert. peft auto-marks LoRA A/B as trainable.
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,
        )
        self.bert = get_peft_model(bert, lora_cfg)
        self.pooler = pooler

        # Task head
        if head_type == "binary":
            self.head = nn.Linear(hidden_size, 1)
        elif head_type == "mlp":
            mid = head_hidden if head_hidden is not None else hidden_size
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(mid, 1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")
        self.head_type = head_type
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        all_hidden = out.hidden_states if self.pooler.needs_all_hidden_states else None
        z = self.pooler(out.hidden_states[-1], attention_mask, all_hidden)
        logits = self.head(z)
        return logits.squeeze(-1) if self.head_type == "binary" else logits

    def count_trainable(self):
        """Return (n_trainable, n_total) -- sanity check that LoRA + head dominate."""
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_trainable, n_total

    def save_adapter(self, save_path: str, downstream_config: dict = None):
        """Save peft adapter, head, and downstream metadata (no backbone, it stays static)."""
        import json
        os.makedirs(save_path, exist_ok=True)
        self.bert.save_pretrained(os.path.join(save_path, "adapter"))
        torch.save(self.head.state_dict(), os.path.join(save_path, "head.pt"))
        if downstream_config is not None:
            with open(os.path.join(save_path, "downstream_config.json"), "w") as f:
                json.dump(downstream_config, f, indent=2)
