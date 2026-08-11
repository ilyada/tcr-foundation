"""
perchain_attention.py -- per-chain relative-position self-attention for TCRFoundation.

WHY THIS EXISTS
---------------
The foundation backbone is pretrained beta-only (single-chain contexts <= ~47 tokens).
Vanilla HF relative position encoding (RPE) uses ONE continuous distance over the whole
sequence (distance = arange_l - arange_r). For paired Stage 2 (up to ~88 tokens) that would
require cross-chain distances the beta-only backbone never trained, and a thin LoRA cannot
install them from scratch.

FIX: reset the relative position at the chain boundary.
  - within a chain: relative distance = (pos_l - pos_r) on per-chain-reset positions
    (so each chain stays in the <=47 range the backbone trained on),
  - across chains (alpha <-> beta): a single learned "cross-chain" bucket.
Each chain then looks to the backbone like a familiar single chain; the only NEW positional
signal in Stage 2 is the cross-chain bucket (learned alongside the LoRA delta).

HOW
---
The relative bias is computed EXACTLY like HF's `relative_key_query`, except the [L, L]
distance matrix is replaced by a per-sample bucket-index matrix rel_idx[B, L, L] built from
per-chain position_ids + token_type_ids (see `build_rel_idx`). The distance_embedding has
one EXTRA row (index 2*max_pos-1) for the shared cross-chain bucket.

`PerChainRelativeSelfAttention` is a drop-in replacement for `BertSelfAttention`: same
query/key/value Linear names (so peft LoRA targets them unchanged) and the same forward
signature. It is eager-only (encoder training; no KV cache / cross-attention / SDPA).

The rel_idx tensor is batch-dependent (chain boundary differs per sample), so it cannot live
in the forward signature that `BertAttention` controls. Instead it is stashed on each module
as `_rel_idx` by `apply_rel_idx(...)` immediately before the model forward.
"""

import math
import torch
import torch.nn as nn


def build_rel_idx(position_ids: torch.Tensor,
                  token_type_ids: torch.Tensor,
                  max_pos: int) -> torch.Tensor:
    """
    Build the per-chain relative-bucket index matrix.

    Args:
        position_ids:   LongTensor [B, L] -- per-chain-reset positions (0..n WITHIN each chain).
        token_type_ids: LongTensor [B, L] -- chain id per token (0 = alpha, 1 = beta).
        max_pos:        int -- max_position_embeddings (per-chain table size).

    Returns:
        LongTensor [B, L, L] of indices into a distance_embedding of size 2*max_pos:
          same chain:  clamp((pos_l - pos_r), -(max-1), max-1) + (max_pos - 1)  -> rows 0..2*max-2
          cross chain:  2*max_pos - 1                                            -> the cross-chain row
    """
    pos_l = position_ids.unsqueeze(2)            # [B, L, 1]
    pos_r = position_ids.unsqueeze(1)            # [B, 1, L]
    dist = (pos_l - pos_r)                        # [B, L, L]  signed within-chain distance
    dist = dist.clamp(-(max_pos - 1), max_pos - 1) + (max_pos - 1)   # -> [0, 2*max-2]
    same_chain = token_type_ids.unsqueeze(2) == token_type_ids.unsqueeze(1)  # [B, L, L]
    cross_idx = 2 * max_pos - 1
    rel_idx = torch.where(same_chain, dist, torch.full_like(dist, cross_idx))
    return rel_idx


class PerChainRelativeSelfAttention(nn.Module):
    """
    Drop-in replacement for HF BertSelfAttention with per-chain relative positions.

    Reads the precomputed bucket matrix from `self._rel_idx` (set per forward via
    `apply_rel_idx`). Module names query/key/value match HF so peft LoRA wraps them unchanged.
    """

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.max_position_embeddings = config.max_position_embeddings
        # 2*max_pos rows = standard relative range (2*max_pos - 1) + one cross-chain bucket.
        self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings,
                                               self.attention_head_size)
        self._rel_idx = None  # [B, L, L] long; set externally before forward

    def _shape(self, x, B):
        return x.view(B, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, past_key_values=None,
                output_attentions=False, cache_position=None, **kwargs):
        B, L, _ = hidden_states.shape
        q = self._shape(self.query(hidden_states), B)   # [B, h, L, d]
        k = self._shape(self.key(hidden_states),   B)
        v = self._shape(self.value(hidden_states), B)

        scores = torch.matmul(q, k.transpose(-1, -2))    # [B, h, L, L]

        # --- per-chain relative bias (relative_key_query form) ---
        if self._rel_idx is None:
            raise RuntimeError(
                "PerChainRelativeSelfAttention._rel_idx is not set; call "
                "apply_rel_idx(model, position_ids, token_type_ids) before the forward pass."
            )
        # pos_emb is BATCH-dependent here (HF's is [L,L,d]; ours is [B,L,L,d]).
        pos_emb = self.distance_embedding(self._rel_idx).to(dtype=q.dtype)   # [B, L, L, d]
        rel_q = torch.einsum("bhld,blrd->bhlr", q, pos_emb)
        rel_k = torch.einsum("bhrd,blrd->bhlr", k, pos_emb)
        scores = scores + rel_q + rel_k

        scores = scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask           # additive mask (precomputed by BertModel)

        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        ctx = torch.matmul(probs, v)                    # [B, h, L, d]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        return ctx, probs                               # match HF (context, attention_probs)


def swap_to_perchain(bert_model) -> list:
    """
    Replace every encoder layer's `attention.self` with a PerChainRelativeSelfAttention,
    in place. Works on a `BertModel` or anything exposing `.bert.encoder.layer` /
    `.encoder.layer`.

    Returns the list of inserted PerChainRelativeSelfAttention modules (for fast rel_idx
    setting via apply_rel_idx).
    """
    # locate the encoder layers regardless of wrapper (BertForMaskedLM.bert vs BertModel)
    if hasattr(bert_model, "encoder"):
        layers = bert_model.encoder.layer
        config = bert_model.config
    elif hasattr(bert_model, "bert"):
        layers = bert_model.bert.encoder.layer
        config = bert_model.bert.config
    else:
        raise ValueError("Cannot find encoder.layer on the provided model")

    inserted = []
    for layer in layers:
        new_attn = PerChainRelativeSelfAttention(config)
        layer.attention.self = new_attn
        inserted.append(new_attn)
    return inserted


def apply_rel_idx(perchain_modules, position_ids, token_type_ids, max_pos):
    """
    Compute rel_idx once and stash it on every PerChainRelativeSelfAttention module before a
    forward pass.

    Args:
        perchain_modules: list returned by swap_to_perchain (or model._perchain_attns).
        position_ids:     LongTensor [B, L] per-chain-reset positions.
        token_type_ids:   LongTensor [B, L] chain ids (0 alpha / 1 beta).
        max_pos:          int -- max_position_embeddings.
    """
    rel_idx = build_rel_idx(position_ids, token_type_ids, max_pos)
    for m in perchain_modules:
        m._rel_idx = rel_idx
    return rel_idx
