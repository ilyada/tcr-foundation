"""Patient-level CDR3 nucleotide generation conditioned on an IGoR germline token.

The input IGoR feature vector is encoded once by the shared IGoR autoencoder
into a 128-dimensional donor token.  The nucleotide decoder never predicts V
or J as sequence tokens.  Instead, it receives the fixed three-token memory
``[germline token, V embedding, J embedding]`` in every Transformer layer.

The complete patient generator factors as ``P(V|G) P(J|V,G)
P(cdr3nt|G,V,J)``.  The first two factors are read from the IGoR autoencoder
decoder heads; the causal Transformer models only the declared CDR3 length and
the nucleotide sequence.  This module deliberately provides model operations
and sampling only.  Dataset wiring, donor splits, and training objectives are
separate experimental work.
"""
from __future__ import annotations

from typing import Any

import comet_ml  # noqa: F401  # initialize Comet before this module ever imports Torch


BASES = "ACGT"
BOS, PAD = 4, 5
LENGTH_OFFSET = 6
NO_BASE = 4


def build_igor_conditioned_generator(autoencoder: Any, *, d_model: int = 128,
                                     n_layer: int = 6, n_head: int = 4,
                                     max_junction: int = 96,
                                     v_templates: Any | None = None,
                                     j_templates: Any | None = None):
    """Build a decoder whose every block cross-attends to donor, V, and J tokens.

    ``autoencoder`` is the trained ordinary-IGoR autoencoder.  Its bottleneck
    must already have width ``d_model``: the latent therefore is the germline
    token itself, not an input to an additional adapter network.  Template
    arrays, when supplied, have shapes ``[n_V, max_junction]`` and
    ``[n_J, max_junction]`` and use base codes ``A,C,G,T,NO_BASE``.
    """
    if int(getattr(autoencoder, "latent_dim", -1)) != d_model:
        raise ValueError(f"autoencoder latent_dim must equal d_model={d_model}")
    if n_layer < 1 or n_head < 1 or d_model % n_head:
        raise ValueError("n_layer and n_head must be positive, and d_model must divide n_head")
    if max_junction < 1:
        raise ValueError("max_junction must be positive")
    try:
        v_genes = tuple(autoencoder.schema.axes["V_gene"])
        j_genes = tuple(autoencoder.schema.axes["J_gene"])
    except (AttributeError, KeyError) as error:
        raise ValueError("autoencoder must expose a gene-level IGoRFeatureSchema") from error

    import torch
    import torch.nn as nn
    from transformers import GPT2Config, GPT2LMHeadModel

    n_v, n_j = len(v_genes), len(j_genes)
    vocab_size = LENGTH_OFFSET + max_junction
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_junction + 2,  # [BOS], length, then at most max_junction bases
        n_embd=d_model,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * d_model,
        bos_token_id=BOS,
        pad_token_id=PAD,
        add_cross_attention=True,
        is_decoder=True,
    )

    def _templates(value: Any | None, n_gene: int, name: str) -> torch.Tensor:
        if value is None:
            return torch.full((n_gene, max_junction), NO_BASE, dtype=torch.long)
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tuple(tensor.shape) != (n_gene, max_junction):
            raise ValueError(f"{name} must have shape [{n_gene}, {max_junction}]")
        if ((tensor < 0) | (tensor > NO_BASE)).any():
            raise ValueError(f"{name} values must be base codes in [0, {NO_BASE}]")
        return tensor

    class IGoRConditionedGenerator(nn.Module):
        """Autoregressive nucleotide decoder with a three-token condition memory."""

        def __init__(self) -> None:
            super().__init__()
            self.autoencoder = autoencoder
            self.d_model = d_model
            self.max_junction = max_junction
            self.v_genes, self.j_genes = v_genes, j_genes
            self.decoder = GPT2LMHeadModel(config)
            self.v_embedding = nn.Embedding(n_v, d_model)
            self.j_embedding = nn.Embedding(n_j, d_model)
            # The V/J template bases are positional input features. They do not
            # replace the donor token or make V/J output tokens; they expose the
            # known germline-derived terminal bases of the selected background.
            self.germ_v = nn.Embedding(NO_BASE + 1, d_model)
            self.germ_j = nn.Embedding(NO_BASE + 1, d_model)
            self.register_buffer("v_templates", _templates(v_templates, n_v, "v_templates"))
            self.register_buffer("j_templates", _templates(j_templates, n_j, "j_templates"))

        @staticmethod
        def _require_vector(name: str, value: torch.Tensor, batch_size: int) -> None:
            if value.ndim != 1 or value.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [batch]")

        def germline_token(self, igor_features: torch.Tensor) -> torch.Tensor:
            """Encode one complete canonical IGoR vector into one fixed donor token."""
            token = self.autoencoder.encode(igor_features)
            if token.shape[-1] != self.d_model:
                raise RuntimeError("IGoR autoencoder returned an unexpected latent width")
            return token

        def condition_memory(self, igor_features: torch.Tensor, v_gene: torch.Tensor,
                             j_gene: torch.Tensor) -> torch.Tensor:
            """Return ``[G_i, e_V(V), e_J(J)]`` for every example in the batch."""
            batch_size = igor_features.shape[0]
            self._require_vector("v_gene", v_gene, batch_size)
            self._require_vector("j_gene", j_gene, batch_size)
            if (v_gene < 0).any() or (v_gene >= len(self.v_genes)).any():
                raise ValueError("v_gene contains an index outside the canonical IGoR V axis")
            if (j_gene < 0).any() or (j_gene >= len(self.j_genes)).any():
                raise ValueError("j_gene contains an index outside the canonical IGoR J axis")
            return torch.stack((self.germline_token(igor_features), self.v_embedding(v_gene),
                                self.j_embedding(j_gene)), dim=1)

        def gene_probabilities(self, igor_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Recover the patient-specific ``P(V)`` and ``P(J|V)`` from the IGoR decoder."""
            probabilities = self.autoencoder.probabilities(self.germline_token(igor_features))
            return probabilities["v_choice"], probabilities["j_choice"]

        def sample_vj(self, igor_features: torch.Tensor, *, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample the V/J background internally for patient-level generation."""
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            p_v, p_j_given_v = self.gene_probabilities(igor_features)
            v_gene = torch.multinomial(torch.softmax(torch.log(p_v.clamp_min(1e-12)) / temperature, dim=-1), 1)[:, 0]
            j_rows = p_j_given_v[torch.arange(v_gene.shape[0], device=v_gene.device), v_gene]
            j_gene = torch.multinomial(torch.softmax(torch.log(j_rows.clamp_min(1e-12)) / temperature, dim=-1), 1)[:, 0]
            return v_gene, j_gene

        def _template_features(self, v_gene: torch.Tensor, j_gene: torch.Tensor,
                               token_count: int, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            """Create germline-base inputs aligned to the input position predicting each base."""
            batch_size = v_gene.shape[0]
            gv = torch.full((batch_size, token_count), NO_BASE, dtype=torch.long, device=v_gene.device)
            gj = torch.full_like(gv, NO_BASE)
            # Input position one is the declared-length token and predicts base
            # zero. Position two predicts base one, and so on. BOS predicts the
            # length and therefore has no germline-base feature.
            positions = torch.arange(max(token_count - 1, 0), device=v_gene.device)
            if len(positions):
                gv[:, 1:] = self.v_templates[v_gene][:, :len(positions)]
                if lengths is not None:
                    j_index = (lengths[:, None] - 1 - positions[None, :]).clamp(0, self.max_junction - 1)
                    j_values = torch.gather(self.j_templates[j_gene], 1, j_index)
                    gj[:, 1:] = torch.where(positions[None, :] < lengths[:, None], j_values,
                                            torch.full_like(j_values, NO_BASE))
            return gv, gj

        def logits(self, token_ids: torch.Tensor, igor_features: torch.Tensor, v_gene: torch.Tensor,
                   j_gene: torch.Tensor, *, lengths: torch.Tensor | None = None) -> torch.Tensor:
            """Return logits for grammar ``[BOS, length, CDR3 nucleotides]``.

            GPT-2 cross-attention is enabled in every decoder block.  The
            condition memory is therefore visible at every layer, rather than
            being added once to the input embeddings and potentially attenuated
            by deeper layers.
            """
            if token_ids.ndim != 2:
                raise ValueError("token_ids must have shape [batch, time]")
            batch_size = token_ids.shape[0]
            if igor_features.ndim != 2 or igor_features.shape[0] != batch_size:
                raise ValueError("igor_features must have shape [batch, feature]")
            if lengths is not None:
                self._require_vector("lengths", lengths, batch_size)
            gv, gj = self._template_features(v_gene, j_gene, token_ids.shape[1], lengths)
            embeddings = self.decoder.transformer.wte(token_ids) + self.germ_v(gv) + self.germ_j(gj)
            memory = self.condition_memory(igor_features, v_gene, j_gene)
            memory_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
            return self.decoder(inputs_embeds=embeddings, encoder_hidden_states=memory,
                                encoder_attention_mask=memory_mask).logits

        def sequence_log_likelihood(self, token_ids: torch.Tensor, igor_features: torch.Tensor,
                                    v_gene: torch.Tensor, j_gene: torch.Tensor,
                                    lengths: torch.Tensor) -> torch.Tensor:
            """Log likelihood of a declared-length nucleotide sequence, one value per example."""
            logits = self.logits(token_ids, igor_features, v_gene, j_gene, lengths=lengths)[:, :-1]
            targets = token_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1).gather(2, targets[..., None])[..., 0]
            return (log_probs * (targets != PAD)).sum(dim=-1)

        @torch.no_grad()
        def generate(self, igor_features: torch.Tensor, *, temperature: float = 1.0) -> list[tuple[str, str, str]]:
            """Sample V, J, length, and then the full CDR3 nucleotide sequence for each donor feature row."""
            if igor_features.ndim != 2:
                raise ValueError("igor_features must have shape [batch, feature]")
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            batch_size, device = igor_features.shape[0], igor_features.device
            v_gene, j_gene = self.sample_vj(igor_features, temperature=temperature)
            tokens = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)

            def draw(logits: torch.Tensor, low: int, high: int) -> torch.Tensor:
                restricted = torch.full_like(logits, float("-inf"))
                restricted[:, low:high] = logits[:, low:high] / temperature
                return torch.multinomial(torch.softmax(restricted, dim=-1), 1)[:, 0]

            # Length is predicted from BOS.  No template bases are needed before
            # the V/J background and length have been selected.
            length_ids = draw(self.logits(tokens, igor_features, v_gene, j_gene)[:, -1],
                              LENGTH_OFFSET, LENGTH_OFFSET + self.max_junction)
            lengths = length_ids - LENGTH_OFFSET + 1
            tokens = torch.cat((tokens, length_ids[:, None]), dim=1)
            for position in range(self.max_junction):
                next_base = draw(self.logits(tokens, igor_features, v_gene, j_gene, lengths=lengths)[:, -1], 0, 4)
                next_base = torch.where(position < lengths, next_base, torch.full_like(next_base, PAD))
                tokens = torch.cat((tokens, next_base[:, None]), dim=1)

            result = []
            for v_index, j_index, row, length in zip(v_gene.tolist(), j_gene.tolist(), tokens.tolist(), lengths.tolist()):
                sequence = "".join(BASES[base] for base in row[2:2 + length] if base < 4)
                result.append((self.v_genes[v_index], self.j_genes[j_index], sequence))
            return result

    return IGoRConditionedGenerator()
