"""Canonical IGoR features and a shared-latent probabilistic autoencoder.

The encoder consumes every supported gene-level IGoR marginal in one fixed
canonical order.  It produces one donor-level latent vector.  Reconstruction
uses a separate head per probability block because the blocks have distinct
conditional-simplex constraints; those heads are training-only diagnostics and
are not a collection of donor-specific encoders.

This module deliberately does not connect the latent to the CDR3 decoder yet.
The latent dimension is independent of a decoder's ``d_model`` and is selected
by held-out IGoR reconstruction, not by Transformer width.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import comet_ml  # noqa: F401  # initialize Comet before this module ever imports Torch
import numpy as np

from .igor_gene_level import REQUIRED_BLOCKS


# These labels are the only axes whose length can change when a donor lacks a
# gene. Deletion and insertion axes have model-defined numeric coordinates and
# must therefore agree exactly across stores.
GENE_AXES = ("V_gene", "D_gene", "J_gene")

# For each block, list only the leading axes that index genes. The remaining
# axes are child-event categories. For example, ``j_choice[V, J]`` has V as a
# parent gene axis and J as the categorical child axis.
GENE_BLOCK_AXES = {
    "v_choice": ("V_gene",),
    "j_choice": ("V_gene", "J_gene"),
    "d_gene": ("V_gene", "J_gene", "D_gene"),
    "v_3_del": ("V_gene",),
    "d_5_del": ("D_gene",),
    "d_3_del": ("D_gene",),
    "j_5_del": ("J_gene",),
}
# IGoR serialises a 4-by-4 nucleotide transition matrix as 16 values. It must
# be reshaped before the final-axis softmax and row-wise KL are applied.
DINUCLEOTIDE_BLOCKS = {"vd_dinucl", "dj_dinucl"}


@dataclass(frozen=True)
class IGoRFeatureSchema:
    """Canonical axes, native block shapes, and a stable flattened layout."""

    axes: dict[str, tuple[str, ...]]
    block_shapes: dict[str, tuple[int, ...]]

    @property
    def block_sizes(self) -> dict[str, int]:
        return {name: int(np.prod(self.block_shapes[name])) for name in REQUIRED_BLOCKS}

    @property
    def probability_dim(self) -> int:
        return sum(self.block_sizes.values())

    @property
    def block_slices(self) -> dict[str, slice]:
        # The order is REQUIRED_BLOCKS, rather than dictionary insertion order,
        # so a saved schema gives the same coordinate meaning to every donor.
        offset, result = 0, {}
        for name in REQUIRED_BLOCKS:
            result[name] = slice(offset, offset + self.block_sizes[name])
            offset += self.block_sizes[name]
        return result

    def probability_shape(self, name: str) -> tuple[int, ...]:
        """Return the shape on which the last axis is a categorical simplex."""
        if name in DINUCLEOTIDE_BLOCKS:
            return (4, 4)
        return self.block_shapes[name]

    @classmethod
    def from_store_paths(cls, store_paths: Iterable[str | Path]) -> "IGoRFeatureSchema":
        """Build a canonical gene schema from stores, retaining a deterministic union."""
        paths = [Path(path) for path in store_paths]
        if not paths:
            raise ValueError("at least one gene-level store is required")
        schemas = [_read_store_schema(path) for path in paths]
        # A gene absent from one donor still receives a canonical coordinate.
        # This makes all donors comparable and lets an unseen donor be aligned
        # to a schema fitted on training donors.
        axes = {
            axis: tuple(sorted({gene for schema in schemas for gene in schema["axes"][axis]}))
            for axis in GENE_AXES
        }
        child_shapes: dict[str, tuple[int, ...]] = {}
        for name in REQUIRED_BLOCKS:
            observed = {tuple(schema["blocks"][name]["shape"]) for schema in schemas}
            expected_children = {
                shape[len(GENE_BLOCK_AXES.get(name, ())):]
                for shape in observed
            }
            # A shared encoder cannot safely flatten two different deletion or
            # insertion supports, so reject a mismatch before training.
            if len(expected_children) != 1:
                raise ValueError(f"{name}: inconsistent non-gene dimensions across stores: {sorted(expected_children)}")
            child_shapes[name] = next(iter(expected_children))
        block_shapes = {
            name: tuple(len(axes[axis]) for axis in GENE_BLOCK_AXES.get(name, ())) + child_shapes[name]
            for name in REQUIRED_BLOCKS
        }
        if any(block_shapes[name] != (16,) for name in DINUCLEOTIDE_BLOCKS):
            raise ValueError("dinucleotide blocks must have 16 stored probabilities")
        return cls(axes=axes, block_shapes=block_shapes)

    def align_store(self, store_path: str | Path) -> dict[str, np.ndarray]:
        """Place one gene-level store in this schema, filling absent gene states with zero."""
        path = Path(store_path)
        source_schema = _read_store_schema(path)
        source_axes = {axis: tuple(source_schema["axes"][axis]) for axis in GENE_AXES}
        # Alignment only permits a source subset of the canonical genes. A
        # source gene outside it would have no stable coordinate in theta_i.
        for axis, labels in source_axes.items():
            unknown = set(labels) - set(self.axes[axis])
            if unknown:
                raise ValueError(f"{path}: {axis} labels absent from canonical schema: {sorted(unknown)}")
        with np.load(path) as archive:
            result = {}
            for name in REQUIRED_BLOCKS:
                if name not in archive:
                    raise ValueError(f"{path}: missing block {name!r}")
                value = np.asarray(archive[name], dtype=np.float32)
                expected = tuple(source_schema["blocks"][name]["shape"])
                if value.shape != expected:
                    raise ValueError(f"{path}: {name} has shape {value.shape}, schema records {expected}")
                result[name] = self._align_block(name, value, source_axes)
        return result

    def _align_block(self, name: str, value: np.ndarray, source_axes: dict[str, tuple[str, ...]]) -> np.ndarray:
        gene_axes = GENE_BLOCK_AXES.get(name, ())
        if not gene_axes:
            if value.shape != self.block_shapes[name]:
                raise ValueError(f"{name}: expected shape {self.block_shapes[name]}, got {value.shape}")
            return value.copy()
        # Missing parent genes are represented by all-zero rows. They retain a
        # fixed position in the input; reconstruction loss later gives such
        # rows zero weight because their parent probability is zero.
        target = np.zeros(self.block_shapes[name], dtype=np.float32)
        indices = []
        for axis in gene_axes:
            target_index = {label: index for index, label in enumerate(self.axes[axis])}
            indices.append([target_index[label] for label in source_axes[axis]])
        # np.ix_ preserves every child-event coordinate while moving only the
        # named gene axes into their canonical positions.
        target[np.ix_(*indices, *[np.arange(size) for size in value.shape[len(gene_axes):]])] = value
        return target

    def flatten(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        """Return one fixed-order probability feature vector for a donor."""
        missing = sorted(set(REQUIRED_BLOCKS) - set(blocks))
        if missing:
            raise ValueError(f"missing IGoR blocks {missing}")
        pieces = []
        for name in REQUIRED_BLOCKS:
            value = np.asarray(blocks[name], dtype=np.float32)
            if value.shape != self.block_shapes[name]:
                raise ValueError(f"{name}: expected {self.block_shapes[name]}, got {value.shape}")
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name}: expected finite non-negative probabilities")
            # No block-specific neural encoder is used here: semantic meaning
            # is carried by the fixed named coordinate and the global encoder
            # learns joint variation across the complete IGoR model.
            pieces.append(value.ravel())
        return np.concatenate(pieces)

    def unflatten(self, features: Any) -> dict[str, Any]:
        """Split a NumPy array or tensor whose last axis is ``probability_dim``."""
        if int(features.shape[-1]) != self.probability_dim:
            raise ValueError(f"expected final dimension {self.probability_dim}, got {features.shape[-1]}")
        return {name: features[..., section].reshape(*features.shape[:-1], *self.block_shapes[name])
                for name, section in self.block_slices.items()}


def _read_store_schema(store_path: Path) -> dict[str, Any]:
    schema_path = store_path.with_name("gene_level_schema.json")
    if not schema_path.is_file():
        raise FileNotFoundError(f"{store_path}: expected sibling {schema_path.name}")
    schema = json.loads(schema_path.read_text())
    if set(schema.get("axes", ())) != set(GENE_AXES):
        raise ValueError(f"{schema_path}: expected axes {GENE_AXES}")
    missing = sorted(set(REQUIRED_BLOCKS) - set(schema.get("blocks", ())))
    if missing:
        raise ValueError(f"{schema_path}: missing blocks {missing}")
    return schema


def load_feature_matrix(store_paths: Iterable[str | Path], schema: IGoRFeatureSchema | None = None) -> tuple[np.ndarray, IGoRFeatureSchema]:
    """Load gene-level stores into a donor-by-feature probability matrix."""
    paths = [Path(path) for path in store_paths]
    schema = schema or IGoRFeatureSchema.from_store_paths(paths)
    # Rows are donors and columns are fixed IGoR coordinates. The returned
    # schema must be saved with any trained model to align a new donor later.
    features = [schema.flatten(schema.align_store(path)) for path in paths]
    return np.stack(features), schema


def build_autoencoder(schema: IGoRFeatureSchema, *, latent_dim: int, hidden_dim: int = 256,
                      dropout: float = 0.1):
    """Build the optional-Torch autoencoder without importing Torch at package import time."""
    if latent_dim < 1 or hidden_dim < 1 or not 0.0 <= dropout < 1.0:
        raise ValueError("latent_dim and hidden_dim must be positive, and dropout must lie in [0, 1)")
    # Torch is optional for the package. Delaying this import keeps IGoR store
    # parsing usable in analysis environments without the neural extra.
    import torch
    from torch import nn

    class IGoRAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.schema = schema
            self.latent_dim = latent_dim
            # One encoder, not one encoder per IGoR block: z is intended to
            # describe one recombination machinery rather than eleven unrelated
            # representations. Dropout regularises against donor-level
            # memorisation when only a few hundred donors are available.
            self.encoder = nn.Sequential(
                nn.Linear(schema.probability_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            # Heads are separate only because their output simplices differ.
            # They reconstruct from the same z and do not create separate
            # latent spaces.
            self.heads = nn.ModuleDict({
                name: nn.Linear(latent_dim, int(np.prod(schema.probability_shape(name))))
                for name in REQUIRED_BLOCKS
            })

        def encode(self, features):
            if features.ndim != 2 or features.shape[1] != self.schema.probability_dim:
                raise ValueError(f"features must have shape [batch, {self.schema.probability_dim}]")
            return self.encoder(features)

        def logits(self, latent):
            if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
                raise ValueError(f"latent must have shape [batch, {self.latent_dim}]")
            # A head emits unconstrained logits. ``probabilities`` applies the
            # correct last-axis softmax rather than treating a matrix/tensor as
            # one unconstrained flat regression target.
            return {name: head(latent).reshape(latent.shape[0], *self.schema.probability_shape(name))
                    for name, head in self.heads.items()}

        def probabilities(self, latent):
            # Every IGoR factor is categorical in its final axis: V for P(V),
            # J for P(J|V), D for P(D|V,J), deletion length for deletion blocks,
            # and next nucleotide for each dinucleotide transition row.
            return {name: torch.softmax(logits, dim=-1).reshape(latent.shape[0], *self.schema.block_shapes[name])
                    for name, logits in self.logits(latent).items()}

        def forward(self, features):
            latent = self.encode(features)
            return latent, self.logits(latent)

    return IGoRAutoencoder()


def reconstruction_loss(model: Any, features: Any) -> tuple[Any, dict[str, Any]]:
    """Equal-block KL reconstruction loss respecting IGoR conditional factorisation."""
    import torch
    import torch.nn.functional as functional

    latent, logits = model(features)
    target = model.schema.unflatten(features)

    # Recover the parent-state marginals from the IGoR Bayesian factorisation.
    # They determine which conditional rows can affect a donor's generated
    # repertoire and therefore how strongly their reconstruction is scored.
    p_v = target["v_choice"]
    p_vj = p_v.unsqueeze(-1) * target["j_choice"]
    p_vjd = p_vj.unsqueeze(-1) * target["d_gene"]
    parents = {
        "v_choice": None,
        "j_choice": p_v,
        "d_gene": p_vj,
        "v_3_del": p_v,
        "d_5_del": p_vjd.sum(dim=(1, 2)),
        "d_3_del": p_vjd.sum(dim=(1, 2)).unsqueeze(-1) * target["d_5_del"],
        "j_5_del": p_vj.sum(dim=1),
        "vd_ins": None,
        "dj_ins": None,
        # No parent-base marginal is present in IGoR's dinucleotide block, so
        # each of the four transition rows contributes equally.
        "vd_dinucl": torch.ones_like(target["vd_dinucl"]).reshape(-1, 4, 4)[..., 0],
        "dj_dinucl": torch.ones_like(target["dj_dinucl"]).reshape(-1, 4, 4)[..., 0],
    }
    losses = {}
    for name in REQUIRED_BLOCKS:
        target_block = target[name]
        if name in DINUCLEOTIDE_BLOCKS:
            target_block = target_block.reshape(features.shape[0], 4, 4)
        log_probability = torch.log_softmax(logits[name], dim=-1)
        # KL is summed only over the categorical child axis. What remains is
        # one divergence for each parent state, such as each V row in P(J|V).
        row_kl = functional.kl_div(log_probability, target_block, reduction="none").sum(dim=-1)
        parent = parents[name]
        if parent is None:
            losses[name] = row_kl.mean()
        else:
            # Weighting by the true parent probability avoids both overvaluing
            # rare conditional rows and penalising all-zero rows for absent
            # parents. Normalisation makes one block's loss comparable across
            # donors before all eleven block losses are averaged below.
            losses[name] = (row_kl * parent).sum(dim=tuple(range(1, parent.ndim))) / parent.sum(dim=tuple(range(1, parent.ndim))).clamp_min(1e-12)
            losses[name] = losses[name].mean()
    # Equal block weighting prevents large tensors, notably P(D|V,J), from
    # dominating simply because they contain more stored probabilities.
    return torch.stack(tuple(losses.values())).mean(), {"latent": latent, **losses}


def fit_autoencoder(features: np.ndarray, schema: IGoRFeatureSchema, *, latent_dim: int,
                    hidden_dim: int = 256, dropout: float = 0.1, steps: int = 1_000,
                    learning_rate: float = 1e-3, device: str = "cpu") -> tuple[Any, list[float]]:
    """Optimise a full-batch autoencoder and return it with its loss trajectory.

    This small trainer is intentionally limited to the IGoR representation stage.
    It does not save weights, select a latent dimension, split donors, or attach a
    CDR3 decoder. Those decisions belong to the later experimental protocol.
    """
    if features.ndim != 2 or features.shape[1] != schema.probability_dim:
        raise ValueError(f"features must have shape [donors, {schema.probability_dim}]")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    import torch

    # This smoke-friendly trainer uses every supplied donor on every step. The
    # production experiment will instead use donor-level train/dev/held-out
    # splits, but a full batch makes a small HD integration check deterministic.
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    model = build_autoencoder(schema, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trajectory = []
    for _ in range(steps):
        model.train()
        loss, _ = reconstruction_loss(model, tensor)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        trajectory.append(float(loss.detach().cpu()))
    return model, trajectory
