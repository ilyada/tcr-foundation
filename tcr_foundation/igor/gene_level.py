"""Aggregate an IGoR TRB allele-level model to gene-level probability blocks.

The input is an IGoR ``final_marginals.txt`` file together with the matching
``final_parms.txt`` file.  The latter supplies the correspondence between IGoR
indices and V/D/J allele names.  Alleles are merged by removing the ``*allele``
suffix.  Conditional distributions are not averaged: the implementation first
forms their joint distribution, sums allele states within a gene, then restores
the conditional factorisation.  This preserves probability mass and avoids an
implicit, and generally wrong, equal weighting of alleles.

The command writes a new directory containing a compressed numerical store and a
human-readable schema.  It never modifies the native IGoR files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_BLOCKS = (
    "v_choice",
    "j_choice",
    "d_gene",
    "v_3_del",
    "d_5_del",
    "d_3_del",
    "j_5_del",
    "vd_ins",
    "vd_dinucl",
    "dj_ins",
    "dj_dinucl",
)


@dataclass(frozen=True)
class GeneAxes:
    """Allele labels from IGoR and their canonical gene-level counterparts."""

    v_alleles: tuple[str, ...]
    d_alleles: tuple[str, ...]
    j_alleles: tuple[str, ...]
    v_genes: tuple[str, ...]
    d_genes: tuple[str, ...]
    j_genes: tuple[str, ...]


def parse_final_marginals(path: str | Path) -> dict[str, np.ndarray]:
    """Parse IGoR's text serialisation into named arrays with native shapes."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    current_name: str | None = None
    current_shape: tuple[int, ...] | None = None
    values: list[float] = []

    def finish_current() -> None:
        if current_name is None:
            return
        if current_shape is None:
            raise ValueError(f"{path}: {current_name!r} has no $Dim declaration")
        expected = math.prod(current_shape)
        if len(values) != expected:
            raise ValueError(f"{path}: {current_name!r} has {len(values)} values, expected {expected}")
        if current_name in arrays:
            raise ValueError(f"{path}: repeated marginal block {current_name!r}")
        array = np.asarray(values, dtype=np.float64).reshape(current_shape)
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"{path}: {current_name!r} contains non-finite or negative probabilities")
        arrays[current_name] = array

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@"):
            finish_current()
            current_name, current_shape, values = line[1:].strip(), None, []
            if not current_name:
                raise ValueError(f"{path}: empty marginal block name")
        elif line.startswith("$Dim["):
            if current_name is None:
                raise ValueError(f"{path}: $Dim declaration before a marginal block")
            match = re.fullmatch(r"\$Dim\[([0-9, ]+)\]", line)
            if match is None:
                raise ValueError(f"{path}: invalid dimension declaration {line!r}")
            current_shape = tuple(int(part.strip()) for part in match.group(1).split(","))
            if not current_shape or any(size < 1 for size in current_shape):
                raise ValueError(f"{path}: invalid dimensions for {current_name!r}")
        elif line.startswith("%"):
            if current_name is None or current_shape is None:
                raise ValueError(f"{path}: probability row before a named, shaped block")
            payload = line[1:].strip()
            if payload:
                try:
                    values.extend(float(value) for value in payload.split(",") if value.strip())
                except ValueError as error:
                    raise ValueError(f"{path}: invalid probability in {current_name!r}") from error
    finish_current()
    missing = sorted(set(REQUIRED_BLOCKS) - set(arrays))
    if missing:
        raise ValueError(f"{path}: missing required marginal blocks {missing}")
    return arrays


def _gene_name(allele: str) -> str:
    gene, separator, _ = allele.partition("*")
    if not separator or not gene:
        raise ValueError(f"cannot derive a gene name from allele label {allele!r}")
    return gene


def _ordered_labels(indexed_labels: dict[int, str], event: str, path: Path) -> tuple[str, ...]:
    expected = list(range(len(indexed_labels)))
    if sorted(indexed_labels) != expected:
        raise ValueError(f"{path}: {event} indices are not exactly {expected}")
    return tuple(indexed_labels[index] for index in expected)


def parse_gene_axes(path: str | Path) -> GeneAxes:
    """Read the V/D/J allele labels and indices from IGoR's final parameters."""
    path = Path(path)
    indexed: dict[str, dict[int, str]] = {"v_choice": {}, "d_gene": {}, "j_choice": {}}
    current_event: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("#GeneChoice;"):
            fields = line[1:].split(";")
            current_event = fields[-1] if fields else None
            if current_event not in indexed:
                current_event = None
        elif line.startswith("#"):
            current_event = None
        elif line.startswith("%") and current_event is not None:
            try:
                payload, index_text = line[1:].rsplit(";", 1)
                index = int(index_text.strip())
            except ValueError as error:
                raise ValueError(f"{path}: invalid {current_event} gene entry {line!r}") from error
            allele = payload.split("|")[1].strip() if "|" in payload else payload.strip().split(";", 1)[0].strip()
            if not allele:
                raise ValueError(f"{path}: empty allele label for {current_event}[{index}]")
            if index in indexed[current_event]:
                raise ValueError(f"{path}: repeated {current_event} index {index}")
            indexed[current_event][index] = allele
    v_alleles = _ordered_labels(indexed["v_choice"], "v_choice", path)
    d_alleles = _ordered_labels(indexed["d_gene"], "d_gene", path)
    j_alleles = _ordered_labels(indexed["j_choice"], "j_choice", path)
    return GeneAxes(v_alleles=v_alleles, d_alleles=d_alleles, j_alleles=j_alleles,
                    v_genes=tuple(dict.fromkeys(_gene_name(label) for label in v_alleles)),
                    d_genes=tuple(dict.fromkeys(_gene_name(label) for label in d_alleles)),
                    j_genes=tuple(dict.fromkeys(_gene_name(label) for label in j_alleles)))


def _normalise(joint: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide by parent probability and encode an impossible parent as an all-zero row."""
    return np.divide(joint, denominator, out=np.zeros_like(joint), where=denominator > 0)


def _membership(alleles: tuple[str, ...], genes: tuple[str, ...]) -> np.ndarray:
    lookup = {gene: index for index, gene in enumerate(genes)}
    membership = np.zeros((len(alleles), len(genes)), dtype=np.float64)
    for allele_index, allele in enumerate(alleles):
        membership[allele_index, lookup[_gene_name(allele)]] = 1.0
    return membership


def _check_conditional(name: str, probabilities: np.ndarray, parent: np.ndarray, *, tolerance: float) -> None:
    if probabilities.shape[:-1] != parent.shape:
        raise ValueError(f"{name}: child axes do not agree with parent axes")
    sums = probabilities.sum(axis=-1)
    possible = parent > 0
    if not np.allclose(sums[possible], 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"{name}: a conditional row with positive parent probability does not sum to one")
    if not np.allclose(sums[~possible], 0.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"{name}: a conditional row with zero parent probability is not all zero")


def validate_igor_blocks(blocks: dict[str, np.ndarray], *, tolerance: float = 2e-5) -> None:
    """Check factor dimensions and normalisation for the supported TRB IGoR model."""
    missing = sorted(set(REQUIRED_BLOCKS) - set(blocks))
    if missing:
        raise ValueError(f"missing required marginal blocks {missing}")
    for name in REQUIRED_BLOCKS:
        value = blocks[name]
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name}: probabilities must be finite and non-negative")
    p_v = blocks["v_choice"]
    p_j_given_v = blocks["j_choice"]
    p_d_given_vj = blocks["d_gene"]
    if p_v.ndim != 1 or p_j_given_v.shape[:1] != p_v.shape or p_d_given_vj.shape[:2] != p_j_given_v.shape:
        raise ValueError("V, J|V, and D|V,J dimensions are inconsistent")
    if not np.isclose(p_v.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("v_choice does not sum to one")
    _check_conditional("j_choice", p_j_given_v, p_v, tolerance=tolerance)
    p_vj = p_v[:, None] * p_j_given_v
    _check_conditional("d_gene", p_d_given_vj, p_vj, tolerance=tolerance)
    p_vjd = p_vj[:, :, None] * p_d_given_vj
    p_j = p_vj.sum(axis=0)
    p_d = p_vjd.sum(axis=(0, 1))
    _check_conditional("v_3_del", blocks["v_3_del"], p_v, tolerance=tolerance)
    _check_conditional("d_5_del", blocks["d_5_del"], p_d, tolerance=tolerance)
    p_d5 = p_d[:, None] * blocks["d_5_del"]
    _check_conditional("d_3_del", blocks["d_3_del"], p_d5, tolerance=tolerance)
    _check_conditional("j_5_del", blocks["j_5_del"], p_j, tolerance=tolerance)
    for name in ("vd_ins", "dj_ins"):
        if blocks[name].ndim != 1 or not np.isclose(blocks[name].sum(), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"{name}: insertion-length probabilities do not sum to one")
    for name in ("vd_dinucl", "dj_dinucl"):
        if blocks[name].size != 16:
            raise ValueError(f"{name}: expected 16 dinucleotide probabilities")
        if not np.allclose(blocks[name].reshape(4, 4).sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"{name}: dinucleotide-transition rows do not sum to one")


def reduce_to_gene_level(blocks: dict[str, np.ndarray], axes: GeneAxes) -> tuple[dict[str, np.ndarray], GeneAxes]:
    """Merge V/D/J allele states through joint marginalisation and refactorisation."""
    validate_igor_blocks(blocks)
    p_v, p_j_given_v, p_d_given_vj = blocks["v_choice"], blocks["j_choice"], blocks["d_gene"]
    n_v, n_j, n_d = len(axes.v_alleles), len(axes.j_alleles), len(axes.d_alleles)
    if (p_v.shape, p_j_given_v.shape, p_d_given_vj.shape) != ((n_v,), (n_v, n_j), (n_v, n_j, n_d)):
        raise ValueError("final_parms V/D/J indices do not match final_marginals dimensions")
    a_v, a_j, a_d = (_membership(axes.v_alleles, axes.v_genes), _membership(axes.j_alleles, axes.j_genes),
                      _membership(axes.d_alleles, axes.d_genes))
    p_vj = p_v[:, None] * p_j_given_v
    p_vjd = p_vj[:, :, None] * p_d_given_vj
    p_v_gene = np.einsum("v,vg->g", p_v, a_v)
    p_vj_gene = np.einsum("vj,vg,jh->gh", p_vj, a_v, a_j)
    p_vjd_gene = np.einsum("vjd,vg,jh,dk->ghk", p_vjd, a_v, a_j, a_d)
    p_j_given_v_gene = _normalise(p_vj_gene, p_v_gene[:, None])
    p_d_given_vj_gene = _normalise(p_vjd_gene, p_vj_gene[:, :, None])

    p_vdel_joint = np.einsum("v,ve,vg->ge", p_v, blocks["v_3_del"], a_v)
    p_vdel_gene = _normalise(p_vdel_joint, p_v_gene[:, None])
    p_j = p_vj.sum(axis=0)
    p_j_gene = np.einsum("j,jh->h", p_j, a_j)
    p_jdel_joint = np.einsum("j,je,jh->he", p_j, blocks["j_5_del"], a_j)
    p_jdel_gene = _normalise(p_jdel_joint, p_j_gene[:, None])
    p_d = p_vjd.sum(axis=(0, 1))
    p_d_gene = np.einsum("d,dk->k", p_d, a_d)
    p_d5_joint = np.einsum("d,de,dk->ke", p_d, blocks["d_5_del"], a_d)
    p_d5_gene = _normalise(p_d5_joint, p_d_gene[:, None])
    p_d5d3_joint = p_d[:, None, None] * blocks["d_5_del"][:, :, None] * blocks["d_3_del"]
    p_d5d3_gene_joint = np.einsum("dab,dk->kab", p_d5d3_joint, a_d)
    p_d3_gene = _normalise(p_d5d3_gene_joint, p_d5_joint[:, :, None])

    result = {
        "v_choice": p_v_gene,
        "j_choice": p_j_given_v_gene,
        "d_gene": p_d_given_vj_gene,
        "v_3_del": p_vdel_gene,
        "d_5_del": p_d5_gene,
        "d_3_del": p_d3_gene,
        "j_5_del": p_jdel_gene,
        "vd_ins": blocks["vd_ins"].copy(),
        "vd_dinucl": blocks["vd_dinucl"].copy(),
        "dj_ins": blocks["dj_ins"].copy(),
        "dj_dinucl": blocks["dj_dinucl"].copy(),
    }
    validate_igor_blocks(result)
    gene_axes = GeneAxes(v_alleles=axes.v_genes, d_alleles=axes.d_genes, j_alleles=axes.j_genes,
                         v_genes=axes.v_genes, d_genes=axes.d_genes, j_genes=axes.j_genes)
    return result, gene_axes


def write_gene_level_store(blocks: dict[str, np.ndarray], axes: GeneAxes, out_dir: str | Path) -> tuple[Path, Path]:
    """Write numerical arrays and their axis labels into a newly created directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = out_dir / "gene_level_marginals.npz"
    schema_path = out_dir / "gene_level_schema.json"
    np.savez_compressed(arrays_path, **blocks)
    schema = {
        "schema_version": 1,
        "aggregation": "allele_to_gene_joint_marginalise_refactorise",
        "axes": {"V_gene": list(axes.v_genes), "D_gene": list(axes.d_genes), "J_gene": list(axes.j_genes)},
        "blocks": {name: {"shape": list(value.shape)} for name, value in blocks.items()},
    }
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    return arrays_path, schema_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate IGoR TRB final marginals from allele to gene level.")
    parser.add_argument("--marginals", required=True, help="IGoR final_marginals.txt")
    parser.add_argument("--parms", required=True, help="matching IGoR final_parms.txt")
    parser.add_argument("--out", required=True, help="new output directory")
    args = parser.parse_args()
    blocks, axes = parse_final_marginals(args.marginals), parse_gene_axes(args.parms)
    gene_blocks, gene_axes = reduce_to_gene_level(blocks, axes)
    arrays_path, schema_path = write_gene_level_store(gene_blocks, gene_axes, args.out)
    print(arrays_path)
    print(schema_path)


if __name__ == "__main__":
    main()
