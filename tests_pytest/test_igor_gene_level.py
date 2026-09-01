import numpy as np

from tcr_foundation.igor_gene_level import parse_final_marginals, parse_gene_axes, reduce_to_gene_level, write_gene_level_store


def _write_marginals(path, blocks):
    lines = []
    for name, value in blocks.items():
        lines.extend((f"@{name}", "$Dim[" + ",".join(map(str, value.shape)) + "]", "%" + ",".join(map(str, value.ravel()))))
    path.write_text("\n".join(lines) + "\n")


def _write_parms(path):
    path.write_text("\n".join((
        "@Event_list",
        "#GeneChoice;V_gene;Undefined_side;7;v_choice",
        "%x|TRBV1*01|x;ACGT;0",
        "%x|TRBV1*02|x;ACGT;1",
        "%x|TRBV2*01|x;ACGT;2",
        "#GeneChoice;D_gene;Undefined_side;6;d_gene",
        "% TRBD1*01;ACGT;0",
        "% TRBD1*02;ACGT;1",
        "% TRBD2*01;ACGT;2",
        "#GeneChoice;J_gene;Undefined_side;7;j_choice",
        "%x|TRBJ1-1*01|x;ACGT;0",
        "%x|TRBJ1-1*02|x;ACGT;1",
        "%x|TRBJ2-1*01|x;ACGT;2",
    )) + "\n")


def _conditional(rows):
    return np.asarray(rows, dtype=float)


def test_joint_marginalisation_preserves_mass_and_weights_alleles_by_usage(tmp_path):
    marginals, parms = tmp_path / "final_marginals.txt", tmp_path / "final_parms.txt"
    blocks = {
        "v_choice": np.array([0.2, 0.3, 0.5]),
        "j_choice": _conditional([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.4, 0.2, 0.4]]),
        "d_gene": np.full((3, 3, 3), 1 / 3),
        "v_3_del": _conditional([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]),
        "d_5_del": _conditional([[0.4, 0.6], [0.5, 0.5], [0.7, 0.3]]),
        "d_3_del": np.full((3, 2, 2), 0.5),
        "j_5_del": _conditional([[0.3, 0.7], [0.6, 0.4], [0.9, 0.1]]),
        "vd_ins": np.array([0.4, 0.6]),
        "vd_dinucl": np.full(16, 0.25),
        "dj_ins": np.array([0.7, 0.3]),
        "dj_dinucl": np.full(16, 0.25),
    }
    _write_marginals(marginals, blocks)
    _write_parms(parms)

    parsed, axes = parse_final_marginals(marginals), parse_gene_axes(parms)
    reduced, gene_axes = reduce_to_gene_level(parsed, axes)

    assert axes.d_alleles == ("TRBD1*01", "TRBD1*02", "TRBD2*01")
    assert gene_axes.v_genes == ("TRBV1", "TRBV2")
    assert gene_axes.d_genes == ("TRBD1", "TRBD2")
    assert gene_axes.j_genes == ("TRBJ1-1", "TRBJ2-1")
    np.testing.assert_allclose(reduced["v_choice"], [0.5, 0.5])
    np.testing.assert_allclose(reduced["j_choice"], [[0.74, 0.26], [0.6, 0.4]])
    np.testing.assert_allclose(reduced["v_3_del"], [[0.44, 0.56], [0.3, 0.7]])
    assert reduced["d_gene"].shape == (2, 2, 2)
    assert reduced["d_3_del"].shape == (2, 2, 2)
    np.testing.assert_allclose(reduced["j_choice"].sum(axis=1), 1.0)
    np.testing.assert_allclose(reduced["d_gene"].sum(axis=2), 1.0)

    arrays_path, schema_path = write_gene_level_store(reduced, gene_axes, tmp_path / "gene_level")
    with np.load(arrays_path) as stored:
        np.testing.assert_allclose(stored["v_choice"], reduced["v_choice"])
    assert '"V_gene"' in schema_path.read_text()
