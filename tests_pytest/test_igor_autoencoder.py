import json

import numpy as np
import pytest

from tcr_foundation.igor_autoencoder import IGoRFeatureSchema, load_feature_matrix


def _store(path, *, v_labels=("TRBV1", "TRBV2"), j_labels=("TRBJ1", "TRBJ2"), d_labels=("TRBD1",)):
    blocks = {
        "v_choice": np.array([0.4, 0.6], dtype=np.float32),
        "j_choice": np.full((2, 2), 0.5, dtype=np.float32),
        "d_gene": np.ones((2, 2, 1), dtype=np.float32),
        "v_3_del": np.full((2, 2), 0.5, dtype=np.float32),
        "d_5_del": np.full((1, 2), 0.5, dtype=np.float32),
        "d_3_del": np.full((1, 2, 2), 0.5, dtype=np.float32),
        "j_5_del": np.full((2, 2), 0.5, dtype=np.float32),
        "vd_ins": np.array([0.4, 0.6], dtype=np.float32),
        "vd_dinucl": np.full(16, 0.25, dtype=np.float32),
        "dj_ins": np.array([0.7, 0.3], dtype=np.float32),
        "dj_dinucl": np.full(16, 0.25, dtype=np.float32),
    }
    path.mkdir()
    np.savez_compressed(path / "gene_level_marginals.npz", **blocks)
    schema = {"axes": {"V_gene": list(v_labels), "D_gene": list(d_labels), "J_gene": list(j_labels)},
              "blocks": {name: {"shape": list(value.shape)} for name, value in blocks.items()}}
    (path / "gene_level_schema.json").write_text(json.dumps(schema))
    return path / "gene_level_marginals.npz"


def test_feature_schema_aligns_absent_gene_states_and_preserves_fixed_layout(tmp_path):
    full = _store(tmp_path / "full")
    partial = _store(tmp_path / "partial", v_labels=("TRBV2", "TRBV1"))
    schema = IGoRFeatureSchema.from_store_paths([full, partial])
    features, loaded = load_feature_matrix([full, partial], schema)
    assert loaded == schema
    assert features.shape == (2, schema.probability_dim)
    v_slice = schema.block_slices["v_choice"]
    np.testing.assert_allclose(features[0, v_slice], [0.4, 0.6])
    np.testing.assert_allclose(features[1, v_slice], [0.6, 0.4])


def test_autoencoder_emits_probability_shapes_and_finite_block_losses(tmp_path):
    pytest.importorskip("comet_ml")
    torch = pytest.importorskip("torch")
    from tcr_foundation.igor_autoencoder import build_autoencoder, reconstruction_loss

    first, second = _store(tmp_path / "one"), _store(tmp_path / "two")
    features, schema = load_feature_matrix([first, second])
    model = build_autoencoder(schema, latent_dim=3, hidden_dim=8, dropout=0.0)
    loss, diagnostics = reconstruction_loss(model, torch.as_tensor(features))
    assert diagnostics["latent"].shape == (2, 3)
    assert torch.isfinite(loss)
    assert set(diagnostics) == {"latent", *schema.block_shapes}
    probabilities = model.probabilities(diagnostics["latent"])
    assert probabilities["j_choice"].shape == (2, 2, 2)
    assert torch.allclose(probabilities["j_choice"].sum(dim=-1), torch.ones(2, 2))
    assert torch.allclose(probabilities["vd_dinucl"].reshape(2, 4, 4).sum(dim=-1), torch.ones(2, 4))


def test_full_batch_smoke_optimisation_reduces_reconstruction_loss(tmp_path):
    pytest.importorskip("comet_ml")
    torch = pytest.importorskip("torch")
    from tcr_foundation.igor_autoencoder import fit_autoencoder, reconstruction_loss

    first, second = _store(tmp_path / "one"), _store(tmp_path / "two")
    features, schema = load_feature_matrix([first, second])
    torch.manual_seed(0)
    model, trajectory = fit_autoencoder(features, schema, latent_dim=2, hidden_dim=8,
                                        dropout=0.0, steps=20, learning_rate=1e-2)
    assert len(trajectory) == 20
    assert trajectory[-1] < trajectory[0]
    loss, diagnostics = reconstruction_loss(model, torch.as_tensor(features))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for name, value in diagnostics.items() if name != "latent")
