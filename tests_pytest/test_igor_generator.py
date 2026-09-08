import pytest

pytest.importorskip("comet_ml")
from tcr_foundation.igor_autoencoder import IGoRFeatureSchema, build_autoencoder
from tcr_foundation.igor_generator import BOS, LENGTH_OFFSET, build_igor_conditioned_generator
torch = pytest.importorskip("torch")


def _schema():
    axes = {"V_gene": ("TRBV1", "TRBV2"), "D_gene": ("TRBD1",), "J_gene": ("TRBJ1", "TRBJ2")}
    shapes = {
        "v_choice": (2,), "j_choice": (2, 2), "d_gene": (2, 2, 1),
        "v_3_del": (2, 2), "d_5_del": (1, 2), "d_3_del": (1, 2, 2),
        "j_5_del": (2, 2), "vd_ins": (2,), "vd_dinucl": (16,),
        "dj_ins": (2,), "dj_dinucl": (16,),
    }
    return IGoRFeatureSchema(axes=axes, block_shapes=shapes)


def test_every_decoder_layer_has_cross_attention_to_three_condition_tokens():
    schema = _schema()
    autoencoder = build_autoencoder(schema, latent_dim=8, hidden_dim=12, dropout=0.0)
    model = build_igor_conditioned_generator(autoencoder, d_model=8, n_layer=2, n_head=2, max_junction=4)
    features = torch.rand(3, schema.probability_dim)
    v_gene, j_gene = torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])
    tokens = torch.tensor([[BOS, LENGTH_OFFSET + 2, 0, 1, 2], [BOS, LENGTH_OFFSET + 1, 3, 0, 5],
                           [BOS, LENGTH_OFFSET + 3, 2, 1, 0]])
    lengths = torch.tensor([3, 2, 4])

    memory = model.condition_memory(features, v_gene, j_gene)
    logits = model.logits(tokens, features, v_gene, j_gene, lengths=lengths)
    likelihood = model.sequence_log_likelihood(tokens, features, v_gene, j_gene, lengths)

    assert memory.shape == (3, 3, 8)
    assert logits.shape == (3, 5, LENGTH_OFFSET + 4)
    assert torch.isfinite(likelihood).all()
    assert all(hasattr(block, "crossattention") for block in model.decoder.transformer.h)


def test_patient_generation_samples_vj_from_igor_heads_and_emits_only_nucleotides():
    schema = _schema()
    autoencoder = build_autoencoder(schema, latent_dim=8, hidden_dim=12, dropout=0.0)
    model = build_igor_conditioned_generator(autoencoder, d_model=8, n_layer=1, n_head=2, max_junction=4)
    generated = model.generate(torch.rand(2, schema.probability_dim))

    assert len(generated) == 2
    for v_gene, j_gene, sequence in generated:
        assert v_gene in schema.axes["V_gene"]
        assert j_gene in schema.axes["J_gene"]
        assert 1 <= len(sequence) <= 4
        assert set(sequence) <= {"A", "C", "G", "T"}
