import numpy as np
import pytest

from tcr_foundation import generator as G


class Corpus:
    donor_names = ["D1", "D2", "D3", "D4"]
    cohort = {"D1": "Cohort 01", "D2": "Cohort 01", "D3": "Cohort 02", "D4": "Cohort 02"}
    donor = np.array([0, 0, 1, 1, 2, 2, 3, 3])


def test_split_requires_a_real_holdout_and_keeps_donors_disjoint():
    train, dev, hold = G.split_donors(Corpus(), dev_donors=1, seed=0)
    assert set(train).isdisjoint(dev)
    assert set(train).isdisjoint(hold)
    assert set(dev).isdisjoint(hold)
    assert set(Corpus.donor[hold]) == {2, 3}


def test_split_rejects_missing_cohort_metadata():
    corpus = Corpus()
    corpus.cohort = {"D1": "Cohort 01"}
    with pytest.raises(ValueError, match="missing cohort metadata"):
        G.split_donors(corpus)


def test_id_table_representation_is_removed_before_loading_torch():
    with pytest.raises(ValueError, match="unknown donor representation"):
        G.build_model(None, donor_representation="id_table")
