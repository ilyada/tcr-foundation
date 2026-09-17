import numpy as np

from tcr_foundation.evaluation.emerson_cmv_fisher_geometry import _nearest_publicness_pools


def test_nearest_publicness_pools_matches_v_length_and_uses_deterministic_ties():
    separator = "\x1f"
    anchor = separator.join(("TRBV1", "CASS", "TRBJ1"))
    same_distance_first = separator.join(("TRBV1", "CATS", "TRBJ2"))
    same_distance_second = separator.join(("TRBV1", "CATS", "TRBJ3"))
    other_length = separator.join(("TRBV1", "CASSA", "TRBJ1"))
    other_v = separator.join(("TRBV2", "CATS", "TRBJ1"))
    pools, qc = _nearest_publicness_pools(
        [anchor],
        {anchor: 10, same_distance_second: 11, same_distance_first: 9, other_length: 10, other_v: 10},
        controls_per_anchor=2,
    )
    assert pools[anchor].tolist() == [same_distance_first, same_distance_second]
    assert bool(qc.loc[0, "retained"])
    assert qc.loc[0, "maximum_selected_publicness_difference"] == 1


def test_nearest_publicness_pools_marks_ineligible_anchor_without_controls():
    separator = "\x1f"
    anchor = separator.join(("TRBV1", "CASS", "TRBJ1"))
    pools, qc = _nearest_publicness_pools([anchor], {anchor: 2}, controls_per_anchor=2)
    assert pools == {}
    assert not bool(qc.loc[0, "retained"])
    assert np.isnan(qc.loc[0, "maximum_selected_publicness_difference"])
