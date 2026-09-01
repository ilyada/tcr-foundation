import pandas as pd
import pytest

from tcr_foundation import events as E


def test_identity_map_preserves_biological_and_library_keys():
    events = pd.DataFrame({"sample": ["S1", "S1"], "donor": ["S1", "S1"], "library": ["S1", "S1"],
                           "timepoint": ["", ""]})
    identity = pd.DataFrame({"sample": ["S1"], "donor": ["D1"], "library": ["L1"], "timepoint": ["pre"]})
    got = E._apply_identity(events, E._normalise_identity_map(identity))
    assert got["donor"].astype(str).tolist() == ["D1", "D1"]
    assert got["library"].astype(str).tolist() == ["L1", "L1"]
    assert got["timepoint"].astype(str).tolist() == ["pre", "pre"]


def test_identity_map_rejects_an_unmapped_sample():
    events = pd.DataFrame({"sample": ["S2"], "donor": ["S2"], "library": ["S2"], "timepoint": [""]})
    identity = pd.DataFrame({"sample": ["S1"], "donor": ["D1"]})
    with pytest.raises(ValueError, match="no row for sample"):
        E._apply_identity(events, E._normalise_identity_map(identity))
