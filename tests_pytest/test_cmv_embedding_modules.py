"""Pure unit checks for the leakage-safe CMV embedding-module primitives."""

import numpy as np

from tcr_foundation.evaluation.emerson_cmv_embedding_modules import (
    LocalConfiguration,
    TrainingState,
    _nearest_publicness_controls,
    _nearest_reference_sets,
    _rule,
)


SEP = "\x1f"


def _identity(cdr3aa: str) -> str:
    return SEP.join(("TRBV1", cdr3aa, "TRBJ1"))


def test_controls_are_nearest_in_publicness_within_v_and_length():
    anchor, close, second, distant, wrong_length = (_identity("CASS"), _identity("CATS"), _identity("CATT"), _identity("CAQQ"), _identity("CAT"))
    publicness = {anchor: 10, close: 11, second: 8, distant: 30, wrong_length: 10}
    controls = _nearest_publicness_controls([anchor], publicness, maximum_controls=2, seed=9)
    assert controls[anchor] == [close, second]


def test_reference_neighbours_exclude_the_query_identity():
    identities = [_identity("CASS"), _identity("CATS"), _identity("CATSX")]
    vectors = np.asarray(((1.0, 0.0), (0.9, 0.1), (0.0, 1.0)), dtype=np.float32)
    state = TrainingState(set(), [], np.empty((0, 2), dtype=np.float32), {}, {}, np.asarray(identities, dtype=object), vectors, {identity: index for index, identity in enumerate(identities)}, ["e0", "e1"], None, None)
    neighbours = _nearest_reference_sets([identities[0]], vectors[:1], state, neighbours=1)
    assert identities[0] not in neighbours[0]
    assert neighbours[0] == frozenset((identities[1],))


def test_topology_rule_uses_shared_neighbours_without_replacing_distance():
    anchor, control_one, control_two, reference = (_identity("CASS"), _identity("CATS"), _identity("CATSX"), _identity("CATSY"))
    embeddings = {
        anchor: np.asarray((1.0, 0.0), dtype=np.float32),
        control_one: np.asarray((0.99, 0.01), dtype=np.float32),
        control_two: np.asarray((0.8, 0.2), dtype=np.float32),
        reference: np.asarray((0.0, 1.0), dtype=np.float32),
    }
    reference_ids = np.asarray((anchor, control_one, control_two, reference), dtype=object)
    vectors = np.vstack([embeddings[identity] / np.linalg.norm(embeddings[identity]) for identity in reference_ids])
    state = TrainingState({anchor}, [anchor], vectors[:1], {anchor: [control_one, control_two]}, embeddings, reference_ids, vectors, {identity: index for index, identity in enumerate(reference_ids)}, ["e0", "e1"], None, None)
    direct, topology, anchor_sets = _rule(state, LocalConfiguration("topology", controls=2, direct_rank=1, neighbours=1, topology_rank=1, logistic_c=1.0))
    assert direct.shape == (1,)
    assert topology is not None and topology.shape == (1,)
    assert anchor_sets is not None and len(anchor_sets) == 1


if __name__ == "__main__":
    test_controls_are_nearest_in_publicness_within_v_and_length()
    test_reference_neighbours_exclude_the_query_identity()
    test_topology_rule_uses_shared_neighbours_without_replacing_distance()
    print("cmv-embedding-unit-smoke-ok")
