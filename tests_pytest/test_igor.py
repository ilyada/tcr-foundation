import pandas as pd

from tcr_foundation.igor import build_scopes, read_input


def test_mixcr_and_canonical_inputs_produce_the_same_cdr3nt(tmp_path):
    mixcr = tmp_path / "patient.tsv"
    mixcr.write_text("nSeqCDR3\taaSeqCDR3\nAACCGGTT\tCA_G\nTTGGCCAA\tCA*G\nCCCCCCCC\tCASS\n")
    canonical = tmp_path / "patient.parquet"
    pd.DataFrame({"frame_type": ["out", "stop", "in"], "rearrangement": ["XAACCGGTTY", "TTGGCCAA", "CCCCCCCC"],
                  "v_index": [1, 0, 0], "cdr3_length": [8, 8, 8]}).to_parquet(canonical, index=False)
    assert read_input(mixcr).cdr3nt.tolist() == ["AACCGGTT", "TTGGCCAA", "CCCCCCCC"]
    assert read_input(canonical).cdr3nt.tolist() == ["AACCGGTT", "TTGGCCAA", "CCCCCCCC"]


def test_individual_deduplicates_within_file_and_group_keeps_between_file_repetitions(tmp_path):
    for name, rows in (("a", [("AAA", "A_A"), ("AAA", "A_A"), ("CCC", "A*A")]),
                       ("b", [("AAA", "A_A"), ("GGG", "A*A")])):
        (tmp_path / f"{name}.tsv").write_text("nSeqCDR3\taaSeqCDR3\n" + "".join(f"{nt}\t{aa}\n" for nt, aa in rows))
    individual, group = build_scopes(tmp_path, "individual"), build_scopes(tmp_path, "group")
    assert [(scope.name, len(scope.sequences)) for scope in individual] == [("a", 2), ("b", 2)]
    assert group[0].sequences.count("AAA") == 2
