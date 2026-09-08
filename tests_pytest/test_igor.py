from pathlib import Path

import pandas as pd

from tcr_foundation.igor import Scope, build_scopes, fit_scope, read_input


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


def test_patient_pooling_unites_cd4_cd8_and_retains_single_library_donor(tmp_path):
    tables = {
        "HD_AK_CD4_TRB.tsv": [("AAA", "A_A"), ("CCC", "A*A")],
        "HD_AK_CD8_TRB.tsv": [("AAA", "A_A"), ("GGG", "A*A")],
        "HD_BM_CD4_TRB.tsv": [("TTT", "A_A")],
    }
    for name, rows in tables.items():
        (tmp_path / name).write_text("nSeqCDR3\taaSeqCDR3\n" + "".join(f"{nt}\t{aa}\n" for nt, aa in rows))
    scopes = build_scopes(tmp_path, "patient", patient_pattern=r"^(HD_[A-Z]+)_")
    assert [(scope.name, scope.kind, len(scope.source_files), scope.sequences) for scope in scopes] == [
        ("HD_AK", "patient", 2, ("AAA", "CCC", "GGG")),
        ("HD_BM", "patient", 1, ("TTT",)),
    ]


def test_patient_mode_requires_one_explicit_donor_capture_group(tmp_path):
    (tmp_path / "HD_AK_CD4_TRB.tsv").write_text("nSeqCDR3\taaSeqCDR3\nAAA\tA_A\n")
    try:
        build_scopes(tmp_path, "patient")
    except ValueError as error:
        assert "patient-pattern" in str(error)
    else:
        raise AssertionError("patient mode accepted no donor grouping rule")


def test_resume_skips_native_final_files_without_requiring_manifest(tmp_path):
    destination = tmp_path / "D1" / "model_inference"
    destination.mkdir(parents=True)
    (destination / "final_marginals.txt").write_text("done\n")
    (destination / "final_parms.txt").write_text("done\n")
    scope = Scope("D1", (), (), ("AAA",), "patient")
    assert fit_scope(scope, igor="not-called", out_dir=str(tmp_path), iterations=5, igor_threads=1,
                     keep_temp=False, resume=True) == Path(tmp_path) / "D1"


def test_restart_incomplete_archives_only_that_scope(tmp_path, monkeypatch):
    from tcr_foundation import igor as igor_module

    incomplete = tmp_path / "D1"
    incomplete.mkdir()
    (incomplete / "partial.txt").write_text("unfinished\n")
    monkeypatch.setattr(igor_module, "igor_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(igor_module, "_igor_version", lambda _igor: "test")
    scope = Scope("D1", (), (), ("AAA",), "patient")
    result = fit_scope(scope, igor="not-called", out_dir=str(tmp_path), iterations=5, igor_threads=1,
                       keep_temp=False, resume=True, restart_incomplete=True)
    assert result == tmp_path / "D1"
    assert (tmp_path / "D1.interrupted" / "partial.txt").is_file()
    assert (tmp_path / "D1" / "manifest.json").is_file()
