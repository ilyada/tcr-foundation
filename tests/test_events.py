"""
test_events.py -- the event schema, the Adaptive reader, and the frame-leak instrument.

This is a test of the INSTRUMENT, not of the data: it needs no cluster, no raw files and no GPU. It builds
recombination events whose coordinates are independent by construction, so the true frame leak is known to
be near zero, and then plants a dependence the instrument is obliged to see. A diagnostic that cannot fail
on a planted signal cannot be trusted when it reports a null.

Run: python tests/test_events.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # import tcr_foundation without installing

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from tcr_foundation import events as E               # noqa: E402
from tcr_foundation import diagnostics as D          # noqa: E402
from tcr_foundation import schema as S               # noqa: E402

fails = []
rng = np.random.default_rng(0)


def make_events(n_donors=20, per_donor=3000, plant=False, seed=0):
    """Independent coordinates; junction length is their SUM, so productivity is a deterministic function
    of the event exactly as in real data. With `plant`, ins_n1 is shifted by residue class -- the failure
    mode the diagnostic exists to catch."""
    r = np.random.default_rng(seed)
    n = n_donors * per_donor
    del_v = np.clip(r.poisson(4, n), 0, 20)
    del_d5 = np.clip(r.poisson(3, n), 0, 12)
    del_d3 = np.clip(r.poisson(4, n), 0, 12)
    del_j = np.clip(r.poisson(4, n), 0, 20)
    ins_n1 = np.clip(r.geometric(0.22, n) - 1, 0, 30)
    ins_n2 = np.clip(r.geometric(0.22, n) - 1, 0, 30)
    v_gene = r.choice([f"TRBV{i}" for i in range(1, 31)], n)
    j_gene = r.choice([f"TRBJ{i}" for i in range(1, 14)], n)

    length = 40 - del_v - del_d5 - del_d3 - del_j + ins_n1 + ins_n2
    if plant:                                        # make ins_n1 carry the residue directly
        ins_n1 = ins_n1 + 4 * (length % 3 == 1)
    stop = r.random(n) < 0.02
    frame = np.where(length % 3 != 0, E.FRAME_OUT, np.where(stop, E.FRAME_STOP, E.FRAME_IN))
    return pd.DataFrame({
        "donor": np.repeat([f"D{i:03d}" for i in range(n_donors)], per_donor),
        "frame_type": frame, "cdr3_length": length.astype("int32"),
        "del_v": del_v, "del_d5": del_d5, "del_d3": del_d3, "del_j": del_j,
        "ins_n1": ins_n1, "ins_n2": ins_n2, "ins_vj": ins_n1 + ins_n2,
        "v_gene": v_gene, "j_gene": j_gene, "d_present": True,
    })


# --- 1. schema carries nucleotides and productivity, and still ingests a messy frame -------------------

raw = pd.DataFrame({"v": ["TRBV20-1", "TRBV5-1"], "cdr3aa": ["CASSF", "CASTF"],
                    "nSeqCDR3": ["tgtgcc", "tgtacc"], "productive": ["True", "False"], "count": [3, 1]})
ing = S.ingest(raw)
ok = S.CDR3_NT in ing.columns and S.PRODUCTIVE in ing.columns
print(f"schema: cdr3_nt+productive ingested = {ok} (expect True); "
      f"nt upper-cased = {ing[S.CDR3_NT].iloc[0]!r} (expect 'TGTGCC'); "
      f"productive dtype = {ing[S.PRODUCTIVE].dtype} (expect bool)")
if not ok or ing[S.CDR3_NT].iloc[0] != "TGTGCC" or ing[S.PRODUCTIVE].dtype != bool:
    fails.append("schema nt/productive ingest")
if list(ing[S.PRODUCTIVE]) != [True, False]:
    fails.append("schema productive coercion")


# --- 2. Adaptive reader: sentinel rule, frame classes, mod-3 invariant ---------------------------------

tmp = os.path.join(_HERE, "_tmp_adaptive.tsv")
n_rows = 300
r = np.random.default_rng(1)
ins1 = np.clip(r.geometric(0.3, n_rows) - 1, 0, 20)
ins2 = np.clip(r.geometric(0.3, n_rows) - 1, 0, 20)
v_idx = np.full(n_rows, 30)
n1_idx = np.where(ins1 > 0, v_idx, -1)               # Adaptive writes -1 when there is no insertion
d_idx = v_idx + ins1
n2_idx = np.where(ins2 > 0, d_idx + 5, -1)
j_idx = d_idx + 5 + ins2
lengths = j_idx + 10
frames = np.where(lengths % 3 != 0, "Out", np.where(r.random(n_rows) < 0.05, "Stop", "In"))
# force the frame label to agree with the length, as the real export does
lengths = np.where(frames == "Out", lengths + ((3 - lengths % 3) % 3 + 1) % 3, lengths - lengths % 3)
frames = np.where(lengths % 3 != 0, "Out", frames)
pd.DataFrame({
    "sample_name": "TESTSAMPLE", "rearrangement": ["ACGT" * 10] * n_rows, "amino_acid": "",
    "frame_type": frames, "rearrangement_type": "VDJ", "templates": 1, "cdr3_length": lengths,
    "v_gene": "TCRBV05-01", "d_gene": "TCRBD01-01", "j_gene": "TCRBJ02-07",
    "v_deletions": 2, "d5_deletions": 1, "d3_deletions": 3, "j_deletions": 2,
    "n1_insertions": ins1, "n2_insertions": ins2,
    "v_index": v_idx, "n1_index": n1_idx, "n2_index": n2_idx, "d_index": d_idx, "j_index": j_idx,
    "v_resolved": "TCRBV05-01", "d_resolved": "TCRBD01-01", "j_resolved": "TCRBJ02-07",
    "sample_tags": "Cohort:Cohort 01,Virus Diseases:Cytomegalovirus +,HLA MHC class I:HLA-A*02,"
                   "Inferred HLA type:Inferred HLA-A*01,Age:41 Years,Biological Sex:Male",
}).to_csv(tmp, sep="\t", index=False)

ev = E.AdaptiveReader().read(tmp)
counts = E.validate_events(ev, platform="adaptive")
print(f"reader: rows {counts['rows']} (expect {n_rows}); "
      f"n1 sentinel {counts['n1_sentinel_ok']}/{counts['n1_sentinel_n']}, "
      f"gap {counts['n1_gap_ok']}/{counts['n1_gap_n']}; "
      f"n2 sentinel {counts['n2_sentinel_ok']}/{counts['n2_sentinel_n']}, "
      f"gap {counts['n2_gap_ok']}/{counts['n2_gap_n']} (all must be equal pairs)")
for tag in ("n1_sentinel", "n1_gap", "n2_sentinel", "n2_gap"):
    if counts[f"{tag}_ok"] != counts[f"{tag}_n"]:
        fails.append(f"adaptive {tag} {counts[f'{tag}_ok']}/{counts[f'{tag}_n']}")
if not counts["mod3_rule_ok"]:
    fails.append("adaptive mod3 rule")
if set(ev.columns) != set(E.EVENT_COLUMNS):
    fails.append("adaptive column set")

meta = E.read_sample_metadata(tmp)
print(f"tags: cohort={meta['cohort']!r} cmv={meta['cmv']!r} age={meta['age']} "
      f"hla_known={meta['hla_known']!r} hla_inferred={meta['hla_inferred']!r} "
      f"(known and inferred must NOT be the same field)")
if (meta["cohort"], meta["cmv"], meta["age"], meta["sex"]) != ("Cohort 01", "+", 41, "Male"):
    fails.append("sample tag parse")
if meta["hla_known"] != "HLA-A*02" or meta["hla_inferred"] != "HLA-A*01":
    fails.append("hla known/inferred separation")
os.remove(tmp)


# --- 3. the instrument returns the noise floor when the coordinates really are independent -------------

clean = make_events(seed=2)
rep = D.frame_leak_report(clean, n_perm=50, seed=0, max_rows=200_000)
main = rep[(rep.arm == "main") & (rep.coordinate == "TOTAL")].iloc[0]
cal = rep[(rep.arm == "calibration") & (rep.coordinate == "TOTAL")].iloc[0]
print(f"independent coords: main TOTAL {main.kl_nats:.4f} nats (floor {main.null_mean:.4f}, z {main.z:.1f}) "
      f"-- expect well under the 0.21 counting bar")
if main.kl_nats > 0.05:
    fails.append(f"main arm on independent coordinates = {main.kl_nats:.4f} nats, expected < 0.05")
print(f"calibration TOTAL {cal.kl_nats:.4f} nats -- productive vs non-productive on the SAME generator, "
      f"so this is the frame effect alone")


# --- 4. ... and sees a planted dependence ---------------------------------------------------------------

dirty = make_events(seed=2, plant=True)
rep2 = D.frame_leak_report(dirty, n_perm=50, seed=0, max_rows=200_000)
main2 = rep2[(rep2.arm == "main") & (rep2.coordinate == "TOTAL")].iloc[0]
n1 = rep2[(rep2.arm == "main") & (rep2.coordinate == "ins_n1")].iloc[0]
print(f"planted dependence: main TOTAL {main2.kl_nats:.4f} nats (z {main2.z:.1f}), "
      f"ins_n1 alone {n1.kl_nats:.4f} (z {n1.z:.1f}) -- expect both far above the clean run")
if main2.kl_nats <= max(0.05, 5 * main.kl_nats):
    fails.append(f"planted dependence not detected: {main2.kl_nats:.4f} vs clean {main.kl_nats:.4f}")
if n1.kl_nats <= rep[(rep.arm == "main") & (rep.coordinate == "ins_n1")].iloc[0].kl_nats:
    fails.append("planted coordinate ins_n1 did not rise")

gene = rep2[(rep2.arm == "main") & (rep2.coordinate == "v_gene")].iloc[0]
print(f"negative control: v_gene {gene.kl_nats:.4f} nats (z {gene.z:.1f}) -- the gene is not a term of the "
      f"sum, so the residue cannot move it")
if abs(gene.z) > 8:
    fails.append(f"v_gene control moved (z={gene.z:.1f}); the arms are separating on something else")

print("\nEVENTS_OK" if not fails else f"\nEVENTS_FAIL: {fails}")
sys.exit(0 if not fails else 1)
