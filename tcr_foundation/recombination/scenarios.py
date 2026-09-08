"""
scenarios.py -- every recombination scenario that could have produced an observed junction.

The scenario is latent. An aligner reports one of them, chosen by a deterministic convention (measured on
Emerson: it extends the germline maximally, leaving zero alternatives in the direction of LESS trimming --
0 of 696,838 rows on the J side). Fitting a density to that one parse fits the aligner's convention as much
as the biology. The marginal likelihood sums over all of them instead:

    log P(x | donor) = logsumexp over scenarios s compatible with x of log P(s | donor)

This module produces the compatible set. It contains no model and no parameters -- it is preprocessing, run
once and cached, so that the training objective stays fixed while the model changes.

WHAT MAKES A SCENARIO COMPATIBLE
--------------------------------
The germline-derived parts must match the observed string EXACTLY. There are no gaps and no mismatches in a
recombination model: a mismatch is sequencing error or somatic hypermutation, and allowing one here would
let the model explain error as biology. So the operation is longest-common-prefix from the left for V,
longest-common-suffix from the right for J, and exact substring occurrence for D. No aligner library is
needed or wanted.

Compatibility is one-sided. A V contribution SHORTER than the longest match is always compatible -- the
extra base is simply declared an insertion that happened to equal the germline -- while a longer one is
not. So the scenarios form a lattice indexed by how far each boundary is pushed back, and each step of
pushback costs at least log 4 = 1.39 nats through the inserted base alone. `d_max` caps that depth; the
mass it discards is bounded by the same argument and is reported rather than assumed.

The no-D scenario is always emitted. It is how a rearrangement whose D the aligner could not call enters
the model -- 21.5 percent of Emerson rows -- without a separate branch: whether a D is present becomes a
posterior over scenarios instead of an annotation.

CLI
    python -m tcr_foundation.scenarios --events DIR --reference FILE --out FILE [--limit N]
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

#: A D contribution shorter than this is indistinguishable from insertion, so it is left to the no-D
#: scenario rather than enumerated separately -- otherwise a 2-nt match occurs by chance almost everywhere
#: and the scenario count explodes with terms the model would price at near zero anyway.
MIN_D_LEN = 3


def lcp(a: str, b: str) -> int:
    """Length of the longest common prefix."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def lcs(a: str, b: str) -> int:
    """Length of the longest common suffix."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def contribution_range(match_len: int, d_max: int) -> range:
    """Germline contributions compatible with a match of this length.

    Everything from the full match down to d_max bases short of it: a shorter contribution reassigns the
    difference to the insertion, which is always allowed, while a longer one would mismatch."""
    return range(max(0, match_len - d_max), match_len + 1)


COMP = str.maketrans("ACGTN", "TGCAN")

#: Palindromic runs longer than this are not seen in practice -- the hairpin is nicked a few bases in.
P_MAX = 4


def extend(g: str, side: str, p_max: int = P_MAX) -> str:
    """A template with its own palindromic tails attached, so that trimming and palindrome become ONE
    coordinate: how much of the extended template was used.

    Opening the hairpin copies the terminal bases back out in reverse complement. Writing that copy into
    the template turns "germline minus a trim, or germline plus a palindrome" into a single contiguous
    substring, and the constraint that a palindrome only exists at zero trimming holds automatically --
    a substring reaching into the tail necessarily contains the whole germline end it came from.

    side is "V" (tail on the 3' side), "J" (tail on the 5' side) or "D" (both)."""
    head = "".join(g[i].translate(COMP) for i in range(min(p_max, len(g)) - 1, -1, -1))
    tail = "".join(g[-1 - i].translate(COMP) for i in range(min(p_max, len(g))))
    if side == "V":
        return g + tail
    if side == "J":
        return head + g
    return head + g + tail


def palindrome_forward(g: str, m: int, x: str, p_max: int = P_MAX) -> int:
    """Palindromic bases that may follow a germline contribution of m bases read left to right.

    Opening the hairpin copies the segment's own terminal bases back out in reverse complement, so the base
    after the germline is the complement of its last base, the next is the complement of the one before,
    and so on. This is a third category: neither germline nor a random TdT insertion, and a model that
    knows only those two cannot explain 16.6 percent of D spans or 1.2 percent of V spans in this corpus.

    Only reachable at zero trimming -- once the end is chewed back the hairpin is gone -- which is why it
    costs a few extra options at one boundary rather than a multiplier on every scenario."""
    p = 0
    while p < p_max and m + p < len(x) and m - 1 - p >= 0 and x[m + p] == g[m - 1 - p].translate(COMP):
        p += 1
    return p


def palindrome_backward(g: str, m: int, x: str, p_max: int = P_MAX) -> int:
    """The same, mirrored: palindromic bases preceding a germline contribution of the LAST m bases of g."""
    n, L = len(g), len(x)
    p = 0
    while p < p_max and L - m - 1 - p >= 0 and n - m + p < n \
            and x[L - m - 1 - p] == g[n - m + p].translate(COMP):
        p += 1
    return p


def d_match_table(g: str, x: str) -> np.ndarray:
    """match[d5, p] = length of the common prefix of g[d5:] and x[p:], for every start pair at once.

    Computed by the recurrence match[d5, p] = match[d5+1, p+1] + 1 when the bases agree and 0 otherwise, so
    each cell costs one operation instead of its own character loop. This is the dominant cost of the whole
    enumeration -- a D template is scanned against every position of every junction -- and the recurrence
    makes it a handful of vectorised passes rather than len(g) x len(x) inner loops."""
    n, L = len(g), len(x)
    gi = np.frombuffer(g.encode(), dtype=np.uint8)
    xi = np.frombuffer(x.encode(), dtype=np.uint8)
    m = np.zeros((n + 1, L + 1), dtype=np.int16)
    for d5 in range(n - 1, -1, -1):
        m[d5, :L] = np.where(gi[d5] == xi, m[d5 + 1, 1:L + 1] + 1, 0)
    return m


#: One scenario, as integer columns. The inserted bases are named by spans into the junction rather than
#: copied, so a scenario is 12 small integers whatever the insert length.
SCENARIO_FIELDS = ("v_idx", "d_idx", "j_idx", "del_v", "del_d5", "del_d3", "del_j",
                   "p_v", "p_d5", "p_d3", "p_j",
                   "ins_n1", "ins_n2", "ins1_start", "ins2_start", "d_present")


def enumerate_scenarios(x: str, v_cands, d_cands, j_cands, ref: dict, d_max: int = 4,
                        min_d_len: int = MIN_D_LEN, key_index: dict | None = None) -> dict:
    """Every compatible scenario for one junction, as flat integer arrays.

    Candidates are template KEYS of the reference; key_index maps them to the integer ids the model uses."""
    L = len(x)
    ki = key_index or {}

    # Templates carry their own palindromic tails, so "used length" covers trimming and palindrome at once:
    # below the germline length it is a trim, above it a palindrome.
    v_opts, j_opts = [], []
    for vk in v_cands:
        g = ref.get(vk, {}).get("junction_side")
        if g:
            v_opts.append((ki.get(vk, 0), len(g), contribution_range(lcp(extend(g, "V"), x), d_max)))
    for jk in j_cands:
        g = ref.get(jk, {}).get("junction_side")
        if g:
            j_opts.append((ki.get(jk, 0), len(g), contribution_range(lcs(extend(g, "J"), x), d_max)))
    if not v_opts or not j_opts:
        return {f: np.zeros(0, np.int16) for f in SCENARIO_FIELDS}

    # D matches do not depend on where the V and J boundaries fall, so they are found once against the whole
    # junction and filtered afterwards by the window each (V, J) pair leaves.
    hd, hd5, hd3, hp5, hp3, hp, ht = [], [], [], [], [], [], []
    for dk in d_cands:
        g = ref.get(dk, {}).get("junction_side")
        if not g or len(g) < min_d_len:
            continue
        n = len(g)
        ge = extend(g, "D")
        head = min(P_MAX, n)                       # bases of palindromic tail prepended
        ne, m = len(ge), d_match_table(ge, x)
        for s in range(ne):
            top = np.minimum(m[s, :L].astype(np.int32), ne - s)
            pos = np.nonzero(top >= min_d_len)[0]
            if not len(pos):
                continue
            # Each position admits a run of used lengths. Expanding that ragged run with a cumulative-offset
            # arange keeps the whole hit list in numpy instead of a Python loop.
            cnt = top[pos] - min_d_len + 1
            starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
            t = min_d_len + (np.arange(int(cnt.sum())) - starts)
            p = np.repeat(pos, cnt)
            # Decode the offset in the EXTENDED template back into trimming and palindrome.
            p5 = max(0, head - s)
            d5 = max(0, s - head)
            gend = (s + t) - head                  # germline end index, may exceed n inside the 3' tail
            p3 = np.maximum(0, gend - n)
            d3 = np.maximum(0, n - gend)
            keep = np.nonzero((t - p5 - p3) >= min_d_len)[0]   # the GERMLINE part must clear the floor
            if not len(keep):
                continue
            hp.append(p[keep]); ht.append(t[keep])
            hd.append(np.full(len(keep), ki.get(dk, 0), np.int32))
            hd5.append(np.full(len(keep), d5, np.int32)); hd3.append(d3[keep])
            hp5.append(np.full(len(keep), p5, np.int32)); hp3.append(p3[keep])
    if hp:
        hp, ht = np.concatenate(hp), np.concatenate(ht)
        hd, hd5, hd3 = np.concatenate(hd), np.concatenate(hd5), np.concatenate(hd3)
        hp5, hp3 = np.concatenate(hp5), np.concatenate(hp3)
    else:
        hp = ht = hd = hd5 = hd3 = hp5 = hp3 = np.zeros(0, np.int32)

    # Flatten the boundary options, then take the whole thing as one masked outer product. Looping over the
    # (V, J) combinations and filtering hits inside each was measurably SLOWER than building Python tuples:
    # a few hundred tiny numpy allocations per junction cost more than the arithmetic they save.
    v_id = np.concatenate([np.full(len(r), i, np.int32) for i, _, r in v_opts])
    v_len = np.concatenate([np.fromiter(r, np.int32) for _, _, r in v_opts])
    v_glen = np.concatenate([np.full(len(r), g, np.int32) for _, g, r in v_opts])
    v_del, v_pal = np.maximum(0, v_glen - v_len), np.maximum(0, v_len - v_glen)
    j_id = np.concatenate([np.full(len(r), i, np.int32) for i, _, r in j_opts])
    j_len = np.concatenate([np.fromiter(r, np.int32) for _, _, r in j_opts])
    j_glen = np.concatenate([np.full(len(r), g, np.int32) for _, g, r in j_opts])
    j_del, j_pal = np.maximum(0, j_glen - j_len), np.maximum(0, j_len - j_glen)

    nv, nj = len(v_len), len(j_len)
    V = np.repeat(np.arange(nv), nj)
    J = np.tile(np.arange(nj), nv)
    ws, we = v_len[V], L - j_len[J]
    ok = we >= ws
    V, J, ws, we = V[ok], J[ok], ws[ok], we[ok]
    nC = len(V)
    if nC == 0:
        return {f: np.zeros(0, np.int16) for f in SCENARIO_FIELDS}

    # The no-D scenario for every boundary combination, then every (combination, D hit) pair whose D falls
    # inside that combination's window.
    if len(hp):
        C2 = np.repeat(np.arange(nC), len(hp))
        H2 = np.tile(np.arange(len(hp)), nC)
        m = (hp[H2] >= ws[C2]) & (hp[H2] + ht[H2] <= we[C2])
        C2, H2 = C2[m], H2[m]
    else:
        C2 = H2 = np.zeros(0, np.int32)

    cat = np.concatenate
    z = np.zeros(nC, np.int32)
    out = {
        "v_idx": cat([v_id[V], v_id[V[C2]]]),
        "j_idx": cat([j_id[J], j_id[J[C2]]]),
        "del_v": cat([v_del[V], v_del[V[C2]]]),
        "del_j": cat([j_del[J], j_del[J[C2]]]),
        "p_v": cat([v_pal[V], v_pal[V[C2]]]),
        "p_j": cat([j_pal[J], j_pal[J[C2]]]),
        "ins1_start": cat([ws, ws[C2]]),
        "d_idx": cat([z, hd[H2]]),
        "del_d5": cat([z, hd5[H2]]),
        "del_d3": cat([z, hd3[H2]]),
        "p_d5": cat([z, hp5[H2]]),
        "p_d3": cat([z, hp3[H2]]),
        "d_present": cat([z, np.ones(len(H2), np.int32)]),
        "ins_n1": cat([we - ws, hp[H2] - ws[C2]]),
        "ins_n2": cat([z, we[C2] - hp[H2] - ht[H2]]),
        "ins2_start": cat([we, hp[H2] + ht[H2]]),
    }
    return {f: out[f].astype(np.int16) for f in SCENARIO_FIELDS}


def candidates_for(row, ref: dict, by_symbol: dict, canonical) -> tuple:
    """Template candidates for one row: the reported call narrowed to what the reference actually holds.

    The reported gene is a hint used to prune, never a constraint -- a call is wrong on some genes
    (TRBV6-1 explains only 78 percent of its own spans) and family-level on others, and the enumeration
    resolves both by keeping only templates that match the sequence."""
    from .germline import lookup_symbols
    out = []
    for side, raw in (("V", row.bestv), ("D", row.d_gene), ("J", row.bestj)):
        alleles = set(lookup_symbols(by_symbol, side, canonical(raw)))
        keys = [k for k, rec in ref.items() if rec["section"] == side and rec["allele"] in alleles]
        if not keys:
            # No usable call: every template of that segment is a candidate, and the enumeration keeps
            # whichever ones actually match. An uncalled gene costs breadth, not correctness.
            keys = [k for k, rec in ref.items() if rec["section"] == side]
        out.append(keys)
    return tuple(out)


def rebuild(x: str, sc: dict, i: int, ref: dict, keys_by_id: dict) -> str:
    """Reassemble the junction from one scenario, to be compared with the observed string.

    This is the enumerator's own invariant and it refers to nothing outside itself. The earlier check --
    "the aligner's reported parse must be among the scenarios" -- was mis-specified: Adaptive's D assignment
    tolerates a mismatched base, ours requires exact germline, so its parse is sometimes not one of ours by
    construction rather than by error."""
    gv = ref[keys_by_id["V"][int(sc["v_idx"][i])]]["junction_side"]
    gj = ref[keys_by_id["J"][int(sc["j_idx"][i])]]["junction_side"]
    v_use = len(gv) - int(sc["del_v"][i]) + int(sc["p_v"][i])
    j_use = len(gj) - int(sc["del_j"][i]) + int(sc["p_j"][i])
    v_part = extend(gv, "V")[:v_use]
    j_part = extend(gj, "J")[len(extend(gj, "J")) - j_use:]
    ins1 = x[int(sc["ins1_start"][i]):int(sc["ins1_start"][i]) + int(sc["ins_n1"][i])]
    ins2 = x[int(sc["ins2_start"][i]):int(sc["ins2_start"][i]) + int(sc["ins_n2"][i])]
    d_part = ""
    if int(sc["d_present"][i]):
        gd = ref[keys_by_id["D"][int(sc["d_idx"][i])]]["junction_side"]
        ge, head = extend(gd, "D"), min(P_MAX, len(gd))
        s = head - int(sc["p_d5"][i]) + int(sc["del_d5"][i])
        e = head + len(gd) - int(sc["del_d3"][i]) + int(sc["p_d3"][i])
        d_part = ge[s:e]
    return v_part + ins1 + d_part + ins2 + j_part


def check_reconstruction(events_dir: str, reference_path: str, limit: int = 200, d_max: int = 4,
                         min_d_len: int = MIN_D_LEN) -> dict:
    """Every enumerated scenario must rebuild the observed junction exactly."""
    import glob
    from .events import best_gene
    from .germline import canonical_name, load_reference

    ref = load_reference(reference_path)
    by_symbol = {}
    for key, rec in ref.items():
        by_symbol.setdefault(rec["section"], {}).setdefault(rec["symbol"], []).append(rec["allele"])
    sec = {s: sorted(k for k, r in ref.items() if r["section"] == s) for s in "VDJ"}
    key_index, keys_by_id = {}, {}
    for s, ks in sec.items():
        off = 1 if s == "D" else 0
        key_index.update({k: i + off for i, k in enumerate(ks)})
        keys_by_id[s] = {i + off: k for i, k in enumerate(ks)}

    f = sorted(glob.glob(os.path.join(events_dir, "*.parquet")))[0]
    df = pd.read_parquet(f)
    df = df[(df.v_index >= 0) & (df.cdr3_length > 0)].head(limit)
    df = df.assign(bestv=best_gene(df, "v").values, bestj=best_gene(df, "j").values)

    total = bad = 0
    examples = []
    for r in df.itertuples():
        x = r.rearrangement[r.v_index:r.v_index + r.cdr3_length]
        vc, dc, jc = candidates_for(r, ref, by_symbol, canonical_name)
        sc = enumerate_scenarios(x, vc, dc, jc, ref, d_max=d_max, min_d_len=min_d_len,
                                 key_index=key_index)
        for i in range(len(sc["v_idx"])):
            total += 1
            got = rebuild(x, sc, i, ref, keys_by_id)
            if got != x:
                bad += 1
                if len(examples) < 3:
                    examples.append((x, got))
    return {"scenarios": total, "mismatched": bad, "examples": examples}


def measure(events_dir: str, reference_path: str, limit: int = 2000, d_max: int = 4,
            min_d_len: int = MIN_D_LEN) -> dict:
    """Rate and shape of the enumeration on a sample, before committing to a corpus-wide run.

    Reports the scenarios per sequence, the fraction with an EMPTY set -- those are sequences no scenario
    explains, i.e. sequencing error or somatic hypermutation, which a v1 without an error term must drop --
    and the throughput, which decides whether this stays in Python."""
    import glob
    from .events import best_gene
    from .germline import canonical_name, load_reference

    ref = load_reference(reference_path)
    by_symbol = {}
    for key, rec in ref.items():
        by_symbol.setdefault(rec["section"], {}).setdefault(rec["symbol"], []).append(rec["allele"])

    f = sorted(glob.glob(os.path.join(events_dir, "*.parquet")))[0]
    df = pd.read_parquet(f)
    df = df[(df.v_index >= 0) & (df.cdr3_length > 0)].head(limit)
    df = df.assign(bestv=best_gene(df, "v").values, bestj=best_gene(df, "j").values)

    key_index = {k: i for i, k in enumerate(sorted(ref))}
    counts, empties, t0 = [], 0, time.time()
    for r in df.itertuples():
        x = r.rearrangement[r.v_index:r.v_index + r.cdr3_length]
        vc, dc, jc = candidates_for(r, ref, by_symbol, canonical_name)
        sc = enumerate_scenarios(x, vc, dc, jc, ref, d_max=d_max, min_d_len=min_d_len,
                                 key_index=key_index)
        n = len(sc["v_idx"])
        counts.append(n)
        empties += (n == 0)
    dt = time.time() - t0
    c = np.array(counts)
    return {"n": len(c), "empty_frac": empties / max(len(c), 1),
            "scenarios_median": float(np.median(c)), "scenarios_p90": float(np.percentile(c, 90)),
            "scenarios_max": int(c.max()) if len(c) else 0, "seq_per_sec": len(c) / max(dt, 1e-9),
            "d_max": d_max, "min_d_len": min_d_len}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Enumerate recombination scenarios for observed junctions.")
    p.add_argument("--events", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--d-max", type=int, default=4)
    p.add_argument("--min-d-len", type=int, default=MIN_D_LEN)
    p.add_argument("--check", action="store_true", help="verify every scenario rebuilds its junction")
    a = p.parse_args()
    if a.check:
        r = check_reconstruction(a.events, a.reference, limit=200, d_max=a.d_max, min_d_len=a.min_d_len)
        print(f"  scenarios checked: {r['scenarios']}, mismatched: {r['mismatched']}")
        for x, got in r["examples"]:
            print(f"    want {x}\n    got  {got}")
        print("  RECONSTRUCT_OK" if r["mismatched"] == 0 else "  RECONSTRUCT_FAIL")
    res = measure(a.events, a.reference, limit=a.limit, d_max=a.d_max, min_d_len=a.min_d_len)
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
