"""
germline.py -- one germline reference table for the project, in the project's own naming.

Everything that reconstructs or enumerates a rearrangement needs the same three things per segment: the
nucleotide template, where the CDR3 starts or ends inside it, and a name that matches what the corpus
actually writes. Those live in three different places today -- OLGA ships the templates and the anchors
under IMGT names, the corpora write Adaptive or MiXCR names, and the encoder side canonicalises with
tidytcells. This module resolves all of that ONCE into a table and freezes it.

WHY THE ALLELE IS CHOSEN BY MEASUREMENT
---------------------------------------
A gene name like TCRBV05-01 does not say which allele was sequenced, and defaulting to *01 is a guess that
fails silently: the junction-side end of the template is exactly where alleles differ, so a wrong allele
shows up as a germline span that is not a prefix of the reference and is then misread as trimming. So the
allele is picked by agreement with the corpus -- for each candidate, how many observed germline spans of
that gene are prefixes (V) or suffixes (J) of its junction-side sequence -- and the margin over the runner
up is reported, because a small margin means the choice is not determined by the data.

WHAT "JUNCTION-SIDE" MEANS
--------------------------
Only part of each template can appear in the junction. For V it is the template from the CDR3 anchor
onward; for J it is the template up to and including the conserved codon at the anchor; for D the whole
template, since D lies inside the junction. Trimming eats these from the junction-facing end: del_v removes
from the END of the V part, del_j from the START of the J part, del_d5 and del_d3 from the two ends of D.

CLI
    python -m tcr_foundation.germline --events DIR --out reference_trb.json [--files 8]
"""
from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

#: OLGA ships the human TRB templates and anchors; using them keeps us comparable with the field's own P_gen.
OLGA_MODEL = ("default_models", "human_T_beta")


def _olga_dir() -> str:
    import olga
    return os.path.join(os.path.dirname(olga.__file__), *OLGA_MODEL)


def load_olga_templates(model_dir: str | None = None) -> dict:
    """Parse model_params.txt. Sections open with `#GeneChoice;V_gene;...` and list `%NAME;SEQUENCE`."""
    model_dir = model_dir or _olga_dir()
    out, section = {"V": {}, "D": {}, "J": {}}, None
    with open(os.path.join(model_dir, "model_params.txt")) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#GeneChoice"):
                section = "V" if "V_gene" in line else ("D" if "D_gene" in line else "J")
            elif line.startswith(("#", "@")):
                section = None
            elif line.startswith("%") and section:
                parts = line[1:].split(";")
                if len(parts) >= 2 and parts[1] and set(parts[1].upper()) <= set("ACGTN"):
                    out[section][parts[0]] = parts[1].upper()
    return out


def load_olga_anchors(model_dir: str | None = None) -> dict:
    model_dir = model_dir or _olga_dir()
    out = {}
    for side, fn in (("V", "V_gene_CDR3_anchors.csv"), ("J", "J_gene_CDR3_anchors.csv")):
        a = pd.read_csv(os.path.join(model_dir, fn))
        out[side] = dict(zip(a.iloc[:, 0].astype(str), a.iloc[:, 1].astype(int)))
    return out


_NUM = re.compile(r"0*(\d+)")
_TIDY_CACHE: dict = {}


def _tidy(raw: str):
    """tidytcells is the project's canonicaliser on the encoder side, so it decides here too. Lazy and
    cached: it is an optional dependency and is absent on the cluster, which is fine because the reference
    table is built once where it IS available and then loaded as data."""
    if raw in _TIDY_CACHE:
        return _TIDY_CACHE[raw]
    out = None
    try:
        import tidytcells as tt
        out = tt.tr.standardize(symbol=str(raw), species="homosapiens", suppress_warnings=True)
        if out:
            out = out.split("*")[0]
    except Exception:
        out = None
    _TIDY_CACHE[raw] = out
    return out


def load_name_map(path: str) -> int:
    """Pre-fill the canonicalisation cache from a file, so a machine without tidytcells still gets its
    answers. Canonicalisation needs tidytcells, allele resolution needs the corpus, and the two live on
    different machines -- this is the seam between them."""
    with open(path) as fh:
        _TIDY_CACHE.update(json.load(fh))
    return len(_TIDY_CACHE)


def dump_name_map(raw_names, path: str) -> int:
    """Resolve every spelling once, here, and freeze it."""
    m = {str(r): _tidy(r) for r in sorted(set(map(str, raw_names)))}
    with open(path, "w") as fh:
        json.dump(m, fh, indent=1)
    n_ok = sum(1 for v in m.values() if v)
    print(f"wrote {path}: {n_ok}/{len(m)} spellings resolved")
    return n_ok


def _fallback_name(raw: str) -> str:
    s = str(raw).strip().upper().replace("TCRB", "TRB").split("*")[0]
    m = re.match(r"(TRB[VDJ])(.*)", s)
    if not m:
        return s
    tag, rest = m.groups()
    nums = [_NUM.sub(r"\1", p) for p in rest.split("-") if p]
    if tag == "TRBD":                                    # IMGT has TRBD1 / TRBD2, no member suffix
        nums = nums[:1]
    return tag + "-".join(nums) if nums else tag


def canonical_name(raw: str) -> str:
    """Corpus spelling -> the project's canonical IMGT symbol, allele stripped.

    tidytcells first, because the amino-acid side already canonicalises with it (resolve_cdr12 in
    repertoire/encode_repertoires.py) and the two halves of the project must name a gene identically or the
    join between an event and its clonotype silently drops rows. Measured on the Emerson corpus, a
    hand-written regex rule disagrees with tidytcells on 11.7 percent of rows -- every case being a gene
    IMGT leaves unnumbered (TCRBV28-01 is TRBV28, not TRBV28-1), which no amount of zero-stripping can know.
    """
    t = _tidy(raw)
    return t if t else _fallback_name(raw)


#: What a platform writes when it could not call the gene. These rows belong to the unresolved class, not
#: to a failed lookup, and are reported separately so a naming bug is never hidden among them.
UNCALLED = frozenset({"TRB", "TRBV", "TRBD", "TRBJ", "UNRESOLVED", "NAN", "NONE", ""})


def lookup_symbols(by_symbol: dict, section: str, symbol: str) -> list:
    """Templates compatible with a reported symbol. More than one is a legitimate answer.

    Three cases, tried in order. The symbol as written. Then the bare form, because Adaptive pads a gene
    IMGT leaves unnumbered -- TCRBV28-01 is TRBV28, not a member of a family. Then every member of the
    family, because a family-level call like TCRBV20 means the read could not separate the members: that is
    real ambiguity, not a defect, and returning all of them is what lets the scenario sum weigh them."""
    hits = by_symbol[section].get(symbol, [])
    if not hits and symbol.endswith("-1"):
        hits = by_symbol[section].get(symbol[:-2], [])
    if not hits:
        hits = [f for sym, fulls in by_symbol[section].items() if sym.startswith(symbol + "-")
                for f in fulls]
    return hits


def junction_side(section: str, name: str, templates: dict, anchors: dict) -> str | None:
    """The part of a template that can appear inside the junction."""
    seq = templates[section].get(name)
    if seq is None:
        return None
    if section == "V":
        a = anchors["V"].get(name)
        return None if a is None else seq[a:]
    if section == "J":
        a = anchors["J"].get(name)
        return None if a is None else seq[:a + 3]
    return seq                                           # D lies wholly inside the junction


def _observed_spans(events_dir: str, n_files: int, cap: int = 400_000):
    """Germline-derived spans as the corpus reports them, per gene symbol."""
    from .events import best_gene
    files = sorted(glob.glob(os.path.join(events_dir, "*.parquet")))[:n_files]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df.v_index >= 0) & (df.j_index >= 0) & (df.n1_index >= 0) & (df.n2_index >= 0)].head(cap)
    # The resolved column, not the raw one: `v_gene` is the literal "unresolved" on a fifth of Emerson rows
    # while `v_resolved` carries at least the family.
    # No leading underscore: itertuples renames such columns positionally.
    df = df.assign(bestv=best_gene(df, "v").values, bestj=best_gene(df, "j").values)
    v, j = {}, {}
    for r in df.itertuples():
        s = r.rearrangement
        v.setdefault(canonical_name(r.bestv), []).append(s[r.v_index:r.n1_index])
        j.setdefault(canonical_name(r.bestj), []).append(s[r.j_index:r.v_index + r.cdr3_length])
    return v, j, len(df)


def build_reference(events_dir: str, n_files: int = 8, model_dir: str | None = None) -> dict:
    """Resolve every gene symbol the corpus uses to a template and an allele, choosing the allele by
    agreement with the observed germline spans. Returns the table plus the evidence for each choice."""
    templates, anchors = load_olga_templates(model_dir), load_olga_anchors(model_dir)
    by_symbol = {sec: {} for sec in ("V", "D", "J")}
    for sec, d in templates.items():
        for full in d:
            by_symbol[sec].setdefault(canonical_name(full), []).append(full)

    v_spans, j_spans, n_rows = _observed_spans(events_dir, n_files)
    print(f"observed {n_rows} rearrangements; {len(v_spans)} V symbols, {len(j_spans)} J symbols", flush=True)

    table, unresolved, uncalled = {}, {"V": [], "D": [], "J": []}, {}
    for sec, spans in (("V", v_spans), ("J", j_spans)):
        for sym, obs in sorted(spans.items()):
            if sym.upper() in UNCALLED:
                uncalled[f"{sec}:{sym}"] = len(obs)
                continue
            cands = lookup_symbols(by_symbol, sec, sym)
            if not cands:
                unresolved[sec].append((sym, len(obs), "no template"))
                continue
            scored = []
            for full in cands:
                js = junction_side(sec, full, templates, anchors)
                if not js:
                    continue
                ok = sum(1 for s in obs if s and (js.startswith(s) if sec == "V" else js.endswith(s)))
                scored.append((ok / max(len(obs), 1), full, js))
            if not scored:
                unresolved[sec].append((sym, len(obs), "no anchor"))
                continue
            scored.sort(reverse=True)
            frac, full, js = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            table[f"{sec}:{sym}"] = {"section": sec, "symbol": sym, "allele": full, "junction_side": js,
                                     "n_obs": len(obs), "agreement": round(frac, 4),
                                     "margin": round(frac - runner, 4), "n_alleles": len(scored)}
    for sym, cands in sorted(by_symbol["D"].items()):     # D has no anchor and no span to match against
        full = sorted(cands)[0]
        table[f"D:{sym}"] = {"section": "D", "symbol": sym, "allele": full,
                             "junction_side": templates["D"][full], "n_obs": 0,
                             "agreement": None, "margin": None, "n_alleles": len(cands)}
    return {"table": table, "unresolved": unresolved, "uncalled": uncalled, "n_rows": n_rows}


def report(ref: dict) -> None:
    rows = [r for r in ref["table"].values() if r["agreement"] is not None]
    tot = sum(r["n_obs"] for r in rows)
    weighted = sum(r["agreement"] * r["n_obs"] for r in rows) / max(tot, 1)
    print(f"\nresolved {len(rows)} V/J symbols covering {tot} spans; "
          f"span-weighted agreement {weighted:.4f}")
    poor = sorted((r for r in rows if r["agreement"] < 0.9), key=lambda r: -r["n_obs"])
    print(f"symbols below 0.90 agreement: {len(poor)}")
    for r in poor[:10]:
        print(f"  {r['symbol']:<12} allele {r['allele']:<14} agreement {r['agreement']:.3f} "
              f"margin {r['margin']:.3f} on {r['n_obs']} spans")
    thin = [r for r in rows if r["n_alleles"] > 1 and r["margin"] < 0.02]
    print(f"allele choices decided by a margin under 0.02: {len(thin)}"
          + (f" -> {[r['symbol'] for r in thin[:8]]}" if thin else ""))
    if ref.get("uncalled"):
        n = sum(ref["uncalled"].values())
        print(f"spans whose gene the platform did not call: {n} ({100*n/max(ref['n_rows'],1):.2f}% of rows) "
              f"-> {ref['uncalled']}")
    for sec, items in ref["unresolved"].items():
        if items:
            items.sort(key=lambda t: -t[1])
            print(f"UNRESOLVED {sec}: {len(items)} symbols, worst {items[:6]}")
        else:
            print(f"unresolved {sec}: none")


def save(ref: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(ref, fh, indent=1)
    print("saved", path)


def load_reference(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)["table"]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build the project's germline reference table.")
    p.add_argument("--events", required=True, help="event corpus directory, used to resolve alleles")
    p.add_argument("--out", default="reference_trb.json")
    p.add_argument("--files", type=int, default=8)
    p.add_argument("--olga-dir", default=None)
    p.add_argument("--name-map", default=None,
                   help="frozen spelling -> canonical map, for machines without tidytcells")
    a = p.parse_args()
    if a.name_map and os.path.exists(a.name_map):
        print(f"loaded {load_name_map(a.name_map)} frozen gene spellings")
    ref = build_reference(a.events, n_files=a.files, model_dir=a.olga_dir)
    report(ref)
    save(ref, a.out)


if __name__ == "__main__":
    main()
