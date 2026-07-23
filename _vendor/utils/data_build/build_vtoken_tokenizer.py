"""
build_vtoken_tokenizer.py -- build the custom tokenizer for the V-token joint model: the amino-acid tokenizer
(wukevin/tcr-bert) EXTENDED with one atomic token per V gene. V-gene symbols are standardised through tidytcells
so different spellings of the same gene collapse to one token; unresolvable symbols map to [V_UNK] (the clonotype
is kept, not dropped). TRAV and TRBV symbols are disjoint so alpha and beta get separate tokens with no conflict.

Sources scanned: Emerson beta parquet (v_gene) + paired TSV (v_gene_alpha, v_gene_beta). Saves the extended
tokenizer dir + a raw->token map (JSON) so the data loaders apply the identical standardisation. Run from scripts/.
"""
import argparse
import json
import os
import sys

import pandas as pd
import tidytcells as tt
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/


def collect_raw_vgenes(emerson_path, paired_path):
    raw = set()
    if os.path.exists(emerson_path):
        raw |= set(pd.read_parquet(emerson_path, columns=["v_gene"])["v_gene"].astype(str).unique())
    if os.path.exists(paired_path):
        p = pd.read_csv(paired_path, sep="\t", usecols=["v_gene_beta", "v_gene_alpha"], low_memory=False)
        raw |= set(p["v_gene_beta"].astype(str).unique()) | set(p["v_gene_alpha"].astype(str).unique())
    raw.discard("nan"); raw.discard("")
    return sorted(raw)


def standardize(raw_vgenes):
    """raw v_gene symbol -> '[V:<standard>]' token (or '[V_UNK]'). Returns (map, n_unresolved)."""
    vmap, n_unres = {}, 0
    for v in raw_vgenes:
        try:
            s = tt.tr.standardize(gene=v, species="homosapiens")            # canonical IMGT symbol (or None)
        except Exception:
            s = None
        if s:
            vmap[v] = f"[V:{s}]"
        else:
            vmap[v] = "[V_UNK]"; n_unres += 1
    return vmap, n_unres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emerson", default="../data/processed/emerson_uniform_train.parquet")
    ap.add_argument("--paired", default="../data/processed/pretrain_unique_paired_sequences.tsv")
    ap.add_argument("--out", default="../models/tokenizers/tcr-vtoken")
    ap.add_argument("--base", default="wukevin/tcr-bert")
    args = ap.parse_args()

    raw = collect_raw_vgenes(args.emerson, args.paired)
    print(f"{len(raw)} unique raw V-gene symbols across corpora")
    vmap, n_unres = standardize(raw)
    tokens = sorted(set(vmap.values()) | {"[V_UNK]"})                       # always keep [V_UNK] for unseen genes at inference
    print(f"-> {len(tokens)} V-gene tokens ({len([t for t in tokens if t != '[V_UNK]'])} standardised, "
          f"{n_unres} raw symbols unresolved -> [V_UNK])")
    print("  sample map:", dict(list(vmap.items())[:5]))

    tok = AutoTokenizer.from_pretrained(args.base)
    before = len(tok)
    tok.add_tokens(tokens, special_tokens=True)                             # atomic V tokens, never split
    print(f"tokenizer vocab: {before} -> {len(tok)} (+{len(tok) - before})")

    os.makedirs(args.out, exist_ok=True)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "vgene_map.json"), "w") as f:
        json.dump(vmap, f, indent=0)
    print(f"saved tokenizer + vgene_map.json -> {args.out}")


if __name__ == "__main__":
    main()
