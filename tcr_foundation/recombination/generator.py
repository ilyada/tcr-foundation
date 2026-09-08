"""
generator.py -- a donor-conditioned autoregressive model of the rearranged junction, in nucleotides.

TWO SEPARATE MODELS
-------------------
    P_pre   fitted on the NON-PRODUCTIVE rearrangements
    P_post  fitted on the PRODUCTIVE ones
    log Q(x | donor) = log P_post(x | L, donor) - log P_pre(x | L, donor)   on productive x

Separate parameter sets share the same technical covariates, so effects those covariates induce divide out of
the difference. A donor representation will be added only after the IGoR cache is available.

WHY THE LENGTH IS DECLARED AND NOT PREDICTED INSIDE log Q
---------------------------------------------------------
An autoregressive model is self-normalised over every string it can emit, and the two corpora do not live
on the same strings: productivity IS the junction length modulo three, so P_pre puts almost no mass where
P_post lives. Their pointwise ratio then carries -log Z_pre, the mass P_pre assigns to the productive
region -- and since P_pre is donor-conditioned, that offset varies BY DONOR, so it does not cancel and it
looks exactly like the donor-specific selection we are trying to measure.

The model therefore emits the length as its own token and the bases are conditioned on it:

    P(x | donor) = P(L | donor, V, J) x P(bases | donor, V, J, L)

At a given L both models normalise over the same set -- all base strings of that length -- so Z_pre = Z_post
= 1 exactly and the offset is gone by construction rather than by approximation. The whole frame lives in
the length factor, which is thus separable and inspectable; log Q simply does not use it. `P(x)` for
anything that compares sequences of DIFFERENT lengths, publicness among them, multiplies the factor back in.

WHAT EACH MODEL EMITS
---------------------
    [BOS]  V gene  J gene  LEN  b1 b2 ... bL

Genes are emitted, not given, so gene usage is part of the likelihood -- and it is where most of the
difference between the two classes sits.

THE GERMLINE IS AN INPUT, NOT SOMETHING TO MEMORISE
---------------------------------------------------
Measured on a trained model: 78 percent of a junction's likelihood is in the ~19 inserted bases, 22 percent
in the choice of genes, and under 1 percent in the ~30 germline-derived bases, which are determined once the
gene is chosen. Making the model re-derive them from its weights spends capacity where there is no
information. Each position therefore carries two extra inputs -- the base the chosen V germline would have
there, and the base the chosen J germline would have, counted from the end -- so the germline stretches are
a copy and the capacity goes to the junction. Indexing from the end is only possible because L is declared,
which is a second reason the length token pays for itself.

CLI
    python -m tcr_foundation.generator --events DIR --reference FILE --out DIR --train
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

BASES = "ACGT"
NO_BASE = 4                                      # "the germline does not reach this position"
BOS, PAD = 4, 5
N_SPECIAL = 6
MAX_JUNCTION = 96
MAX_LEN = MAX_JUNCTION + 8
VALID_DONOR_REPRESENTATIONS = frozenset({"none"})
IGOR_FEATURE_KEY = "igor_features"


class JunctionCorpus:
    """Junctions, donors, class, gene calls, germline templates and sequencing depth, as compact arrays.

    Junctions live in one bytes buffer with an offsets array rather than a list of Python strings: fifteen
    million strings is over a gigabyte of separately refcounted objects and every DataLoader worker would
    copy it on first touch."""

    def __init__(self, events_dir: str, reference: dict, samples_path: str | None = None,
                 donors: int | None = None, rows_per_donor: int | None = None, seed: int = 0):
        from .events import best_gene
        from .germline import canonical_name

        # Germline junction-side templates, keyed by canonical symbol. A corpus symbol may be family-level
        # (Adaptive writes TCRBV20 when the read cannot separate the members) while the reference is keyed
        # by member, so the same fallback the reference lookup uses applies here: exact symbol, then the
        # bare form, then any member of the family. Without it a quarter of V genes arrive with no template
        # and the germline feature is silently empty exactly where it would have helped most.
        from .germline import lookup_symbols
        by_symbol = {}
        for rec in reference.values():
            by_symbol.setdefault(rec["section"], {}).setdefault(rec["symbol"], []).append(rec["allele"])
        allele_seq = {rec["allele"]: rec["junction_side"] for rec in reference.values()}
        self._by_symbol, self._allele_seq = by_symbol, allele_seq

        def template(section, symbol):
            hits = lookup_symbols(by_symbol, section, symbol)
            for h in hits:                                  # deterministic: first by name
                if allele_seq.get(h):
                    return allele_seq[h]
            return ""

        self._template = template
        self.germ = {"V": {}, "J": {}}
        for rec in reference.values():
            if rec["section"] in ("V", "J"):
                self.germ[rec["section"]][rec["symbol"]] = rec["junction_side"]

        files = sorted(glob.glob(os.path.join(events_dir, "*.parquet")))
        rng = np.random.default_rng(seed)
        if donors and donors < len(files):
            files = [files[i] for i in sorted(rng.choice(len(files), donors, replace=False))]

        buf, offs, t0 = bytearray(), [0], time.time()
        d_id, lib_id, is_post, v_raw, j_raw = [], [], [], [], []
        donor_names, library_names, seen_donor, seen_library, library_depth = [], [], {}, {}, []
        for i, f in enumerate(files, 1):
            import pyarrow.parquet as pq
            base_columns = ["donor", "frame_type", "rearrangement", "v_index", "cdr3_length", "v_gene",
                            "v_resolved", "j_gene", "j_resolved"]
            names = set(pq.ParquetFile(f).schema.names)
            extra_columns = [c for c in ("library", "timepoint") if c in names]
            df = pd.read_parquet(f, columns=base_columns + extra_columns)
            if "library" not in df:
                df["library"] = df["donor"].astype(str)
            if "timepoint" not in df:
                df["timepoint"] = ""
            depth_by_library = df["library"].astype(str).value_counts().to_dict()
            ft = df["frame_type"].astype(str)
            df = df[ft.isin(("in", "out")) & (df.v_index >= 0) & (df.cdr3_length > 0)
                    & (df.cdr3_length <= MAX_JUNCTION)]
            if rows_per_donor and len(df) > rows_per_donor:
                df = df.iloc[rng.choice(len(df), rows_per_donor, replace=False)]
            if not len(df):
                continue
            df = df.assign(bv=best_gene(df, "v").values, bj=best_gene(df, "j").values)
            if df["donor"].astype(str).nunique() != 1:
                raise ValueError(f"{f}: one event file must represent one donor")
            if df["library"].astype(str).nunique() != 1:
                raise ValueError(f"{f}: one event file must represent one library")
            d = str(df["donor"].iloc[0])
            if d not in seen_donor:
                seen_donor[d] = len(donor_names)
                donor_names.append(d)
            libraries = df["library"].astype(str).tolist()
            for library in dict.fromkeys(libraries):
                if library not in seen_library:
                    seen_library[library] = len(library_names)
                    library_names.append(library)
                    library_depth.append(depth_by_library[library])
            for r in df.itertuples():
                buf.extend(r.rearrangement[r.v_index:r.v_index + r.cdr3_length].encode())
                offs.append(len(buf))
            d_id.extend([seen_donor[d]] * len(df))
            lib_id.extend([seen_library[str(x)] for x in df["library"].astype(str)])
            is_post.extend((df["frame_type"].astype(str) == "in").tolist())
            v_raw.extend(df["bv"].map(canonical_name).tolist())
            j_raw.extend(df["bj"].map(canonical_name).tolist())
            if i % 100 == 0:
                print(f"  {i}/{len(files)} donors, {len(offs)-1} junctions, {time.time()-t0:.0f}s",
                      flush=True)

        self.v_vocab = sorted(set(v_raw))
        self.j_vocab = sorted(set(j_raw))
        vi = {g: k for k, g in enumerate(self.v_vocab)}
        ji = {g: k for k, g in enumerate(self.j_vocab)}
        self.buf = bytes(buf)
        self.off = np.asarray(offs, np.int64)
        self.donor = np.asarray(d_id, np.int32)
        self.library = np.asarray(lib_id, np.int32)
        self.is_post = np.asarray(is_post, bool)
        self.v = np.asarray([vi[g] for g in v_raw], np.int16)
        self.j = np.asarray([ji[g] for g in j_raw], np.int16)
        self.donor_names = donor_names

        # Depth is a library-level technical covariate: given to BOTH densities, it divides out of their
        # difference while remaining separate from the biological donor identity. Bucketed by log decile.
        self.library_names = library_names
        self.library_log_depth = np.log1p(np.asarray(library_depth, np.float64))
        self.depth_bucket = None
        self.n_depth = None

        # Token ids: bases, BOS, PAD, then one token per junction length, then per V gene, then per J gene.
        self.len_off = N_SPECIAL
        self.v_off = self.len_off + MAX_JUNCTION + 1
        self.j_off = self.v_off + len(self.v_vocab)
        self.vocab_size = self.j_off + len(self.j_vocab)

        # Germline templates as base codes, padded, so a position lookup is an array index.
        self.gv = np.full((len(self.v_vocab), MAX_JUNCTION), NO_BASE, np.int8)
        for k, g in enumerate(self.v_vocab):
            s = self._template("V", g)[:MAX_JUNCTION]
            for p, c in enumerate(s):
                self.gv[k, p] = BASES.find(c) if c in BASES else NO_BASE
        self.gj = np.full((len(self.j_vocab), MAX_JUNCTION), NO_BASE, np.int8)
        for k, g in enumerate(self.j_vocab):
            s = self._template("J", g)[-MAX_JUNCTION:]
            for p, c in enumerate(reversed(s)):            # indexed from the END of the junction
                self.gj[k, p] = BASES.find(c) if c in BASES else NO_BASE

        if samples_path and os.path.exists(samples_path):
            meta = pd.read_parquet(samples_path)
            if not {"donor", "cohort"}.issubset(meta.columns):
                raise ValueError("samples metadata must contain donor and cohort columns")
            cohort_counts = meta.assign(donor=meta["donor"].astype(str), cohort=meta["cohort"].astype(str)).groupby("donor")["cohort"].nunique()
            conflicted = cohort_counts[cohort_counts > 1].index.tolist()
            if conflicted:
                raise ValueError(f"samples metadata assigns multiple cohorts to donor(s) {conflicted[:5]}")
            self.cohort = dict(zip(meta["donor"].astype(str), meta["cohort"].astype(str)))
        else:
            self.cohort = {}
        miss_v = sum(1 for g in self.v_vocab if not self._template("V", g))
        miss_j = sum(1 for g in self.j_vocab if not self._template("J", g))
        print(f"{len(self.donor)} junctions, {len(donor_names)} donors, V {len(self.v_vocab)} "
              f"({miss_v} without a template), J {len(self.j_vocab)} ({miss_j} without), "
              f"vocab {self.vocab_size}, libraries {len(self.library_names)}, {time.time()-t0:.0f}s", flush=True)

    def __len__(self):
        return len(self.donor)

    def fit_depth_buckets(self, train_rows: np.ndarray) -> None:
        """Fit library-depth bins on training donors only, then transform every library with that fixed frame."""
        train_libraries = np.unique(self.library[train_rows])
        if not len(train_libraries):
            raise ValueError("cannot fit depth buckets without training libraries")
        train_depth = self.library_log_depth[train_libraries]
        edges = np.quantile(train_depth, np.linspace(0, 1, 11)[1:-1]) if len(train_depth) > 10 else np.array([])
        self.depth_bucket = np.digitize(self.library_log_depth, edges).astype(np.int16)
        self.n_depth = int(self.depth_bucket.max()) + 1

    def junction(self, i: int) -> str:
        return self.buf[self.off[i]:self.off[i + 1]].decode()

    def item(self, i: int) -> dict:
        if self.depth_bucket is None:
            raise RuntimeError("fit_depth_buckets(train_rows) must be called before creating generator batches")
        b = np.frombuffer(self.buf[self.off[i]:self.off[i + 1]], np.uint8)
        base = np.zeros(len(b), np.int64)
        for k, c in enumerate(BASES):
            base[b == ord(c)] = k
        L = len(base)
        tok = np.concatenate([[self.v_off + self.v[i], self.j_off + self.j[i], self.len_off + L], base])
        # Germline the chosen genes would put at each junction position: V from the left, J from the right.
        gv = np.full(L, NO_BASE, np.int8)
        gj = np.full(L, NO_BASE, np.int8)
        m = min(L, MAX_JUNCTION)
        gv[:m] = self.gv[self.v[i], :m]
        gj[:m] = self.gj[self.j[i], :m][::-1]
        return {"tok": tok, "gv": gv, "gj": gj, "L": L,
                "depth": int(self.depth_bucket[self.library[i]])}


class TokenDataset:
    def __init__(self, corpus: JunctionCorpus, rows: np.ndarray):
        self.c, self.rows = corpus, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k: int):
        return self.c.item(int(self.rows[k]))


def collate(items):
    """Pad to the longest item. The germline features are aligned to the INPUT position that predicts each
    base: once V, J and L are emitted, the germline at every junction position is known, so the feature is
    available exactly where the model needs it."""
    import torch
    L = max(len(it["tok"]) for it in items) + 1                # +1 for the leading BOS
    n = len(items)
    tok = np.full((n, L), PAD, np.int64)
    gv = np.full((n, L), NO_BASE, np.int64)
    gj = np.full((n, L), NO_BASE, np.int64)
    for k, it in enumerate(items):
        t = it["tok"]
        tok[k, 0] = BOS
        tok[k, 1:1 + len(t)] = t
        # tok index 3 holds LEN; base i sits at index 4+i, and is PREDICTED from index 3+i.
        gv[k, 3:3 + it["L"]] = it["gv"]
        gj[k, 3:3 + it["L"]] = it["gj"]
    return {"tok": torch.from_numpy(tok), "gv": torch.from_numpy(gv), "gj": torch.from_numpy(gj),
            "depth": torch.tensor([it["depth"] for it in items], dtype=torch.long)}


class EpisodicSampler:
    """Donor-major batches. Uniform over donors and then over their junctions, never uniform over rows:
    depth runs from thousands to hundreds of thousands per donor, and sampling rows uniformly would let
    sequencing depth set the effective donor weights."""

    def __init__(self, donor_ids, donors_per_batch=8, targets_per_donor=32, steps=1000, seed=0):
        self.by_donor = {d: np.nonzero(donor_ids == d)[0] for d in np.unique(donor_ids)}
        self.donors = np.array(sorted(self.by_donor))
        self.dpb, self.tpd, self.steps = donors_per_batch, targets_per_donor, steps
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            idx = []
            for d in self.rng.choice(self.donors, min(self.dpb, len(self.donors)), replace=False):
                rows = self.by_donor[d]
                idx.extend(self.rng.choice(rows, min(self.tpd, len(rows)), replace=False).tolist())
            yield idx


def build_model(corpus: JunctionCorpus, donor_representation: str = "none", n_layer: int = 6,
                n_head: int = 4, n_embd: int = 256):
    if donor_representation not in VALID_DONOR_REPRESENTATIONS:
        raise ValueError(f"unknown donor representation {donor_representation!r}; expected one of {sorted(VALID_DONOR_REPRESENTATIONS)}")
    import torch
    import torch.nn as nn
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=corpus.vocab_size, n_positions=MAX_LEN, n_embd=n_embd, n_layer=n_layer,
                     n_head=n_head, n_inner=4 * n_embd, bos_token_id=BOS)
    len_lo, len_hi = corpus.len_off, corpus.len_off + MAX_JUNCTION + 1

    class JunctionGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt = GPT2LMHeadModel(cfg)
            self.donor_representation = donor_representation
            self.germ_v = nn.Embedding(NO_BASE + 1, n_embd)
            self.germ_j = nn.Embedding(NO_BASE + 1, n_embd)
            self.depth_emb = nn.Embedding(corpus.n_depth, n_embd)

        def representation_bias(self, batch):
            """Placeholder method for a future IGoR vector projected into token-embedding space.

            The present baseline intentionally has no donor input. When the versioned IGoR cache is added,
            collate will provide batch[IGOR_FEATURE_KEY] with one fitted vector per event and this method will
            validate its shape, apply the IGoR conditioner, and return a [batch, hidden] bias."""
            if IGOR_FEATURE_KEY in batch:
                raise RuntimeError("IGoR features are not integrated; add the versioned IGoR cache and conditioner first")
            return None

        def token_logprobs(self, batch):
            """log P of every emitted token, [B, T-1], aligned with the token it scores."""
            tok = batch["tok"]
            emb = (self.gpt.transformer.wte(tok) + self.germ_v(batch["gv"]) + self.germ_j(batch["gj"])
                   + self.depth_emb(batch["depth"])[:, None, :])
            z = self.representation_bias(batch)
            if z is not None:
                emb = emb + z[:, None, :]
            out = self.gpt(inputs_embeds=emb).logits[:, :-1]
            tgt = tok[:, 1:]
            lp = torch.log_softmax(out, -1).gather(2, tgt[:, :, None])[..., 0]
            return lp * (tgt != PAD), tgt

        def forward(self, batch, given_length: bool = False):
            """log P(x) summed over positions.

            given_length drops the length token's own term, leaving log P(x | L). That is the form
            log Q must use: at a fixed L the two models normalise over the same set of strings, so their
            difference carries no leftover from where each of them put its total mass."""
            lp, tgt = self.token_logprobs(batch)
            if given_length:
                lp = lp * ~((tgt >= len_lo) & (tgt < len_hi))
            return lp.sum(1)

    return JunctionGPT()


# -- splits, training, scoring -------------------------------------------------------------------------

def split_donors(corpus: JunctionCorpus, holdout_cohort="Cohort 02", dev_donors=65, seed=0):
    """Train / dev-donor / holdout-cohort. Dev donors come out of the training cohort: early stopping needs
    a donor-level signal, and stopping on the holdout would burn the number we report."""
    rng = np.random.default_rng(seed)
    coh = np.array([corpus.cohort.get(d, "") for d in corpus.donor_names])
    if not np.all(coh):
        missing = [d for d, c in zip(corpus.donor_names, coh) if not c][:5]
        raise ValueError(f"missing cohort metadata for donor(s) {missing}; a donor-held-out split requires samples metadata")
    held = np.nonzero(coh == holdout_cohort)[0]
    pool = np.nonzero(coh != holdout_cohort)[0]
    if not len(held):
        raise ValueError(f"no donors belong to holdout cohort {holdout_cohort!r}")
    if len(pool) < 2:
        raise ValueError("need at least two non-holdout donors for train/dev splitting")
    dev = rng.choice(pool, min(dev_donors, max(1, len(pool) // 4)), replace=False)
    train = np.setdiff1d(pool, dev)
    r = corpus.donor
    print(f"donors: train {len(train)}, dev {len(dev)}, holdout {len(held)}", flush=True)
    return (np.nonzero(np.isin(r, train))[0], np.nonzero(np.isin(r, dev))[0],
            np.nonzero(np.isin(r, held))[0])


def evaluate(model, corpus, rows, device, batch=256, max_batches=40,
             given_length=False):
    """Mean log P per junction, and bits per base -- the readable form: 2.0 bits is a uniform guess among
    four bases, 0.0 is certainty."""
    import torch
    ds = TokenDataset(corpus, rows)
    model.eval()
    tot, tot_b, n, nb = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            if max_batches and s // batch >= max_batches:
                break
            b = collate([ds[i] for i in range(s, min(s + batch, len(rows)))])
            b = {k: v.to(device) for k, v in b.items()}
            lp, tgt = model.token_logprobs(b)
            if given_length:
                lo = corpus.len_off
                lp = lp * ~((tgt >= lo) & (tgt < lo + MAX_JUNCTION + 1))
            tot += float(lp.sum())
            # bits per base is over BASE tokens only -- gene and length tokens are not bases and would
            # otherwise push the figure above the 2.0 ceiling a four-letter alphabet has.
            base = tgt < 4
            tot_b += float((lp * base).sum())
            nb += int(base.sum())
            n += b["tok"].shape[0]
    model.train()
    return {"nats_per_seq": tot / max(n, 1), "bits_per_base": -tot_b / max(nb, 1) / np.log(2)}


def train_one(corpus, rows_train, rows_dev, donor_representation="none", device="cuda", lr=3e-4, warmup=1000,
               max_steps=200_000, eval_every=1000, patience=3, max_reductions=3, donors_per_batch=8,
               targets_per_donor=32, workers=8, out_dir=".", exp=None, seed=0, tag="pre", rows_hold=None):
    """Train to convergence: warmup, then plateau scheduling with early stopping, keeping the best
    checkpoint by dev-donor likelihood rather than the last one."""
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(seed)

    ds = TokenDataset(corpus, rows_train)
    sampler = EpisodicSampler(corpus.donor[rows_train], donors_per_batch, targets_per_donor,
                              steps=max_steps, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers else None,
                        pin_memory=True)
    model = build_model(corpus, donor_representation=donor_representation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"[{tag}/{donor_representation}] params={sum(p.numel() for p in model.parameters())}", flush=True)

    best, best_step, best_state, since, reductions, step, t0 = -1e30, 0, None, 0, 0, 0, time.time()
    for b in loader:
        step += 1
        if step <= warmup:
            for gp in opt.param_groups:
                gp["lr"] = lr * step / warmup
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        loss = -model(b).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if exp is not None and step % 100 == 0:
            exp.log_metric(f"{tag}_{donor_representation}/train_nats", -float(loss.detach()), step=step)
        if step % eval_every == 0:
            m = evaluate(model, corpus, rows_dev, device)
            dev = m["nats_per_seq"]
            good = dev > best + 1e-4
            print(f"[{tag}/{donor_representation}] step {step}: train {-float(loss.detach()):.2f}  "
                  f"dev {dev:.2f} nats  {m['bits_per_base']:.3f} bits/base  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {'*' if good else ''}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if exp is not None:
                exp.log_metric(f"{tag}_{donor_representation}/dev_nats", dev, step=step)
                exp.log_metric(f"{tag}_{donor_representation}/dev_bits_per_base", m["bits_per_base"], step=step)
            if good:
                best, best_step, since = dev, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state": model.state_dict(), "donor_representation": donor_representation, "class": tag, "step": step,
                            "dev": dev, "bits_per_base": m["bits_per_base"],
                            "v_vocab": corpus.v_vocab, "j_vocab": corpus.j_vocab,
                            "donors": corpus.donor_names}, os.path.join(out_dir, f"{tag}_{donor_representation}_best.pt"))
            else:
                since += 1
                if since >= patience:
                    reductions, since = reductions + 1, 0
                    for gp in opt.param_groups:
                        gp["lr"] *= 0.5
                    print(f"[{tag}/{donor_representation}] plateau -> lr {opt.param_groups[0]['lr']:.2e} "
                          f"({reductions}/{max_reductions})", flush=True)
                    if reductions > max_reductions:
                        print(f"[{tag}/{donor_representation}] early stop at {step}; best {best:.2f} at {best_step}",
                              flush=True)
                        break
    if best_state is None:
        best = evaluate(model, corpus, rows_dev, device)["nats_per_seq"]
        best_step = step
    else:
        model.load_state_dict(best_state)
    held = evaluate(model, corpus, rows_hold, device) if rows_hold is not None else {}
    return {"class": tag, "donor_representation": donor_representation, "best_dev_nats": best, "best_step": best_step, "steps": step,
            "holdout_nats": held.get("nats_per_seq"), "holdout_bits_per_base": held.get("bits_per_base")}


def log_q(model_pre, model_post, corpus, rows, device, batch=256):
    """log Q per rearrangement, AT FIXED LENGTH: how many times more likely this receptor is in the
    selected repertoire than in this donor's own background. The length term is dropped on both sides, so
    the two densities are normalised over the same set and nothing survives of where each put its mass."""
    import torch
    ds = TokenDataset(corpus, rows)
    model_pre.eval()
    model_post.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            b = collate([ds[i] for i in range(s, min(s + batch, len(rows)))])
            b = {k: v.to(device) for k, v in b.items()}
            out.append((model_post(b, given_length=True) - model_pre(b, given_length=True)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def generate(model, corpus, n=1000, device="cuda", temperature=1.0, depth_bucket=None,
             library_id=None):
    """Ancestral sampling: V, then J, then the length, then the bases. The germline features become
    available exactly when the genes and the length have been drawn -- which is why they can be an input."""
    import torch
    model.eval()
    if depth_bucket is None:
        if library_id is None:
            raise ValueError("library_id is required when depth_bucket is not supplied")
        dep = corpus.depth_bucket[library_id]
    else:
        dep = depth_bucket
    depth = torch.full((n,), int(dep), dtype=torch.long, device=device)
    tok = torch.full((n, 1), BOS, dtype=torch.long, device=device)
    gv = torch.full((n, 1), NO_BASE, dtype=torch.long, device=device)
    gj = torch.full((n, 1), NO_BASE, dtype=torch.long, device=device)
    pad_col = torch.full((n, 1), NO_BASE, dtype=torch.long, device=device)

    def step(lo, hi):
        emb = (model.gpt.transformer.wte(tok) + model.germ_v(gv) + model.germ_j(gj)
               + model.depth_emb(depth)[:, None, :])
        z = model.representation_bias({})
        if z is not None:
            emb = emb + z[:, None, :]
        lg = model.gpt(inputs_embeds=emb).logits[:, -1] / temperature
        m = torch.full_like(lg, -1e30)
        m[:, lo:hi] = 0
        return torch.multinomial(torch.softmax(lg + m, -1), 1)

    with torch.no_grad():
        nxt = step(corpus.v_off, corpus.j_off)
        v_id = nxt[:, 0] - corpus.v_off
        tok, gv, gj = torch.cat([tok, nxt], 1), torch.cat([gv, pad_col], 1), torch.cat([gj, pad_col], 1)

        nxt = step(corpus.j_off, corpus.vocab_size)
        j_id = nxt[:, 0] - corpus.j_off
        tok, gv, gj = torch.cat([tok, nxt], 1), torch.cat([gv, pad_col], 1), torch.cat([gj, pad_col], 1)

        nxt = step(corpus.len_off + 1, corpus.len_off + MAX_JUNCTION + 1)
        lengths = nxt[:, 0] - corpus.len_off
        tok = torch.cat([tok, nxt], 1)

        GV = torch.as_tensor(corpus.gv.astype(np.int64), device=device)[v_id]
        GJ = torch.as_tensor(corpus.gj.astype(np.int64), device=device)[j_id]
        Lmax = int(lengths.max())
        pos = torch.arange(Lmax, device=device)[None, :]
        gv_full = GV[:, :Lmax]
        gj_full = torch.gather(GJ, 1, (lengths[:, None] - 1 - pos).clamp(0, MAX_JUNCTION - 1))
        gj_full = torch.where(pos < lengths[:, None], gj_full, torch.full_like(gj_full, NO_BASE))

        for p in range(Lmax):
            gv = torch.cat([gv, gv_full[:, p:p + 1]], 1)
            gj = torch.cat([gj, gj_full[:, p:p + 1]], 1)
            nxt = step(0, 4)
            nxt[lengths <= p] = PAD
            tok = torch.cat([tok, nxt], 1)

    out = []
    for r, L in zip(tok.cpu().numpy(), lengths.cpu().numpy()):
        out.append((corpus.v_vocab[r[1] - corpus.v_off], corpus.j_vocab[r[2] - corpus.j_off],
                    "".join(BASES[c] for c in r[4:4 + int(L)] if c < 4)))
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="Nucleotide generator prepared for IGoR donor representations: P_pre and P_post.")
    p.add_argument("--events", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--samples", default=None)
    p.add_argument("--out", default=".")
    p.add_argument("--donors", type=int, default=None)
    p.add_argument("--rows-per-donor", type=int, default=None)
    p.add_argument("--train", action="store_true")
    p.add_argument("--donor-representations", default="none", help="comma-separated donor representations; only none is available before IGoR integration")
    p.add_argument("--classes", default="pre,post")
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-donors", type=int, default=8)
    p.add_argument("--batch-targets", type=int, default=32)
    p.add_argument("--dev-donors", type=int, default=65)
    p.add_argument("--comet-project", default="tcr-recombination")
    p.add_argument("--no-comet", action="store_true")
    a = p.parse_args()
    donor_representations = [value.strip() for value in a.donor_representations.split(",") if value.strip()]
    classes = [cls.strip() for cls in a.classes.split(",") if cls.strip()]
    unknown_representations = sorted(set(donor_representations) - VALID_DONOR_REPRESENTATIONS)
    if unknown_representations:
        p.error(f"unknown --donor-representations value(s) {unknown_representations}; choose from {sorted(VALID_DONOR_REPRESENTATIONS)}")
    if set(classes) - {"pre", "post"} or not classes:
        p.error("--classes must contain only pre and/or post")

    exp = None
    if a.train and not a.no_comet:
        try:
            import comet_ml                                   # before torch, project convention
            exp = comet_ml.Experiment(project_name=a.comet_project)
            exp.set_name("TCR-Recombination | nucleotide AR | Emerson TRB | " + a.donor_representations)
            exp.log_parameters(vars(a))
        except Exception as e:
            print(f"comet disabled: {e}", flush=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from .germline import load_reference
    corpus = JunctionCorpus(a.events, load_reference(a.reference), a.samples, donors=a.donors,
                            rows_per_donor=a.rows_per_donor)
    tr, dv, hold = split_donors(corpus, dev_donors=a.dev_donors)
    corpus.fit_depth_buckets(tr)
    print(f"junctions: train {len(tr)}, dev {len(dv)}, holdout {len(hold)}", flush=True)
    if not a.train:
        return
    os.makedirs(a.out, exist_ok=True)

    results = []
    for cls in classes:
        sel = corpus.is_post == (cls == "post")
        rt = np.intersect1d(tr, np.nonzero(sel)[0])
        rd = np.intersect1d(dv, np.nonzero(sel)[0])
        rh = np.intersect1d(hold, np.nonzero(sel)[0])
        if not len(rt) or not len(rd) or not len(rh):
            raise ValueError(f"{cls}: empty train, dev, or holdout partition ({len(rt)}, {len(rd)}, {len(rh)})")
        print(f"\n=== {cls}: train {len(rt)}, dev {len(rd)}, holdout {len(rh)} ===", flush=True)
        for donor_representation in donor_representations:
            r = train_one(corpus, rt, rd, donor_representation=donor_representation, device=device, lr=a.lr, max_steps=a.max_steps,
                          eval_every=a.eval_every, donors_per_batch=a.batch_donors,
                          targets_per_donor=a.batch_targets, workers=a.workers, out_dir=a.out,
                          exp=exp, tag=cls, rows_hold=rh)
            results.append(r)
    pd.DataFrame(results).to_csv(os.path.join(a.out, "generator_results.csv"), index=False)
    if exp is not None:
        exp.end()


if __name__ == "__main__":
    main()
