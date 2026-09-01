"""
encoders.py -- ClonotypeEncoder implementations (per-clonotype embeddings).

NeuralEncoder : the maintained per-chain cdr123 foundation model. It mirrors the proven
                benchmark_ram.make_embedder path exactly (so numbers reproduce), just wrapped behind the
                ClonotypeEncoder protocol.
SceptrEncoder : SCEPTR b_sceptr beta-only baseline (dim 64).

Heavy deps (torch/transformers/tidytcells/sceptr) are imported LAZILY at construction, so importing this
module -- or the light core -- costs nothing until you actually build an encoder. Reuses the existing,
unmodified helpers in scripts/ (utils.benchmark_utils, repertoire.encode_repertoires).
"""
from __future__ import annotations

import numpy as np

from . import schema as S


class NeuralEncoder:
    """Per-clonotype embedding from a per-chain joint foundation checkpoint. `encode(df) -> (Z, keep)`."""

    def __init__(self, model_path: str, device=None, chain: str = "beta", batch_size: int = 256):
        import torch  # lazy heavy import
        from utils.benchmark_utils import _get_tokenizer, _load_perchain_joint_model

        self._torch = torch
        self.chain = chain
        self.batch_size = batch_size
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.jcfg = _load_perchain_joint_model(model_path, self.device)
        self.dim = int(self.model.bert.config.hidden_size)
        self.input_format = self.jcfg.get("input_format", "cdr123")
        if self.input_format != "cdr123":
            raise ValueError(f"unsupported checkpoint input format {self.input_format!r}; use the maintained cdr123 model")
        self.tokenizer = _get_tokenizer()

    def encode(self, df):
        from repertoire.encode_repertoires import resolve_cdr12, embed_clonotypes
        c3 = df[S.CDR3].astype(str).tolist()
        # cdr123 needs CDR1/2 from tidytcells; drop rows that do not resolve.
        c12 = df[S.V_GENE].astype(str).map(resolve_cdr12)
        c1 = [c[0] for c in c12]; c2 = [c[1] for c in c12]
        keep = np.array([a is not None and b is not None for a, b in zip(c1, c2)])
        idx = np.where(keep)[0]
        Z = embed_clonotypes(self.model, self.tokenizer, self.jcfg,
                             [c1[i] for i in idx], [c2[i] for i in idx], [c3[i] for i in idx],
                             self.chain, self.device, self.batch_size)
        return Z.astype(np.float32), keep


class SceptrEncoder:
    """SCEPTR b_sceptr beta-only baseline. dim 64. Drops non-TRBV rows (keep mask)."""

    dim = 64

    def __init__(self):
        import sceptr           # lazy
        import tidytcells as tt
        self._m = sceptr.variant.b_sceptr()
        self._tt = tt

    def _std(self, v):
        try:
            s = self._tt.tr.standardize(gene=str(v), species="homosapiens", enforce_functional=True)
            return s if s and str(s).startswith("TRBV") else None
        except Exception:
            return None

    def encode(self, df):
        import pandas as pd
        vs = df[S.V_GENE].astype(str).map(self._std)
        keep = vs.notna().to_numpy()
        sdf = pd.DataFrame({"TRBV": (vs[keep] + "*01").tolist(),
                            "CDR3B": df[S.CDR3].astype(str).to_numpy()[keep], "TRBJ": np.nan})
        Z = np.asarray(self._m.calc_vector_representations(sdf), np.float32)
        return Z, keep
