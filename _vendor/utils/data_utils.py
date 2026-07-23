"""
Data loading and dataset classes for TCR-BERT training.
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np


# ================= CONSTANTS =================

# Property columns in v4 data
PROPERTY_COLUMNS = [
    "cdr3Length", "ndnSize", "insertSize",
    "hydropathy_avg", "charge_avg", "polarity_avg", "strength_avg"
]

# Complete human TRBV gene vocabulary (functional + ORF + pseudogenes that may appear in data)
ALL_TRBV_GENES = [
    "TRBV1", "TRBV2",
    "TRBV3-1", "TRBV3-2",
    "TRBV4-1", "TRBV4-2", "TRBV4-3",
    "TRBV5-1", "TRBV5-2", "TRBV5-3", "TRBV5-4", "TRBV5-5", "TRBV5-6", "TRBV5-7", "TRBV5-8",
    "TRBV6-1", "TRBV6-2", "TRBV6-3", "TRBV6-4", "TRBV6-5", "TRBV6-6", "TRBV6-7", "TRBV6-8", "TRBV6-9",
    "TRBV7-1", "TRBV7-2", "TRBV7-3", "TRBV7-4", "TRBV7-5", "TRBV7-6", "TRBV7-7", "TRBV7-8", "TRBV7-9",
    "TRBV8-1", "TRBV8-2",
    "TRBV9",
    "TRBV10-1", "TRBV10-2", "TRBV10-3",
    "TRBV11-1", "TRBV11-2", "TRBV11-3",
    "TRBV12-1", "TRBV12-2", "TRBV12-3", "TRBV12-4", "TRBV12-5",
    "TRBV13", "TRBV14", "TRBV15", "TRBV16", "TRBV17", "TRBV18", "TRBV19",
    "TRBV20-1", "TRBV21-1", "TRBV22-1", "TRBV23-1", "TRBV24-1", "TRBV25-1",
    "TRBV26", "TRBV27", "TRBV28", "TRBV29-1", "TRBV30",
    "UNKNOWN",
]

# Gene name → integer ID mapping (fixed across all runs)
TRBV_GENE_TO_ID = {gene: i for i, gene in enumerate(ALL_TRBV_GENES)}
NUM_TRBV_GENES = len(ALL_TRBV_GENES)

# Pseudogene V genes to exclude (stop codons in germline CDR sequences)
PSEUDOGENE_V_GENES = {"TRBV12-2"}

# Emerson zero-padded V gene → IMGT plain format (e.g. TRBV10-01 → TRBV10-1, TRBV2-01 → TRBV2)
# Built from ALL_TRBV_GENES so it stays consistent with the training vocabulary.
_EMERSON_TO_IMGT: dict = {}
for _g in ALL_TRBV_GENES:
    if _g == "UNKNOWN":
        continue
    if "-" in _g:
        _base, _sub = _g.rsplit("-", 1)
        _EMERSON_TO_IMGT[f"{_base}-{int(_sub):02d}"] = _g
    else:
        _EMERSON_TO_IMGT[f"{_g}-01"] = _g

# Fixed-maximum normalization constants (from VDJtools Annotate output ranges)
PROPERTY_NORMS = {
    "cdr3Length": 100.0,      # VDJtools reports nucleotides; max ~99 nt in data
    "ndnSize": 70.0,          # N/D/N region in nucleotides; max ~65 in data
    "insertSize": 60.0,       # Insert size in nucleotides; max ~57 in data
    "hydropathy_avg": 4.5,    # Kyte-Doolittle scale: [-4.5, 4.5]
    "charge_avg": 1.0,        # Per-residue average charge
    "polarity_avg": 1.0,      # Per-residue average polarity (already 0-1 from VDJtools)
    "strength_avg": 1.0,      # Per-residue average strength (already 0-1 from VDJtools)
}


# ================= DATA LOADING FUNCTIONS =================

def load_and_filter_sequences(file_path: str) -> dict:
    """
    Load TCR data from TSV for MLM pretraining.

    Removes sequences appearing in both CD4 and CD8 populations.

    Args:
        file_path: Path to preprocessed TSV file

    Returns:
        Dict with keys 'cdr1' (if available), 'cdr2', 'cdr3' containing sequence lists
    """
    df = pd.read_csv(file_path, sep="\t")
    has_cdr1 = "aaSeqCDR1OfGermline" in df.columns

    # Split by population
    cd4 = df[df["Cell_Type"] == "CD4"].copy()
    cd8 = df[df["Cell_Type"] == "CD8"].copy()

    # Create tuples for intersection (include CDR1 if available)
    if has_cdr1:
        cd4_tuples = list(zip(cd4["aaSeqCDR3"], cd4["aaSeqCDR2OfGermline"], cd4["aaSeqCDR1OfGermline"]))
        cd8_tuples = list(zip(cd8["aaSeqCDR3"], cd8["aaSeqCDR2OfGermline"], cd8["aaSeqCDR1OfGermline"]))
    else:
        cd4_tuples = list(zip(cd4["aaSeqCDR3"], cd4["aaSeqCDR2OfGermline"]))
        cd8_tuples = list(zip(cd8["aaSeqCDR3"], cd8["aaSeqCDR2OfGermline"]))

    # Intersection across populations
    inter = set(cd4_tuples).intersection(set(cd8_tuples))

    # Keep only unique sequences per population
    cd4["tuple"] = cd4_tuples
    cd4 = cd4[~cd4["tuple"].isin(inter)]

    cd8["tuple"] = cd8_tuples
    cd8 = cd8[~cd8["tuple"].isin(inter)]

    # Combine back
    df_filtered = pd.concat([cd4, cd8], ignore_index=True)

    result = {
        "cdr2": df_filtered["aaSeqCDR2OfGermline"].tolist(),
        "cdr3": df_filtered["aaSeqCDR3"].tolist()
    }

    if has_cdr1:
        result["cdr1"] = df_filtered["aaSeqCDR1OfGermline"].tolist()

    # Include physicochemical properties if available (v4 data)
    available_props = [c for c in PROPERTY_COLUMNS if c in df_filtered.columns]
    if available_props:
        props = df_filtered[available_props].values.astype(np.float32)
        # Apply fixed-maximum normalization
        for i, col in enumerate(available_props):
            norm = PROPERTY_NORMS.get(col, 1.0)
            props[:, i] = props[:, i] / norm
        result["properties"] = props
        result["property_names"] = available_props

    # Include V gene if available (v4 data)
    if "v_gene" in df_filtered.columns:
        result["v_gene"] = df_filtered["v_gene"].tolist()

    return result


def leave_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate CDR3 sequences and sequences appearing in both CD4 and CD8.

    Args:
        df: DataFrame with columns: cdr3aa, Cell_population

    Returns:
        Filtered DataFrame with unique sequences per population
    """
    cd4 = df[df["Cell_population"] == "CD4"].drop_duplicates("cdr3aa")
    cd8 = df[df["Cell_population"] == "CD8"].drop_duplicates("cdr3aa")
    inter = set(cd4.cdr3aa) & set(cd8.cdr3aa)
    return pd.concat([
        cd4[~cd4.cdr3aa.isin(inter)],
        cd8[~cd8.cdr3aa.isin(inter)]
    ])


def load_split(base: str, name: str) -> Tuple[List[str], np.ndarray]:
    """
    Load classification split for fine-tuning.

    Args:
        base: Base directory containing split TSV files
        name: Split name ("hd", "t1d", or "ms")

    Returns:
        Tuple of (sequences, labels) where labels are binary (CD8=1, CD4=0)
    """
    meta = pd.read_csv(f"{base}/meta_{name}_tcrbert.tsv", sep="\t")
    meta = leave_unique(meta)
    meta = meta[(meta.cdr3aa.str.startswith("C")) & (meta.cdr3aa.str.len() >= 8)]
    y = (meta.Cell_population == "CD8").astype(int).to_numpy()
    return meta.cdr3aa.tolist(), y


def load_split_by_condition(
    data_path: str,
    condition,  # str or list[str]
) -> Tuple[dict, np.ndarray]:
    """
    Load classification split from preprocessed training data by Cohort.

    Supports both the legacy v4 TSV format (aaSeqCDR1OfGermline / aaSeqCDR3 columns)
    and the unified parquet format (cdr1 / cdr3 columns).

    Args:
        data_path: Path to training_data_all_cohorts_v4.tsv or training_data_unified.parquet
        condition: Cohort name or list of cohort names ("HD", "T1D", "MS", "VDJdb")

    Returns:
        Tuple of (sequences_dict, labels)
        sequences_dict has keys 'cdr1', 'cdr2', 'cdr3', and 'v_gene' when available.
        labels are binary (CD8=1, CD4=0)
    """
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, sep="\t")
        # Normalize legacy column names to short form
        df = df.rename(columns={
            "aaSeqCDR1OfGermline": "cdr1",
            "aaSeqCDR2OfGermline": "cdr2",
            "aaSeqCDR3":           "cdr3",
        })

    conditions = [condition] if isinstance(condition, str) else list(condition)
    cohort_col = "cohort" if "cohort" in df.columns else "Cohort"
    df = df[df[cohort_col].isin(conditions)].copy()

    cdr3_col = "cdr3aa" if "cdr3aa" in df.columns else "cdr3"
    sequences = {
        "cdr2": df["cdr2"].tolist(),
        "cdr3": df[cdr3_col].tolist(),
    }
    if "cdr1" in df.columns:
        sequences["cdr1"] = df["cdr1"].tolist()

    cell_type_col = "cell_type" if "cell_type" in df.columns else "Cell_Type"
    labels = (df[cell_type_col] == "CD8").astype(int).to_numpy()

    return sequences, labels


def load_split_with_patient_frac(
    data_path: str,
    cohort: str,
    frac: float,
    seed: int = 42,
    complement: bool = False,
) -> Tuple[dict, np.ndarray]:
    """
    Load a patient-level fraction of a single cohort from the unified parquet.

    Used to realise the MS 80/20 patient-aware split (and analogous splits) without
    sequence-level leakage: a patient's sequences live entirely in train OR val, never
    split across both.

    Args:
        data_path: Path to a parquet with columns cohort, patient, cdr1, cdr2, cdr3aa,
                   cell_type (and optionally v_gene).
        cohort:    Cohort name to pull from (e.g. "MS").
        frac:      Fraction of unique patients to keep (e.g. 0.8 for train side).
        seed:      RNG seed for patient sampling. Same seed across the two sides of a
                   split (train/val) guarantees disjoint patient sets when paired with
                   complement=True on the other side.
        complement: If True, return the OTHER 1-frac of patients instead. Used to build
                   the val side from the same seed: call once with complement=False (train)
                   and once with complement=True (val).

    Returns:
        Tuple of (sequences_dict, labels) -- same shape as load_split_by_condition.
    """
    df = pd.read_parquet(data_path)
    cohort_col = "cohort" if "cohort" in df.columns else "Cohort"
    df = df[df[cohort_col] == cohort].copy()
    if "patient" not in df.columns:
        raise ValueError(f"Expected 'patient' column in {data_path} for patient-level split")

    rng = np.random.default_rng(seed)
    patients = sorted(df["patient"].unique().tolist())
    rng.shuffle(patients)
    n_keep = int(round(len(patients) * frac))
    kept = set(patients[:n_keep])
    if complement:
        kept = set(patients) - kept
    df = df[df["patient"].isin(kept)].copy()

    cdr3_col = "cdr3aa" if "cdr3aa" in df.columns else "cdr3"
    sequences = {
        "cdr2": df["cdr2"].tolist(),
        "cdr3": df[cdr3_col].tolist(),
    }
    if "cdr1" in df.columns:
        sequences["cdr1"] = df["cdr1"].tolist()

    cell_type_col = "cell_type" if "cell_type" in df.columns else "Cell_Type"
    labels = (df[cell_type_col] == "CD8").astype(int).to_numpy()
    return sequences, labels



# ================= DATASET CLASSES =================

class TCRPairMLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling on CDR2+CDR3 pairs.

    Concatenates CDR2 and CDR3 sequences with [SEP] token, applies MLM masking,
    and generates segment IDs to distinguish CDR2 from CDR3.
    """

    def __init__(self, cdr2: List[str], cdr3: List[str], tokenizer,
                 max_len: int = 64, mlm_prob: float = 0.15,
                 properties: np.ndarray = None):
        """
        Args:
            cdr2: List of CDR2 sequences
            cdr3: List of CDR3 sequences (same length as cdr2)
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum sequence length
            mlm_prob: Probability of masking each token (default 0.15)
            properties: Optional [N, num_props] array of normalized property targets
        """
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.properties = properties

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking to input token IDs.

        Masking strategy:
        - 80% of masked tokens → [MASK]
        - 10% of masked tokens → random token
        - 10% of masked tokens → unchanged

        Args:
            inputs: Token IDs tensor

        Returns:
            Tuple of (masked_inputs, labels) where labels=-100 for non-masked positions
        """
        labels = inputs.clone()

        # Create probability matrix
        prob = torch.full(labels.shape, self.mlm_prob)

        # Don't mask special tokens
        special = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        prob.masked_fill_(torch.tensor(special, dtype=torch.bool), 0.0)

        # Select positions to mask
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100  # Only calculate loss on masked positions

        # 80% → [MASK]
        mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        inputs[mask] = self.tokenizer.mask_token_id

        # 10% → random token
        rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~mask
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[rand] = random_tokens[rand]

        # 10% → unchanged (already in inputs)

        return inputs, labels

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys: input_ids, attention_mask, token_type_ids, labels
        """
        # Space-separate amino acids
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])

        # Concatenate CDR2 and CDR3 with [SEP]
        text = f"{s2} {self.tokenizer.sep_token} {s3}"

        # Tokenize
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Generate segment IDs (0 for CDR2, 1 for CDR3)
        token_type_ids = torch.zeros_like(input_ids)
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 1:
            first_sep = sep_positions[0]
            token_type_ids[first_sep + 1:] = 1

        # Apply MLM masking
        input_ids, labels = self.mask_tokens(input_ids)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        if self.properties is not None:
            item["property_targets"] = torch.tensor(
                self.properties[idx], dtype=torch.float32)
        return item


# ================= V GENE ENCODING =================

def encode_v_genes(*v_gene_lists: List[str]):
    """
    Encode V gene names to integer IDs using the fixed ALL_TRBV_GENES vocabulary.

    Uses a complete vocabulary of all known human TRBV segments so the model
    has prototypes for every possible V gene, even unseen ones.

    Args:
        *v_gene_lists: Variable number of V gene string lists (e.g., train, val, test)

    Returns:
        Tuple of (encoded_array_1, encoded_array_2, ..., num_v_genes)
        Each encoded array is np.ndarray of int64 V gene IDs.
        The last element is the total number of V genes (NUM_TRBV_GENES).
    """
    unknown_id = TRBV_GENE_TO_ID["UNKNOWN"]
    encoded = []
    for vg_list in v_gene_lists:
        ids = np.array(
            [TRBV_GENE_TO_ID.get(g, unknown_id) for g in vg_list],
            dtype=np.int64
        )
        encoded.append(ids)

    return (*encoded, NUM_TRBV_GENES)


def load_vdjdb(vdjdb_path: str, exclude_tuples: set = None) -> Tuple[dict, np.ndarray]:
    """
    Load VDJdb dataset for finetuning augmentation.

    Args:
        vdjdb_path: Path to vdj_cd4_cd8.csv (columns: CDR3, V, Cell Type, CDR1, CDR2)
        exclude_tuples: Set of (cdr1, cdr2, cdr3) tuples to exclude (deduplication)

    Returns:
        Tuple of (sequences_dict, labels) matching load_split_by_condition format.
        sequences_dict has keys 'cdr1', 'cdr2', 'cdr3', 'v_gene'.
        labels are binary (CD8=1, CD4=0).
    """
    df = pd.read_csv(vdjdb_path)

    # CDR3 quality filter (same as cohort data)
    df = df[df["CDR3"].str.startswith("C", na=False) & (df["CDR3"].str.len() >= 8)]

    # Deduplicate against cohort data
    if exclude_tuples:
        df["_tuple"] = list(zip(df["CDR1"], df["CDR2"], df["CDR3"]))
        n_before = len(df)
        df = df[~df["_tuple"].isin(exclude_tuples)]
        n_removed = n_before - len(df)
        print(f"  VDJdb dedup: removed {n_removed} sequences shared with cohort data")
        df = df.drop(columns=["_tuple"])

    df = df.reset_index(drop=True)

    sequences = {
        "cdr1": df["CDR1"].tolist(),
        "cdr2": df["CDR2"].tolist(),
        "cdr3": df["CDR3"].tolist(),
        "v_gene": df["V"].tolist(),
    }
    labels = (df["Cell Type"] == "CD8").astype(int).to_numpy()

    return sequences, labels


class TCRDataset(Dataset):
    """
    Dataset for binary classification fine-tuning (CD4 vs CD8).

    Tokenizes CDR3 sequences and returns with binary labels.
    """

    def __init__(self, seqs: List[str], labels: np.ndarray, tokenizer, max_len: int = 64):
        """
        Args:
            seqs: List of CDR3 sequences
            labels: Binary labels (0 or 1)
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum sequence length
        """
        self.seqs = seqs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys: input_ids, attention_mask, labels
        """
        # Space-separate amino acids
        seq = " ".join(self.seqs[idx])

        # Tokenize
        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TCRPairDataset(Dataset):
    """
    Dataset for binary classification with CDR2+CDR3 pairs.

    Similar to TCRPairMLMDataset but for classification instead of MLM.
    Concatenates CDR2 and CDR3 with [SEP] token and generates segment IDs.
    """

    def __init__(self, cdr2: List[str], cdr3: List[str], labels: np.ndarray,
                 tokenizer, max_len: int = 64):
        """
        Args:
            cdr2: List of CDR2 sequences
            cdr3: List of CDR3 sequences (same length as cdr2)
            labels: Binary labels (0 or 1) for CD4 vs CD8
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum sequence length
        """
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys: input_ids, attention_mask, token_type_ids, labels
        """
        # Space-separate amino acids
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])

        # Concatenate CDR2 and CDR3 with [SEP]
        text = f"{s2} {self.tokenizer.sep_token} {s3}"

        # Tokenize
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Generate segment IDs (0 for CDR2, 1 for CDR3)
        token_type_ids = torch.zeros_like(input_ids)
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 1:
            first_sep = sep_positions[0]
            token_type_ids[first_sep + 1:] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TCRTripleDataset(Dataset):
    """
    Dataset for classification with CDR1+CDR2+CDR3.

    Uses custom token_type_ids: 0 for CDR1+CDR2 (germline), 1 for CDR3 (somatic).
    Format: [CLS] CDR1 [SEP] CDR2 [SEP] CDR3 [SEP]
    """

    def __init__(self, cdr1: List[str], cdr2: List[str], cdr3: List[str],
                 labels: np.ndarray, tokenizer, max_len: int = 64):
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        # Space-separate amino acids
        s1 = " ".join(self.cdr1[idx])
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])

        # Format: CDR1 [SEP] CDR2 [SEP] CDR3
        text = f"{s1} {self.tokenizer.sep_token} {s2} {self.tokenizer.sep_token} {s3}"

        # Tokenize
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Custom token_type_ids: 0 until SECOND [SEP], then 1
        token_type_ids = torch.zeros_like(input_ids)
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) >= 2:
            second_sep = sep_positions[1]
            token_type_ids[second_sep + 1:] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TCRTripleMLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling on CDR1+CDR2+CDR3 triples.

    Uses custom token_type_ids: 0 for CDR1+CDR2 (germline), 1 for CDR3 (somatic).
    Format: [CLS] CDR1 [SEP] CDR2 [SEP] CDR3 [SEP]
    """

    def __init__(self, cdr1: List[str], cdr2: List[str], cdr3: List[str],
                 tokenizer, max_len: int = 64, mlm_prob: float = 0.15,
                 properties: np.ndarray = None, v_gene_ids: np.ndarray = None,
                 n_donors: np.ndarray = None, n_convergence: np.ndarray = None,
                 chain_token: str = ""):
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.properties = properties
        self.v_gene_ids = v_gene_ids
        self.n_donors = n_donors
        self.n_convergence = n_convergence
        self.chain_token = chain_token
        self.chain_token_id = (tokenizer.convert_tokens_to_ids(chain_token)
                               if chain_token else None)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking to input token IDs."""
        labels = inputs.clone()

        # Create probability matrix
        prob = torch.full(labels.shape, self.mlm_prob)

        # Don't mask special tokens
        special = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        prob.masked_fill_(torch.tensor(special, dtype=torch.bool), 0.0)
        if self.chain_token_id is not None:
            prob.masked_fill_(labels == self.chain_token_id, 0.0)

        # Select positions to mask
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100  # Only calculate loss on masked positions

        # 80% → [MASK]
        mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        inputs[mask] = self.tokenizer.mask_token_id

        # 10% → random token
        rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~mask
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[rand] = random_tokens[rand]

        # 10% → unchanged (already in inputs)

        return inputs, labels

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        # Space-separate amino acids
        s1 = " ".join(self.cdr1[idx])
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])

        # Format: [chain_token] CDR1 [SEP] CDR2 [SEP] CDR3
        prefix = f"{self.chain_token} " if self.chain_token else ""
        text = f"{prefix}{s1} {self.tokenizer.sep_token} {s2} {self.tokenizer.sep_token} {s3}"

        # Tokenize
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Custom token_type_ids: 0 until SECOND [SEP], then 1
        token_type_ids = torch.zeros_like(input_ids)
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) >= 2:
            second_sep = sep_positions[1]
            token_type_ids[second_sep + 1:] = 1

        # Apply MLM masking
        input_ids, labels = self.mask_tokens(input_ids)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        if self.properties is not None:
            item["property_targets"] = torch.tensor(
                self.properties[idx], dtype=torch.float32)
        if self.v_gene_ids is not None:
            item["v_gene_ids"] = torch.tensor(
                self.v_gene_ids[idx], dtype=torch.long)
        if self.n_donors is not None:
            item["n_donors"] = torch.tensor(float(self.n_donors[idx]), dtype=torch.float32)
        if self.n_convergence is not None:
            item["n_convergence"] = torch.tensor(float(self.n_convergence[idx]), dtype=torch.float32)
        return item


class TCRTripleCausalDataset(Dataset):
    """
    Dataset for autoregressive (causal LM) pretraining on CDR1+CDR2+CDR3 triples.

    Format: [CLS] CDR1 [SEP] CDR2 [SEP] CDR3 [SEP] [PAD]...
    [CLS] serves as BOS. Labels = input_ids with -100 for [PAD] positions.
    GPT2LMHeadModel shifts labels internally, so no manual shift is needed.
    """

    def __init__(self, cdr1: List[str], cdr2: List[str], cdr3: List[str],
                 tokenizer, max_len: int = 64,
                 n_donors: np.ndarray = None, n_convergence: np.ndarray = None):
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_donors = n_donors
        self.n_convergence = n_convergence

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        s1 = " ".join(self.cdr1[idx])
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])
        text = f"{s1} {self.tokenizer.sep_token} {s2} {self.tokenizer.sep_token} {s3}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if self.n_donors is not None:
            item["n_donors"] = torch.tensor(float(self.n_donors[idx]), dtype=torch.float32)
        if self.n_convergence is not None:
            item["n_convergence"] = torch.tensor(float(self.n_convergence[idx]), dtype=torch.float32)
        return item


# ================= SONIA DATA PIPELINE =================

def load_vgene_cdr_dict(dict_path: str) -> dict:
    """
    Load the pre-built V gene → [CDR1, CDR2] lookup table from JSON.

    The file at data/processed/v_cdr_dict.json maps each TRBV gene name to a
    two-element list [cdr1_seq, cdr2_seq].  CDR1 and CDR2 are fully germline-encoded
    (determined entirely by which V gene was used), so this single file covers every
    OLGA-generated sequence regardless of which patient cohort it came from.

    Args:
        dict_path: Path to v_cdr_dict.json (e.g. "../data/processed/v_cdr_dict.json")

    Returns:
        Dict mapping V gene name → [cdr1_str, cdr2_str]
    """
    with open(dict_path, "r") as f:
        return json.load(f)


def lookup_germline_cdrs(
    v_genes: List[str],
    vgene_cdr_dict: dict,
    fallback_cdr1: str = "",
    fallback_cdr2: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Look up CDR1 and CDR2 sequences for a list of V gene calls.

    Args:
        v_genes: List of V gene names (e.g. ["TRBV20-1", "TRBV7-2", ...])
        vgene_cdr_dict: Dict from load_vgene_cdr_dict — format {v_gene: [cdr1, cdr2]}
        fallback_cdr1: CDR1 to use when V gene is not in the dict (default "")
        fallback_cdr2: CDR2 to use when V gene is not in the dict (default "")

    Returns:
        Tuple of (cdr1_list, cdr2_list)
    """
    fallback = [fallback_cdr1, fallback_cdr2]
    cdr1_list = [vgene_cdr_dict.get(v, fallback)[0] for v in v_genes]
    cdr2_list = [vgene_cdr_dict.get(v, fallback)[1] for v in v_genes]
    return cdr1_list, cdr2_list


class SoNNiaDataset(Dataset):
    """
    Dataset for soNNia selection model training (stages 3a and 3b).

    Combines CD4, CD8, and NCE negative sequences into one dataset.
    Uses the same CDR1+CDR2+CDR3 triple tokenisation as TCRTripleDataset so
    BERT sees the same input format it was pretrained on.

    Key difference from TCRTripleDataset: outputs 'sequence_type' (0/1/2) rather
    than 'labels' (0/1), keeping the three-way soNNia semantics explicit and
    separate from the binary classification convention used elsewhere.

    Token type IDs:
        0 → CDR1 + CDR2 (germline-encoded, stable)
        1 → CDR3 (somatic recombination product)

    sequence_type values:
        0 (SoNNiaDataset.CD4) → productive CD4 T-cell sequences
        1 (SoNNiaDataset.CD8) → productive CD8 T-cell sequences
        2 (SoNNiaDataset.NEG) → NCE negative sequences (e.g. Emerson bulk or OLGA)
    """

    CD4 = 0
    CD8 = 1
    NEG = 2

    def __init__(
        self,
        cdr1: List[str],
        cdr2: List[str],
        cdr3: List[str],
        sequence_types: np.ndarray,
        tokenizer,
        max_len: int = 64,
        q_cd4_scores: np.ndarray = None,
        q_cd8_scores: np.ndarray = None,
    ):
        """
        Args:
            cdr1: CDR1 sequences (CD4 + CD8 + Emerson concatenated in any order)
            cdr2: CDR2 sequences — same length as cdr1
            cdr3: CDR3 sequences — same length as cdr1
            sequence_types: Integer array with values 0 (CD4), 1 (CD8), or 2 (EMERSON)
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum token length (default 64)
            q_cd4_scores: Optional float array of pre-computed Q^CD4 scores (stage 3b)
            q_cd8_scores: Optional float array of pre-computed Q^CD8 scores (stage 3b)
        """
        assert len(cdr1) == len(cdr2) == len(cdr3) == len(sequence_types), \
            "All input lists must have the same length"
        if q_cd4_scores is not None:
            assert len(q_cd4_scores) == len(cdr3), \
                "q_cd4_scores length must match number of sequences"
        if q_cd8_scores is not None:
            assert len(q_cd8_scores) == len(cdr3), \
                "q_cd8_scores length must match number of sequences"
        self.cdr1           = cdr1
        self.cdr2           = cdr2
        self.cdr3           = cdr3
        self.sequence_types = sequence_types
        self.tokenizer      = tokenizer
        self.max_len        = max_len
        self.q_cd4_scores   = q_cd4_scores
        self.q_cd8_scores   = q_cd8_scores

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        s1 = " ".join(self.cdr1[idx])
        s2 = " ".join(self.cdr2[idx])
        s3 = " ".join(self.cdr3[idx])

        # Format: [CLS] CDR1 [SEP] CDR2 [SEP] CDR3 [SEP]
        text = f"{s1} {self.tokenizer.sep_token} {s2} {self.tokenizer.sep_token} {s3}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # token_type_ids: 0 for [CLS]+CDR1+[SEP]+CDR2, then 1 from second [SEP] onward
        token_type_ids = torch.zeros_like(input_ids)
        sep_positions  = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 2:
            token_type_ids[sep_positions[1] + 1:] = 1

        item = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sequence_type":  torch.tensor(self.sequence_types[idx], dtype=torch.long),
        }
        if self.q_cd4_scores is not None:
            item["score_cd4"] = torch.tensor(self.q_cd4_scores[idx], dtype=torch.float32)
            item["score_cd8"] = torch.tensor(self.q_cd8_scores[idx], dtype=torch.float32)
        return item


def load_olga_sequences(olga_path: str) -> dict:
    """
    Load OLGA-generated sequences for P_gen baseline estimation.

    OLGA output format: comma-separated, no header, columns are
    nt_seq, aa_cdr3, v_gene, j_gene. Files with a header row are also
    accepted (auto-detected).

    No quality filtering is applied — OLGA sequences are the P_gen baseline
    and must represent raw recombination without selection bias.

    Args:
        olga_path: Path to OLGA-generated CSV file

    Returns:
        Dict with keys 'cdr3' (List[str]) and 'v_gene' (List[str])
    """
    # Detect whether the file has a header by checking if the first value looks
    # like a nucleotide sequence (all ACGT) vs a column name string.
    with open(olga_path) as _f:
        first = _f.readline().split(",")[0].strip().upper()
    has_header = not all(c in "ACGT" for c in first)

    if has_header:
        df = pd.read_csv(olga_path)
        # Accept common OLGA column name variants
        col_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ("aa_seq", "amino_acid", "cdr3_aa", "cdr3"):
                col_map["cdr3_aa"] = col
            elif lc in ("v_gene", "v_call", "v_segment"):
                col_map["v_gene"] = col
        if "cdr3_aa" not in col_map or "v_gene" not in col_map:
            raise ValueError(
                f"load_olga_sequences: could not find cdr3/v_gene columns in {list(df.columns)}"
            )
        df = df.rename(columns={col_map["cdr3_aa"]: "cdr3_aa", col_map["v_gene"]: "v_gene"})
    else:
        # Standard OLGA output: no header, columns are nt_seq, aa_cdr3, v_gene, j_gene
        df = pd.read_csv(olga_path, header=None,
                         names=["nt_seq", "cdr3_aa", "v_gene", "j_gene"])

    # Drop rows with null CDR3 only — no further filtering.
    # OLGA sequences are the P_gen baseline and must not be pre-filtered;
    # removing sequences would bias Q = P_data / P_gen away from true recombination.
    df = df.dropna(subset=["cdr3_aa"]).reset_index(drop=True)

    return {
        "cdr3": df["cdr3_aa"].tolist(),
        "v_gene": df["v_gene"].tolist(),
    }


def load_emerson_sequences(emerson_path: str, n_samples: int = None,
                           compute_pdata: bool = False) -> dict:
    """
    Load Emerson bulk sequences from parquet.

    Reads the standard schema produced by build_cohort_parquet.py:
    cdr1, cdr2, cdr3aa, v_gene. Also loads n_donors and n_convergence
    when present (produced by build_pretrain_targets.py).
    Backward compat: parquets with n_multiplicity column are transparently
    loaded and exposed as n_convergence.

    Args:
        emerson_path:   Path to emerson parquet.
        n_samples:      If given, randomly sample this many rows (before dedup).
        compute_pdata:  If True, compute p_data = count(x) / N_total for each unique
                        clonotype (x identified by cdr3aa + v_gene), counting duplicate
                        rows before deduplication. Useful for holdouts built with
                        replace=True where duplicates encode empirical frequency.
                        Also deduplicates the result so each clonotype is scored once.

    Returns:
        Dict with keys 'cdr1', 'cdr2', 'cdr3', 'v_gene' and optionally
        'n_donors', 'n_convergence' (numpy int32 arrays), and 'p_data' (float64)
        when compute_pdata=True.
    """
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(emerson_path).names)
    cols = ["cdr1", "cdr2", "cdr3aa", "v_gene"]
    if "n_donors" in schema_names:
        cols.append("n_donors")
    # n_convergence is the canonical name; n_multiplicity is the legacy name
    nc_col = "n_convergence" if "n_convergence" in schema_names else (
        "n_multiplicity" if "n_multiplicity" in schema_names else None)
    if nc_col:
        cols.append(nc_col)

    if compute_pdata:
        df = pd.read_parquet(emerson_path, columns=cols)
        if n_samples is not None and n_samples < len(df):
            df = df.sample(n=n_samples, replace=False, random_state=42).reset_index(drop=True)
        # Count occurrences per unique clonotype before dedup → p_data
        df["p_data"] = (df.groupby(["cdr3aa", "v_gene"])["cdr3aa"]
                          .transform("count") / len(df))
        df = df.drop_duplicates(subset=["cdr3aa", "v_gene"]).reset_index(drop=True)
    else:
        table = pq.read_table(emerson_path, columns=cols)
        if n_samples is not None and n_samples < table.num_rows:
            idx = np.random.choice(table.num_rows, size=n_samples, replace=False)
            idx.sort()
            table = table.take(idx)
        df = table.to_pandas()

    result = {
        "cdr1":   df["cdr1"].tolist(),
        "cdr2":   df["cdr2"].tolist(),
        "cdr3":   df["cdr3aa"].tolist(),
        "v_gene": df["v_gene"].tolist(),
    }
    if "n_donors" in df.columns:
        result["n_donors"] = df["n_donors"].fillna(0).astype("int32").values
    if nc_col and nc_col in df.columns:
        result["n_convergence"] = df[nc_col].fillna(0).astype("int32").values
    if compute_pdata:
        result["p_data"] = df["p_data"].values
    return result


# ================= PAIRED CHAIN DATA =================

def load_paired_sequences(path: str, dataset_source: str = None, vgene_map: dict = None) -> dict:
    """
    Load paired alpha-beta TCR sequences from TSV.

    Args:
        path:           Path to pretrain_unique_paired_sequences.tsv
        dataset_source: If given, keep only rows where dataset_source == this value.
        vgene_map:      Optional {raw_v_gene: token} (from build_vtoken_tokenizer vgene_map.json). If given, also
                        return alpha_vtoken / beta_vtoken (raw V-gene -> atomic V token, unseen -> [V_UNK]) for the
                        V-token model. Backward-compatible: omitted -> only the CDR1/2/3 keys.

    Returns:
        Dict with keys: alpha_cdr1, alpha_cdr2, alpha_cdr3, beta_cdr1, beta_cdr2, beta_cdr3 (+ *_vtoken if mapped).
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if dataset_source is not None:
        # accept a single source string OR a list/tuple of sources (e.g. OTS + pan_disease)
        if isinstance(dataset_source, (list, tuple, set)):
            df = df[df["dataset_source"].isin(list(dataset_source))].reset_index(drop=True)
        else:
            df = df[df["dataset_source"] == dataset_source].reset_index(drop=True)
    out = {
        "alpha_cdr1": df["aaSeqCDR1_alpha"].tolist(),
        "alpha_cdr2": df["aaSeqCDR2_alpha"].tolist(),
        "alpha_cdr3": df["aaSeqCDR3_alpha"].tolist(),
        "beta_cdr1":  df["aaSeqCDR1_beta"].tolist(),
        "beta_cdr2":  df["aaSeqCDR2_beta"].tolist(),
        "beta_cdr3":  df["aaSeqCDR3_beta"].tolist(),
    }
    if vgene_map is not None:
        out["alpha_vtoken"] = [vgene_map.get(str(v), "[V_UNK]") for v in df["v_gene_alpha"]]
        out["beta_vtoken"] = [vgene_map.get(str(v), "[V_UNK]") for v in df["v_gene_beta"]]
    return out


class TCRPairedDataset(Dataset):
    """
    Dataset for paired alpha-beta TCR chain pretraining.

    Each item returns two independently tokenized and MLM-masked sequences
    (alpha and beta chains) for a single observed single-cell pair.
    """

    def __init__(self, alpha_cdr1: List[str], alpha_cdr2: List[str], alpha_cdr3: List[str],
                 beta_cdr1: List[str],  beta_cdr2: List[str],  beta_cdr3: List[str],
                 tokenizer, max_len: int = 64, mlm_prob: float = 0.15):
        self.alpha = (alpha_cdr1, alpha_cdr2, alpha_cdr3)
        self.beta  = (beta_cdr1,  beta_cdr2,  beta_cdr3)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.mlm_prob  = mlm_prob
        self.special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        }
        self.alpha_chain_tok = "[unused0]"
        self.beta_chain_tok  = "[unused1]"
        self.special_ids |= {
            tokenizer.convert_tokens_to_ids("[unused0]"),
            tokenizer.convert_tokens_to_ids("[unused1]"),
        }
        self.vocab_size = tokenizer.vocab_size

    def _format(self, cdr1: List[str], cdr2: List[str], cdr3: List[str], idx: int,
                chain_token: str = "") -> str:
        prefix = chain_token + " " if chain_token else ""
        return (
            prefix +
            " ".join(str(c) for c in cdr1[idx]) + " [SEP] " +
            " ".join(str(c) for c in cdr2[idx]) + " [SEP] " +
            " ".join(str(c) for c in cdr3[idx])
        )

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _apply_mlm(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, -100)
        candidates = [
            i for i in range(len(input_ids))
            if attention_mask[i] == 1 and input_ids[i].item() not in self.special_ids
        ]
        if not candidates:
            return input_ids.clone(), labels

        n_mask = max(1, int(len(candidates) * self.mlm_prob))
        chosen = np.random.choice(candidates, size=n_mask, replace=False)

        masked_ids = input_ids.clone()
        for pos in chosen:
            labels[pos] = masked_ids[pos]
            r = np.random.random()
            if r < 0.8:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                masked_ids[pos] = np.random.randint(0, self.vocab_size)
            # else: keep original (10%)

        return masked_ids, labels

    def __len__(self) -> int:
        return len(self.alpha[0])

    def __getitem__(self, idx: int) -> dict:
        alpha_ids, alpha_mask = self._tokenize(self._format(*self.alpha, idx, self.alpha_chain_tok))
        beta_ids,  beta_mask  = self._tokenize(self._format(*self.beta,  idx, self.beta_chain_tok))

        alpha_ids_v1, alpha_labels_v1 = self._apply_mlm(alpha_ids, alpha_mask)
        beta_ids_v1,  beta_labels_v1  = self._apply_mlm(beta_ids,  beta_mask)
        alpha_ids_v2, alpha_labels_v2 = self._apply_mlm(alpha_ids, alpha_mask)
        beta_ids_v2,  beta_labels_v2  = self._apply_mlm(beta_ids,  beta_mask)

        return {
            "alpha_input_ids_v1":  alpha_ids_v1,  "alpha_attention_mask": alpha_mask,
            "alpha_labels_v1":     alpha_labels_v1,
            "beta_input_ids_v1":   beta_ids_v1,   "beta_attention_mask":  beta_mask,
            "beta_labels_v1":      beta_labels_v1,
            "alpha_input_ids_v2":  alpha_ids_v2,  "alpha_labels_v2": alpha_labels_v2,
            "beta_input_ids_v2":   beta_ids_v2,   "beta_labels_v2":  beta_labels_v2,
        }


class TCRJointDataset(Dataset):
    """
    Dataset for joint (single-sequence) paired alpha-beta TCR pretraining (Exp 9 / SCEPTR
    replica family).

    Both chains are encoded in ONE sequence so BERT self-attention models inter-chain
    interactions directly (SCEPTR-style), unlike the siamese TCRPairedDataset.

    Joint format (post chain-identity fix, 2026-05-22):
      [CLS] aCDR1 | aCDR2 | aCDR3 | bCDR1 | bCDR2 | bCDR3 [SEP]
    where every "|" is the tokenizer's real sep token (id 24), and [CLS] + trailing
    [SEP] are added by the tokenizer's add_special_tokens=True. No string chain markers.

    Chain identity is carried by token_type_ids in {0, 1}: 0 across CLS + alpha portion,
    1 across the middle SEP + beta portion + trailing SEP. Single-chain views use the
    chain's own type id throughout.

    Behaviour is selected via `view_strategy`:

      "exp9_chain_drop_loss" (default) -- emits 4 sets of inputs per item:
        - full v1 (MLM-masked), full v2 (MLM-masked, separate mask)
        - alpha-only view (beta dropped), beta-only view (alpha dropped)
        Returns: full_input_ids_v1, full_labels_v1, full_input_ids_v2, full_labels_v2,
                 full_attention_mask (pre-censor, shared) + full_attention_mask_v1/_v2 (censored),
                 full_token_type_ids, full_position_ids,
                 alpha_input_ids, alpha_attention_mask, alpha_token_type_ids, alpha_position_ids,
                 beta_input_ids,  beta_attention_mask,  beta_token_type_ids,  beta_position_ids.
        When censor_apply_prob > 0, CDR3-only per-chain censoring is applied per view before MLM
        (germline CDR1/CDR2 untouched); v1 and v2 then carry different attention masks.

      "sceptr_replica" -- emits 2 independently-censored views per item (chain-drop is
        a view augmentation, not a separate loss):
        Returns: full_input_ids_v1, full_attention_mask_v1, full_labels_v1,
                 full_input_ids_v2, full_attention_mask_v2,
                 full_token_type_ids_v1, full_token_type_ids_v2.
    """

    def __init__(self, alpha_cdr1: List[str], alpha_cdr2: List[str], alpha_cdr3: List[str],
                 beta_cdr1: List[str],  beta_cdr2: List[str],  beta_cdr3: List[str],
                 tokenizer, max_len: int = 128, mlm_prob: float = 0.15,
                 view_strategy: str = "exp9_chain_drop_loss",
                 censor_residue_prob: float = 0.20,
                 chain_drop_prob: float = 0.5,
                 censor_apply_prob: float = 0.0,
                 input_format: str = "cdr123",
                 alpha_vtoken: List[str] = None,
                 beta_vtoken: List[str] = None,
                 alpha_simcse: bool = False):
        self.alpha = (alpha_cdr1, alpha_cdr2, alpha_cdr3)
        self.beta  = (beta_cdr1,  beta_cdr2,  beta_cdr3)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.mlm_prob  = mlm_prob
        # No string chain markers in the joint format anymore -- they tokenised to UNK
        # under the wukevin tokenizer and added only noise. Chain identity is now
        # carried by token_type_ids (segment embeddings) instead.
        self.sep = tokenizer.sep_token
        self.special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        }
        self.vocab_size = tokenizer.vocab_size
        # ---- V-token mode ("vtoken"): each chain is "[V:gene] | CDR3", the atomic germline V token
        #      replacing the V-determined CDR1/CDR2. alpha_simcse -> emit TWO alpha views (alpha's own
        #      SimCSE) instead of the single alpha chain-drop view (alpha detached from the pair). ----
        if input_format not in ("cdr123", "vtoken"):
            raise ValueError(f"Unknown input_format: {input_format}")
        self.input_format = input_format
        self.alpha_v = alpha_vtoken
        self.beta_v  = beta_vtoken
        self.alpha_simcse = alpha_simcse
        if input_format == "vtoken":
            if alpha_vtoken is None or beta_vtoken is None:
                raise ValueError("input_format='vtoken' requires alpha_vtoken and beta_vtoken lists")
            # The atomic V tokens are germline -> must NEVER be MLM-masked. Add their ids to special_ids
            # so _apply_mlm (which skips special_ids) leaves them intact; CDR3-only censoring is span-
            # bounded and never reaches them anyway.
            vtoks = set(alpha_vtoken) | set(beta_vtoken) | {"[V_UNK]"}
            self.special_ids |= {tokenizer.convert_tokens_to_ids(t) for t in vtoks}
        self.view_strategy = view_strategy
        self.censor_residue_prob = censor_residue_prob
        self.chain_drop_prob = chain_drop_prob
        # Per-view, per-chain probability of applying CDR3 censoring (0.0 = OFF, the default,
        # so existing exp9 consumers like tcr_bert_joint_pretrain stay uncensored). Stage 2
        # sets 0.5. `censor_res_prob` is the per-residue drop prob (alias of censor_residue_prob,
        # used by _apply_censoring_cdr3_only which mirrors the Stage 1 beta dataset).
        self.censor_apply_prob = censor_apply_prob
        self.censor_res_prob   = censor_residue_prob
        if view_strategy not in ("exp9_chain_drop_loss", "sceptr_replica"):
            raise ValueError(f"Unknown view_strategy: {view_strategy}")

    def _chain_str(self, cdrs, idx: int) -> str:
        cdr1, cdr2, cdr3 = cdrs
        sep = self.sep
        if self.input_format == "vtoken":
            # "[V:gene] | CDR3" -- the atomic V token stands in for the V-determined CDR1/CDR2.
            # cdrs is self.alpha or self.beta (identity), so pick the matching V-token list.
            vtok = self.alpha_v if cdrs is self.alpha else self.beta_v
            return str(vtok[idx]) + " " + sep + " " + " ".join(str(c) for c in cdr3[idx])
        return (  # legacy: CDR1 | CDR2 | CDR3
            " ".join(str(c) for c in cdr1[idx]) + " " + sep + " " +
            " ".join(str(c) for c in cdr2[idx]) + " " + sep + " " +
            " ".join(str(c) for c in cdr3[idx])
        )

    def _format_joint(self, idx: int) -> str:
        # Joint pair: aCDR1 | aCDR2 | aCDR3 | bCDR1 | bCDR2 | bCDR3
        return (self._chain_str(self.alpha, idx) + " " + self.sep + " " +
                self._chain_str(self.beta,  idx))

    def _format_single(self, cdrs, idx: int) -> str:
        return self._chain_str(cdrs, idx)

    def _joint_token_type_ids(self, n_alpha: int, n_beta: int) -> torch.Tensor:
        """Build (max_len,) long tensor of segment ids: 0 across [CLS..alpha-end],
        1 across [middle_sep..beta-end + trailing_sep], 0 in padding tail."""
        type_ids = torch.zeros(self.max_len, dtype=torch.long)
        # Positions 0..n_alpha = CLS + alpha tokens stay 0.
        # Middle SEP is at position 1 + n_alpha; flip to 1 from there onward up to and
        # including the trailing [SEP] at position 1 + n_alpha + 1 + n_beta.
        start_b = 1 + n_alpha
        end_b   = min(self.max_len, 1 + n_alpha + 1 + n_beta + 1)
        type_ids[start_b:end_b] = 1
        return type_ids

    def _joint_position_ids(self, n_alpha: int, n_beta: int) -> torch.Tensor:
        """Per-chain RESET positions for the joint paired layout (mirrors _joint_token_type_ids):
        alpha span [0, 1+n_alpha) -> positions 0..n_alpha (CLS + alpha tokens);
        beta span [1+n_alpha, ...) -> positions RESET to 0.. (middle_sep + beta + trailing_sep);
        padding tail stays 0. Built on the token-count layout (pre-censor) so positions are
        bounded by the per-chain length (<= ~47) and reset at the chain boundary."""
        pos = torch.zeros(self.max_len, dtype=torch.long)
        a_end = min(1 + n_alpha, self.max_len)
        pos[0:a_end] = torch.arange(a_end)
        b_start = 1 + n_alpha
        b_end   = min(self.max_len, 1 + n_alpha + 1 + n_beta + 1)
        if b_start < self.max_len:
            pos[b_start:b_end] = torch.arange(b_end - b_start)
        return pos

    @staticmethod
    def _single_chain_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Contiguous positions 0..span over a single-chain view's active span; pad stays 0."""
        span = int(attention_mask.sum())
        pos = torch.zeros_like(attention_mask, dtype=torch.long)
        pos[:span] = torch.arange(span)
        return pos

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _apply_mlm(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, -100)
        candidates = [
            i for i in range(len(input_ids))
            if attention_mask[i] == 1 and input_ids[i].item() not in self.special_ids
        ]
        if not candidates:
            return input_ids.clone(), labels

        n_mask = max(1, int(len(candidates) * self.mlm_prob))
        chosen = np.random.choice(candidates, size=n_mask, replace=False)

        masked_ids = input_ids.clone()
        for pos in chosen:
            labels[pos] = masked_ids[pos]
            r = np.random.random()
            if r < 0.8:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                masked_ids[pos] = np.random.randint(0, self.vocab_size)
            # else: keep original (10%)

        return masked_ids, labels

    def _apply_censoring(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         alpha_span: Tuple[int, int],
                         beta_span: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SCEPTR-style view augmentation: with prob chain_drop_prob drop one whole chain,
        then remove censor_residue_prob (default 0.20) of remaining residue tokens.
        Removed positions become PAD with attention_mask=0; positions of surviving tokens
        are preserved (positional encoding "remains fixed" per the SCEPTR Methods).

        alpha_span / beta_span are [start, end) half-open token-index ranges, precomputed
        by __getitem__ from the alpha/beta token lengths (we cannot detect spans from
        token IDs alone because the wukevin tokenizer has no chain-marker tokens).
        """
        new_ids = input_ids.clone()
        new_mask = attention_mask.clone()
        pad_id = self.tokenizer.pad_token_id

        # Step 1: with prob chain_drop_prob, drop one whole chain
        if np.random.random() < self.chain_drop_prob:
            if np.random.random() < 0.5:
                start, end = alpha_span
            else:
                start, end = beta_span
            new_ids[start:end] = pad_id
            new_mask[start:end] = 0

        # Step 2: remove censor_residue_prob of remaining residue tokens.
        # special_ids includes CLS, the real SEP, PAD, MASK, UNK -- so CLS, the real
        # CDR/chain separators, and any unexpected UNKs are never picked for removal.
        residue_positions = [
            i for i in range(len(new_ids))
            if new_mask[i] == 1 and new_ids[i].item() not in self.special_ids
        ]
        if residue_positions:
            n_remove = int(len(residue_positions) * self.censor_residue_prob)
            if n_remove > 0:
                chosen = np.random.choice(residue_positions, size=n_remove, replace=False)
                for pos in chosen:
                    new_ids[pos] = pad_id
                    new_mask[pos] = 0

        return new_ids, new_mask

    def _seg_counts(self, cdrs, idx: int) -> Tuple[int, int, int]:
        """Token counts of (CDR1, CDR2, CDR3) for one chain -- used to locate the CDR3 sub-span.
        The wukevin tokenizer is residue-level (1 token per amino acid), so the count equals the
        number of residues == len(cdr_string) (0 UNK verified on this data). Using len() instead of
        tokenize() removes 6 tokenizer calls per paired item -- a big data-loading saving."""
        return tuple(len(seg[idx]) for seg in cdrs)

    def _cdr3_span(self, n1: int, n2: int, n3: int, start_offset: int = 1) -> Tuple[int, int]:
        """Half-open [start, end) token range of CDR3 in a C1|C2|C3 block whose C1 begins at
        index `start_offset` (1 = just [CLS] for a single chain; for the beta chain inside the
        joint pair, start_offset = number of tokens before bCDR1). Capped at max_len."""
        start = start_offset + n1 + 1 + n2 + 1     # + C1 + sep + C2 + sep
        end = start + n3
        return min(start, self.max_len), min(end, self.max_len)

    def _cdr3_span_vtoken(self, n3: int, start_offset: int = 1) -> Tuple[int, int]:
        """Half-open [start, end) CDR3 range in a `[V-token] | CDR3` block whose V-token is at index
        `start_offset` (1 = just [CLS] for a single chain). CDR3 begins after the V-token (+1) and the
        sep (+1), i.e. start_offset + 2. Capped at max_len. V-token layout replaces the C1|C2 prefix."""
        start = start_offset + 2
        end = start + n3
        return min(start, self.max_len), min(end, self.max_len)

    def _apply_censoring_cdr3_only(self, input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    cdr3_span: Tuple[int, int]
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SCEPTR residue censoring on CDR3 ONLY (germline CDR1/CDR2 never touched), mirroring
        the Stage 1 beta dataset. With prob censor_apply_prob apply to this view, else return
        as-is; when applied, drop each CDR3 residue independently with prob censor_res_prob
        (-> PAD + mask 0). Called once per chain so each CDR3 gets an independent apply draw."""
        if self.censor_apply_prob <= 0.0 or np.random.random() >= self.censor_apply_prob:
            return input_ids.clone(), attention_mask.clone()
        new_ids = input_ids.clone()
        new_mask = attention_mask.clone()
        pad_id = self.tokenizer.pad_token_id
        start, end = cdr3_span
        for pos in range(start, end):
            if new_mask[pos] == 1 and new_ids[pos].item() not in self.special_ids:
                if np.random.random() < self.censor_res_prob:
                    new_ids[pos] = pad_id
                    new_mask[pos] = 0
        return new_ids, new_mask

    def __len__(self) -> int:
        return len(self.alpha[0])

    def __getitem__(self, idx: int) -> dict:
        full_ids, full_mask = self._tokenize(self._format_joint(idx))

        # Per-chain block token counts (used for token_type_ids, and by SCEPTR-replica for spans).
        # The tokenizer is residue-level (1 token/AA, 0 UNK verified), so these are EXACT arithmetic --
        # no tokenize() call needed (this is the per-item data-loading hot path). vtoken chain =
        # [V]+sep+CDR3 = 2 + n_cdr3; cdr123 chain = n1 + sep + n2 + sep + n3.
        na = self._seg_counts(self.alpha, idx)   # (n_a1, n_a2, n_a3)
        nb = self._seg_counts(self.beta,  idx)   # (n_b1, n_b2, n_b3)
        if self.input_format == "vtoken":
            n_alpha, n_beta = 2 + na[2], 2 + nb[2]
        else:
            n_alpha = na[0] + 1 + na[1] + 1 + na[2]
            n_beta  = nb[0] + 1 + nb[1] + 1 + nb[2]
        full_token_type_ids = self._joint_token_type_ids(n_alpha, n_beta)

        if self.view_strategy == "sceptr_replica":
            # alpha span = [CLS+1, CLS+1+n_alpha); beta span = [middle_sep+1, middle_sep+1+n_beta)
            alpha_span = (1, 1 + n_alpha)
            beta_span  = (1 + n_alpha + 1, 1 + n_alpha + 1 + n_beta)

            # Two independently censored views; MLM only on view 1 (single MLM term, per paper).
            v1_ids, v1_mask = self._apply_censoring(full_ids, full_mask, alpha_span, beta_span)
            v2_ids, v2_mask = self._apply_censoring(full_ids, full_mask, alpha_span, beta_span)
            v1_ids_m, v1_labels = self._apply_mlm(v1_ids, v1_mask)
            return {
                "full_input_ids_v1":      v1_ids_m,
                "full_attention_mask_v1": v1_mask,
                "full_labels_v1":         v1_labels,
                "full_input_ids_v2":      v2_ids,
                "full_attention_mask_v2": v2_mask,
                "full_token_type_ids_v1": full_token_type_ids,
                "full_token_type_ids_v2": full_token_type_ids.clone(),
            }

        # Default: exp9_chain_drop_loss
        alpha_ids, alpha_mask = self._tokenize(self._format_single(self.alpha, idx))
        beta_ids,  beta_mask  = self._tokenize(self._format_single(self.beta,  idx))

        # Positions are built on the PRE-censor masks/lengths, so censoring (which only PADs
        # slots) leaves survivor positions fixed (SCEPTR "positions remain fixed").
        full_position_ids  = self._joint_position_ids(n_alpha, n_beta)
        alpha_position_ids = self._single_chain_position_ids(alpha_mask)
        beta_position_ids  = self._single_chain_position_ids(beta_mask)

        # Per-chain CDR3 spans (germline CDR1/CDR2 -- or the V-token -- never censored). Alpha sits at
        # the front, so its span is the same alone and in the pair; beta's block starts after CLS + the
        # whole alpha block + the alpha|beta sep. (na/nb already computed at the top of __getitem__.)
        if self.input_format == "vtoken":
            # alpha block = [Va] | aCDR3 = 2 + n_a3 tokens -> beta V-token at index 1 + (2+n_a3) + 1 = 4 + n_a3
            na3, nb3 = na[2], nb[2]
            a_cdr3      = self._cdr3_span_vtoken(na3, start_offset=1)
            b_cdr3_full = self._cdr3_span_vtoken(nb3, start_offset=4 + na3)
            b_cdr3_solo = self._cdr3_span_vtoken(nb3, start_offset=1)
        else:
            a_cdr3      = self._cdr3_span(*na, start_offset=1)
            b_cdr3_full = self._cdr3_span(*nb, start_offset=4 + na[0] + na[1] + na[2])
            b_cdr3_solo = self._cdr3_span(*nb, start_offset=1)

        # Two independent full views: per-chain CDR3 censoring (independent apply draw per chain
        # AND per view) THEN MLM mask. v1 carries the MLM labels; both feed the paired InfoNCE.
        f1_ids, f1_mask = self._apply_censoring_cdr3_only(full_ids, full_mask, a_cdr3)
        f1_ids, f1_mask = self._apply_censoring_cdr3_only(f1_ids,   f1_mask,   b_cdr3_full)
        f2_ids, f2_mask = self._apply_censoring_cdr3_only(full_ids, full_mask, a_cdr3)
        f2_ids, f2_mask = self._apply_censoring_cdr3_only(f2_ids,   f2_mask,   b_cdr3_full)
        full_ids_v1, full_labels_v1 = self._apply_mlm(f1_ids, f1_mask)
        full_ids_v2, full_labels_v2 = self._apply_mlm(f2_ids, f2_mask)

        # Beta single-chain view (chain-drop student -> pulled to the pair): censor own CDR3
        # (independent draw) then MLM (labels discarded -- feeds z only).
        b_ids_c, b_mask_c = self._apply_censoring_cdr3_only(beta_ids, beta_mask, b_cdr3_solo)
        beta_ids_m, _ = self._apply_mlm(b_ids_c, b_mask_c)

        item = {
            "full_input_ids_v1": full_ids_v1, "full_labels_v1": full_labels_v1,
            "full_input_ids_v2": full_ids_v2, "full_labels_v2": full_labels_v2,
            # Per-view CENSORED masks for Stage 2. "full_attention_mask" (= pre-censor) is kept
            # for the uncensored exp9 consumer (tcr_bert_joint_pretrain), which shares one mask.
            "full_attention_mask":    full_mask,
            "full_attention_mask_v1": f1_mask,
            "full_attention_mask_v2": f2_mask,
            "full_token_type_ids": full_token_type_ids,
            "full_position_ids":   full_position_ids,
            "beta_input_ids":  beta_ids_m,  "beta_attention_mask":  b_mask_c,
            "beta_token_type_ids":  torch.ones(self.max_len, dtype=torch.long),
            "beta_position_ids":   beta_position_ids,
        }

        if self.alpha_simcse:
            # Alpha DETACHED from the pair: instead of one chain-drop view pulled to z_pair, emit TWO
            # independent alpha views (own SimCSE) -- each an independent CDR3 censor + MLM (labels
            # discarded). The loss pulls the two views together (InfoNCE), so alpha learns its own
            # representation. Positions are the fixed pre-censor ones (shared across both views).
            a1_c, a1_m = self._apply_censoring_cdr3_only(alpha_ids, alpha_mask, a_cdr3)
            a2_c, a2_m = self._apply_censoring_cdr3_only(alpha_ids, alpha_mask, a_cdr3)
            alpha_ids_v1, _ = self._apply_mlm(a1_c, a1_m)
            alpha_ids_v2, _ = self._apply_mlm(a2_c, a2_m)
            item.update({
                "alpha_input_ids_v1": alpha_ids_v1, "alpha_attention_mask_v1": a1_m,
                "alpha_input_ids_v2": alpha_ids_v2, "alpha_attention_mask_v2": a2_m,
                "alpha_token_type_ids": torch.zeros(self.max_len, dtype=torch.long),
                "alpha_position_ids":  alpha_position_ids,
            })
        else:
            # Legacy: single alpha chain-drop view (pulled to the pair, like beta).
            a_ids_c, a_mask_c = self._apply_censoring_cdr3_only(alpha_ids, alpha_mask, a_cdr3)
            alpha_ids_m, _ = self._apply_mlm(a_ids_c, a_mask_c)
            item.update({
                "alpha_input_ids": alpha_ids_m, "alpha_attention_mask": a_mask_c,
                "alpha_token_type_ids": torch.zeros(self.max_len, dtype=torch.long),
                "alpha_position_ids":  alpha_position_ids,
            })

        return item


class TCRJointSingleChainLabeledDataset(Dataset):
    """
    Labeled single-chain dataset in the joint-encoder format. Used for downstream
    supervised tasks (CD4/CD8, etc.) where input is one chain (alpha or beta) and the
    label is per-sequence.

    Format mirrors TCRJointDataset._chain_str and benchmark_utils._joint_single_text:
    real sep_token "|" between CDRs, no string chain markers. Chain identity is carried
    by token_type_ids = 0 (alpha) or 1 (beta) across all positions.
    """

    def __init__(self, cdr1: List[str], cdr2: List[str], cdr3: List[str],
                 labels: np.ndarray, tokenizer, max_len: int = 64,
                 chain: str = "beta"):
        if chain not in ("alpha", "beta"):
            raise ValueError(f"chain must be 'alpha' or 'beta', got {chain!r}")
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chain = chain
        self.type_id_value = 0 if chain == "alpha" else 1
        self.sep = tokenizer.sep_token

    def __len__(self) -> int:
        return len(self.cdr3)

    def __getitem__(self, idx: int) -> dict:
        sep = self.sep
        text = (
            " ".join(str(c) for c in self.cdr1[idx]) + " " + sep + " " +
            " ".join(str(c) for c in self.cdr2[idx]) + " " + sep + " " +
            " ".join(str(c) for c in self.cdr3[idx])
        )
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = torch.full((self.max_len,), self.type_id_value, dtype=torch.long)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label":          torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        }


class TCRBetaOnlyMLMDataset(Dataset):
    """
    Two-view beta-only dataset for TCRFoundation Stage 1 (MLM + InfoNCE).

    Format mirrors the joint-encoder single-chain layout:
        [CLS] cdr1 | cdr2 | cdr3 [SEP]
    with `token_type_ids = 1` across all attention-active positions.

    Per __getitem__: two independent views of the same clonotype. Each view is built
    as (probabilistic SCEPTR-style residue censoring on CDR3) then (MLM masking).
    Only view 1 carries MLM labels; view 2 is for the autocontrastive InfoNCE only.

    Censoring is applied to CDR3 only (germline CDR1/CDR2 are V-gene-determined and
    never censored). Apply probability is per-view: with prob `censor_apply_prob`
    apply censoring, else leave the view pristine. This 50/50 mix keeps the MLM head
    calibrated for full-length contexts (B1 PLL evaluation uses uncensored input).
    """

    def __init__(self, cdr1: List[str], cdr2: List[str], cdr3: List[str],
                 tokenizer, max_len: int = 64,
                 mlm_prob: float = 0.15,
                 censor_apply_prob: float = 0.5,
                 censor_res_prob: float = 0.20,
                 input_format: str = "cdr123",
                 beta_vtoken: List[str] = None):
        self.cdr1 = cdr1
        self.cdr2 = cdr2
        self.cdr3 = cdr3
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.censor_apply_prob = censor_apply_prob
        self.censor_res_prob = censor_res_prob
        self.sep = tokenizer.sep_token
        self.special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        }
        self.vocab_size = tokenizer.vocab_size
        # ---- V-token mode: "[V:gene] | CDR3" (same grammar as the paired encoder), so beta-only and
        #      paired stay in ONE input space. The V-token is germline -> protected from MLM. ----
        if input_format not in ("cdr123", "vtoken"):
            raise ValueError(f"Unknown input_format: {input_format}")
        self.input_format = input_format
        self.beta_v = beta_vtoken
        if input_format == "vtoken":
            if beta_vtoken is None:
                raise ValueError("input_format='vtoken' requires beta_vtoken")
            vtoks = set(beta_vtoken) | {"[V_UNK]"}
            self.special_ids |= {tokenizer.convert_tokens_to_ids(t) for t in vtoks}

    def __len__(self) -> int:
        return len(self.cdr3)

    def _format_text(self, idx: int) -> Tuple[str, int, int, int]:
        """Build the single-chain text and return it together with the token counts
        of CDR1, CDR2, CDR3 (needed to locate the CDR3 span for censoring). In vtoken mode
        the text is "[V:gene] | CDR3" and (n1, n2) are unused for the span (see _cdr3_span)."""
        c3 = " ".join(str(c) for c in self.cdr3[idx])
        n3 = len(self.cdr3[idx])
        if self.input_format == "vtoken":
            text = f"{self.beta_v[idx]} {self.sep} {c3}"
            return text, 0, 0, n3
        c1 = " ".join(str(c) for c in self.cdr1[idx])
        c2 = " ".join(str(c) for c in self.cdr2[idx])
        # Residue-level tokenizer: #tokens == #residues == len(cdr) (0 UNK verified), so skip the
        # 3 tokenize() calls and read lengths directly -- cheaper per item.
        n1, n2 = len(self.cdr1[idx]), len(self.cdr2[idx])
        text = f"{c1} {self.sep} {c2} {self.sep} {c3}"
        return text, n1, n2, n3

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _cdr3_span(self, n1: int, n2: int, n3: int) -> Tuple[int, int]:
        """Half-open token-index range [start, end) of the CDR3 segment in the tokenised sequence.
        cdr123 layout: [CLS] cdr1 | cdr2 | cdr3 [SEP] -> start = 1 (CLS) + n1 + 1 (SEP) + n2 + 1 (SEP).
        vtoken layout: [CLS] [V] | cdr3 [SEP] -> start = 3 (CLS + V-token + SEP). End = start + n3,
        capped at max_len."""
        start = 3 if self.input_format == "vtoken" else (1 + n1 + 1 + n2 + 1)
        end = min(self.max_len, start + n3)
        start = min(start, self.max_len)
        return start, end

    def _apply_censoring_cdr3_only(self, input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    cdr3_span: Tuple[int, int]
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """With prob censor_apply_prob apply censoring to this view, else return as-is.
        When applied, drop each CDR3 residue independently with prob censor_res_prob:
        replace the token with PAD and zero its attention mask. Germline CDR1/CDR2
        positions are NEVER touched."""
        if np.random.random() >= self.censor_apply_prob:
            return input_ids.clone(), attention_mask.clone()

        new_ids = input_ids.clone()
        new_mask = attention_mask.clone()
        pad_id = self.tokenizer.pad_token_id
        start, end = cdr3_span

        for pos in range(start, end):
            if new_mask[pos] == 1 and new_ids[pos].item() not in self.special_ids:
                if np.random.random() < self.censor_res_prob:
                    new_ids[pos] = pad_id
                    new_mask[pos] = 0
        return new_ids, new_mask

    def _apply_mlm(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, -100)
        candidates = [
            i for i in range(len(input_ids))
            if attention_mask[i] == 1 and input_ids[i].item() not in self.special_ids
        ]
        if not candidates:
            return input_ids.clone(), labels

        n_mask = max(1, int(len(candidates) * self.mlm_prob))
        chosen = np.random.choice(candidates, size=n_mask, replace=False)

        masked_ids = input_ids.clone()
        for pos in chosen:
            labels[pos] = masked_ids[pos]
            r = np.random.random()
            if r < 0.8:
                masked_ids[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                masked_ids[pos] = np.random.randint(0, self.vocab_size)
            # else: keep original (10%)
        return masked_ids, labels

    def __getitem__(self, idx: int) -> dict:
        text, n1, n2, n3 = self._format_text(idx)
        base_ids, base_mask = self._tokenize(text)
        cdr3_span = self._cdr3_span(n1, n2, n3)

        # Two independent (censoring, MLM) pipelines. View 1 carries labels;
        # view 2 is for the autocontrastive pair only.
        v1_ids_c, v1_mask = self._apply_censoring_cdr3_only(base_ids, base_mask, cdr3_span)
        v2_ids_c, v2_mask = self._apply_censoring_cdr3_only(base_ids, base_mask, cdr3_span)
        v1_ids, v1_labels = self._apply_mlm(v1_ids_c, v1_mask)
        v2_ids, _         = self._apply_mlm(v2_ids_c, v2_mask)

        # Per-chain layout is built on the PRE-censor span (base_mask), so that censoring
        # (which only masks slots) leaves survivor positions intact and padding stays at 0.
        span = int(base_mask.sum())
        position_ids = torch.zeros(self.max_len, dtype=torch.long)
        position_ids[:span] = torch.arange(span)
        # token_type_ids = 1 (beta identity) across the chain span; pad stays 0
        type_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids[:span] = 1

        return {
            "input_ids_v1":      v1_ids,
            "attention_mask_v1": v1_mask,
            "token_type_ids_v1": type_ids,
            "position_ids_v1":   position_ids,
            "labels_v1":         v1_labels,
            "input_ids_v2":      v2_ids,
            "attention_mask_v2": v2_mask,
            "token_type_ids_v2": type_ids.clone(),
            "position_ids_v2":   position_ids.clone(),
        }


def load_emerson_beta_pretrain(data_path: str) -> dict:
    """
    Load a pre-built Emerson beta-chain pretrain parquet for TCRFoundation Stage 1.

    The carving (uniform / freq-weighted train + holdouts) is done upstream by
    scripts/utils/build_pretrain_targets.py, which guarantees train and holdouts are
    disjoint by construction. This loader just reads the chosen train file and
    returns CDR lists ready for TCRBetaOnlyMLMDataset.

    For TCRFoundation Stage 1 the recommended train file is
    `emerson_uniform_train.parquet` (each unique clonotype appears once -- matches
    the universal-donor view from the spec). The corresponding B1 evaluation set is
    `emerson_uniform_holdout.parquet`, loaded separately by the evaluation script.

    Args:
        data_path: Path to a pre-built train parquet (uniform or freq-weighted).

    Returns:
        Dict with keys {cdr1, cdr2, cdr3, v_gene, n_donors} -- lists of length n_rows.
    """
    print(f"Loading Emerson pretrain corpus from {data_path} ...")
    df = pd.read_parquet(
        data_path,
        columns=["cdr1", "cdr2", "cdr3aa", "v_gene", "n_donors"],
    )
    print(f"  rows: {len(df):,}")
    print(f"  n_donors range: [{df['n_donors'].min()}, {df['n_donors'].max()}]")
    return {
        "cdr1":     df["cdr1"].tolist(),
        "cdr2":     df["cdr2"].tolist(),
        "cdr3":     df["cdr3aa"].tolist(),
        "v_gene":   df["v_gene"].tolist(),
        "n_donors": df["n_donors"].astype(int).tolist(),
    }
