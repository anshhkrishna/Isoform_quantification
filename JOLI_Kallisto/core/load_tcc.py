"""
load_tcc.py
===========
Load kallisto/bustools output files into Python data structures for JOLI EM.

Inputs (from a single sample's kallisto_output/<sample_stem>/ directory):
  - count.mtx       : MatrixMarket sparse matrix (rows=samples, cols=ECs, 1-indexed)
  - matrix.ec       : EC definitions, format: "ec_id<TAB>tx_idx1,tx_idx2,..."
  - transcripts.txt : One transcript name per line (0-indexed, matches tx indices in matrix.ec)
  - run_info.json   : Kallisto run metadata (n_processed, n_pseudoaligned, n_targets)

Outputs (returned as TCCData dataclass):
  - ec_counts       : np.ndarray, shape (n_ecs,)  -- read count per EC
  - ec_transcripts  : list[list[int]]             -- transcript indices per EC
  - transcript_names: list[str]                   -- transcript name at each index
  - total_reads     : int                         -- total pseudoaligned reads (for est_counts)
  - n_targets       : int                         -- total number of transcripts in index

Usage:
  from load_tcc import load_tcc_data
  data = load_tcc_data("/path/to/kallisto_output/sample_stem/")
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.io import mmread


# Sentinel value used by kallisto quant-tcc in flens.txt for transcripts
# that had no observed reads. UINT32_MAX = 4294967295.
# Defined once here and imported by weights.py to avoid divergence.
FLENS_SENTINEL = 4294967295.0


# ============================================================
# Data container
# ============================================================

@dataclass
class TCCData:
    """
    Container for all data loaded from a single-sample kallisto/bustools output.

    Attributes:
        ec_counts        (np.ndarray, int64, shape [n_ecs])       : Read count per EC.
        ec_transcripts   (list[list[int]], length n_ecs)           : 0-based transcript
                                                                     indices per EC.
        transcript_names (list[str], length n_transcripts)         : Transcript name at
                                                                     each 0-based index.
        total_reads      (int)                                     : Total pseudoaligned
                                                                     reads (n_pseudoaligned
                                                                     from run_info.json).
        n_targets        (int)                                     : Total transcripts in
                                                                     the kallisto index.
        sample_dir       (str)                                     : Source directory path.
    """
    ec_counts: np.ndarray
    ec_transcripts: list
    transcript_names: list
    total_reads: int
    n_targets: int
    sample_dir: str


# ============================================================
# Individual file parsers
# ============================================================

def _load_count_mtx(mtx_path: str, n_ecs: int) -> np.ndarray:
    """
    Parse count.mtx (MatrixMarket sparse format) into a 1-D array of EC counts.

    In bulk single-sample mode, bustools count outputs a matrix with:
      - 1 row  (the single sample / barcode)
      - N cols (one per EC, 1-indexed)
    Each non-zero entry: row col count  →  ec_id = col - 1

    Args:
        mtx_path : str  -- Full path to count.mtx
        n_ecs    : int  -- Number of ECs (from matrix.ec)

    Returns:
        np.ndarray (int64, shape [n_ecs]) -- Count for each EC; 0 if not in matrix.
    """
    print(f"  Loading count matrix: {mtx_path}")

    # mmread returns a sparse COO matrix; convert to dense for small-to-medium datasets.
    # For very large datasets (>500k ECs) consider keeping sparse.
    sparse_mat = mmread(mtx_path).tocsr()

    # In bulk mode the matrix is (1 sample × n_ecs).
    # We want a 1-D array indexed by ec_id.
    n_rows, n_cols = sparse_mat.shape
    print(f"    Matrix shape: {n_rows} rows (samples) x {n_cols} cols (ECs)")

    if n_rows == 1:
        # Standard bulk single-sample: flatten to 1-D
        ec_counts = np.asarray(sparse_mat.todense()).flatten().astype(np.int64)
    else:
        # Multi-sample TCC (future): sum across samples as a fallback
        # This branch is not used in Phase 1 and will be replaced in Phase 3.
        print(f"    WARNING: {n_rows} rows found; summing across rows for Phase 1.")
        ec_counts = np.asarray(sparse_mat.sum(axis=0)).flatten().astype(np.int64)

    # Pad or trim to exactly n_ecs entries
    if len(ec_counts) < n_ecs:
        ec_counts = np.pad(ec_counts, (0, n_ecs - len(ec_counts)))
    elif len(ec_counts) > n_ecs:
        ec_counts = ec_counts[:n_ecs]

    total_count = int(ec_counts.sum())
    print(f"    Total read counts in matrix: {total_count}")
    return ec_counts


def _load_matrix_ec(ec_path: str) -> list:
    """
    Parse matrix.ec into a list of transcript-index lists.

    File format (one EC per line):
        ec_id<TAB>tx_idx1,tx_idx2,...

    Lines are expected in order (ec_id 0, 1, 2, ...).

    Args:
        ec_path : str -- Full path to matrix.ec

    Returns:
        list[list[int]] -- ec_transcripts[ec_id] = sorted list of 0-based transcript indices.
    """
    print(f"  Loading EC definitions: {ec_path}")
    ec_transcripts = []

    with open(ec_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected format in matrix.ec line: '{line}'. "
                    f"Expected 'ec_id<TAB>tx1,tx2,...'"
                )
            # ec_id = int(parts[0])  # we assume lines are in order; index = position
            tx_indices = [int(t) for t in parts[1].split(",")]
            ec_transcripts.append(tx_indices)

    print(f"    Loaded {len(ec_transcripts)} equivalence classes")

    # Checkpoint: report EC size distribution
    sizes = [len(ec) for ec in ec_transcripts]
    n_single = sum(1 for s in sizes if s == 1)
    n_multi = len(sizes) - n_single
    print(f"    Single-transcript ECs: {n_single}  |  Multi-transcript ECs: {n_multi}")

    return ec_transcripts


def _load_transcripts(tx_path: str) -> list:
    """
    Parse transcripts.txt into an ordered list of transcript names.

    File format: one transcript name per line, 0-indexed by line position.

    Args:
        tx_path : str -- Full path to transcripts.txt

    Returns:
        list[str] -- transcript_names[tx_id] = transcript name string.
    """
    print(f"  Loading transcript names: {tx_path}")
    transcript_names = []

    with open(tx_path, "r") as fh:
        for line in fh:
            name = line.strip()
            if name:
                transcript_names.append(name)

    print(f"    Loaded {len(transcript_names)} transcript names")
    return transcript_names


def _load_run_info(json_path: str) -> dict:
    """
    Parse run_info.json for kallisto run metadata.

    Args:
        json_path : str -- Full path to run_info.json

    Returns:
        dict with keys: n_processed, n_pseudoaligned, n_unique, n_targets, etc.
        Returns default dict if file is missing (non-fatal).
    """
    if not os.path.exists(json_path):
        print(f"  WARNING: run_info.json not found at {json_path}. Using defaults.")
        return {"n_pseudoaligned": 0, "n_targets": 0}

    print(f"  Loading run info: {json_path}")
    with open(json_path, "r") as fh:
        info = json.load(fh)

    print(f"    n_processed:    {info.get('n_processed', 'N/A')}")
    print(f"    n_pseudoaligned:{info.get('n_pseudoaligned', 'N/A')}")
    print(f"    n_targets:      {info.get('n_targets', 'N/A')}")
    return info


# ============================================================
# Fragment / effective length loader (Fix B)
# ============================================================

def load_flens(sample_dir: str, n_transcripts: int) -> np.ndarray:
    """
    Load flens.txt — per-transcript effective lengths computed by kallisto quant-tcc.

    flens.txt is a space-separated flat list of one value per transcript.
    Sentinel value 4294967295 (UINT32_MAX) means no reads were observed for that
    transcript; 1/sentinel ≈ 0, giving near-zero EM weight (same behaviour as kallisto).

    Args:
        sample_dir     : str -- Directory containing flens.txt.
        n_transcripts  : int -- Expected number of transcripts (for validation).

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- Effective length per transcript.

    Raises:
        FileNotFoundError : If flens.txt does not exist in sample_dir.
        ValueError        : If the number of values does not match n_transcripts.
    """
    flens_path = os.path.join(sample_dir, "flens.txt")
    if not os.path.exists(flens_path):
        raise FileNotFoundError(
            f"flens.txt not found in {sample_dir}. "
            "It is produced by kallisto quant-tcc. "
            "Use --eff_len_mode uniform if it is unavailable."
        )

    print(f"  Loading fragment/effective lengths: {flens_path}")
    with open(flens_path, "r") as fh:
        raw = fh.read()

    values = np.array(raw.split(), dtype=np.float64)

    if len(values) != n_transcripts:
        raise ValueError(
            f"flens.txt has {len(values)} entries but n_transcripts={n_transcripts}. "
            "Ensure flens.txt was generated with the same index as transcripts.txt."
        )

    # Checkpoint: report sentinel count
    n_sentinel = int((values >= FLENS_SENTINEL).sum())
    print(f"    Loaded {len(values)} effective lengths "
          f"({n_sentinel} sentinel/unobserved transcripts)")

    return values


# ============================================================
# Main loader function
# ============================================================

def load_tcc_data(sample_dir: str) -> TCCData:
    """
    Load all bustools/kallisto output files from one sample directory.

    Expected files in sample_dir:
        count.mtx       -- Sparse EC count matrix (MatrixMarket)
        matrix.ec       -- EC-to-transcript-index mapping
        transcripts.txt -- Ordered list of transcript names
        run_info.json   -- Kallisto run metadata (optional but recommended)

    Args:
        sample_dir : str -- Path to <data_dir>/kallisto_output/<sample_stem>/

    Returns:
        TCCData -- Fully populated data container ready for JoliEM.

    Raises:
        FileNotFoundError : If count.mtx, matrix.ec, or transcripts.txt is missing.
        ValueError        : If the loaded data is internally inconsistent.
    """
    sample_dir = os.path.abspath(sample_dir)
    print(f"\n[load_tcc] Loading TCC data from: {sample_dir}")

    # --- Verify required files exist ---
    required = ["count.mtx", "matrix.ec", "transcripts.txt"]
    for fname in required:
        fpath = os.path.join(sample_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Required file missing: {fpath}\n"
                f"Run kallisto bus + bustools sort + bustools count first."
            )

    # --- Load each file ---
    ec_transcripts  = _load_matrix_ec(os.path.join(sample_dir, "matrix.ec"))
    transcript_names = _load_transcripts(os.path.join(sample_dir, "transcripts.txt"))
    run_info        = _load_run_info(os.path.join(sample_dir, "run_info.json"))
    ec_counts       = _load_count_mtx(os.path.join(sample_dir, "count.mtx"),
                                       n_ecs=len(ec_transcripts))

    # --- Consistency checks ---
    n_transcripts = len(transcript_names)
    n_ecs = len(ec_transcripts)

    # Every transcript index in matrix.ec must be within range
    max_tx_idx = max(
        (max(txs) for txs in ec_transcripts if txs),
        default=-1
    )
    if max_tx_idx >= n_transcripts:
        raise ValueError(
            f"matrix.ec references transcript index {max_tx_idx}, "
            f"but transcripts.txt only has {n_transcripts} entries."
        )

    # ec_counts length must match number of ECs
    if len(ec_counts) != n_ecs:
        raise ValueError(
            f"count.mtx has {len(ec_counts)} EC entries but matrix.ec has {n_ecs}."
        )

    # total_reads: prefer n_pseudoaligned from run_info; fall back to sum of ec_counts
    total_reads = int(run_info.get("n_pseudoaligned", 0))
    if total_reads == 0:
        total_reads = int(ec_counts.sum())
        print(f"  INFO: n_pseudoaligned not in run_info; using sum(ec_counts) = {total_reads}")

    n_targets = int(run_info.get("n_targets", n_transcripts))

    print(f"\n[load_tcc] Done.")
    print(f"  ECs loaded:          {n_ecs}")
    print(f"  Transcripts loaded:  {n_transcripts}")
    print(f"  Total reads:         {total_reads}")
    print(f"  ECs with count > 0:  {int((ec_counts > 0).sum())}")

    return TCCData(
        ec_counts=ec_counts,
        ec_transcripts=ec_transcripts,
        transcript_names=transcript_names,
        total_reads=total_reads,
        n_targets=n_targets,
        sample_dir=sample_dir,
    )
