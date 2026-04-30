"""
weights.py
==========
Step 1.2 of JOLI-Kallisto Phase 1: compute effective lengths and per-EC
transcript weights for the EM algorithm.

Mirrors kallisto's weights.cpp logic. Two modes:

  "uniform"  (Phase 1 default):
      eff_lens[t] = 1.0 for all transcripts.
      Equivalent to kallisto's behaviour when no fragment length distribution
      is available (--long mode on TCC-only input).

  "kallisto" (Phase 2+):
      eff_lens[t] = max(1.0, transcript_length[t] - mean_frag_len + 1)
      Uses actual transcript lengths and a mean fragment length.
      Guard: if the result < 1.0, falls back to transcript_length[t].

EC weights:
  For each EC, for each transcript t in that EC:
      ec_weights[ec][i] = 1.0 / eff_lens[t]
  where i is the position of t within the EC's transcript list.

  This matches kallisto's weight_map_ construction in weights.cpp.

Inputs:
  - TCCData   : from load_tcc.py
  - transcript_lengths : optional np.ndarray, shape (n_transcripts,) [bp]
  - mean_frag_len      : optional float, mean fragment length [bp]
  - mode               : "uniform" | "kallisto"

Outputs:
  - WeightData dataclass:
      eff_lens    : np.ndarray, shape (n_transcripts,) -- effective length per tx
      ec_weights  : list[np.ndarray]                  -- ec_weights[ec][i] = weight
                                                          for i-th tx in that EC
"""

from dataclasses import dataclass

import numpy as np

from load_tcc import TCCData, FLENS_SENTINEL


# ============================================================
# Data container
# ============================================================

@dataclass
class WeightData:
    """
    Container for computed effective lengths and EC-level transcript weights.

    Attributes:
        eff_lens   (np.ndarray, float64, shape [n_transcripts]):
                    Effective length for each transcript. In "uniform" mode
                    all values are 1.0. In "kallisto" mode: max(1, len - mfl + 1).

        ec_weights (list[np.ndarray], length n_ecs):
                    ec_weights[ec_id] is a float64 array of length
                    len(ec_transcripts[ec_id]). Element i is the weight
                    (1/eff_len) for the i-th transcript in that EC.
                    This is the direct input to the EM E-step denominator.
    """
    eff_lens: np.ndarray
    ec_weights: list

    def __repr__(self) -> str:
        n_tx = len(self.eff_lens) if self.eff_lens is not None else 0
        n_ec = len(self.ec_weights) if self.ec_weights is not None else 0
        return f"WeightData(n_transcripts={n_tx}, n_ecs={n_ec})"


# ============================================================
# Effective length computation
# ============================================================

def _compute_eff_lens_uniform(n_transcripts: int) -> np.ndarray:
    """
    Uniform mode: every transcript gets effective length 1.0.

    Equivalent to treating all transcripts equally, which matches kallisto's
    behaviour when no fragment length distribution (FLD) is provided for
    long-read TCC input.

    Args:
        n_transcripts : int -- Total number of transcripts.

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- all ones.
    """
    return np.ones(n_transcripts, dtype=np.float64)


def _compute_eff_lens_kallisto(
    transcript_lengths: np.ndarray,
    mean_frag_len: float,
) -> np.ndarray:
    """
    Kallisto mode: eff_len[t] = max(1.0, length[t] - mean_frag_len + 1).

    Guard (matches kallisto weights.cpp line ~70):
      if length[t] - mean_frag_len + 1 < 1.0:
          eff_lens[t] = length[t]   (use raw length as fallback)

    Args:
        transcript_lengths : np.ndarray (float64, shape [n_transcripts]) -- lengths in bp.
        mean_frag_len      : float -- Mean fragment length in bp.

    Returns:
        np.ndarray (float64, shape [n_transcripts]) -- Effective lengths.
    """
    raw = transcript_lengths - mean_frag_len + 1.0
    eff = np.where(raw >= 1.0, raw, transcript_lengths)
    return eff.astype(np.float64)


# ============================================================
# EC weight construction
# ============================================================

def _build_ec_weights(
    ec_transcripts: list,
    eff_lens: np.ndarray,
) -> list:
    """
    Build per-EC weight arrays from effective lengths.

    For each EC, creates a float64 array where element i =
    1.0 / eff_lens[ec_transcripts[ec][i]].

    This is the weight used in the EM E-step:
      denom = sum( theta[t] * ec_weights[ec][i]  for i, t in enumerate(EC) )

    Args:
        ec_transcripts : list[list[int]] -- transcript indices per EC.
        eff_lens       : np.ndarray      -- effective length per transcript.

    Returns:
        list[np.ndarray] -- ec_weights[ec_id][i] = 1 / eff_lens[tx_i].
    """
    ec_weights = []
    for txs in ec_transcripts:
        # Gather effective lengths for transcripts in this EC, then invert
        lens = eff_lens[txs]          # shape (len(txs),)
        weights = 1.0 / lens          # element-wise reciprocal
        ec_weights.append(weights)
    return ec_weights


# ============================================================
# Main entry point
# ============================================================

def compute_weights(
    tcc_data: TCCData,
    transcript_lengths: np.ndarray = None,
    mean_frag_len: float = None,
    flens: np.ndarray = None,
    mode: str = "uniform",
) -> WeightData:
    """
    Compute effective lengths and EC transcript weights.

    Args:
        tcc_data           : TCCData    -- Loaded TCC data (from load_tcc.py).
        transcript_lengths : np.ndarray -- Shape (n_transcripts,), lengths in bp.
                                          Used for mode="kallisto" when flens is None.
        mean_frag_len      : float      -- Mean fragment length in bp.
                                          Used for mode="kallisto" when flens is None.
        flens              : np.ndarray -- Shape (n_transcripts,), effective lengths
                                          loaded directly from flens.txt (Fix B).
                                          When provided for mode="kallisto", takes
                                          priority over transcript_lengths/mean_frag_len.
        mode               : str        -- "uniform" (Phase 1) or "kallisto" (Phase 2+).

    Returns:
        WeightData -- Contains eff_lens and ec_weights arrays.

    Raises:
        ValueError : If mode="kallisto" but neither flens nor transcript_lengths
                     (+ mean_frag_len) are provided.
        ValueError : If provided arrays length doesn't match n_transcripts.
    """
    n_transcripts = len(tcc_data.transcript_names)
    print(f"\n[weights] Computing weights (mode='{mode}', "
          f"n_transcripts={n_transcripts}, n_ecs={len(tcc_data.ec_transcripts)})")

    # --- Compute effective lengths ---
    if mode == "uniform":
        eff_lens = _compute_eff_lens_uniform(n_transcripts)
        print(f"  Uniform mode: eff_lens = 1.0 for all {n_transcripts} transcripts")

    elif mode == "kallisto":
        if flens is not None:
            # Fix B: use effective lengths loaded directly from flens.txt.
            # These are pre-computed by kallisto quant-tcc (one value per transcript).
            if len(flens) != n_transcripts:
                raise ValueError(
                    f"flens has {len(flens)} entries but n_transcripts={n_transcripts}."
                )
            eff_lens = flens.astype(np.float64)
            n_sentinel = int((eff_lens >= FLENS_SENTINEL).sum())
            print(f"  Kallisto mode (flens): loaded from flens.txt")
            print(f"  eff_lens: min={eff_lens[eff_lens < FLENS_SENTINEL].min():.1f}, "
                  f"max={eff_lens[eff_lens < FLENS_SENTINEL].max():.1f}, "
                  f"sentinel (unobserved): {n_sentinel}")
        elif transcript_lengths is not None and mean_frag_len is not None:
            # Legacy path: compute from raw lengths + mean fragment length.
            if len(transcript_lengths) != n_transcripts:
                raise ValueError(
                    f"transcript_lengths has {len(transcript_lengths)} entries "
                    f"but n_transcripts={n_transcripts}."
                )
            eff_lens = _compute_eff_lens_kallisto(
                transcript_lengths.astype(np.float64), float(mean_frag_len)
            )
            print(f"  Kallisto mode (lengths): mean_frag_len={mean_frag_len:.1f}")
            print(f"  eff_lens: min={eff_lens.min():.1f}, "
                  f"max={eff_lens.max():.1f}, mean={eff_lens.mean():.1f}")
            n_guarded = int((transcript_lengths - mean_frag_len + 1 < 1.0).sum())
            print(f"  Guard applied (used raw length) for {n_guarded} transcripts")
        else:
            raise ValueError(
                "mode='kallisto' requires either:\n"
                "  flens=<np.ndarray from flens.txt>  (Fix B, recommended), or\n"
                "  transcript_lengths + mean_frag_len (legacy formula)."
            )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'uniform' or 'kallisto'.")

    # --- Build EC weight arrays ---
    ec_weights = _build_ec_weights(tcc_data.ec_transcripts, eff_lens)

    # Checkpoint: verify shapes
    assert len(ec_weights) == len(tcc_data.ec_transcripts), "EC weight count mismatch"
    for ec_id, (txs, w) in enumerate(zip(tcc_data.ec_transcripts, ec_weights)):
        assert len(w) == len(txs), \
            f"EC {ec_id}: weight array length {len(w)} != transcript count {len(txs)}"

    print(f"  EC weights computed for {len(ec_weights)} ECs")
    print(f"[weights] Done.")

    return WeightData(eff_lens=eff_lens, ec_weights=ec_weights)
