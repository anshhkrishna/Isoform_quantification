"""
output_writer.py
================
Step 1.4 of JOLI-Kallisto Phase 1: write abundance.tsv identical to
kallisto's PlaintextWriter.cpp output format.

Output file columns (tab-separated, matches kallisto abundance.tsv exactly):
  target_id   : transcript name
  length      : transcript length in bp (0 if unavailable in Phase 1)
  eff_length  : effective length used by EM
  est_counts  : raw expected read counts from EM = alpha[t]
                (multi-tx fractional counts + single-tx raw counts)
  tpm         : transcripts per million = (alpha[t]/eff_len[t]) / sum(alpha/eff_len) * 1e6

All transcripts are written (including those with alpha=0), sorted by
transcript index — same ordering as kallisto.

TPM formula (from kallisto PlaintextWriter.cpp compute_rho()):
  rho[t]  = alpha[t] / eff_len[t]
  tpm[t]  = rho[t] / sum(rho) * 1e6

Note: alpha here is EMResult.alpha (raw expected counts, not normalized).
This matches kallisto's alpha_ after the EM loop + single-tx additions.

Inputs:
  - alpha             : np.ndarray from EMResult.alpha (raw expected counts)
  - eff_lens          : np.ndarray from WeightData.eff_lens
  - transcript_names  : list[str] from TCCData.transcript_names
  - output_path       : str -- where to write abundance.tsv
  - transcript_lengths: np.ndarray (optional) -- raw lengths in bp; zeros if absent

Outputs:
  - abundance.tsv written to output_path
  - Returns dict of summary statistics for checkpointing
"""

import os

import numpy as np


def write_abundance(
    alpha: np.ndarray,
    eff_lens: np.ndarray,
    transcript_names: list,
    output_path: str,
    transcript_lengths: np.ndarray = None,
) -> dict:
    """
    Compute TPM and estimated counts from raw expected counts, then write abundance.tsv.

    Column format (tab-separated, identical to kallisto):
      target_id  length  eff_length  est_counts  tpm

    TPM computation (mirrors kallisto PlaintextWriter.cpp compute_rho()):
      rho[t]  = alpha[t] / eff_lens[t]
      tpm[t]  = rho[t] / sum(rho) * 1e6

    Estimated counts:
      est_counts[t] = alpha[t]  (raw expected counts from EM; matches kallisto alpha_)

    Args:
        alpha             : np.ndarray (float64, shape [n_transcripts])
                            Raw expected counts from EMResult.alpha.
                            Multi-tx fractional counts + single-tx raw counts.
                            Does NOT need to sum to 1.
        eff_lens          : np.ndarray (float64, shape [n_transcripts])
                            Effective lengths from WeightData.eff_lens.
        transcript_names  : list[str]
                            Transcript name at each index (from TCCData.transcript_names).
        output_path       : str
                            Full path to output file (e.g. .../abundance.tsv).
        transcript_lengths: np.ndarray (float64, shape [n_transcripts]), optional
                            Raw transcript lengths in bp. Written to the 'length' column.
                            If None, the 'length' column is written as 0.

    Returns:
        dict with summary statistics:
          "n_transcripts"       : total transcripts written
          "n_nonzero"           : transcripts with tpm > 0
          "total_tpm"           : sum of TPM values (should be ~1e6)
          "total_est_counts"    : sum of alpha (total expected reads assigned)
          "output_path"         : path written to

    Raises:
        ValueError : If array lengths don't match n_transcripts.
    """
    n_tx = len(transcript_names)

    # --- Input validation ---
    if len(alpha) != n_tx:
        raise ValueError(f"alpha length {len(alpha)} != n_transcripts {n_tx}")
    if len(eff_lens) != n_tx:
        raise ValueError(f"eff_lens length {len(eff_lens)} != n_transcripts {n_tx}")
    if transcript_lengths is not None and len(transcript_lengths) != n_tx:
        raise ValueError(
            f"transcript_lengths length {len(transcript_lengths)} != n_transcripts {n_tx}"
        )

    # --- Compute rho (length-normalized expected counts; matches kallisto compute_rho()) ---
    # Sentinel values (e.g. UINT32_MAX from flens.txt) produce near-zero weights;
    # guard against true zero eff_lens to avoid division by zero.
    zero_eff_mask = eff_lens <= 0
    if zero_eff_mask.any():
        n_zero = int(zero_eff_mask.sum())
        print(f"[output_writer] WARNING: {n_zero} transcript(s) have eff_len <= 0 "
              f"(not a sentinel — genuinely missing/short). Falling back to eff_len=1.0. "
              f"TPM for these transcripts may be inflated.")
    safe_eff_lens = np.where(eff_lens > 0, eff_lens, 1.0)
    rho = alpha / safe_eff_lens          # shape (n_tx,)

    # --- TPM: normalize rho to parts per million ---
    rho_sum = rho.sum()
    if rho_sum > 0:
        tpm = rho / rho_sum * 1e6
    else:
        tpm = np.zeros(n_tx, dtype=np.float64)

    # --- Estimated counts: alpha directly (raw expected reads per transcript) ---
    est_counts = alpha                   # shape (n_tx,)

    # --- Lengths ---
    if transcript_lengths is not None:
        lengths = transcript_lengths.astype(np.float64)
    else:
        lengths = np.zeros(n_tx, dtype=np.float64)

    # --- Write output ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\n[output_writer] Writing abundance.tsv -> {output_path}")

    with open(output_path, "w") as fh:
        # Header — identical to kallisto
        fh.write("target_id\tlength\teff_length\test_counts\ttpm\n")

        for i in range(n_tx):
            fh.write(
                f"{transcript_names[i]}\t"
                f"{int(lengths[i])}\t"
                f"{eff_lens[i]:.6f}\t"
                f"{est_counts[i]:.6f}\t"
                f"{tpm[i]:.6f}\n"
            )

    # --- Summary stats / checkpoint ---
    n_nonzero    = int((tpm > 0).sum())
    total_tpm    = float(tpm.sum())
    total_counts = float(est_counts.sum())

    print(f"  Transcripts written:  {n_tx}")
    print(f"  Non-zero TPM:         {n_nonzero}")
    print(f"  Sum TPM:              {total_tpm:.2f}  (expected ~1,000,000)")
    print(f"  Sum est_counts:       {total_counts:.2f}  (= total reads assigned)")

    return {
        "n_transcripts":    n_tx,
        "n_nonzero":        n_nonzero,
        "total_tpm":        total_tpm,
        "total_est_counts": total_counts,
        "output_path":      output_path,
    }
