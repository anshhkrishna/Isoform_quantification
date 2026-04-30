"""
main_joli.py
============
Step 1.5 — JOLI EM entry point (Python only, no subprocess calls).

NOTE: This script does NOT run kallisto bus or bustools.
It assumes count.mtx, matrix.ec, and transcripts.txt already exist in --sample_dir.
To run the full pipeline (bustools + JOLI EM), use run_joli_kallisto.sh instead.

Takes a directory already containing bustools TCC output files:
  count.mtx, matrix.ec, transcripts.txt, run_info.json

Runs ONLY the JOLI EM pipeline (Step 4):
  Step 1.1  load_tcc.py     : parse bustools output -> TCCData
  Step 1.2  weights.py      : compute effective lengths and EC weights
  Step 1.3  em_algorithm.py : run JoliEM on equivalence classes
  Step 1.4  output_writer.py: write abundance.tsv in kallisto format

Use this script when:
  - The bustools TCC files already exist (cache hit in run_joli_kallisto.sh)
  - You want to set breakpoints in the EM logic
  - You want to re-run the EM with different settings without re-running bustools

Inputs:
  --sample_dir   : directory with count.mtx, matrix.ec, transcripts.txt
  --output_dir   : where to write abundance.tsv (default: same as sample_dir)
  --eff_len_mode : "uniform" (Phase 1) | "kallisto" (Phase 2+)
  --max_em_rounds: max EM iterations (default: 10000)
  --min_rounds   : min EM iterations before convergence check (default: 50)
  --em_type           : "plain" (Phase 1) | "MAP" (Phase 2) | "VI" (Phase 4)
  --convergence_mode  : "kallisto" (raw count threshold, matches lr-kallisto exactly) |
                        "joli"     (normalized theta threshold, faster, for MAP/VI)

Outputs:
  <output_dir>/abundance.tsv  : per-transcript quantification
  <output_dir>/runtime.txt    : wall-clock time for this run

Run:
  conda activate NanoCount_5
  python main_joli.py --sample_dir /path/to/kallisto_output/sample_stem/
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

# Add core/ to path for library module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

from load_tcc import load_tcc_data, load_flens
from weights import compute_weights
from em_algorithm import JoliEM
from output_writer import write_abundance


# ============================================================
# CONFIG — default values for all CLI arguments; edit here or pass as flags
# ============================================================
DEFAULT_SAMPLE_DIR   = "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_fastq/PacBio/reads/long/downsampled/kallisto_output/toy"
DEFAULT_OUTPUT_DIR   = ""           # defaults to sample_dir if empty
DEFAULT_EFF_LEN_MODE = "uniform"    # "uniform" (Phase 1) | "kallisto" (Phase 2+)
DEFAULT_MAX_EM_ROUNDS = 10000
DEFAULT_MIN_ROUNDS    = 50
DEFAULT_EM_TYPE            = "plain"     # "plain" | "MAP" | "VI"
DEFAULT_CONVERGENCE_MODE   = "kallisto"  # "kallisto" (raw count threshold, matches LK)
                                         # "joli"     (normalized theta, faster, for MAP/VI)
SAVE_SNAPSHOTS             = False       # True = save theta snapshots every SNAPSHOT_INTERVAL
#                                        #   rounds to <output_dir>/theta_snapshots.pkl
#                                        #   Used by plot_convergence_animation.py
SNAPSHOT_INTERVAL          = 5          # save a snapshot every N EM rounds
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace -- parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="JOLI EM: isoform quantification from kallisto/bustools TCC files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir", default=DEFAULT_SAMPLE_DIR,
        help="Directory containing count.mtx, matrix.ec, transcripts.txt "
             "(output of bustools count)."
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write abundance.tsv. Defaults to --sample_dir if empty."
    )
    parser.add_argument(
        "--eff_len_mode", default=DEFAULT_EFF_LEN_MODE,
        choices=["uniform", "kallisto"],
        help="Effective length mode. 'uniform'=all 1.0 (Phase 1); "
             "'kallisto'=length-based (Phase 2+, requires transcript lengths)."
    )
    parser.add_argument(
        "--max_em_rounds", type=int, default=DEFAULT_MAX_EM_ROUNDS,
        help="Maximum EM iterations."
    )
    parser.add_argument(
        "--min_rounds", type=int, default=DEFAULT_MIN_ROUNDS,
        help="Minimum EM iterations before checking convergence (matches kallisto default)."
    )
    parser.add_argument(
        "--em_type", default=DEFAULT_EM_TYPE,
        choices=["plain", "MAP", "VI"],
        help="EM mode. 'plain'=standard EM (Phase 1); 'MAP'=Dirichlet MAP (Phase 2); "
             "'VI'=variational inference (Phase 4)."
    )
    parser.add_argument(
        "--convergence_mode", default=DEFAULT_CONVERGENCE_MODE,
        choices=["kallisto", "joli"],
        help="Convergence criterion. 'kallisto'=raw expected count threshold (matches "
             "lr-kallisto exactly, recommended for plain EM comparison); "
             "'joli'=normalized theta threshold (faster, recommended for MAP/VI)."
    )
    parser.add_argument(
        "--save_snapshots", default=str(SAVE_SNAPSHOTS).lower(),
        choices=["true", "false"],
        help="Save theta snapshots every --snapshot_interval rounds to theta_snapshots.pkl."
    )
    parser.add_argument(
        "--snapshot_interval", type=int, default=SNAPSHOT_INTERVAL,
        help="Save a theta snapshot every N EM rounds (only used when --save_snapshots true)."
    )
    return parser.parse_args()


def save_runtime(output_dir: str, elapsed: float) -> None:
    """
    Write elapsed wall-clock time to runtime.txt.

    Args:
        output_dir : str   -- Output directory.
        elapsed    : float -- Elapsed seconds.
    """
    path = os.path.join(output_dir, "runtime.txt")
    with open(path, "w") as fh:
        fh.write(f"{elapsed:.2f} seconds\n")
    print(f"[main_joli] Runtime saved: {elapsed:.2f}s -> {path}")


def main() -> None:
    """
    JOLI EM pipeline: load TCC -> compute weights -> run EM -> write abundance.tsv.
    """
    run_start = time.time()
    args = parse_args()

    # Resolve paths
    sample_dir = os.path.abspath(args.sample_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else sample_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("JOLI-Kallisto: main_joli.py")
    print("=" * 60)
    print(f"  sample_dir   : {sample_dir}")
    print(f"  output_dir   : {output_dir}")
    print(f"  eff_len_mode : {args.eff_len_mode}")
    print(f"  em_type      : {args.em_type}")
    print(f"  max_em_rounds    : {args.max_em_rounds}")
    print(f"  min_rounds       : {args.min_rounds}")
    print(f"  convergence_mode : {args.convergence_mode}")
    print(f"  save_snapshots   : {args.save_snapshots}")
    print(f"  snapshot_interval: {args.snapshot_interval}")
    print("=" * 60)

    # Resolve save_snapshots flag (CLI passes string "true"/"false")
    save_snapshots = args.save_snapshots.lower() == "true"

    # --- Step 1.1: Load TCC data ---
    tcc_data = load_tcc_data(sample_dir)

    # --- Step 1.2: Compute weights ---
    # Fix B: "kallisto" mode loads effective lengths from flens.txt (produced by
    # kallisto quant-tcc), matching kallisto's calc_eff_lens() exactly.
    if args.eff_len_mode == "kallisto":
        flens = load_flens(sample_dir, n_transcripts=len(tcc_data.transcript_names))
        weight_data = compute_weights(tcc_data, flens=flens, mode="kallisto")
    else:
        weight_data = compute_weights(tcc_data, mode=args.eff_len_mode)

    # --- Step 1.3: Run EM ---
    if args.em_type == "VI":
        raise NotImplementedError(
            "em_type='VI' (variational inference) is not yet implemented. "
            "Use --em_type plain or --em_type MAP."
        )

    # MAP mode: uniform Dirichlet prior (alpha[t] = 1.0 for all t)
    if args.em_type == "MAP":
        alpha_prior = np.ones(len(tcc_data.transcript_names), dtype=np.float64)
    else:
        alpha_prior = None  # plain EM

    em = JoliEM(tcc_data, weight_data)
    em_result = em.run(
        max_em_rounds     = args.max_em_rounds,
        min_rounds        = args.min_rounds,
        convergence_mode  = args.convergence_mode,
        alpha_prior       = alpha_prior,
        snapshot_interval = args.snapshot_interval if save_snapshots else 0,
    )

    print(f"\n[main_joli] EM finished: "
          f"rounds={em_result.n_rounds}, converged={em_result.converged}, "
          f"nonzero_tx={int((em_result.alpha > 0).sum())}")

    # --- Save theta snapshots (if enabled) ---
    if save_snapshots and em_result.snapshots:
        snap_path = os.path.join(output_dir, "theta_snapshots.pkl")
        with open(snap_path, "wb") as fh:
            pickle.dump({
                "transcript_names": tcc_data.transcript_names,
                "snapshots":        em_result.snapshots,   # list of (round_num, theta)
            }, fh)
        print(f"[main_joli] Snapshots saved ({len(em_result.snapshots)} frames): {snap_path}")

    # --- Step 1.4: Write abundance.tsv ---
    output_path = os.path.join(output_dir, "abundance.tsv")
    summary = write_abundance(
        alpha=em_result.alpha,
        eff_lens=weight_data.eff_lens,
        transcript_names=tcc_data.transcript_names,
        output_path=output_path,
    )

    # --- Save runtime ---
    elapsed = time.time() - run_start
    save_runtime(output_dir, elapsed)

    print(f"\n[main_joli] Done.")
    print(f"  abundance.tsv   : {output_path}")
    print(f"  non-zero TPM    : {summary['n_nonzero']} / {summary['n_transcripts']}")
    print(f"  total TPM       : {summary['total_tpm']:.2f}")
    print(f"  total est_counts: {summary['total_est_counts']:.2f}")
    print(f"  total elapsed   : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
