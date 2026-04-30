"""
main_pipeline.py
================
JOLI-Kallisto full pipeline orchestrator (Python entry point).

Runs the complete pipeline for one or more samples:
  Step 1  [optional, cached]: kallisto bus  -> output.bus
  Step 2  [optional, cached]: bustools sort -> sorted.bus
  Step 3  [optional, cached]: bustools count -> count.mtx, matrix.ec, transcripts.txt
  Step 4  [JOLI EM]         : load TCC + run EM -> abundance.tsv

Bustools cache behaviour:
  - Outputs from steps 1-3 are saved to:
      <reads_dir>/kallisto_output/<sample_stem>/
  - If count.mtx, matrix.ec, and transcripts.txt already exist there,
    steps 1-3 are skipped entirely.
  - This means re-runs (e.g. to test different EM settings) go straight
    to Step 4 without re-running the expensive pseudoalignment.

Inputs:
  - CONFIG section below (edit before running)
  - FASTQ/FASTA reads files per sample

Outputs (per sample, inside <output_base>/exprmnt_<timestamp>/<sample_name>/):
  - abundance.tsv       : JOLI EM quantification
  - (future steps will add est_counts, tpm columns)

Run locally:
  conda activate NanoCount_5
  python main_pipeline.py

Run on SLURM:
  bash submit_joli_pipeline.sh
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add core/ to path for library module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

from load_tcc import load_tcc_data      # Step 1.1 — data loader
from weights import compute_weights    # Step 1.2 — effective lengths and EC weights
from em_algorithm import JoliEM        # Step 1.3 — EM algorithm
from output_writer import write_abundance  # Step 1.4 — output writer

# ============================================================
# CONFIG — edit everything here; do not touch pipeline logic below
# ============================================================

# --- Tool paths ---
KALLISTO  = "/gpfs/commons/home/atalukder/RNA_Splicing/data/Shree_stuff/SOTA/lr-kallisto/kallisto/build/src/kallisto"
BUSTOOLS  = "bustools"   # assumes bustools is on PATH; use full path if not

# --- Reference files ---
INDEX_FILE = "/gpfs/commons/home/atalukder/RNA_Splicing/data/Shree_stuff/SOTA/lr-kallisto/new_index.idx"
T2G_FILE   = "/gpfs/commons/home/atalukder/RNA_Splicing/data/Shree_stuff/SOTA/lr-kallisto/t2g.txt"

# --- Run settings ---
READ_TYPE  = "long"    # "long" (PacBio/ONT) or "short" (paired Illumina)
PLATFORM   = "PacBio"  # used only when READ_TYPE == "long": "PacBio" or "ONT"
THREADS    = 32
THRESHOLD  = 0.8       # kallisto bus alignment threshold

# --- Output base ---
OUTPUT_BASE = "/gpfs/commons/home/atalukder/RNA_Splicing/files/results"

# --- Samples ---
# Each entry: (sample_name, reads_dir, reads_file)
# For paired short-read, provide two files: (sample_name, reads_dir, r1_file, r2_file)
SAMPLES = [
    ("ds52_250k_reads",
     "/gpfs/commons/groups/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_fastq/PacBio/reads/long/downsampled",
     "ds_52_furtherDownsampled.fastq"),
]

# --- EM settings (Phase 1: plain EM; later phases add MAP/VI/multi-sample) ---
EM_TYPE        = "plain"  # "plain" | "MAP" | "VI"
MAX_EM_ROUNDS  = 10000
EFF_LEN_MODE   = "uniform"  # "uniform" (Phase 1) | "kallisto" (Phase 2+)

# --- Comparison flag ---
# Set to True to also run the original `kallisto quant-tcc` for numeric comparison
RUN_KALLISTO_QUANT_TCC = False

# ============================================================
# END CONFIG
# ============================================================


# ---- Constants ----
CACHE_REQUIRED_FILES = ["count.mtx", "matrix.ec", "transcripts.txt"]


# ============================================================
# Logging setup
# ============================================================

def setup_logging(log_path: str) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and a log file.

    Args:
        log_path : str -- Full path to the log file.

    Returns:
        logging.Logger -- Configured logger.
    """
    logger = logging.getLogger("joli_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Utility helpers
# ============================================================

def make_timestamp() -> str:
    """Return a timestamp string in the project's standard format."""
    return datetime.now().strftime("exprmnt_%Y_%m_%d__%H_%M_%S")


def get_sample_stem(reads_file: str) -> str:
    """
    Derive the cache directory stem from the reads filename.
    Strips all extensions: 'ds_52.fastq.gz' -> 'ds_52'

    Args:
        reads_file : str -- Just the filename (not the full path).

    Returns:
        str -- Stem used as the cache subdirectory name.
    """
    stem = reads_file
    for ext in (".fastq.gz", ".fasta.gz", ".fastq", ".fasta", ".fa", ".fq"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def run_command(cmd: list, log: logging.Logger, cwd: str = None) -> None:
    """
    Run a shell command via subprocess, streaming output to the logger.

    Args:
        cmd : list[str] -- Command and arguments.
        log : Logger    -- Logger to write stdout/stderr to.
        cwd : str       -- Working directory (optional).

    Raises:
        subprocess.CalledProcessError -- If the command exits with non-zero status.
    """
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )
    for line in result.stdout.splitlines():
        log.debug(f"  | {line}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout)


# ============================================================
# Cache check
# ============================================================

def check_bustools_cache(cache_dir: str, log: logging.Logger) -> bool:
    """
    Check whether all required bustools output files already exist in cache_dir.

    This is checked BEFORE running kallisto bus + bustools steps.
    If all required files are present, the bustools pipeline is skipped entirely.

    Required files: count.mtx, matrix.ec, transcripts.txt

    Args:
        cache_dir : str    -- Path to <reads_dir>/kallisto_output/<sample_stem>/
        log       : Logger

    Returns:
        bool -- True if cache is complete (skip bustools); False if must run.
    """
    if not os.path.isdir(cache_dir):
        log.info(f"Cache miss (directory absent): {cache_dir}")
        return False

    missing = [f for f in CACHE_REQUIRED_FILES
               if not os.path.isfile(os.path.join(cache_dir, f))]
    if missing:
        log.info(f"Cache miss (missing files: {missing}): {cache_dir}")
        return False

    log.info(f"Cache hit — skipping bustools pipeline: {cache_dir}")
    return True


# ============================================================
# Bustools pipeline (Steps 1-3)
# ============================================================

def run_bustools_pipeline(
    sample_name: str,
    reads_dir: str,
    reads_files: list,
    cache_dir: str,
    log: logging.Logger,
) -> None:
    """
    Run kallisto bus + bustools sort + bustools count for one sample.
    Saves all outputs to cache_dir for future re-use.

    Steps:
      1. kallisto bus  -> output.bus
      2. bustools sort -> sorted.bus
      3. bustools count -> count.mtx, matrix.ec, transcripts.txt

    Args:
        sample_name  : str       -- Human-readable sample label.
        reads_dir    : str       -- Directory containing the reads file(s).
        reads_files  : list[str] -- Reads filename(s) (1 for long, 2 for paired short).
        cache_dir    : str       -- Output directory for bustools files.
        log          : Logger

    Raises:
        subprocess.CalledProcessError -- If any step fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    log.info(f"[{sample_name}] Running bustools pipeline -> {cache_dir}")

    reads_paths = [os.path.join(reads_dir, f) for f in reads_files]

    # --- Step 1: kallisto bus ---
    bus_out = os.path.join(cache_dir, "output.bus")
    cmd_bus = [KALLISTO, "bus", "-i", INDEX_FILE, "-o", cache_dir,
               "-t", str(THREADS)]
    if READ_TYPE == "long":
        cmd_bus += ["--long", f"--threshold={THRESHOLD}", "--unmapped"]
    else:
        # Short-read paired: no extra flags needed
        pass
    cmd_bus += reads_paths

    log.info(f"[{sample_name}] Step 1: kallisto bus")
    run_command(cmd_bus, log)

    # Verify output.bus was created
    if not os.path.isfile(bus_out):
        raise FileNotFoundError(f"kallisto bus did not produce: {bus_out}")

    # --- Step 2: bustools sort ---
    sorted_bus = os.path.join(cache_dir, "sorted.bus")
    cmd_sort = [BUSTOOLS, "sort", "-t", str(THREADS), "-o", sorted_bus, bus_out]

    log.info(f"[{sample_name}] Step 2: bustools sort")
    run_command(cmd_sort, log)

    # --- Step 3: bustools count ---
    count_prefix = os.path.join(cache_dir, "count")
    cmd_count = [BUSTOOLS, "count",
                 "-o", count_prefix,
                 "-g", T2G_FILE,
                 "-e", os.path.join(cache_dir, "matrix.ec"),
                 "-t", os.path.join(cache_dir, "transcripts.txt"),
                 "--genecounts",
                 sorted_bus]
    # Note: bustools count writes count.mtx to <count_prefix>.mtx
    # and reads the EC file it already wrote during `kallisto bus`

    log.info(f"[{sample_name}] Step 3: bustools count")
    # bustools count reads matrix.ec and transcripts.txt from the bus output dir,
    # then writes count.mtx. We pass them explicitly.
    cmd_count = [BUSTOOLS, "count",
                 "-o", count_prefix,
                 "-g", T2G_FILE,
                 "-e", os.path.join(cache_dir, "matrix.ec"),
                 "-t", os.path.join(cache_dir, "transcripts.txt"),
                 sorted_bus]
    run_command(cmd_count, log)

    # Verify required outputs
    for fname in CACHE_REQUIRED_FILES:
        fpath = os.path.join(cache_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"bustools count did not produce: {fpath}"
            )

    log.info(f"[{sample_name}] Bustools pipeline complete.")


# ============================================================
# JOLI EM (Step 4 — Phase 1: load + stub; EM added in Step 1.3)
# ============================================================

def run_joli_em(
    sample_name: str,
    cache_dir: str,
    sample_result_dir: str,
    log: logging.Logger,
) -> None:
    """
    Run JOLI EM on the TCC matrix for one sample and write abundance.tsv.

    Currently Phase 1 Step 1.1: loads TCC data and reports stats.
    Steps 1.2-1.5 (weights, EM, output writer) will be added incrementally.

    Args:
        sample_name       : str    -- Human-readable sample label.
        cache_dir         : str    -- Path to bustools output cache dir.
        sample_result_dir : str    -- Where to write abundance.tsv.
        log               : Logger
    """
    log.info(f"[{sample_name}] Step 4: JOLI EM")

    # --- Step 1.1: Load TCC data ---
    tcc_data = load_tcc_data(cache_dir)
    log.info(f"[{sample_name}] Loaded {len(tcc_data.ec_transcripts)} ECs, "
             f"{len(tcc_data.transcript_names)} transcripts, "
             f"{tcc_data.total_reads} total reads")

    # --- Step 1.2: Compute weights ---
    weight_data = compute_weights(tcc_data, mode=EFF_LEN_MODE)
    log.info(f"[{sample_name}] Weights computed (mode='{EFF_LEN_MODE}')")

    # --- Step 1.3: Run JOLI EM ---
    em = JoliEM(tcc_data, weight_data)
    em_result = em.run(
        max_em_rounds    = MAX_EM_ROUNDS,
        convergence_mode = "kallisto",
    )
    log.info(f"[{sample_name}] EM done: rounds={em_result.n_rounds}, "
             f"converged={em_result.converged}, "
             f"nonzero_tx={int((em_result.alpha > 0).sum())}")

    # --- Step 1.4: Write abundance.tsv ---
    out_path = os.path.join(sample_result_dir, "abundance.tsv")
    summary = write_abundance(
        alpha            = em_result.alpha,
        eff_lens         = weight_data.eff_lens,
        transcript_names = tcc_data.transcript_names,
        output_path      = out_path,
    )
    log.info(f"[{sample_name}] abundance.tsv written: "
             f"non-zero TPM={summary['n_nonzero']}, "
             f"total TPM={summary['total_tpm']:.2f}")


# ============================================================
# Per-sample orchestrator
# ============================================================

def process_sample(sample_entry: tuple, run_dir: str, log: logging.Logger) -> None:
    """
    Run the full pipeline (bustools cache check + JOLI EM) for one sample.

    Args:
        sample_entry : tuple  -- (sample_name, reads_dir, reads_file[, reads_file2])
        run_dir      : str    -- Timestamped result directory for this experiment.
        log          : Logger
    """
    sample_name = sample_entry[0]
    reads_dir   = sample_entry[1]
    reads_files = list(sample_entry[2:])  # one file (long) or two (short paired)
    stem        = get_sample_stem(reads_files[0])

    # Cache dir: lives next to the input data, not in results
    cache_dir = os.path.join(reads_dir, "kallisto_output", stem)

    # Result dir for this sample (inside the timestamped experiment folder)
    sample_result_dir = os.path.join(run_dir, sample_name)
    os.makedirs(sample_result_dir, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"Sample: {sample_name}")
    log.info(f"  Reads:      {[os.path.join(reads_dir, f) for f in reads_files]}")
    log.info(f"  Cache dir:  {cache_dir}")
    log.info(f"  Result dir: {sample_result_dir}")
    log.info(f"{'='*60}")

    # --- Cache check: skip bustools if outputs already exist ---
    if check_bustools_cache(cache_dir, log):
        log.info(f"[{sample_name}] Using cached bustools output.")
    else:
        run_bustools_pipeline(sample_name, reads_dir, reads_files, cache_dir, log)

    # --- JOLI EM ---
    run_joli_em(sample_name, cache_dir, sample_result_dir, log)


# ============================================================
# Result directory setup
# ============================================================

def setup_run_dir(log_placeholder: logging.Logger) -> tuple:
    """
    Create timestamped result directory and start the run log.

    Args:
        log_placeholder : Logger -- Temporary logger before run_dir exists.

    Returns:
        (run_dir: str, log: Logger) -- Finalized run dir and logger with file handler.
    """
    timestamp = make_timestamp()
    run_dir   = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "running.log")
    log      = setup_logging(log_path)
    log.info(f"Run directory: {run_dir}")
    return run_dir, log


def save_experiment_description(run_dir: str) -> None:
    """
    Write experiment_description.log with full config dump.

    Args:
        run_dir : str -- Timestamped result directory.
    """
    lines = [
        "JOLI-Kallisto Experiment",
        f"Date: {datetime.now().isoformat()}",
        "",
        "=== CONFIG ===",
        f"KALLISTO:              {KALLISTO}",
        f"BUSTOOLS:              {BUSTOOLS}",
        f"INDEX_FILE:            {INDEX_FILE}",
        f"T2G_FILE:              {T2G_FILE}",
        f"READ_TYPE:             {READ_TYPE}",
        f"PLATFORM:              {PLATFORM}",
        f"THREADS:               {THREADS}",
        f"THRESHOLD:             {THRESHOLD}",
        f"EM_TYPE:               {EM_TYPE}",
        f"MAX_EM_ROUNDS:         {MAX_EM_ROUNDS}",
        f"EFF_LEN_MODE:          {EFF_LEN_MODE}",
        f"RUN_KALLISTO_QUANT_TCC:{RUN_KALLISTO_QUANT_TCC}",
        "",
        "=== SAMPLES ===",
    ]
    for s in SAMPLES:
        lines.append(f"  {s}")

    desc_path = os.path.join(run_dir, "experiment_description.log")
    with open(desc_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save_code_snapshot(run_dir: str) -> None:
    """
    Copy all .py and .sh files from JOLI_Kallisto/ into run_dir/code_snapshot/.

    Args:
        run_dir : str -- Timestamped result directory.
    """
    snapshot_dir = os.path.join(run_dir, "code_snapshot")
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    exts         = {".py", ".sh", ".txt", ".yml", ".yaml"}

    for fpath in Path(src_dir).iterdir():
        if fpath.suffix in exts:
            os.makedirs(snapshot_dir, exist_ok=True)
            shutil.copy2(str(fpath), os.path.join(snapshot_dir, fpath.name))


def save_runtime(run_dir: str, elapsed: float) -> None:
    """
    Write total wall-clock time to runtime.txt.

    Args:
        run_dir : str   -- Timestamped result directory.
        elapsed : float -- Elapsed seconds.
    """
    with open(os.path.join(run_dir, "runtime.txt"), "w") as fh:
        fh.write(f"{elapsed:.2f} seconds\n")


# ============================================================
# Main
# ============================================================

def main():
    """
    Entry point. Creates run directory, iterates over SAMPLES,
    checks bustools cache, runs JOLI EM, saves outputs.
    """
    run_start = time.time()

    # Bootstrap logger before run_dir exists
    bootstrap_log = logging.getLogger("bootstrap")
    bootstrap_log.addHandler(logging.StreamHandler(sys.stdout))
    bootstrap_log.setLevel(logging.INFO)

    # Create run directory and proper logger
    run_dir, log = setup_run_dir(bootstrap_log)

    # Save experiment metadata
    save_experiment_description(run_dir)
    save_code_snapshot(run_dir)

    log.info(f"Starting JOLI-Kallisto pipeline")
    log.info(f"Samples to process: {len(SAMPLES)}")

    errors = []
    for sample_entry in SAMPLES:
        try:
            process_sample(sample_entry, run_dir, log)
        except Exception as e:
            log.error(f"Sample {sample_entry[0]} FAILED: {e}", exc_info=True)
            errors.append(sample_entry[0])

    # Save runtime
    elapsed = time.time() - run_start
    save_runtime(run_dir, elapsed)

    log.info(f"\nPipeline complete in {elapsed:.1f}s")
    log.info(f"Results: {run_dir}")
    if errors:
        log.error(f"Failed samples: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
