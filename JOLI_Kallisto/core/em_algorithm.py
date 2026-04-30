"""
em_algorithm.py
===============
Step 1.3 of JOLI-Kallisto Phase 1: EC-based EM algorithm.

Implements JoliEM — a vectorized EM that operates on equivalence classes (ECs)
rather than individual reads. Mathematically equivalent to kallisto's C++ EM
(EMAlgorithm.h) but written in Python/NumPy, and designed to extend to
Dirichlet MAP (Phase 2), multi-sample GD (Phase 3), and VI (Phase 4).

Single-transcript EC handling (Fix A — matches kallisto EMAlgorithm.h exactly):
  Single-tx ECs are excluded from the EM loop entirely.  After convergence,
  their raw read counts are added directly to alpha (expected counts).  This
  prevents single-tx counts from biasing the fractional assignment of
  multi-tx ECs in cases where a transcript appears in both EC types.

Variable mapping to kallisto EMAlgorithm.h:
  alpha       <->  alpha_          (raw expected counts; NOT normalized)
  theta       <->  (internal only) normalized abundances used for convergence
  n           <->  next_alpha      (accumulated expected counts per M-step)
  ec_counts   <->  counts_         (read count per EC)
  ec_weights  <->  weight_map_     (1/eff_len per transcript per EC)
  TOLERANCE   <->  TOLERANCE       (numeric floor for denominator)

Variable mapping to AT_code EM_VIorMAP_GD_vector.py:
  alpha       <->  all_theta[sample] (after scaling back to raw counts)
  n           <->  all_n[sample]
  phi         <->  all_Phi_ri      (fractional assignment; implicit here via n)

Inputs:
  - TCCData   (from load_tcc.py)
  - WeightData (from weights.py)
  - EM hyperparameters (max_em_rounds, min_rounds, convergence thresholds)

Outputs:
  - EMResult dataclass:
      alpha      : np.ndarray (float64, shape [n_transcripts]) -- raw expected counts
                   (multi-tx EM counts + single-tx raw counts; does NOT sum to 1)
      n_rounds   : int  -- number of EM iterations completed
      converged  : bool -- whether the convergence criterion was met
"""

from dataclasses import dataclass

import numpy as np

from load_tcc import TCCData
from weights import WeightData


# ============================================================
# Constants — match kallisto EMAlgorithm.h exactly
# ============================================================

TOLERANCE              = np.finfo(np.float64).tiny   # ~2.2e-308; floor for denominators
ALPHA_LIMIT            = 1e-7    # transcripts below this × 0.1 are zeroed out
CONVERGENCE_MIN_COUNT  = 1e-2    # minimum raw expected count (kallisto mode) or normalized
                                 # theta floor (joli mode) for a transcript to be monitored:
                                 #   "kallisto" mode: raw expected counts > 0.01  (matches C++ EM)
                                 #   "joli"    mode: not used (uses 1e-10 floor instead)
CONVERGENCE_REL_CHANGE = 1e-2    # 1% relative change threshold for declaring convergence
DEFAULT_MIN_ROUNDS     = 50      # minimum EM iterations before convergence is checked


# ============================================================
# Result container
# ============================================================

@dataclass
class EMResult:
    """
    Output from one run of JoliEM.

    Attributes:
        alpha     (np.ndarray, float64, shape [n_transcripts]):
                   Raw expected read counts per transcript. Multi-tx EM expected
                   counts + single-tx raw counts added post-convergence.
                   Does NOT sum to 1. Matches kallisto's alpha_ after run().
                   Use alpha / eff_lens → rho → TPM for output (see output_writer.py).

        n_rounds  (int): Number of EM iterations completed.

        converged (bool): True if the convergence criterion was satisfied before
                          reaching max_em_rounds.

        snapshots (list | None): When snapshot_interval > 0 in run(), a list of
                   (round_num, theta) tuples saved every snapshot_interval rounds
                   plus the final theta. theta is normalized (sums to ~1).
                   None when snapshots are disabled (snapshot_interval=0).
    """
    alpha:     np.ndarray
    n_rounds:  int
    converged: bool
    snapshots: list = None


# ============================================================
# JoliEM class
# ============================================================

class JoliEM:
    """
    EC-based EM algorithm for isoform quantification.

    The E-step and M-step are combined in each iteration (same structure as
    kallisto's EMAlgorithm.h::run()):

      For each EC with count > 0:
        denom = sum_t ( theta[t] * ec_weight[t] )
        For each t in EC:
          n[t] += ec_count * theta[t] * ec_weight[t] / denom

    Single-transcript ECs are handled deterministically (no division needed),
    matching kallisto's optimization for singleton ECs.

    Multi-transcript ECs are vectorized using pre-built flat index arrays to
    avoid a Python loop over EC positions. See _preprocess() for details.
    """

    def __init__(self, tcc_data: TCCData, weight_data: WeightData):
        """
        Pre-process TCC and weight data into vectorization-friendly flat arrays.

        Args:
            tcc_data    : TCCData    -- Parsed bustools output.
            weight_data : WeightData -- Effective lengths and EC weights.
        """
        self.n_transcripts  = len(tcc_data.transcript_names)
        self.ec_counts      = tcc_data.ec_counts          # shape (n_ecs,)
        self.ec_transcripts = tcc_data.ec_transcripts     # list[list[int]]
        self.eff_lens       = weight_data.eff_lens        # shape (n_transcripts,)
        self.ec_weights     = weight_data.ec_weights      # list[np.ndarray]

        self._preprocess()

    def _preprocess(self):
        """
        Split ECs into single-transcript and multi-transcript groups.
        Build flat index arrays for vectorized multi-tx EM computation.

        Single-tx ECs: deterministic assignment, handled with np.add.at once.
        Multi-tx ECs:  fractional assignment, vectorized via flat arrays.

        Flat arrays built here (all parallel, indexed by EC position):
          _multi_flat_tx      : transcript index for each (ec, position) pair
          _multi_flat_weights : weight for each (ec, position) pair
          _multi_flat_ec_idx  : which multi-EC each position belongs to
          _multi_ec_counts    : count for each multi-EC (one per EC, not per position)
        """
        # --- Separate single-tx and multi-tx ECs ---
        single_ec_ids = []
        multi_ec_ids  = []
        for ec_id, txs in enumerate(self.ec_transcripts):
            if len(txs) == 1:
                single_ec_ids.append(ec_id)
            else:
                multi_ec_ids.append(ec_id)

        # Single-tx: static arrays (independent of theta, computed once)
        if single_ec_ids:
            single_ec_ids = np.array(single_ec_ids, dtype=np.int64)
            self._single_tx_ids    = np.array(
                [self.ec_transcripts[i][0] for i in single_ec_ids], dtype=np.int64)
            self._single_ec_counts = self.ec_counts[single_ec_ids]  # shape (n_single,)
        else:
            self._single_tx_ids    = np.array([], dtype=np.int64)
            self._single_ec_counts = np.array([], dtype=np.float64)

        # Multi-tx: build flat arrays
        flat_tx      = []
        flat_weights = []
        flat_ec_idx  = []
        multi_counts = []

        for local_idx, ec_id in enumerate(multi_ec_ids):
            txs     = self.ec_transcripts[ec_id]
            weights = self.ec_weights[ec_id]           # np.ndarray, shape (len(txs),)
            for pos, (tx, w) in enumerate(zip(txs, weights)):
                flat_tx.append(tx)
                flat_weights.append(w)
                flat_ec_idx.append(local_idx)
            multi_counts.append(self.ec_counts[ec_id])

        if flat_tx:
            self._multi_flat_tx      = np.array(flat_tx,      dtype=np.int64)
            self._multi_flat_weights = np.array(flat_weights,  dtype=np.float64)
            self._multi_flat_ec_idx  = np.array(flat_ec_idx,   dtype=np.int64)
            self._multi_ec_counts    = np.array(multi_counts,   dtype=np.float64)
            self._n_multi_ecs        = len(multi_ec_ids)
        else:
            self._multi_flat_tx      = np.array([], dtype=np.int64)
            self._multi_flat_weights = np.array([], dtype=np.float64)
            self._multi_flat_ec_idx  = np.array([], dtype=np.int64)
            self._multi_ec_counts    = np.array([], dtype=np.float64)
            self._n_multi_ecs        = 0

        # Active-transcript mask: True only for transcripts that appear in at
        # least one EC for this sample. Transcripts with no reads must stay
        # zero even when alpha_prior > 0 (prior must not invent abundance).
        self._active_tx_mask = np.zeros(self.n_transcripts, dtype=bool)
        if len(self._single_tx_ids) > 0:
            self._active_tx_mask[self._single_tx_ids] = True
        if len(self._multi_flat_tx) > 0:
            self._active_tx_mask[self._multi_flat_tx] = True
        n_active = int(self._active_tx_mask.sum())

        # Total multi-tx read count — fixed (depends only on data, not theta).
        # Cached here so both run() and em_step() can use it without recomputing.
        self._total_multi_reads = (
            float(self._multi_ec_counts.sum()) if len(self._multi_ec_counts) > 0 else 0.0
        )

        assert len(self._single_tx_ids) == len(self._single_ec_counts), (
            "single_tx_ids / single_ec_counts length mismatch — preprocessing bug"
        )

        print(f"[JoliEM] Pre-processed: {len(single_ec_ids)} single-tx ECs, "
              f"{self._n_multi_ecs} multi-tx ECs, "
              f"{len(self._multi_flat_tx)} total EC-transcript positions, "
              f"{n_active} active transcripts (have reads in this sample)")

    def _em_step(self, theta: np.ndarray) -> np.ndarray:
        """
        Run one combined E+M step and return updated n (expected counts per tx).

        Only multi-transcript ECs are processed here.  Single-tx ECs are excluded
        from the EM loop and added as raw counts post-convergence (Fix A — matches
        kallisto EMAlgorithm.h long-read branch exactly).

        The E-step computes fractional assignment weights; the M-step accumulates
        them into n[t]. Both happen in one pass, matching kallisto's structure.

        Args:
            theta : np.ndarray (float64, shape [n_transcripts]) -- current normalized
                    abundances (derived from multi-tx EM only; excludes single-tx).

        Returns:
            np.ndarray (float64, shape [n_transcripts]) -- n[t] = expected counts
            from multi-tx ECs only.
        """
        n = np.zeros(self.n_transcripts, dtype=np.float64)

        # --- Multi-transcript ECs: vectorized fractional assignment ---
        # Single-tx ECs are intentionally excluded here; added post-convergence.
        if self._n_multi_ecs > 0:
            # theta values and weights for every (ec, position) in flat arrays
            theta_per_pos   = theta[self._multi_flat_tx]              # shape (n_positions,)
            weighted_theta  = theta_per_pos * self._multi_flat_weights # theta[t] * w[t,ec]

            # Denominator per multi-EC: sum of weighted_theta over positions in each EC
            denominators = np.zeros(self._n_multi_ecs, dtype=np.float64)
            np.add.at(denominators, self._multi_flat_ec_idx, weighted_theta)

            # Broadcast denominator and EC count back to each position
            denom_per_pos = denominators[self._multi_flat_ec_idx]    # shape (n_positions,)
            count_per_pos = self._multi_ec_counts[self._multi_flat_ec_idx]

            # Mask positions where denominator is too small (avoids div-by-zero)
            valid = denom_per_pos >= TOLERANCE
            contributions = np.where(
                valid,
                count_per_pos * weighted_theta / np.where(valid, denom_per_pos, 1.0),
                0.0
            )

            # Accumulate contributions into n
            np.add.at(n, self._multi_flat_tx, contributions)

        return n

    def run(
        self,
        max_em_rounds: int      = 10000,
        min_rounds: int         = DEFAULT_MIN_ROUNDS,
        convergence_mode: str   = "kallisto",
        alpha_prior: np.ndarray = None,
        init_theta: np.ndarray  = None,
        min_read_support: float = 0.0,
        snapshot_interval: int  = 0,
    ) -> EMResult:
        """
        Run EM until convergence or max_em_rounds, then zero small abundances.

        Two convergence modes (set via convergence_mode):

          "kallisto" (default for plain EM):
            Matches kallisto EMAlgorithm.h exactly. The convergence threshold
            CONVERGENCE_MIN_COUNT is applied to RAW expected counts (alpha = theta *
            total_multi_reads), so even transcripts with tiny fractional counts
            (> 0.01 reads) are monitored. This forces JK to run until ALL small
            transcripts have truly converged, matching kallisto's isoform set.
            Also runs one extra EM round after zeroing (finalRound mechanism from
            kallisto), allowing reads to redistribute away from zeroed transcripts.

          "joli" (recommended for MAP / VI):
            Applies CONVERGENCE_MIN_COUNT to NORMALIZED theta (sums to 1). Only
            transcripts with theta > 0.01 (> 1% of total reads) are monitored.
            Converges faster. Small transcripts are left to the Dirichlet prior
            in MAP/VI mode rather than being driven to zero by more EM rounds.

        Fix A — gating the Dirichlet prior on read support (min_read_support):
            When min_read_support > 0.0, the prior is only applied to transcripts
            where the E-step produced n[t] >= min_read_support expected reads.
            Transcripts below the threshold get plain EM (no prior) — plain EM
            naturally drives multi-mapping leakage transcripts toward zero.
            When min_read_support = 0.0 (default), the prior is always applied
            (original behaviour, Fix A disabled).

        Args:
            max_em_rounds     : int         -- Maximum EM iterations (default: 10000).
            min_rounds        : int         -- Minimum iterations before convergence check
                                              (default: 50).
            convergence_mode  : str         -- "kallisto" (raw count threshold, matches LK)
                                              or "joli" (normalized theta threshold, faster).
            alpha_prior       : np.ndarray  -- Optional Dirichlet concentration vector,
                                              shape [n_transcripts]. When provided, the
                                              M-step uses the posterior mean:
                                                theta_new = (n + alpha) / sum(n + alpha)
                                              instead of plain EM (n / sum(n)).
                                              Pass None for plain EM (default).
            init_theta        : np.ndarray  -- Optional starting theta, shape [n_transcripts].
                                              When provided (e.g. carried over from a previous
                                              GD round), EM starts from this theta instead of
                                              uniform. Pass None to start from uniform (default).
            min_read_support  : float       -- Fix A flag. Minimum expected read count n[t]
                                              required to apply alpha_prior in the M-step.
                                              0.0 = disabled (prior always applied).
                                              Typical value to try: 0.1 reads.
            snapshot_interval : int         -- Save a (round_num, theta) snapshot every
                                              this many rounds. 0 = disabled (default).
                                              Used by plot_convergence_animation.py to
                                              animate convergence across training rounds.

        Returns:
            EMResult -- alpha (raw expected counts), n_rounds, converged, snapshots.
        """
        if convergence_mode not in ("kallisto", "joli"):
            raise ValueError(
                f"convergence_mode must be 'kallisto' or 'joli', got '{convergence_mode}'"
            )

        # Validate alpha_prior shape if provided
        if alpha_prior is not None:
            if alpha_prior.shape != (self.n_transcripts,):
                raise ValueError(
                    f"alpha_prior shape {alpha_prior.shape} does not match "
                    f"n_transcripts={self.n_transcripts}"
                )

        # Total multi-tx reads — needed for kallisto mode convergence check and
        # for scaling theta back to raw counts at the end.
        total_multi_reads = self._total_multi_reads

        print(f"\n[JoliEM] Starting EM: max_rounds={max_em_rounds}, "
              f"min_rounds={min_rounds}, convergence_mode={convergence_mode}, "
              f"mode={'MAP(posterior_mean)' if alpha_prior is not None else 'plain'}, "
              f"n_transcripts={self.n_transcripts}, "
              f"total_multi_reads={total_multi_reads:.0f}, "
              f"warm_start={'yes' if init_theta is not None else 'no'}")

        # --- Initialization ---
        # Use provided init_theta (warm-start from previous GD round) or uniform.
        if init_theta is not None:
            theta = init_theta.copy().astype(np.float64)
            total_init = theta.sum()
            if total_init > 0:
                theta /= total_init     # re-normalize for safety
            else:
                theta = np.full(self.n_transcripts, 1.0 / self.n_transcripts,
                                dtype=np.float64)
        else:
            # Uniform initialization — matches kallisto's alpha_[i] = 1/num_targets
            theta = np.full(self.n_transcripts, 1.0 / self.n_transcripts,
                            dtype=np.float64)
        converged = False

        # Snapshot list: populated when snapshot_interval > 0
        snapshots = [] if snapshot_interval > 0 else None

        # Record initialization snapshot (round=-1) before any EM step.
        # theta at this point is uniform 1/T.
        if snapshots is not None:
            snapshots.append((-1, theta.copy()))

        for round_num in range(max_em_rounds):
            # One E+M step
            n = self._em_step(theta)

            # M-step: posterior-mean EM (with Dirichlet prior) or plain EM
            if alpha_prior is not None:
                if min_read_support > 0.0:
                    # Fix A: gate the prior on read support.
                    # Transcripts with n[t] >= min_read_support get the MAP update;
                    # those below threshold get plain EM — natural convergence toward
                    # zero eliminates multi-mapping leakage without harming true TPs.
                    has_support = n >= min_read_support
                    numerator = np.where(has_support, n + alpha_prior, n)
                else:
                    # Fix A disabled (default): prior always applied.
                    numerator = n + alpha_prior
            else:
                numerator = n

            # Zero out transcripts with no reads in this sample before normalizing.
            # Without this, alpha_prior > 0 would assign positive theta to transcripts
            # that have zero reads here, inflating nonzero-transcript counts.
            numerator[~self._active_tx_mask] = 0.0

            total = numerator.sum()
            if total > 0:
                theta_new = numerator / total
            else:
                theta_new = theta.copy()

            # --- Convergence check ---
            if convergence_mode == "kallisto":
                # Compare raw expected counts against 0.01 reads — matches kallisto C++.
                # alpha_new[t] = theta_new[t] * total_multi_reads
                # Monitor: alpha_new > 0.01  AND  |alpha_new - alpha_old| / alpha_new > 1%
                alpha_new = theta_new * total_multi_reads
                alpha_old = theta     * total_multi_reads
                changed = int(np.sum(
                    (alpha_new > CONVERGENCE_MIN_COUNT) &
                    (np.abs(alpha_new - alpha_old) / np.maximum(alpha_new, TOLERANCE)
                     > CONVERGENCE_REL_CHANGE)
                ))
            else:
                # "joli" mode: compare normalized theta directly.
                # Monitor all active transcripts (theta > numerical floor).
                # CONVERGENCE_MIN_COUNT is NOT used here — it was designed for raw
                # counts in kallisto mode and incorrectly filters out most
                # transcripts when applied to normalized theta over 200k+ entries.
                changed = int(np.sum(
                    (theta_new > 1e-10) &
                    (np.abs(theta_new - theta) / np.maximum(theta_new, TOLERANCE)
                     > CONVERGENCE_REL_CHANGE)
                ))

            theta = theta_new

            # Save snapshot every snapshot_interval rounds
            if snapshots is not None and round_num % snapshot_interval == 0:
                snapshots.append((round_num, theta.copy()))

            # Log how many transcripts are being monitored (first 3 rounds only)
            if round_num < 3:
                if convergence_mode == "joli":
                    n_monitored = int((theta_new > 1e-10).sum())
                else:
                    n_monitored = int((theta_new * total_multi_reads > CONVERGENCE_MIN_COUNT).sum())
                print(f"[JoliEM] Round {round_num + 1}: "
                      f"monitored={n_monitored}, changed={changed}")

            if changed == 0 and round_num >= min_rounds:
                converged = True
                print(f"[JoliEM] Converged at round {round_num + 1}")
                break

            # Progress checkpoint every 100 rounds
            if (round_num + 1) % 100 == 0:
                print(f"[JoliEM] Round {round_num + 1}: "
                      f"changed={changed}, "
                      f"nonzero_tx={int((theta > 0).sum())}")

        if not converged:
            print(f"[JoliEM] Reached max_em_rounds={max_em_rounds} without convergence.")

        # --- Zero small multi-tx abundances ---
        if convergence_mode == "kallisto":
            # Matches kallisto EMAlgorithm.h: threshold applied to RAW expected counts
            # (alpha = theta * total_multi_reads), not normalized theta.
            # Avoids over-zeroing on large samples where 1e-8 on theta = ~0.1 raw reads.
            alpha_temp = theta * total_multi_reads
            n_zeroed = int(((alpha_temp > 0) & (alpha_temp < ALPHA_LIMIT / 10)).sum())
            theta[alpha_temp < ALPHA_LIMIT / 10] = 0.0
        else:
            # "joli" mode: threshold on normalized theta (faster, for MAP/VI)
            n_zeroed = int(((theta > 0) & (theta < ALPHA_LIMIT / 10)).sum())
            theta[theta < ALPHA_LIMIT / 10] = 0.0
        if n_zeroed:
            print(f"[JoliEM] Zeroed {n_zeroed} multi-tx transcripts below {ALPHA_LIMIT/10:.1e}")

        # --- finalRound (kallisto mode only) ---
        # Kallisto runs one extra EM round after zeroing so reads can redistribute
        # away from the newly-zeroed transcripts (EMAlgorithm.h finalRound mechanism).
        if convergence_mode == "kallisto":
            n_final = self._em_step(theta)
            total_final = n_final.sum()
            if total_final > 0:
                theta = n_final / total_final
                # Apply raw-count zeroing again after redistribution
                alpha_final = theta * total_multi_reads
                n_zeroed_final = int(
                    ((alpha_final > 0) & (alpha_final < ALPHA_LIMIT / 10)).sum()
                )
                theta[alpha_final < ALPHA_LIMIT / 10] = 0.0
                print(f"[JoliEM] finalRound done. "
                      f"Additional transcripts zeroed: {n_zeroed_final}")

        # --- Build alpha: raw expected counts ---
        # Scale multi-tx theta back to expected counts, then add single-tx raw counts.
        alpha = theta * total_multi_reads

        # Add single-tx raw counts post-convergence (Fix A).
        if len(self._single_tx_ids) > 0:
            np.add.at(alpha, self._single_tx_ids, self._single_ec_counts)

        print(f"[JoliEM] Done. Rounds={round_num + 1}, "
              f"nonzero_transcripts={int((alpha > 0).sum())}")

        # Always append the final theta as the last snapshot.
        # Tag with round_num (the last *executed* round), not round_num + 1.
        if snapshots is not None:
            snapshots.append((round_num, theta.copy()))

        return EMResult(alpha=alpha, n_rounds=round_num + 1, converged=converged,
                        snapshots=snapshots)

    def em_step(
        self,
        theta: np.ndarray,
        alpha_prior: np.ndarray = None,
        min_read_support: float = 0.0,
        convergence_mode: str   = "joli",
    ) -> tuple:
        """
        Run one combined E+M step without zeroing or finalRound.

        Used exclusively by MultiSampleJoliEM in em_wrapper (AT_code) mode,
        where 1 EM step per sample is interleaved with N GD steps per outer
        iteration, and the outer loop runs until EM convergence.

        Unlike run(), this method:
          - Does NOT zero small transcripts (no ALPHA_LIMIT thresholding).
          - Does NOT run finalRound.
          - Does NOT add single-tx raw counts to alpha.
        Those finalisation steps happen once after the outer loop ends, via a
        single call to run(max_em_rounds=1, init_theta=...) in write_results().

        Args:
            theta            : np.ndarray  -- current normalized abundances,
                                             shape [n_transcripts].
            alpha_prior      : np.ndarray  -- optional Dirichlet prior,
                                             shape [n_transcripts]. None = plain EM.
            min_read_support : float       -- Fix A threshold. Prior only applied
                                             when n[t] >= this. 0.0 = disabled.
            convergence_mode : str         -- "joli" (normalized theta change) or
                                             "kallisto" (raw expected count change).

        Returns:
            tuple(np.ndarray, int):
                theta_new : updated normalized abundances, shape [n_transcripts].
                n_changed : number of monitored transcripts still changing.
                            0 means this sample has converged for this step.
        """
        # E+M step: compute expected counts from multi-tx ECs
        n = self._em_step(theta)

        # M-step: MAP posterior mean (with prior) or plain EM (without)
        if alpha_prior is not None:
            if min_read_support > 0.0:
                # Fix A: gate prior on read support
                has_support = n >= min_read_support
                numerator = np.where(has_support, n + alpha_prior, n)
            else:
                numerator = n + alpha_prior
        else:
            numerator = n

        # Zero out transcripts with no reads in this sample
        numerator[~self._active_tx_mask] = 0.0

        total = numerator.sum()
        theta_new = numerator / total if total > 0 else theta.copy()

        # Convergence check — mirrors run() logic exactly
        if convergence_mode == "kallisto":
            alpha_new = theta_new * self._total_multi_reads
            alpha_old = theta     * self._total_multi_reads
            n_changed = int(np.sum(
                (alpha_new > CONVERGENCE_MIN_COUNT) &
                (np.abs(alpha_new - alpha_old) /
                 np.maximum(alpha_new, TOLERANCE) > CONVERGENCE_REL_CHANGE)
            ))
        else:
            n_changed = int(np.sum(
                (theta_new > 1e-10) &
                (np.abs(theta_new - theta) /
                 np.maximum(theta_new, TOLERANCE) > CONVERGENCE_REL_CHANGE)
            ))

        return theta_new, n_changed
