"""
training_tracker.py
===================
Collects and stores per-round training metrics for the multi-sample MAP EM loop.

Tracked per GD round:
  - gd_loss              : final GD loss value for this round
  - alpha_sum            : sum of all alpha[t] — grows as prior concentrates
  - alpha_entropy        : entropy of normalized alpha — spread of shared prior
  - alpha_max_change     : max |alpha_t - alpha_t_prev| — alpha convergence
  - em_rounds            : EM iterations to convergence per sample
  - em_converged         : EM convergence flag per sample
  - inter_sample_corr    : Spearman + Pearson between all theta_i / theta_j pairs
  - theta_vs_alpha_corr  : Spearman + Pearson between each theta_s and normalized alpha
  - nonzero_per_sample   : number of nonzero transcripts per sample

Saved as training_stats.pkl in the experiment output folder.

Usage:
  tracker = TrainingTracker(sample_names)
  tracker.record(gd_round, theta_list, alpha, em_rounds, em_converged, gd_loss)
  tracker.save(path)
  tracker = TrainingTracker.load(path)
"""

import pickle

import numpy as np
from scipy.stats import spearmanr, pearsonr


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return Spearman correlation coefficient between two 1-D arrays.

    Compatible with scipy < 1.11 (uses .correlation) and >= 1.11 (uses .statistic).

    Args:
        a : np.ndarray -- First array.
        b : np.ndarray -- Second array.

    Returns:
        float -- Spearman r.
    """
    result = spearmanr(a, b)
    try:
        return float(result.statistic)       # scipy >= 1.11
    except AttributeError:
        return float(result.correlation)     # scipy < 1.11


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return Pearson correlation coefficient between two 1-D arrays.

    Args:
        a : np.ndarray -- First array.
        b : np.ndarray -- Second array.

    Returns:
        float -- Pearson r.
    """
    return float(pearsonr(a, b)[0])


class TrainingTracker:
    """
    Accumulates per-round metrics during the multi-sample MAP EM outer GD loop.

    Stores a list of round-dicts (one per GD round) that can be saved to disk
    and later loaded for plotting.
    """

    def __init__(self, sample_names: list):
        """
        Initialize the tracker.

        Args:
            sample_names : list[str] -- Names of the samples in order.
        """
        self.sample_names = list(sample_names)
        self.history      = []         # list of per-round metric dicts
        self._prev_alpha  = None       # alpha from previous round for change tracking
        self.snapshots    = []         # list of snapshot dicts (populated by record_snapshot)

    def record(
        self,
        gd_round:        int,
        theta_list:      list,
        alpha:           np.ndarray,
        em_rounds_list:  list,
        em_converged_list: list,
        gd_loss:         float,
    ) -> None:
        """
        Compute and store metrics for one completed GD round.

        Args:
            gd_round          : int          -- 0-indexed GD round number.
            theta_list        : list[ndarray]-- Per-sample normalized theta (T,).
            alpha             : np.ndarray   -- Current shared alpha (T,).
            em_rounds_list    : list[int]    -- EM iterations per sample this round.
            em_converged_list : list[bool]   -- EM convergence flag per sample.
            gd_loss           : float        -- Final GD loss for this round.
        """
        # Normalized alpha (used as reference for theta-vs-alpha correlations)
        alpha_sum  = float(alpha.sum())
        alpha_norm = alpha / alpha_sum if alpha_sum > 0 else alpha.copy()

        # Alpha entropy: H = -sum(p * log(p))
        p             = np.clip(alpha_norm, 1e-300, None)
        alpha_entropy = float(-np.sum(p * np.log(p)))

        # Max absolute change in alpha since last round (None on first round)
        if self._prev_alpha is not None:
            alpha_max_change = float(np.abs(alpha - self._prev_alpha).max())
        else:
            alpha_max_change = None
        self._prev_alpha = alpha.copy()

        # Non-zero transcripts per sample
        nonzero_per_sample = [int((th > 0).sum()) for th in theta_list]

        # ---- Inter-sample correlations: all (i, j) pairs ----
        inter_sample_corr = {}
        n = len(theta_list)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (self.sample_names[i], self.sample_names[j])
                inter_sample_corr[pair] = {
                    "spearman": _spearman(theta_list[i], theta_list[j]),
                    "pearson":  _pearson(theta_list[i],  theta_list[j]),
                }

        # ---- Theta vs alpha correlations: one per sample ----
        theta_vs_alpha_corr = []
        for th in theta_list:
            theta_vs_alpha_corr.append({
                "spearman": _spearman(th, alpha_norm),
                "pearson":  _pearson(th,  alpha_norm),
            })

        self.history.append({
            "gd_round":            gd_round,
            "gd_loss":             float(gd_loss),
            "alpha_sum":           alpha_sum,
            "alpha_entropy":       alpha_entropy,
            "alpha_max_change":    alpha_max_change,
            "em_rounds":           list(em_rounds_list),
            "em_converged":        list(em_converged_list),
            "inter_sample_corr":   inter_sample_corr,
            "theta_vs_alpha_corr": theta_vs_alpha_corr,
            "nonzero_per_sample":  nonzero_per_sample,
        })

    def record_snapshot(
        self,
        round_num:  int,
        alpha:      np.ndarray,
        theta_list: list,
    ) -> None:
        """
        Store a full alpha + per-sample theta snapshot for one round.

        Called every snapshot_interval rounds from _run_em_wrapper when
        save_snapshots=True. Snapshots are saved to snapshots.pkl via
        save_snapshots() in write_results().

        Args:
            round_num  : int          -- 0-indexed outer round number.
            alpha      : np.ndarray   -- Current shared alpha, shape (T,).
            theta_list : list[ndarray]-- Per-sample normalized theta, each (T,).
        """
        self.snapshots.append({
            "round":  round_num,
            "alpha":  alpha.copy(),
            "thetas": [t.copy() for t in theta_list],
        })

    def save_snapshots(self, path: str) -> None:
        """
        Pickle the snapshot list to disk.

        Args:
            path : str -- Output file path (e.g. snapshots.pkl).
        """
        with open(path, "wb") as fh:
            pickle.dump({
                "sample_names": self.sample_names,
                "snapshots":    self.snapshots,
            }, fh)
        print(f"[TrainingTracker] Snapshots saved ({len(self.snapshots)} frames): {path}")

    def print_round_summary(self, gd_round: int) -> None:
        """
        Print a compact summary of the most recently recorded round to stdout.

        Args:
            gd_round : int -- GD round index (used as label only).
        """
        if not self.history:
            return
        rec = self.history[-1]

        print(f"\n  [Tracker] Round {gd_round + 1} summary:")
        print(f"    GD loss       : {rec['gd_loss']:.4f}")
        print(f"    alpha sum     : {rec['alpha_sum']:.4f}")
        print(f"    alpha entropy : {rec['alpha_entropy']:.4f}", end="")
        if rec["alpha_max_change"] is not None:
            print(f"  |alpha change| : {rec['alpha_max_change']:.4e}")
        else:
            print()

        for i, sname in enumerate(self.sample_names):
            nz  = rec["nonzero_per_sample"][i]
            cvg = "Y" if rec["em_converged"][i] else "N"
            sp  = rec["theta_vs_alpha_corr"][i]["spearman"]
            print(f"    {sname}: em_rounds={rec['em_rounds'][i]}  "
                  f"converged={cvg}  nonzero={nz}  "
                  f"spearman(theta,alpha)={sp:.4f}")

        for pair, corr in rec["inter_sample_corr"].items():
            print(f"    {pair[0]} ↔ {pair[1]}: "
                  f"spearman={corr['spearman']:.4f}  "
                  f"pearson={corr['pearson']:.4f}")

    def save(self, path: str) -> None:
        """
        Pickle the tracker to disk.

        Args:
            path : str -- Output file path (e.g. training_stats.pkl).
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"[TrainingTracker] Saved to: {path}")

    @staticmethod
    def load(path: str) -> "TrainingTracker":
        """
        Load a previously saved TrainingTracker from disk.

        Args:
            path : str -- Path to the pkl file.

        Returns:
            TrainingTracker -- Loaded tracker with full history.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)
