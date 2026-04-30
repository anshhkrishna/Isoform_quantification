"""
dirichlet_optimizer.py
======================
Gradient descent optimizer for the shared Dirichlet hyperparameter alpha
used in multi-sample MAP EM (Phase 2).

Adapted from AT_code/DirichletOptimizer_vector.py. Key simplification:
all JOLI samples share the same transcript index (same transcripts.txt),
so the theta_matrix input is already a dense (S x T) array — no isoform
name alignment or np.searchsorted needed.

Math:
  Dirichlet log-likelihood over S samples:
    L(alpha) = S * [log Gamma(sum_t alpha[t]) - sum_t log Gamma(alpha[t])]
             + sum_s sum_t (alpha[t] - 1) * log(theta_s[t])

  alpha is parameterized as exp(log_alpha) to keep it strictly positive
  throughout training. Adam optimizer updates log_alpha.

Inputs:
  - theta_matrix : np.ndarray, shape (S, T) — per-sample normalized theta,
                   one row per sample, same transcript order for all samples.
  - n_transcripts: int — number of transcripts T (= columns of theta_matrix).
  - gd_lr        : float — Adam learning rate.
  - alpha_initial: float — initial value for all alpha[t] (default 1.0).

Outputs:
  - alpha        : np.ndarray, shape (T,) — updated Dirichlet concentrations.
  - loss_history : list[float] — GD loss per Adam iteration.
"""

import math
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Dirichlet


# Small floor applied to theta before log() to avoid log(0)
THETA_FLOOR = 1e-10

# Minimum alpha value kept strictly positive throughout training.
# Prevents digamma(alpha→0) = -inf from producing NaN gradients when
# some transcripts are consistently zero across samples (e.g. under Fix A).
# log(ALPHA_MIN) sets the lower bound on log_alpha after each Adam step.
ALPHA_MIN = 1e-6


class DirichletOptimizer:
    """
    Gradient descent optimizer for the shared Dirichlet prior alpha.

    Maintains log_alpha (unconstrained) as the optimized parameter.
    alpha = exp(log_alpha) is always strictly positive.

    Usage:
        optimizer = DirichletOptimizer(n_transcripts=T, gd_lr=0.01)
        for gd_round in range(max_gd_rounds):
            # ... run per-sample MAP EM, collect theta per sample ...
            theta_matrix = np.stack(all_theta)   # shape (S, T)
            alpha, loss_history = optimizer.update(theta_matrix)
            # alpha is now updated; use in next MAP EM round
    """

    def __init__(
        self,
        n_transcripts: int,
        gd_lr: float,
        alpha_initial: float = 1.0,
    ):
        """
        Initialize the optimizer with uniform alpha.

        Args:
            n_transcripts : int   -- Number of transcripts T (dimension of alpha).
            gd_lr         : float -- Adam learning rate.
            alpha_initial : float -- Initial value for all alpha[t]. Default 1.0
                                     (uniform Dirichlet = no prior preference).
        """
        self.n_transcripts = n_transcripts
        self.gd_lr         = gd_lr

        # Parameterize as log_alpha so alpha = exp(log_alpha) > 0 always
        log_alpha_init = np.full(n_transcripts, np.log(alpha_initial), dtype=np.float32)
        self.log_alpha = torch.tensor(log_alpha_init, requires_grad=True)
        self.optimizer = optim.Adam([self.log_alpha], lr=gd_lr)

        print(f"[DirichletOptimizer] Initialized: "
              f"n_transcripts={n_transcripts}, gd_lr={gd_lr}, "
              f"alpha_initial={alpha_initial:.4f}")

    def update(
        self,
        theta_matrix: np.ndarray,
        max_iterations: int = 10,
        tolerance: float    = 1e-6,
    ) -> tuple:
        """
        Run Adam gradient descent to maximize the Dirichlet log-likelihood
        over all samples.

        Stops early if the change in loss between consecutive iterations
        drops below tolerance.

        Args:
            theta_matrix   : np.ndarray, shape (S, T) -- per-sample normalized
                             theta. Each row must approximately sum to 1.
                             Zero values are floored to THETA_FLOOR before log().
            max_iterations : int   -- Maximum Adam steps (default 10).
            tolerance      : float -- Early-stop threshold on |loss change|.

        Returns:
            tuple:
              alpha        (np.ndarray, shape [T])  -- updated alpha = exp(log_alpha).
              loss_history (list[float])             -- GD loss per iteration.
        """
        n_samples, n_tx = theta_matrix.shape
        if n_tx != self.n_transcripts:
            raise ValueError(
                f"theta_matrix has {n_tx} columns but optimizer expects "
                f"n_transcripts={self.n_transcripts}"
            )

        # Floor zeros to avoid log(0) in Dirichlet log-prob
        theta_floored = np.where(theta_matrix > 0, theta_matrix, THETA_FLOOR)

        # Re-normalize each row after flooring (rows must sum to 1 for Dirichlet)
        row_sums = theta_floored.sum(axis=1, keepdims=True)
        theta_norm = theta_floored / row_sums                   # shape (S, T)

        # Convert to torch tensor once (stays fixed during Adam iterations)
        data = torch.tensor(theta_norm, dtype=torch.float32)   # shape (S, T)

        loss_history     = []
        prev_loss        = None
        total_nan_resets = 0

        log_alpha_floor = math.log(ALPHA_MIN)

        for iteration in range(max_iterations):
            self.optimizer.zero_grad()

            # Clamp log_alpha BEFORE the forward pass so alpha is always >= ALPHA_MIN.
            # Must happen here (not after optimizer.step()) because:
            #   1. torch.clamp(NaN, min=x) returns NaN — clamping after a corrupted
            #      step does nothing.
            #   2. digamma(alpha → 0) = -inf → NaN gradient → corrupts log_alpha.
            # Clamping before each forward pass breaks the NaN chain at its source.
            # Also replaces any NaN/inf that may have crept in from previous rounds.
            with torch.no_grad():
                bad = ~torch.isfinite(self.log_alpha)
                if bad.any():
                    n_bad = bad.sum().item()
                    total_nan_resets += n_bad
                    print(f"[DirichletOptimizer] iter {iteration}: "
                          f"resetting {n_bad} NaN/inf log_alpha → floor")
                    self.log_alpha[bad] = log_alpha_floor
                    # Reset Adam state for corrupted parameters so momentum does not
                    # carry forward the invalid gradient signal
                    for param_state in self.optimizer.state.values():
                        if "exp_avg" in param_state:
                            param_state["exp_avg"][bad]    = 0.0
                        if "exp_avg_sq" in param_state:
                            param_state["exp_avg_sq"][bad] = 0.0
                self.log_alpha.clamp_(min=log_alpha_floor)

            alpha = torch.exp(self.log_alpha)                  # shape (T,), always >= ALPHA_MIN

            # Dirichlet log-likelihood: sum log p(theta_s | alpha) over all samples
            dirichlet       = Dirichlet(alpha)
            log_likelihood  = dirichlet.log_prob(data).sum()   # scalar
            loss            = -log_likelihood                  # minimize negative LL

            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Early stop if improvement is negligible
            if prev_loss is not None and abs(prev_loss - loss_val) < tolerance:
                print(f"[DirichletOptimizer] Early stop at iteration {iteration + 1} "
                      f"(|loss change| < {tolerance:.1e})")
                break

            prev_loss = loss_val

        # Warn if NaN resets occurred — final alpha may be partially invalid
        if total_nan_resets > 0:
            warnings.warn(
                f"[DirichletOptimizer] NaN/inf resets occurred {total_nan_resets} time(s) "
                "during this update call. Final alpha may be partially invalid. "
                "Check training_stats.pkl for cumulative reset counts.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Convert back to numpy for use in EM
        alpha_np = torch.exp(self.log_alpha).detach().numpy().astype(np.float64)

        print(f"[DirichletOptimizer] Updated alpha: "
              f"min={alpha_np.min():.4f}, max={alpha_np.max():.4f}, "
              f"mean={alpha_np.mean():.4f}, loss={loss_history[-1]:.4f}"
              + (f", nan_resets={total_nan_resets}" if total_nan_resets > 0 else ""))

        return alpha_np, loss_history

    def get_alpha(self) -> np.ndarray:
        """
        Return the current alpha (exp(log_alpha)) as a numpy array.

        Returns:
            np.ndarray, shape (T,) -- current Dirichlet concentration parameters.
        """
        return torch.exp(self.log_alpha).detach().numpy().astype(np.float64)
