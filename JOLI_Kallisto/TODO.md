# JOLI-Kallisto TODO

Tracked issues and planned work, organized by priority.  
Cross-reference: `plans/` directory contains full design docs for each phase.

---

## CRITICAL — Pipeline is broken / features don't run

### TODO-1: Wire EM into `main_pipeline.py` (Steps 1.2–1.4 stub)

**File:** `main_pipeline.py`, `run_joli_em()` (~line 345)

The function loads TCC data but then does nothing. Steps 1.2–1.4 are placeholder comments:

```python
# TODO Step 1.2: compute weights (weights.py)
# TODO Step 1.3: run JoliEM (em_algorithm.py)
# TODO Step 1.4: write abundance.tsv (output_writer.py)
log.info("JOLI EM stub complete. EM not yet implemented.")
```

All three modules are fully written — they just aren't called. Fix:

```python
# Step 1.2
weights, eff_lens = compute_weights(tcc_data, mode="kallisto")

# Step 1.3
em = JoliEM(tcc_data, weights)
result = em.run(convergence_mode="kallisto")

# Step 1.4
write_abundance(result.alpha, eff_lens, tcc_data.transcript_names,
                tcc_data.transcript_lengths, out_path)
```

See `plans/JoliToKallistoCLAUDEplan.txt` for exact parameter names.

---

### TODO-2: Implement single-sample MAP mode in `main_joli.py`

**File:** `main_joli.py`, `main()` (~line 192)

`--em_type map` and `--em_type vi` explicitly raise `NotImplementedError`:

```python
if args.em_type != "plain":
    raise NotImplementedError("Use --em_type plain for Phase 1.")
```

The multi-sample MAP code (`multi_sample_em.py`, `dirichlet_optimizer.py`) is complete.
Single-sample MAP just needs `alpha_prior` passed to `em.run()` with a fixed alpha vector.
Either implement it or remove the `--em_type` argument so users aren't misled.

---

## HIGH — Correctness issues

### TODO-3: Fix false positives from Dirichlet prior in multi-sample EM

**File:** `core/multi_sample_em.py`, `write_results()` (~line 308)

**Problem (from `plans/false_positive_fix_plan_2026_03_29.md`):**  
JK MS produces 16,066 false positives vs LK's 6,092 on sim1. The Dirichlet prior prevents
low-abundance transcripts from converging to zero — they stay alive because
`numerator[t] = n[t] + alpha[t]` is always positive even when `n[t] ≈ 0`.

**Secondary bug:** `write_results()` runs one additional MAP EM round during output writing.
Transcripts that were barely zeroed during training get resurrected:
- `theta[t] = 0` → `n[t] = 0` in E-step → `numerator[t] = 0 + alpha[t] > 0` → survives output.

**Fix approaches to evaluate:**
1. Post-training hard threshold: zero transcripts below a minimum expected-count floor
   (e.g., `alpha < 0.01`) before calling `write_results()`.
2. In `write_results()`: skip the extra MAP EM round, or run it with `alpha_prior=None`
   (plain EM) so the prior can't resurrect zeroed transcripts.
3. Sparsity-inducing prior: set `alpha[t] < 1` for unobserved transcripts
   to push them toward zero instead of away from it.

See `plans/false_positive_fix_plan_2026_03_29.md` for full root cause analysis.

---

### TODO-4: Off-by-one in final snapshot round number

**File:** `core/em_algorithm.py`, `run()` (~line 508)

```python
if snapshots is not None:
    snapshots.append((round_num + 1, theta.copy()))
```

When the loop converges early (breaks at round `R`), the final snapshot is tagged `R+1`,
but the last *executed* round was `R`. This makes convergence animation plots (see
`plans/convergence_animation_plan_2026_03_30.md`) show an extra ghost frame.

Fix: tag the final snapshot with `round_num`, not `round_num + 1`.

---

### TODO-5: Silent NaN recovery in `DirichletOptimizer` should surface as a warning

**File:** `core/dirichlet_optimizer.py`, `update()` (~line 108)

When gradient descent corrupts `log_alpha` with NaN/inf, the optimizer silently resets
those entries and clears Adam momentum. The reset is logged only to stdout (easily missed).
If this fires repeatedly, the final `alpha` may be partially invalid with no indication
in the output files.

Fix: count total NaN resets across the full training run and either raise a warning
in the returned result or write a count to `training_stats.pkl` so `plot_training.py`
can surface it.

---

## MEDIUM — Maintainability / robustness

### TODO-6: Consolidate hardcoded UINT32_MAX sentinel into one constant

**File:** `core/weights.py` (~line 204) and `core/load_tcc.py` (~line 251)

The sentinel value `4294967295.0` is defined independently in two files:

```python
# weights.py
sentinel = 4294967295.0  # UINT32_MAX — unobserved transcripts

# load_tcc.py
sentinel = 4294967295.0  # UINT32_MAX — no reads observed for this transcript
```

If one is changed without the other, unobserved transcripts are handled differently
between loading and weight computation — silent data corruption.

Fix: define once, e.g. in `load_tcc.py` as a module-level constant `FLENS_SENTINEL`,
and import it in `weights.py`.

---

### TODO-7: `output_writer.py` — silent fallback for zero effective length may inflate TPM

**File:** `core/output_writer.py` (~line 99)

```python
safe_eff_lens = np.where(eff_lens > 0, eff_lens, 1.0)
rho = alpha / safe_eff_lens
```

If a transcript has effective length 0 (not a sentinel, just a genuinely short/missing
value), its TPM is computed as `alpha / 1.0` — potentially much higher than it should be.
Add a log warning when this fallback fires so it isn't invisible in outputs.

---

### TODO-8: Add length assertion for single-tx EC arrays in `em_algorithm.py`

**File:** `core/em_algorithm.py`, `run()` (~line 500)

`_single_tx_ids` and `_single_ec_counts` are built separately in `_preprocess()` and
consumed together in `np.add.at()`. A preprocessing bug could create a silent length
mismatch (numpy won't raise, it'll just write to wrong indices).

Fix: add one assert after preprocessing:
```python
assert len(self._single_tx_ids) == len(self._single_ec_counts), \
    "single_tx_ids / single_ec_counts length mismatch — preprocessing bug"
```

---

### TODO-9: Replace fragile scipy version compatibility in `training_tracker.py`

**File:** `core/training_tracker.py`, `_spearman()` (~line 92)

```python
return float(getattr(result, "statistic", getattr(result, "correlation", result[0])))
```

Triple-nested getattr fallback is brittle. Fix with an explicit try/except:

```python
try:
    return float(result.statistic)       # scipy >= 1.11
except AttributeError:
    return float(result.correlation)     # scipy < 1.11
```

---

## LOW — Code clarity

### TODO-10: Rename convergence threshold constants in `em_algorithm.py`

**File:** `core/em_algorithm.py` (module-level constants)

`ALPHA_CHANGE_LIMIT` and `ALPHA_CHANGE` are both `1e-2` but mean different things:
- `ALPHA_CHANGE_LIMIT`: minimum raw count for a transcript to be monitored (0.01 reads)
- `ALPHA_CHANGE`: relative change threshold for declaring convergence (1%)

Having the same value with different names creates confusion when tuning convergence.
Rename to `CONVERGENCE_MIN_COUNT` and `CONVERGENCE_REL_CHANGE` (or similar) to make
the roles self-documenting.

---

## Future Phases (not yet started)

- **Phase 3 — Variational Inference (VI):** `--em_type vi` path in `main_joli.py`
  is explicitly not implemented. See `AT_code/EM_VIorMAP_GD_vector.py` for reference
  implementation to port.
- **Short-read support validation:** multi-sample scripts have a `READ_TYPE` flag for
  paired-end short reads but this path hasn't been benchmarked. Needs a sim run.
- **EDA: index discrepancy (Priority 1) and leaked transcripts (Priority 6):** see
  `plans/eda_plan_2026_03_29.md` for analysis plan.
- **Dummy dataset for fast CI testing:** see `plans/dummy_dataset_plan.md`.
