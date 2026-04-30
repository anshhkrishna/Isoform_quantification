# Isoform_quantification

Multi-sample RNA isoform quantification from long-read sequencing (PacBio/ONT).
Forked from [ArghamitraT/Isoform_quantification](https://github.com/ArghamitraT/Isoform_quantification).

---

## AT_code — Original EM/VI/MAP Algorithm

To try the initial algorithm, please run the file `AT_code/main_EM_VI_GD_CLEAN.py`. This is the main file and it will call `EM_VI_GD_CLEAN.py` and `DirichletOptimizer_CLEAN.py`.

**DATA:** Please find some trial SIRV data in this folder: `SIRV_data`

**ENVIRONMENT:** Please install the environment from the following .yml or .txt file:
`AT_code/NanoCount_5.yml`, `AT_code/NanoCount_5.txt`

**Command:** To run with trial SIRV data:
```bash
conda activate NanoCount_5
python main_EM_VI_GD_CLEAN.py --input_folder "data folder name" > "output file name"
```

Example:
```bash
python main_EM_VI_GD_CLEAN.py \
  --input_folder /gpfs/commons/home/spark/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_Liz/transcriptome_aln_pklfiles/ \
  > output.txt
```

The command will automatically create a folder called `result` and store the isoform abundances.

---

## JOLI_Kallisto — Active Development

Integration of the AT_code EM/VI/MAP approach with [lr-kallisto](https://github.com/pachterlab/kallisto) for faster quantification via equivalence classes (TCC).

### Pipelines

| Pipeline | Entry point | When to use |
|----------|-------------|-------------|
| lr-kallisto baseline | `scripts/run_lr_kallisto.sh` | C++ EM, comparison baseline |
| JOLI full (single-sample) | `scripts/run_joli_kallisto.sh` | bustools + Python EM, end-to-end |
| JOLI EM only (single-sample) | `main_joli.py` | bustools output already exists |
| JOLI multi-sample full pipeline | `scripts/run_multisample_joli.sh` | Phase 2: bustools + joint MAP EM |
| JOLI multi-sample MAP EM only | `main_multisample_joli.py` | Phase 2: bustools output already exists |

### Quick start (single sample)

```bash
conda activate NanoCount_5

# Full pipeline (bustools + JOLI EM):
bash JOLI_Kallisto/scripts/run_joli_kallisto.sh

# EM only (if bustools output already exists):
python JOLI_Kallisto/main_joli.py \
  --sample_dir /path/to/kallisto_output/sample_stem/ \
  --em_type plain \
  --convergence_mode kallisto
```

### Quick start (multi-sample MAP)

```bash
python JOLI_Kallisto/main_multisample_joli.py \
  --sample_dirs /path/s1/ /path/s2/ /path/s3/ \
  --results_base /path/to/results/
```

---

## Bug Fixes (this fork)

The following bugs from `JOLI_Kallisto/TODO.md` were fixed in this fork on 2026-04-30.

### Critical — Pipeline was broken

**TODO-1: Wire EM steps into `main_pipeline.py`**
`run_joli_em()` loaded TCC data then did nothing (stub comments). Now calls
`compute_weights()`, `JoliEM.run()`, and `write_abundance()` so the pipeline
actually produces `abundance.tsv`.

**TODO-2: Implement single-sample MAP mode in `main_joli.py`**
`--em_type MAP` previously raised `NotImplementedError`. Now constructs a uniform
Dirichlet prior (`alpha[t] = 1.0` for all transcripts) and passes it to `em.run()`.
`--em_type VI` still raises `NotImplementedError` with a clear message.

### High — Correctness

**TODO-3: Fix false positives from Dirichlet prior in `multi_sample_em.write_results()`**
The finalisation EM round in `write_results()` ran with the learned `alpha_prior`,
which resurrected transcripts zeroed during training (`numerator = 0 + alpha > 0`).
Changed to plain EM (`alpha_prior=None`) so the prior cannot revive zeroed transcripts
at output time, reducing false positive inflation.

**TODO-4: Off-by-one in convergence animation snapshot round numbers**
Final snapshot in `em_algorithm.run()` was tagged `round_num + 1` but the last
executed round was `round_num`. This caused convergence animation plots to show a
ghost frame. Fixed to tag with `round_num`.

**TODO-5: Silent NaN recovery in `DirichletOptimizer` now surfaces as a warning**
NaN/inf resets in `dirichlet_optimizer.update()` were only printed to stdout.
Now tracks total reset count across the full call and emits `warnings.warn(RuntimeWarning)`
if any occurred, so callers and log files capture the signal.

### Medium — Robustness

**TODO-6: Consolidate UINT32_MAX sentinel into one constant**
`FLENS_SENTINEL = 4294967295.0` was defined independently in `load_tcc.py` and
`weights.py`. A divergence between the two would cause silent data corruption in
unobserved-transcript handling. Now defined once in `load_tcc.py` and imported
in `weights.py`.

**TODO-7: Log warning when `eff_len=0` fallback fires in `output_writer.py`**
The silent `np.where(eff_lens > 0, eff_lens, 1.0)` fallback could inflate TPM
for genuinely short/missing transcripts with no indication. Now logs a warning
with the count of affected transcripts.

**TODO-8: Add assertion for single-tx EC array length consistency**
`_single_tx_ids` and `_single_ec_counts` are built separately in `_preprocess()`
and consumed together in `np.add.at()`. A preprocessing bug causing a length mismatch
would silently write to wrong indices. Assertion added at end of `_preprocess()`.

**TODO-9: Replace fragile scipy version compatibility in `training_tracker._spearman()`**
Triple-nested `getattr` fallback replaced with an explicit `try/except AttributeError`
matching the documented scipy >= 1.11 (`.statistic`) vs < 1.11 (`.correlation`) split.

### Low — Clarity

**TODO-10: Rename ambiguous convergence threshold constants in `em_algorithm.py`**
`ALPHA_CHANGE_LIMIT` and `ALPHA_CHANGE` both had value `1e-2` but different roles.
Renamed to `CONVERGENCE_MIN_COUNT` and `CONVERGENCE_REL_CHANGE` to make their
purposes self-documenting.

---

## Environment

```bash
conda activate NanoCount_5
```

Key packages: PyTorch 2.3.1, Pyro-PPL 1.9.1, NanoCount 1.0.0, NumPy, SciPy, pandas, Matplotlib.
Environment files: `AT_code/NanoCount_5.yml`, `AT_code/NanoCount_5.txt`
