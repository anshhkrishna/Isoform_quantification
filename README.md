# isoform_quantification

multi-sample rna isoform quantification from long-read sequencing (pacbio/ont).
forked from [ArghamitraT/Isoform_quantification](https://github.com/ArghamitraT/Isoform_quantification).

---

## at_code — original em/vi/map algorithm

to try the initial algorithm, please run the file `AT_code/main_EM_VI_GD_CLEAN.py`. this is the main file and it will call `EM_VI_GD_CLEAN.py` and `DirichletOptimizer_CLEAN.py`.

**data:** please find some trial sirv data in this folder: `SIRV_data`

**environment:** please install the environment from the following .yml or .txt file:
`AT_code/NanoCount_5.yml`, `AT_code/NanoCount_5.txt`

**command:** to run with trial sirv data:
```bash
conda activate NanoCount_5
python main_EM_VI_GD_CLEAN.py --input_folder "data folder name" > "output file name"
```

example:
```bash
python main_EM_VI_GD_CLEAN.py \
  --input_folder /gpfs/commons/home/spark/knowles_lab/Argha/RNA_Splicing/data/PacBio_data_Liz/transcriptome_aln_pklfiles/ \
  > output.txt
```

the command will automatically create a folder called `result` and store the isoform abundances.

---

## joli_kallisto — active development

integration of the at_code em/vi/map approach with [lr-kallisto](https://github.com/pachterlab/kallisto) for faster quantification via equivalence classes (tcc).

### pipelines

| pipeline | entry point | when to use |
|----------|-------------|-------------|
| lr-kallisto baseline | `scripts/run_lr_kallisto.sh` | c++ em, comparison baseline |
| joli full (single-sample) | `scripts/run_joli_kallisto.sh` | bustools + python em, end-to-end |
| joli em only (single-sample) | `main_joli.py` | bustools output already exists |
| joli multi-sample full pipeline | `scripts/run_multisample_joli.sh` | phase 2: bustools + joint map em |
| joli multi-sample map em only | `main_multisample_joli.py` | phase 2: bustools output already exists |

### quick start (single sample)

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

### quick start (multi-sample map)

```bash
python JOLI_Kallisto/main_multisample_joli.py \
  --sample_dirs /path/s1/ /path/s2/ /path/s3/ \
  --results_base /path/to/results/
```

---

## bug fixes (this fork)

the following bugs from `JOLI_Kallisto/TODO.md` were fixed in this fork on 2026-04-30.

### critical — pipeline was broken

**todo-1: wire em steps into `main_pipeline.py`**
`run_joli_em()` loaded tcc data then did nothing (stub comments). now calls
`compute_weights()`, `JoliEM.run()`, and `write_abundance()` so the pipeline
actually produces `abundance.tsv`.

**todo-2: implement single-sample map mode in `main_joli.py`**
`--em_type MAP` previously raised `NotImplementedError`. now constructs a uniform
dirichlet prior (`alpha[t] = 1.0` for all transcripts) and passes it to `em.run()`.
`--em_type VI` still raises `NotImplementedError` with a clear message.

### high — correctness

**todo-3: fix false positives from dirichlet prior in `multi_sample_em.write_results()`**
the finalisation em round in `write_results()` ran with the learned `alpha_prior`,
which resurrected transcripts zeroed during training (`numerator = 0 + alpha > 0`).
changed to plain em (`alpha_prior=None`) so the prior cannot revive zeroed transcripts
at output time, reducing false positive inflation.

**todo-4: off-by-one in convergence animation snapshot round numbers**
final snapshot in `em_algorithm.run()` was tagged `round_num + 1` but the last
executed round was `round_num`. this caused convergence animation plots to show a
ghost frame. fixed to tag with `round_num`.

**todo-5: silent nan recovery in `DirichletOptimizer` now surfaces as a warning**
nan/inf resets in `dirichlet_optimizer.update()` were only printed to stdout.
now tracks total reset count across the full call and emits `warnings.warn(RuntimeWarning)`
if any occurred, so callers and log files capture the signal.

### medium — robustness

**todo-6: consolidate uint32_max sentinel into one constant**
`FLENS_SENTINEL = 4294967295.0` was defined independently in `load_tcc.py` and
`weights.py`. a divergence between the two would cause silent data corruption in
unobserved-transcript handling. now defined once in `load_tcc.py` and imported
in `weights.py`.

**todo-7: log warning when `eff_len=0` fallback fires in `output_writer.py`**
the silent `np.where(eff_lens > 0, eff_lens, 1.0)` fallback could inflate tpm
for genuinely short/missing transcripts with no indication. now logs a warning
with the count of affected transcripts.

**todo-8: add assertion for single-tx ec array length consistency**
`_single_tx_ids` and `_single_ec_counts` are built separately in `_preprocess()`
and consumed together in `np.add.at()`. a preprocessing bug causing a length mismatch
would silently write to wrong indices. assertion added at end of `_preprocess()`.

**todo-9: replace fragile scipy version compatibility in `training_tracker._spearman()`**
triple-nested `getattr` fallback replaced with an explicit `try/except AttributeError`
matching the documented scipy >= 1.11 (`.statistic`) vs < 1.11 (`.correlation`) split.

### low — clarity

**todo-10: rename ambiguous convergence threshold constants in `em_algorithm.py`**
`ALPHA_CHANGE_LIMIT` and `ALPHA_CHANGE` both had value `1e-2` but different roles.
renamed to `CONVERGENCE_MIN_COUNT` and `CONVERGENCE_REL_CHANGE` to make their
purposes self-documenting.

---

## environment

```bash
conda activate NanoCount_5
```

key packages: pytorch 2.3.1, pyro-ppl 1.9.1, nanocount 1.0.0, numpy, scipy, pandas, matplotlib.
environment files: `AT_code/NanoCount_5.yml`, `AT_code/NanoCount_5.txt`
