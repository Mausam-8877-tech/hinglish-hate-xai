# Issue #3 results: files and provenance

Results of the transformer experiments (issue #3): 7 configurations × 3 seeds (42, 43, 44) = 21 runs.
Test set = official HASOC 2019 Hindi test file (1,314 posts); validation = 463 HASOC posts.
Interpretation and limitations are in the main [README](../../README.md#transformers--rationale-supervision-issue-3).

## Files

| File | What it is |
|---|---|
| `master_results.csv` | One row per run (21 rows), with provenance. **Use this file.** |
| `experiment_summary.csv` | One row per configuration: mean and sample std over 3 seeds, verified/reconstructed counts, validation macro-F1. |
| `accuracy_comparison.png` | Test accuracy, mean ± std per configuration (original plot, unchanged). |
| `macro_f1_comparison.png` | Test macro-F1, mean ± std per configuration (original plot, unchanged). |
| `seed_variability.png` | Accuracy and macro-F1 with seed error bars (original plot, unchanged). |
| `training_curves_val_macro_f1.png` | Validation macro-F1 per epoch, one panel per configuration, one line per seed; the ring marks the epoch kept by early stopping. Verified runs only. |
| `training_curves_loss.png` | Training (solid) and validation (dashed) loss per epoch. Verified runs only. The y-axis differs per panel: the training loss of rationale runs includes the λ-weighted rationale term (≈25 for λ = 10), so it is not comparable across λ; validation loss is plain cross-entropy on HASOC. |
| `training_history.csv` | The numbers behind both curve plots: one row per run and epoch (15 verified runs, 68 rows), from each run's `history.json`. |

| `explanations_hasoc.csv` | Issue #4: faithfulness + Hinglish plausibility of the CNN / BiLSTM explanations (1 seed). |
| `explanations_transformers_hasoc.csv` | Issue #4: same for every transformer run with a checkpoint, mean ± std over seeds (HASOC test, 1,314 posts). |
| `explanations_transformers_hatexplain.csv` | Issue #4: same on the HateXplain test set (seeded sample of 500 posts). |

The explanation CSVs are produced by `python scripts/explanation_table.py --results-dir <folder> --dataset <hasoc|hatexplain> --out <csv>`.

The three training-curve files are produced by `python scripts/plot_training_curves.py --results-dir <folder with the run folders>`.
| `original/master_results.csv`, `original/experiment_summary.csv` | The CSVs exactly as first produced, kept for traceability. |

## Units and rounding

- All metrics in `master_results.csv` and `experiment_summary.csv` are **fractions in [0, 1]**
  (0.8355 = 83.55%). `original/experiment_summary.csv` uses **percent**; the values are the same.
- Per-run values: verified runs are rounded to 6 decimals from the full-precision `metrics.json`
  (trailing zeros dropped, e.g. `0.8379` = `0.837900`). Reconstructed runs keep the precision that was
  recovered from the training log (4 decimals, with the one exception noted below).
- Summary means and standard deviations are rounded to 4 decimals. Std = **sample** standard deviation (n − 1).
  They were recomputed from `master_results.csv` and match `original/experiment_summary.csv`.
- The PNGs show percent, and their y-axis starts at 80. That makes differences of < 1 point look large.

## Provenance (`provenance` column)

| Value | Meaning | Runs |
|---|---|---|
| `verified_metrics_json` | Value matches a saved `results/<run>/metrics.json` exactly (same seed). The run folder (`run_folder`) also has `config.yaml`, `history.json` and `test_predictions.jsonl`. | 15 |
| `reconstructed_from_training_log` | Recovered from earlier training-log output. **No** `metrics.json`, history or predictions survive, so these values cannot be re-checked. | 6 |

The 6 reconstructed runs are:

| Configuration | Seeds |
|---|---|
| MuRIL Rationale λ = 0.1 | 44 |
| MuRIL Rationale λ = 1 | 42, 43, 44 (all of them) |
| MuRIL Rationale λ = 10 | 42, 43 |

`val_macro_f1`, `best_epoch` and `run_folder` are filled only for verified runs (the logs did not keep them).

### Unresolved: λ = 0.1, seed 44

The reconstructed test accuracy (0.821918) is **identical** to the verified accuracy of the
**MuRIL HX λ = 0, seed 44** run (both = 1080/1314 correct). Its accuracy also has 6 decimals while its
macro-F1 has only 4, unlike the other reconstructed rows. The two runs can legitimately have the same
number of correct posts, but the value may also have been copied from the wrong log line. The original log
is not available (the only trace is an empty `muril_rationale_lambda01_lam0.1_seed44/` folder in an earlier
backup), so the value is **kept unchanged and flagged as unresolved**. It can only be settled by re-running
that seed.

## Plots

The three PNGs are the original figures. They were drawn from all 21 runs (their means match
`experiment_summary.csv`) and **do not distinguish reconstructed from verified runs**. In particular, the
λ = 1 bar is built entirely from reconstructed values and the λ = 10 bar mostly from reconstructed values.
