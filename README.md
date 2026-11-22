# hinglish-hate-xai

**Explainable hate speech detection for code-mixed Hindi-English (Hinglish).**

The model detects hateful/offensive posts and **highlights the words responsible** for each decision.

> **Research question:** Does rationale supervision learned from *English* (HateXplain) make
> explanations more **faithful** on *code-mixed Hinglish* text?

Deep Learning course project (Semester 5), Sitare University. Team 4 / Project P04.

## Plan

- **Data:** HASOC Hindi/Hinglish tracks, plus HateXplain (English, with rationales)
- **Models:** CNN/LSTM baseline → MuRIL / XLM-R fine-tuning → rationale-supervised training
- **Explanations:** LIME and attention

## Setup

Use **Python 3.10 – 3.12**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

## Repository structure

```
hinglish-hate-xai/
├── configs/              one YAML file per experiment
├── data/
│   ├── raw/              downloaded datasets (git-ignored)
│   ├── processed/        cleaned data (git-ignored)
│   └── annotations/      our own Hinglish rationale annotations (committed)
├── docs/TASKS.md         task board (Member 1 – 4)
├── notebooks/            EDA and analysis notebooks
├── results/              run outputs (git-ignored)
├── scripts/              command-line entry points
│   ├── prepare_data.py
│   ├── train.py
│   ├── explain.py
│   ├── explanation_table.py
│   └── sample_for_annotation.py
├── tests/                pytest tests
└── xhate/                the Python package
    ├── data/             dataset loaders and preprocessing      (Member 1)
    ├── models/           baseline + transformer models          (Members 2, 3)
    ├── training/         training loop and losses               (Members 2, 3)
    ├── explain/          attention, LIME, visualisation         (Member 4)
    └── evaluation/       classification + faithfulness metrics  (Member 4)
```

## Results

Test set = the official HASOC 2019 Hindi test file (1,314 posts). Models are selected on validation
macro-F1 (early stopping). Seed 42, trained on CPU.

| Model | Train data | Test data | Accuracy | Macro-F1 | Config |
|---|---|---|---|---|---|
| CNN (Kim 2014) | HASOC 2019 Hindi | HASOC 2019 Hindi | 0.7770 | 0.7765 | `configs/baseline_cnn.yaml` |
| BiLSTM + attention | HASOC 2019 Hindi | HASOC 2019 Hindi | 0.8044 | 0.8016 | `configs/baseline_bilstm.yaml` |

Reproduce:

```bash
python scripts/prepare_data.py --datasets hasoc
python scripts/train.py --config configs/baseline_cnn.yaml      # -> results/baseline_cnn/
python scripts/train.py --config configs/baseline_bilstm.yaml   # -> results/baseline_bilstm/
```

### Transformers + rationale supervision (issue #3)

**Methodology.**

- Models: MuRIL (`google/muril-base-cased`) and XLM-R (`xlm-roberta-base`), fine-tuned with a
  linear layer on the [CLS] vector.
- Training setup: AdamW, lr 2e-5, 10% linear warmup, batch 16, max 128 sub-words, at most 5 epochs.
  Early stopping on validation macro-F1 (patience 2) and class-weighted cross-entropy.
- Rationale-supervised runs add `λ · CE(human rationale, [CLS] attention)` (last layer, mean over
  heads), applied only to HateXplain HOF posts that have a rationale.
- "+ HX" means the 15,383 HateXplain train posts are added to the 4,164 HASOC train posts.
  Validation (463 posts) and test (official HASOC 2019 Hindi test file, 1,314 posts) are **HASOC only**
  in every run.
- **7 configurations × 3 seeds (42, 43, 44) = 21 runs**, trained using Kaggle GPU runtime.
- The λ = 0 control uses the same data as the rationale runs without the extra loss. It separates
  "more data" from "rationale supervision".

**Results.** Test accuracy and macro-F1 as mean ± **sample** standard deviation over 3 seeds.
Values are fractions rounded to 4 decimals. "Verified" = how many of the 3 runs are backed by a saved
`metrics.json`; the rest were reconstructed from training logs (see below).

| Model | Train data | λ | Accuracy | Macro-F1 | Verified | Config |
|---|---|---|---|---|---|---|
| MuRIL base | HASOC | – | 0.8354 ± 0.0032 | 0.8339 ± 0.0048 | 3/3 | `configs/muril_vanilla.yaml` |
| XLM-R base | HASOC | – | 0.8326 ± 0.0008 | 0.8312 ± 0.0010 | 3/3 | `configs/xlmr_vanilla.yaml` |
| MuRIL base (control) | HASOC + HX | 0 | 0.8260 ± 0.0036 | 0.8254 ± 0.0037 | 3/3 | `configs/muril_hx_lambda0.yaml` |
| MuRIL base + rationale | HASOC + HX | 0.01 | 0.8343 ± 0.0004 | 0.8328 ± 0.0005 | 3/3 | `configs/muril_rationale.yaml` |
| MuRIL base + rationale | HASOC + HX | 0.1 | 0.8298 ± 0.0111 | 0.8281 ± 0.0108 | 2/3 ‡ | `configs/muril_rationale.yaml` |
| MuRIL base + rationale † | HASOC + HX | 1 | 0.8361 ± 0.0009 | 0.8355 ± 0.0011 | **0/3** | `configs/muril_rationale.yaml` |
| MuRIL base + rationale † | HASOC + HX | 10 | 0.8338 ± 0.0029 | 0.8318 ± 0.0035 | 1/3 | `configs/muril_rationale.yaml` |

† Includes reconstructed runs (λ = 1 entirely). ‡ Includes one reconstructed run with an unresolved
value (see below).

Graphs (original figures, built from all 21 runs, reconstructed runs **not** marked; y-axis starts at 80%):
[test accuracy](docs/results/accuracy_comparison.png) ·
[test macro-F1](docs/results/macro_f1_comparison.png) ·
[seed variability](docs/results/seed_variability.png).
Training curves (verified runs only, from `history.json`):
[validation macro-F1 per epoch](docs/results/training_curves_val_macro_f1.png) ·
[training / validation loss per epoch](docs/results/training_curves_loss.png)
(made with `python scripts/plot_training_curves.py`).
Per-run numbers with provenance: [`docs/results/master_results.csv`](docs/results/master_results.csv);
per-configuration summary: [`docs/results/experiment_summary.csv`](docs/results/experiment_summary.csv);
file guide: [`docs/results/README.md`](docs/results/README.md).

**Disclosure: six runs have reconstructed metrics.** For these runs only the test accuracy and macro-F1
were recovered from earlier training-log output:

- λ = 0.1 seed 44
- λ = 1 seeds 42, 43, 44
- λ = 10 seeds 42, 43

No result folder, `metrics.json`, validation score or predictions survive for them, so they
cannot be re-checked. They are marked `reconstructed_from_training_log` in `master_results.csv`; the other
15 runs are marked `verified_metrics_json`.

The reconstructed λ = 0.1 seed 44 accuracy (0.821918) is identical to that of the verified λ = 0 seed 44
run. The original log is not available, so this value is kept as recorded and flagged **unresolved**.

**Interpretation and limitations.**

- **The differences are small.** All seven configurations lie between 0.825 and 0.836 test macro-F1, a
  spread of about one point. This is comparable to the seed-to-seed variation within a configuration
  (std up to 0.011). With 3 seeds per configuration and no significance test, **none of the differences
  should be read as statistically significant**.
- **No conclusive gain from rationale supervision.** The only pattern consistent across verified seeds is
  that the λ = 0 control scores slightly lower than MuRIL trained on HASOC alone for each matching seed
  (42, 43, 44). Adding English data without supervision did not help. The rationale-supervised runs are
  roughly on par with vanilla MuRIL. λ = 1 has the highest mean, but all of its runs are reconstructed.
- **λ must be chosen on validation, not test.** With the current evidence it cannot be chosen properly:
  λ = 1 has no validation scores (reconstructed), and λ = 10 has a validation score for only 1 seed.
  Verified mean validation macro-F1 is 0.8222 for λ = 0.01 (3 seeds), 0.8196 for λ = 0.1 (2 seeds) and
  0.8337 for λ = 10 (1 seed). The test column above must not be used to pick λ.
- **MuRIL vs XLM-R:** MuRIL is slightly higher on average but varies more across seeds; the gap
  (0.003 macro-F1) is within noise.
- **Accuracy is not faithfulness.** These numbers only show how well each model *classifies*. Whether
  rationale supervision makes the models' explanations more faithful or plausible, which is the project's
  research question, needs a separate explanation evaluation (comprehensiveness, sufficiency, token F1;
  issue #4). It cannot be concluded from this table.
- The data is mostly Devanagari Hindi with some romanised Hinglish, and the rationales are English only.

Needs a GPU (Colab / Kaggle). Reproduce:

```bash
python scripts/prepare_data.py                      # HASOC + HateXplain
python scripts/run_seeds.py --config configs/muril_vanilla.yaml
python scripts/run_seeds.py --config configs/xlmr_vanilla.yaml
python scripts/run_seeds.py --config configs/muril_hx_lambda0.yaml
python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=0.01 run_name=muril_rationale_lambda001
python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=0.1 run_name=muril_rationale_lambda01
python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=1 run_name=muril_rationale_lambda1
python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=10 run_name=muril_rationale_lambda10
```
Run folders are named `<run_name>_seed<N>`, plus `_lam<λ>` for the rationale runs, e.g.
`muril_rationale_lambda001_lam0.01_seed42`, matching `run_folder` in `docs/results/master_results.csv`.

Each model returns its [CLS] attention for explanation:
`TransformerClassifier.get_attention(input_ids, attention_mask)` → `(B, L)`.

### Explainability + faithfulness (issue #4)

**Methods.** Every trained run is explained at the word level with:

- **attention**: [CLS] attention of the supervised layer (mean over heads), summed from sub-words back
  to words and re-normalised; for the BiLSTM, its additive attention. The CNN has no attention.
- **LIME**: one weight per word position (`split_expression=r"\s+"`, `bow=False`, removed words deleted).
- **random**: a seeded random score per word, the sanity baseline every method must beat.

**Metrics.**

- Faithfulness (ERASER, DeYoung et al. 2020), w.r.t. the predicted class, averaged over the top
  1 / 5 / 10 / 20 / 50 % of words: **comprehensiveness ↑** = p(x) − p(x without top words),
  **sufficiency ↓** = p(x) − p(only top words). Reported on all posts and on posts predicted HOF.
- Plausibility against human rationales: **token F1 ↑** (top-k words, k = average human rationale
  length) and **AUPRC ↑**. English: HateXplain test rationales. Hinglish: our own annotations of
  ~250 HASOC test HOF posts ([data/annotations/](data/annotations/README.md)).

**Run.**

```bash
python scripts/explain.py --run results/muril_vanilla_seed42                        # Hinglish test posts
python scripts/explain.py --run results/muril_vanilla_seed42 --dataset hatexplain   # English, with rationales
# -> results/<run>/explain_<dataset>_test/{metrics.json, explanations.jsonl, report.html}
python scripts/explanation_table.py --results-dir results --dataset hasoc           # final table (mean ± std over seeds)
```

Explain every run with the same options (`--lime-samples`, `--seed`, `--max-posts`), so only the model differs.

Hinglish rationale annotation: `python scripts/sample_for_annotation.py` (makes 4 sheets),
then `python scripts/sample_for_annotation.py --collect` → `data/annotations/hasoc_rationales.jsonl`.

**Results: baselines (done).** HASOC 2019 Hindi test set, all 1,314 posts, seed 42, 500 LIME
samples per post, CPU. The models were retrained with their configs for this (BiLSTM test macro-F1
0.8016, the same as above; CNN 0.7740, slightly below the 0.7765 above, from CPU/library differences).
Plausibility is measured on the 218 HASOC test posts with our Hinglish rationales
([data/annotations/](data/annotations/README.md): LLM pre-annotation, reviewed by the four team members,
no suggestion changed; token F1 uses k = the average rationale length).
Per-configuration CSV: [`docs/results/explanations_hasoc.csv`](docs/results/explanations_hasoc.csv).

| Model | Method | Comprehensiveness ↑ | Sufficiency ↓ | Comp. (pred. HOF) ↑ | Suff. (pred. HOF) ↓ | Token F1 ↑ | AUPRC ↑ |
|---|---|---|---|---|---|---|---|
| BiLSTM + attention | attention | 0.1630 | 0.1116 | 0.3913 | −0.0388 | **0.3615** | **0.5355** |
| BiLSTM + attention | LIME | **0.3800** | **−0.1016** | **0.5127** | **−0.0949** | 0.3364 | 0.5099 |
| BiLSTM + attention | random | 0.0316 | 0.2153 | 0.0541 | 0.2779 | 0.1189 | 0.2074 |
| CNN (Kim 2014) | LIME | **0.4328** | **−0.0688** | **0.4523** | **−0.0677** | **0.3559** | **0.5385** |
| CNN (Kim 2014) | random | 0.0373 | 0.2515 | 0.0416 | 0.2773 | 0.1189 | 0.2074 |

- Both methods beat the random baseline by a wide margin, so the explanations carry real information
  about the models.
- LIME is more faithful than the BiLSTM's attention: deleting LIME's top words lowers the model's
  confidence more than twice as much (0.38 vs 0.16). This matches "Attention is not Explanation"
  (Jain & Wallace, 2019): attention points in the right direction but is not the most faithful
  explanation. Attention does much better on posts predicted HOF (0.39) than on all posts.
- LIME's sufficiency is negative: the model is *more* confident on LIME's top words alone than on the
  full post, so the remaining words were pulling it towards the other class.
- **Faithful is not the same as plausible.** For the BiLSTM, LIME is the more *faithful* explanation, but
  attention agrees slightly better with the human rationales (token F1 0.36 vs 0.34, AUPRC 0.54 vs 0.51).
  Both are far above random (0.12 / 0.21).
- Single seed, so no ± yet. The numbers measure the baselines only; they say nothing yet about
  rationale supervision. The Hinglish rationales were LLM-suggested and accepted unchanged by the
  reviewers, so plausibility partly measures agreement with that LLM.

**Results: vanilla vs rationale-supervised transformers.** Every run with a saved checkpoint was explained
with the same options (Colab T4, fp16): HASOC test = all 1,314 posts (plausibility on our 218 Hinglish
rationales), HateXplain test = a seeded sample of 500 posts (302 with rationales), 200 LIME samples, seed 42.
Mean ± sample std over the seeds that have a checkpoint: λ = 1 has none (all three runs were reconstructed
from logs), λ = 0.1 seed 44 has no `config.yaml`, λ = 10 only seed 44 exists.
Full tables (all methods, also "predicted HOF" columns):
[`explanations_transformers_hasoc.csv`](docs/results/explanations_transformers_hasoc.csv) ·
[`explanations_transformers_hatexplain.csv`](docs/results/explanations_transformers_hatexplain.csv).

*Hinglish (HASOC test): [CLS] attention, the explanation the rationale loss trains*

| Model | Train data | λ | Seeds | Comprehensiveness ↑ | Sufficiency ↓ | Token F1 ↑ | AUPRC ↑ |
|---|---|---|---|---|---|---|---|
| MuRIL | HASOC | – | 3 | 0.096 ± 0.011 | 0.057 ± 0.018 | 0.502 ± 0.008 | 0.696 ± 0.011 |
| XLM-R | HASOC | – | 3 | 0.140 ± 0.013 | 0.103 ± 0.020 | 0.476 ± 0.041 | 0.661 ± 0.070 |
| MuRIL (control) | HASOC + HX | 0 | 3 | **0.204 ± 0.007** | 0.082 ± 0.010 | 0.531 ± 0.038 | 0.737 ± 0.054 |
| MuRIL + rationale | HASOC + HX | 0.01 | 3 | 0.193 ± 0.026 | 0.080 ± 0.018 | 0.569 ± 0.006 | 0.803 ± 0.008 |
| MuRIL + rationale | HASOC + HX | 0.1 | 2 | 0.124 ± 0.008 | **0.044 ± 0.001** | **0.585 ± 0.018** | **0.817 ± 0.011** |
| MuRIL + rationale | HASOC + HX | 10 | 1 | 0.162 | 0.059 | 0.575 | 0.816 |
| *random baseline* | | | | *0.02 – 0.04* | *0.13 – 0.25* | *0.119* | *0.207* |

*Same runs, LIME on Hinglish (model-agnostic check)*: comprehensiveness 0.159 (MuRIL), 0.313 (XLM-R),
0.305 (control), 0.279 (λ 0.01), 0.168 (λ 0.1), 0.253 (λ 10); token F1 0.418, 0.459, 0.441, 0.481, 0.508, 0.534.

*English (HateXplain test, 500 posts): attention* token F1 / AUPRC: control 0.438 / 0.792 → λ 0.01
0.480 / 0.899 → λ 0.1 0.487 / 0.919 → λ 10 0.496 / 0.935; comprehensiveness 0.332 → 0.284 → 0.194 → 0.270.

**What this says about the research question** (rationale runs compared with the λ = 0 control, which has
the same training data):

- **Plausibility transfers from English to Hinglish.** Supervising the [CLS] attention with *English*
  HateXplain rationales makes it agree more with *Hinglish* human rationales: token F1 0.53 → 0.57–0.59 and
  AUPRC 0.74 → 0.80–0.82, for every λ tried and by more than the seed-to-seed spread. As expected, the gain
  is largest on English, where the supervision was given (AUPRC 0.79 → 0.90–0.94).
- **Faithfulness does not clearly improve.** Sufficiency gets better (0.082 → 0.044 at λ = 0.1): the
  attended words alone keep the prediction. But comprehensiveness does not rise; it stays level (λ = 0.01)
  or falls (λ = 0.1: 0.204 → 0.124). Deleting the attended words does not hurt the supervised models
  more than the control. So the answer to "does English rationale supervision make explanations more
  faithful on code-mixed text?" is **no on comprehensiveness, partly yes on sufficiency**: it mainly makes
  the explanations look more human-like.
- Much of the comprehensiveness drop is a change in the models' *probability scale*, not only in the
  explanations: the λ = 0.1 models (like MuRIL on HASOC only) never output P(HOF) above ≈ 0.75 or below
  ≈ 0.26 on the test set, while the control spans 0.11–0.90. A deletion can therefore lower the
  confidence by at most ≈ 0.48, which caps comprehensiveness; even LIME and *random* deletions score lower
  (random 0.044 → 0.020). Relative to random, attention comprehensiveness is 4.7× (control),
  5.5× (λ 0.01), 6.1× (λ 0.1) and 5.7× (λ 10).
- Comprehensiveness also underrates explanations of posts with many abusive words: deleting the top 1–2
  words leaves other slurs, so the prediction does not move (e.g. posts with comprehensiveness ≈ 0 whose
  top attention words are all slurs).
- **Adding English data alone already changed the explanations.** MuRIL on HASOC only → control (HASOC + HX,
  λ = 0): attention comprehensiveness 0.096 → 0.204, while test macro-F1 fell slightly (0.834 → 0.825,
  issue #3). The control row is what separates this from the effect of the rationale loss.
- LIME stays more faithful than attention for every model; attention stays more plausible for MuRIL.

**Limitations.** 1–3 seeds per configuration (λ = 1 cannot be evaluated; λ = 10 has a single seed) and no
significance test; the Hinglish rationales were LLM-suggested and accepted unchanged by the reviewers, so
Hinglish plausibility partly measures agreement with that LLM; HateXplain plausibility uses 500 of the
1,924 test posts; LIME used 200 samples per post; models were explained in fp16.

## Tasks

The work is split into four tasks (Member 1 – 4). **Any member can take any one task.**
The actual work is tracked as GitHub Issues. See [docs/TASKS.md](docs/TASKS.md).

## Team

| Name | GitHub |
|---|---|
| Anushri Mishra | |
| Manoj Dewda | |
| Kuldeep Saraswat | |
| Mausam Kumari | [@Mausam-8877-tech](https://github.com/Mausam-8877-tech) |

## References

1. Mathew et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021. <https://arxiv.org/abs/2012.10289>
2. HASOC shared task. <https://hasocfire.github.io>
3. Kim. *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014. <https://arxiv.org/abs/1408.5882>
4. Yang et al. *Hierarchical Attention Networks for Document Classification.* NAACL 2016.
