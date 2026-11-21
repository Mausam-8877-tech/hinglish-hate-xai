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
├── results/              run outputs and checkpoints (git-ignored)
├── scripts/              command-line entry points
│   ├── prepare_data.py
│   ├── train.py
│   ├── explain.py
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

Test = HASOC 2019 Hindi test, mean ± std over seeds 42, 43, 44. "+ HX" = HateXplain train
posts added to training (val/test stay HASOC). λ = weight of the rationale loss.

| Model | Train data | λ | Accuracy | Macro-F1 | Config |
|---|---|---|---|---|---|
| XLM-R base | HASOC | – | _todo_ | _todo_ | `configs/xlmr_vanilla.yaml` |
| MuRIL base | HASOC | – | _todo_ | _todo_ | `configs/muril_vanilla.yaml` |
| MuRIL base (control) | HASOC + HX | 0 | _todo_ | _todo_ | `configs/muril_hx_lambda0.yaml` |
| MuRIL base + rationale | HASOC + HX | 0.01 / 0.1 / 1 / 10 | _todo_ | _todo_ | `configs/muril_rationale.yaml` |

Needs a GPU (Colab / Kaggle). Reproduce:

```bash
python scripts/prepare_data.py                      # HASOC + HateXplain
python scripts/run_seeds.py --config configs/muril_vanilla.yaml
python scripts/run_seeds.py --config configs/xlmr_vanilla.yaml
python scripts/run_seeds.py --config configs/muril_hx_lambda0.yaml
for L in 0.01 0.1 1 10; do
  python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=$L
done
```

Each model returns its [CLS] attention for explanation:
`TransformerClassifier.get_attention(input_ids, attention_mask)` → `(B, L)`.

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
