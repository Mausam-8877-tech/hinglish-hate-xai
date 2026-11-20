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
