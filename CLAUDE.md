# CLAUDE.md

Guidance for Claude (and any AI assistant) working in this repository.

## ⚠️ Git rules (most important)

- **Never run `git commit`, `git push`, `git merge`, `git rebase`, or any other command that
  changes git history or the remote.**
- When work is ready, **print the exact commands** (`git add ...`, `git commit -m ...`,
  `git push ...`) for the user to run themselves.
- Read-only git commands (`git status`, `git diff`, `git log`) are fine.
- Never commit datasets (`data/raw/`, `data/processed/`), checkpoints (`*.pt`), or `results/`.
  The HASOC licence does not allow us to redistribute the data.

## Scope rule

- Do only what the user asks. Work items are defined by **GitHub Issues** raised by the team.
  Don't implement project features that no issue or request asks for.

## Project

Explainable hate speech detection for code-mixed Hindi-English (Hinglish).
Deep Learning course project, Team 4 / P04.

- **Goal:** detect hate/offensive posts and highlight the words responsible.
- **Data:** HASOC Hindi/Hinglish tracks, plus HateXplain (English, with rationales).
- **Models:** CNN/LSTM baseline → MuRIL / XLM-R fine-tuning → rationale-supervised training.
- **Explanations:** LIME and attention.
- **Research question:** does rationale supervision from English improve faithfulness on
  code-mixed text?

## Task division

There are 4 tasks, **Member 1 – Member 4**. They are **not assigned to specific people**:
any team member can take any one task. See [docs/TASKS.md](docs/TASKS.md).

| Task | Area | Folders |
|---|---|---|
| Member 1 | Data & preprocessing | `xhate/data/`, `scripts/prepare_data.py` |
| Member 2 | Baselines (CNN / LSTM) | `xhate/models/baseline.py` |
| Member 3 | Transformers + rationale supervision | `xhate/models/transformer.py`, `xhate/training/` |
| Member 4 | Explainability & evaluation | `xhate/explain/`, `xhate/evaluation/` |

Team: Anushri Mishra, Manoj Dewda, Kuldeep Saraswat, Mausam Kumari.

## Layout

- `xhate/` is the Python package. Its modules are currently stubs, filled in through issues.
- `scripts/` holds the command-line entry points. Run them from the repo root.
- `configs/` has one YAML file per experiment.
- `notebooks/` notebooks are named `NN_<initials>_<topic>.ipynb`.

## Conventions

- Branches: `task1/<short-name>`, `task2/...`, and so on, or `fix/...`. Never work directly on `main`.
- When a branch is ready, it can be merged into `main` directly. A Pull Request and review are
  optional (recommended for big changes, or changes that affect other members' tasks).
- Before merging into `main`: run `pytest` and make sure all tests pass.
- Commit messages are short and imperative, e.g. `Add HASOC loader`.
