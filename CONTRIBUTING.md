# Contributing

## 1. Claim a task
1. Pick a free task in [docs/TASKS.md](docs/TASKS.md). Any member can take any task.
2. Open an Issue using the **Task** template and assign it to yourself.

## 2. Work on a branch (never directly on `main`)
```bash
git checkout main
git pull
git checkout -b task2/bilstm-baseline
```

## 3. Commit and push your branch
```bash
git add <files you changed>
git commit -m "Add BiLSTM attention baseline"
git push -u origin task2/bilstm-baseline
```

## 4. Open a Pull Request
- Open it on GitHub into `main` and fill in the PR template.
- Ask at least **one** other member to review it.
- Run `pytest` before you ask for a review.

## Do not commit
- Datasets (`data/raw/`, `data/processed/`). The HASOC licence forbids sharing them.
- Model checkpoints (`*.pt`) and `results/`. Share these on Google Drive instead.
- Notebook outputs with huge images or tables (clear the outputs first).

## Keep your branch up to date
```bash
git checkout main
git pull
git checkout <your-branch>
git merge main
```
