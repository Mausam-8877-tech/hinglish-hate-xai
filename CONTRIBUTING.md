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

## 4. Merge into `main`
You can merge your branch into `main` **directly**. A Pull Request and review are optional.

1. Run `pytest` and make sure all tests pass.
2. Merge and push:
```bash
git checkout main
git pull
git merge task2/bilstm-baseline
git push origin main
```
3. Tell the team what you merged (comment on your Issue), especially if it changes something
   other members use (data format, model interface, config files).

**Optional:** for big changes, or changes that affect other members' tasks, you can still open a
Pull Request on GitHub into `main` (fill in the PR template) and ask a member to review it.

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
