"""Train a model from a config file.

Usage (from the repo root):

    python scripts/train.py --config configs/baseline_cnn.yaml
    python scripts/train.py --config configs/baseline_bilstm.yaml
    python scripts/train.py --config configs/baseline_cnn.yaml --set train.lr=0.0005 model.dropout=0.3

Everything a run produces goes to results/<run_name>/ (git-ignored; share on Drive):
    config.yaml              the exact settings used (incl. --set overrides)
    vocab.json               word -> id table (needed to reload the model)
    best.pt                  weights of the epoch with the best val macro-F1
    history.json             loss / accuracy / macro-F1 per epoch (for training curves)
    metrics.json             val + test metrics of the best model
    test_predictions.jsonl   one line per test post: text, gold, predicted, P(HOF)
                             -> use it for error analysis

Q&A
---
Q: Which models does this script train?
A: The baselines (model.type: cnn or bilstm). Transformers (MuRIL / XLM-R) are
   issue #3; that task can add its model type here and reuse the same ``fit``.

Q: How is the run reproducible?
A: The seed in the config fixes weight initialisation, dropout and the order in
   which training batches are shuffled. On the CPU two runs with the same config
   give the same numbers; on a GPU small differences are possible.

Q: Why is the test set only scored once, at the end?
A: Choosing settings by looking at test scores would make them optimistic. We
   choose (early stopping, hyperparameters) on val, and report test only.
"""

# argparse: command-line options.
import argparse
# json: write the predictions file.
import json
# sys: make "import xhate" work when the script is run directly.
import sys
# Path: OS-independent paths.
from pathlib import Path

# Repo root (this file is repo/scripts/train.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Make "import xhate" work when the script is run directly.
sys.path.insert(0, str(REPO_ROOT))

# torch: DataLoader and the shuffling generator.
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Our modules.
from xhate.data import load_all  # noqa: E402
from xhate.models.baseline import build_model  # noqa: E402
from xhate.models.vocab import Vocab, WordDataset, collate_words  # noqa: E402
from xhate.training.losses import class_weights  # noqa: E402
from xhate.training.trainer import compute_metrics, default_loss_fn, fit, predict  # noqa: E402
from xhate.utils import get_device, load_config, save_config, save_json, set_seed  # noqa: E402

# Model types this script can train today.
BASELINE_TYPES = {"cnn", "bilstm"}


def parse_args(argv=None):
    """Define and read the command-line options."""
    # The parser, with this file's docstring as help text.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # The YAML config of the experiment.
    parser.add_argument("--config", required=True, type=Path, help="path of a YAML config")
    # Optional overrides, e.g. --set train.lr=0.0005 model.dropout=0.3
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values (dotted keys)")
    # Where run folders are created.
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="folder for run outputs (default: results/)")
    # Return the parsed options.
    return parser.parse_args(argv)


def make_loaders(splits, vocab, cfg, seed):
    """Build the train / val / test DataLoaders."""
    # Settings from the config.
    max_length = cfg["data"].get("max_length", 100)
    batch_size = cfg["train"].get("batch_size", 32)
    # A seeded generator makes the shuffling order the same in every run.
    generator = torch.Generator().manual_seed(seed)
    # One loader per split; only train is shuffled.
    return {
        split: DataLoader(WordDataset(records, vocab, max_length), batch_size=batch_size,
                          shuffle=(split == "train"), collate_fn=collate_words,
                          generator=generator if split == "train" else None)  # shuffle only train
        for split, records in splits.items()
    }


def save_predictions(records, out, path):
    """Write one JSON line per test post (for error analysis)."""
    # Make sure the folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 so Hindi stays readable.
    with Path(path).open("w", encoding="utf-8") as f:
        # records and predictions are in the same order (the test loader is not shuffled).
        for rec, pred, prob in zip(records, out["preds"], out["probs"]):
            f.write(json.dumps({
                "id": rec["id"],                       # post id
                "text": rec["text"],                   # cleaned text
                "label": rec["label"],                 # gold label (0 NOT, 1 HOF)
                "pred": pred,                          # predicted label
                "prob_hof": round(prob, 4),            # model's P(HOF)
                "correct": pred == rec["label"],       # handy filter for error analysis
            }, ensure_ascii=False) + "\n")


def main(argv=None):
    """Run one experiment. Returns the metrics dict (useful for tests)."""
    # Read the options and the config.
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    # Only baselines are supported for now (see Q&A).
    model_type = cfg["model"]["type"]
    if model_type not in BASELINE_TYPES:
        raise SystemExit(f"model.type {model_type!r} is not supported yet "
                         f"(baselines: {sorted(BASELINE_TYPES)}; transformers: issue #3)")
    # Run name = folder name under results/.
    run_name = cfg.get("run_name", args.config.stem)
    run_dir = args.results_dir / run_name
    # Fix every random generator.
    seed = cfg.get("seed", 42)
    set_seed(seed)
    # CPU or GPU.
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"[train] run {run_name} | model {model_type} | device {device}")

    # --- data -----------------------------------------------------------------
    # train / val / test records from Member 1's processed files.
    splits = load_all(cfg["data"]["dataset"], cfg["data"].get("processed_dir"))
    print("[train] posts: " + ", ".join(f"{s}={len(r)}" for s, r in splits.items()))
    # Vocabulary from the TRAINING posts only.
    vocab = Vocab.build([r["tokens"] for r in splits["train"]],
                        min_freq=cfg["data"].get("min_freq", 1),
                        max_size=cfg["data"].get("max_vocab"))
    print(f"[train] vocabulary: {len(vocab)} entries (incl. <pad>, <unk>)")
    # DataLoaders.
    loaders = make_loaders(splits, vocab, cfg, seed)

    # --- model + loss ---------------------------------------------------------
    model = build_model(cfg["model"], vocab_size=len(vocab))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable parameters: {n_params:,}")
    # Class-weighted loss if asked (default: yes).
    weight = None
    if cfg["train"].get("class_weights", True):
        weight = class_weights([r["label"] for r in splits["train"]])
        print(f"[train] class weights NOT={weight[0]:.3f} HOF={weight[1]:.3f}")
    loss_fn = default_loss_fn(weight)

    # --- training ---------------------------------------------------------------
    tcfg = cfg["train"]
    result = fit(model, loaders["train"], loaders["val"], device,
                 epochs=tcfg.get("epochs", 20), lr=tcfg.get("lr", 1e-3),
                 weight_decay=tcfg.get("weight_decay", 0.0), patience=tcfg.get("patience", 3),
                 grad_clip=tcfg.get("grad_clip", 5.0), loss_fn=loss_fn,
                 checkpoint_path=run_dir / "best.pt")

    # --- final scores (best weights are already loaded by fit) ---------------
    test_out = predict(model, loaders["test"], device)
    test_metrics = compute_metrics(test_out["labels"], test_out["preds"])
    metrics = {
        "run_name": run_name,
        "model": model_type,
        "dataset": cfg["data"]["dataset"],
        "best_epoch": result["best_epoch"],
        "trainable_parameters": n_params,
        "vocab_size": len(vocab),
        "n_posts": {s: len(r) for s, r in splits.items()},
        "val": result["best_val"],
        "test": test_metrics,
    }

    # --- save everything ------------------------------------------------------
    save_config(cfg, run_dir / "config.yaml")
    vocab.save(run_dir / "vocab.json")
    save_json(result["history"], run_dir / "history.json")
    save_json(metrics, run_dir / "metrics.json")
    save_predictions(splits["test"], test_out, run_dir / "test_predictions.jsonl")

    # --- summary ----------------------------------------------------------------
    print(f"[train] best epoch {result['best_epoch']} | val macro-F1 {result['best_val']['macro_f1']:.4f}")
    print(f"[train] TEST accuracy {test_metrics['accuracy']:.4f} | TEST macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"[train] saved to {run_dir}")
    # A ready-made row for the README results table.
    print(f"| {run_name} | {cfg['data']['dataset']} | {test_metrics['accuracy']:.4f} | "
          f"{test_metrics['macro_f1']:.4f} |")
    return metrics


# Run main() only when executed as a script (not when imported by a test).
if __name__ == "__main__":
    main()
