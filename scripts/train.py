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

The steps inside ``main()``:
    1. read the config (+ --set overrides) and fix the random seed
    2. load train / val / test from Member 1's processed files
    3. build the vocabulary from TRAIN only, and the DataLoaders
    4. build the model and the (class-weighted) loss
    5. train with early stopping on val macro-F1 (``fit``)
    6. score the TEST set once with the best weights
    7. save every output file and print a summary

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

Q: Why save the config and the vocabulary next to the weights?
A: The weights alone are useless: to rebuild the model you need the same
   settings (layer sizes) and the same word -> id table. Saving all three in one
   folder means any member can reload the exact model later (e.g. for LIME).

Q: Why is only the train split shuffled?
A: Shuffling train gives the model a different mix of posts in every batch,
   which helps learning. Val / test are only scored, so their order does not
   matter; keeping it fixed lets us match predictions back to the posts.

Q: Why does main() accept ``argv``?
A: So a test can call ``main([...])`` with its own options instead of the real
   command line (see tests/test_baseline.py).
"""

# argparse: reads the command-line options (--config, --set, --results-dir).
import argparse
# json: writes the test predictions as one JSON object per line.
import json
# sys: lets us add the repo root to Python's import path (see below).
import sys
# Path: OS-independent file paths (works on Windows and Linux).
from pathlib import Path

# Repo root = one folder above this file (repo/scripts/train.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Put the repo root first on the import path, so "import xhate" works when the
# script is run directly with "python scripts/train.py".
sys.path.insert(0, str(REPO_ROOT))

# torch: creates the random generator that fixes the shuffling order.
import torch  # noqa: E402  (E402 = import not at the top; needed after sys.path.insert)
# DataLoader: cuts a Dataset into batches (and shuffles train).
from torch.utils.data import DataLoader  # noqa: E402

# load_all: Member 1's loader -> {"train": [...], "val": [...], "test": [...]}.
from xhate.data import load_all  # noqa: E402
# build_model: creates the CNN or BiLSTM from the "model:" config section.
from xhate.models.baseline import build_model  # noqa: E402
# Vocab: word <-> id table; WordDataset + collate_words: posts -> padded id batches.
from xhate.models.vocab import Vocab, WordDataset, collate_words  # noqa: E402
# class_weights: "balanced" weights against class imbalance.
from xhate.training.losses import class_weights  # noqa: E402
# The shared training loop, scoring and metrics.
from xhate.training.trainer import compute_metrics, default_loss_fn, fit, predict  # noqa: E402
# Config loading/saving, seeding, device choice and JSON writing.
from xhate.utils import get_device, load_config, save_config, save_json, set_seed  # noqa: E402

# Model types this script can train today (transformers come with issue #3).
BASELINE_TYPES = {"cnn", "bilstm"}


def parse_args(argv=None):
    """Define and read the command-line options.

    Args:
        argv: list of option strings; None means "use the real command line".
    """
    # The parser; this file's docstring becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # keep line breaks in --help
    # The YAML config of the experiment (required).
    parser.add_argument("--config", required=True, type=Path, help="path of a YAML config")
    # Optional overrides, e.g. --set train.lr=0.0005 model.dropout=0.3
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",  # nargs="*": zero or more values
                        help="override config values (dotted keys)")  # shown by --help
    # Where run folders are created (default: results/ in the repo).
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="folder for run outputs (default: results/)")  # shown by --help
    # Read the options and return them (args.config, args.set, args.results_dir).
    return parser.parse_args(argv)


def make_loaders(splits, vocab, cfg, seed):
    """Build the train / val / test DataLoaders.

    Args:
        splits: {"train": records, "val": records, "test": records}.
        vocab: the Vocab built from the training posts.
        cfg: the full config dict.
        seed: random seed for the shuffling order.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    # Posts longer than this many words are cut (default 100).
    max_length = cfg["data"].get("max_length", 100)
    # Number of posts per batch (default 32).
    batch_size = cfg["train"].get("batch_size", 32)
    # A seeded random generator: the shuffling order is the same in every run.
    generator = torch.Generator().manual_seed(seed)
    # One DataLoader per split (dict comprehension over train / val / test).
    return {
        split: DataLoader(WordDataset(records, vocab, max_length),  # posts -> word-id tensors
                          batch_size=batch_size,                    # posts per batch
                          shuffle=(split == "train"),               # shuffle ONLY train (see Q&A)
                          collate_fn=collate_words,                 # pads each batch to equal length
                          generator=generator if split == "train" else None)  # seeded shuffling for train
        for split, records in splits.items()  # one loader per split
    }  # keys: "train", "val", "test"


def save_predictions(records, out, path):
    """Write one JSON line per test post (for error analysis).

    Args:
        records: the test records, in the same order as the predictions.
        out: the dict returned by ``predict`` ("preds", "probs", ...).
        path: output file, e.g. results/<run>/test_predictions.jsonl.
    """
    # Make sure the output folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Open the file as UTF-8 so Hindi text stays readable.
    with Path(path).open("w", encoding="utf-8") as f:
        # Walk over posts and predictions together; they are in the same order
        # because the test loader is NOT shuffled.
        for rec, pred, prob in zip(records, out["preds"], out["probs"]):
            # One JSON object per line.
            f.write(json.dumps({
                "id": rec["id"],                       # post id
                "text": rec["text"],                   # cleaned text
                "label": rec["label"],                 # gold label (0 NOT, 1 HOF)
                "pred": pred,                          # predicted label
                "prob_hof": round(prob, 4),            # model's P(HOF), 4 decimals
                "correct": pred == rec["label"],       # handy filter for error analysis
            }, ensure_ascii=False) + "\n")             # ensure_ascii=False keeps Devanagari readable


def main(argv=None):
    """Run one experiment end to end. Returns the metrics dict (useful for tests)."""
    # ===== 1. config + seed =====================================================
    # Read the command-line options.
    args = parse_args(argv)
    # Read the YAML config and apply the --set overrides.
    cfg = load_config(args.config, overrides=args.set)
    # Which model to train ("cnn" or "bilstm").
    model_type = cfg["model"]["type"]
    # Stop with a clear message for model types this script does not support yet.
    if model_type not in BASELINE_TYPES:
        raise SystemExit(f"model.type {model_type!r} is not supported yet "  # exits with this message
                         f"(baselines: {sorted(BASELINE_TYPES)}; transformers: issue #3)")  # names the issue to follow
    # Run name = folder name under results/ (falls back to the config file name).
    run_name = cfg.get("run_name", args.config.stem)
    # The folder all outputs of this run go into.
    run_dir = args.results_dir / run_name
    # Random seed from the config (default 42).
    seed = cfg.get("seed", 42)
    # Fix Python, NumPy and PyTorch random generators.
    set_seed(seed)
    # Train on the GPU if there is one, otherwise the CPU ("auto").
    device = get_device(cfg["train"].get("device", "auto"))
    # Tell the user what is about to happen.
    print(f"[train] run {run_name} | model {model_type} | device {device}")

    # ===== 2. data ==============================================================
    # train / val / test records from Member 1's processed files.
    splits = load_all(cfg["data"]["dataset"], cfg["data"].get("processed_dir"))
    # Print how many posts each split has.
    print("[train] posts: " + ", ".join(f"{s}={len(r)}" for s, r in splits.items()))

    # ===== 3. vocabulary + loaders ==============================================
    # Vocabulary from the TRAINING posts only (val/test words may be unknown).
    vocab = Vocab.build([r["tokens"] for r in splits["train"]],     # every training post's word list
                        min_freq=cfg["data"].get("min_freq", 1),    # rarer words become <unk>
                        max_size=cfg["data"].get("max_vocab"))      # None = no size limit
    # Print the vocabulary size (= rows of the embedding table).
    print(f"[train] vocabulary: {len(vocab)} entries (incl. <pad>, <unk>)")
    # Batches for train / val / test.
    loaders = make_loaders(splits, vocab, cfg, seed)

    # ===== 4. model + loss ======================================================
    # Create the CNN or BiLSTM; the embedding table needs one row per vocabulary entry.
    model = build_model(cfg["model"], vocab_size=len(vocab))
    # Count the weights that training will update (numel = number of values in a tensor).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print the model size (useful for the report).
    print(f"[train] trainable parameters: {n_params:,}")
    # No class weights unless the config asks for them.
    weight = None
    # Default is True: weight the loss against class imbalance.
    if cfg["train"].get("class_weights", True):
        # Weights from the TRAINING labels only.
        weight = class_weights([r["label"] for r in splits["train"]])
        # Print them (close to 1.0 means the data is nearly balanced).
        print(f"[train] class weights NOT={weight[0]:.3f} HOF={weight[1]:.3f}")
    # Cross-entropy loss, class-weighted if ``weight`` is set.
    loss_fn = default_loss_fn(weight)

    # ===== 5. training ==========================================================
    # Short name for the "train:" section of the config.
    tcfg = cfg["train"]
    # Train with early stopping; the best weights are saved to best.pt and loaded back.
    result = fit(model, loaders["train"], loaders["val"], device,
                 epochs=tcfg.get("epochs", 20),              # maximum number of epochs
                 lr=tcfg.get("lr", 1e-3),                     # Adam learning rate
                 weight_decay=tcfg.get("weight_decay", 0.0),  # L2 penalty (0 = off)
                 patience=tcfg.get("patience", 3),            # early-stopping patience
                 grad_clip=tcfg.get("grad_clip", 5.0),        # maximum gradient norm
                 loss_fn=loss_fn,                             # the loss built above
                 checkpoint_path=run_dir / "best.pt")         # where the best weights go

    # ===== 6. final scores (best weights are already loaded by fit) ============
    # Predictions on the TEST set (scored only once, here).
    test_out = predict(model, loaders["test"], device)
    # Accuracy, macro-F1, per-class scores and confusion matrix on test.
    test_metrics = compute_metrics(test_out["labels"], test_out["preds"])
    # Everything worth keeping about this run, in one dict.
    metrics = {
        "run_name": run_name,                                  # folder name
        "model": model_type,                                   # cnn / bilstm
        "dataset": cfg["data"]["dataset"],                     # hasoc / hatexplain
        "best_epoch": result["best_epoch"],                    # epoch chosen by early stopping
        "trainable_parameters": n_params,                      # model size
        "vocab_size": len(vocab),                              # incl. <pad>, <unk>
        "n_posts": {s: len(r) for s, r in splits.items()},     # posts per split
        "val": result["best_val"],                             # val metrics of the best epoch
        "test": test_metrics,                                  # test metrics of the best model
    }  # saved as metrics.json below

    # ===== 7. save everything ==================================================
    # The exact settings used (incl. --set overrides).
    save_config(cfg, run_dir / "config.yaml")
    # The word -> id table (needed to reload the model).
    vocab.save(run_dir / "vocab.json")
    # Loss / accuracy / macro-F1 per epoch (for training curves in the report).
    save_json(result["history"], run_dir / "history.json")
    # Val + test metrics.
    save_json(metrics, run_dir / "metrics.json")
    # One line per test post, for error analysis.
    save_predictions(splits["test"], test_out, run_dir / "test_predictions.jsonl")

    # ===== summary ==============================================================
    # Which epoch won and its validation score.
    print(f"[train] best epoch {result['best_epoch']} | val macro-F1 {result['best_val']['macro_f1']:.4f}")
    # The numbers that go into the report.
    print(f"[train] TEST accuracy {test_metrics['accuracy']:.4f} | TEST macro-F1 {test_metrics['macro_f1']:.4f}")
    # Where the outputs are.
    print(f"[train] saved to {run_dir}")
    # A ready-made row for the README results table.
    print(f"| {run_name} | {cfg['data']['dataset']} | {test_metrics['accuracy']:.4f} | "
          f"{test_metrics['macro_f1']:.4f} |")  # run | dataset | accuracy | macro-F1
    # Return the metrics (the end-to-end test checks them).
    return metrics


# Run main() only when executed as a script (not when imported by a test).
if __name__ == "__main__":
    main()  # start the script
