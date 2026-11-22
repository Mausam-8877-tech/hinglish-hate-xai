"""Explain a trained model and evaluate the explanations (Member 4, issue #4).

Usage (from the repo root):

    python scripts/explain.py --run results/muril_vanilla_seed42
    python scripts/explain.py --run results/muril_rationale_lambda1_lam1.0_seed42 --dataset hatexplain
    python scripts/explain.py --run results/baseline_bilstm --methods attention lime random
    python scripts/explain.py --run results/baseline_cnn --methods lime random --max-posts 300

Outputs go to <run>/explain_<dataset>_<split>/ (inside results/, so git-ignored):
    explanations.jsonl   one line per post: tokens, gold, prediction, P(HOF), word scores of
                         every method, human rationale (if any), per-post faithfulness
    metrics.json         faithfulness (ERASER) + plausibility per method, + classification scores
    report.html          metrics table + highlighted posts (good / bad examples) -> open in a browser

The steps inside ``main()``:
    1. reload the trained model from its run folder (xhate.runs.load_run)
    2. load the posts (+ our Hinglish rationale annotations for HASOC)
    3. predict every post once
    4. for every method: word scores -> faithfulness -> plausibility
    5. save the three output files and print a summary row for the README table

Q&A
---
Q: Which data should a model be explained on?
A: * --dataset hasoc (default): the Hinglish TEST posts. Faithfulness on these answers the
     research question; plausibility uses our own annotations (data/annotations/).
   * --dataset hatexplain: the English test posts with HateXplain's human rationales
     (plausibility on English, i.e. on the data the rationale loss was trained with).

Q: Why are faithfulness numbers reported twice ("all" and "predicted_hof")?
A: "all" is ERASER's definition: every post, explained w.r.t. its predicted class. But in hate
   detection we mostly care about "why did you call this HOF?", so we also report the
   subset of posts the model predicted HOF. Both come from the same per-post numbers.

Q: Why does the CNN have no "attention" row?
A: The CNN has no attention layer. LIME (and the random baseline) work for every model.

Q: Why --max-posts?
A: LIME needs ``--lime-samples`` model calls PER POST. 1,314 posts x 500 samples is ~650k
   forward passes: fine on a GPU, slow on a laptop CPU. --max-posts takes a seeded random
   sample, so a quick run is still unbiased and reproducible.

Q: Why must every run be explained with the same options?
A: The final table compares vanilla vs rationale-supervised models. Same posts, same seed,
   same LIME samples and same bins -> the only thing that differs is the model.
"""

# argparse: command-line options.
import argparse
# json: one JSON object per line in explanations.jsonl.
import json
# random: seeded sampling for --max-posts.
import random
# sys: import path and console encoding.
import sys
# time: report how long each method takes.
import time
# Path: OS-independent paths.
from pathlib import Path

# Repo root (this file is repo/scripts/explain.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Make "import xhate" work when the script is run directly.
sys.path.insert(0, str(REPO_ROOT))

# numpy: score arrays.
import numpy as np  # noqa: E402

# Member 1's loader.
from xhate.data import load_split  # noqa: E402
# Classification scores (Member 2's implementation, re-exported by evaluation/).
from xhate.evaluation.classification import compute_metrics  # noqa: E402
# Our Hinglish rationale annotations.
from xhate.evaluation.annotations import DEFAULT_ANNOTATIONS, attach_annotations, load_annotations  # noqa: E402
# ERASER faithfulness and plausibility.
from xhate.evaluation.faithfulness import ERASER_BINS, faithfulness, plausibility  # noqa: E402
# The explanation methods.
from xhate.explain import METHODS, explain_posts, scores_for_prediction  # noqa: E402
# Any model -> predict_proba.
from xhate.explain.predictor import Predictor  # noqa: E402
# The HTML report.
from xhate.explain.visualize import write_report  # noqa: E402
# Reload a trained run.
from xhate.runs import load_run  # noqa: E402
# JSON writing (UTF-8, Hindi readable).
from xhate.utils import save_json  # noqa: E402


def parse_args(argv=None):
    """Define and read the command-line options (argv=None -> the real command line)."""
    # Parser with this docstring as --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The run folder written by scripts/train.py.
    parser.add_argument("--run", required=True, type=Path, help="run folder (config.yaml + best.pt)")
    # Which posts to explain.
    parser.add_argument("--dataset", default="hasoc", choices=["hasoc", "hatexplain"], help="posts to explain")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="split to explain")
    # Explanation methods.
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                        help="explanation methods (attention is skipped for the CNN)")
    # Seeded subsample (see Q&A).
    parser.add_argument("--max-posts", type=int, default=None, help="explain a random sample of this many posts")
    # LIME cost / stability trade-off.
    parser.add_argument("--lime-samples", type=int, default=500, help="LIME perturbations per post")
    # Seed for sampling, LIME and the random baseline.
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Hinglish annotations (only used for --dataset hasoc).
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS,
                        help="our Hinglish rationales (jsonl), used for --dataset hasoc")
    # Other folders and settings.
    parser.add_argument("--processed-dir", type=Path, default=None, help="processed data (default data/processed)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default <run>/explain_<dataset>_<split>)")
    parser.add_argument("--batch-size", type=int, default=32, help="posts per forward pass")
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto")
    parser.add_argument("--html-posts", type=int, default=50, help="posts shown in the report's last section")
    # Read the options.
    return parser.parse_args(argv)


def load_posts(args):
    """The posts to explain (+ annotations for HASOC, + seeded subsample)."""
    # Member 1's processed records.
    records = load_split(args.dataset, args.split, args.processed_dir)
    # HASOC has no rationales of its own: add ours where we annotated a post.
    if args.dataset == "hasoc":
        # {id: annotation}, empty if nothing is annotated yet.
        annotations = load_annotations(args.annotations)
        # Copy the masks onto matching posts.
        matched = attach_annotations(records, annotations)
        print(f"[explain] Hinglish rationale annotations matched: {matched} of {len(records)} posts")
    # Seeded random subsample (see Q&A); sorted back into file order.
    if args.max_posts is not None and args.max_posts < len(records):
        # Which positions to keep.
        keep = sorted(random.Random(args.seed).sample(range(len(records)), args.max_posts))
        # The sampled posts.
        records = [records[i] for i in keep]
    # Done.
    return records


def fmt(value):
    """A metric with 4 decimals, or "–" when it does not exist (e.g. no human rationales)."""
    # None -> dash; numbers -> 4 decimals.
    return "–" if value is None else f"{value:.4f}"


def subset_mean(per_post, mask, key):
    """Mean of ``per_post[i][key]`` over the posts where mask[i] is True (None if none)."""
    # The selected values.
    values = [p[key] for p, m in zip(per_post, mask) if m]
    # Mean, or None for an empty subset.
    return float(np.mean(values)) if values else None


def main(argv=None):
    """Explain one run end to end. Returns the metrics dict (useful for tests)."""
    # Print Hindi / emoji safely on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # ===== 1. model ============================================================
    args = parse_args(argv)
    # Rebuild the model with its best weights.
    run = load_run(args.run, device=args.device)
    # Wrap it as predict_proba(token lists).
    predictor = Predictor.from_run(run, batch_size=args.batch_size)
    # Output folder.
    out_dir = args.out or args.run / f"explain_{args.dataset}_{args.split}"
    print(f"[explain] run {run.name} | model {run.model_type} | device {run.device} -> {out_dir}")

    # ===== 2. posts ============================================================
    records = load_posts(args)
    # The word lists everything works on.
    token_lists = [rec["tokens"] for rec in records]
    # Human rationale per post (None where there is none).
    gold = [rec["rationale"] if rec.get("has_rationale") else None for rec in records]
    print(f"[explain] posts: {len(records)} | with human rationale: {sum(g is not None for g in gold)}")

    # ===== 3. predictions ======================================================
    # Predicted class and probabilities of every post (one pass).
    preds, probs = predictor.predict(token_lists)
    # Gold labels.
    labels = [rec["label"] for rec in records]
    # True where the model predicted HOF (for the "predicted_hof" faithfulness subset).
    is_hof = preds == 1

    # ===== 4. explanations + metrics ===========================================
    # Transformers always have [CLS] attention; a BiLSTM only with use_attention; the CNN never.
    has_attention = run.kind == "transformer" or getattr(run.model, "use_attention", False)
    # Drop "attention" for models without it (see Q&A), keep every other method.
    methods = [m for m in args.methods if m != "attention" or has_attention]
    # Say so, so a missing row in the table is not a surprise.
    if len(methods) < len(args.methods):
        print("[explain] this model has no attention: skipping the 'attention' method")
    # Raw scores per method (HOF direction for LIME), and metrics per method.
    all_scores, method_metrics, per_post_faith = {}, {}, {}
    for method in methods:
        # Time this method (LIME is by far the slowest).
        t0 = time.time()
        # One score array per post.
        scores = explain_posts(predictor, token_lists, method, lime_samples=args.lime_samples, seed=args.seed)
        # Orient towards the predicted class for faithfulness (flips LIME for NOT predictions).
        oriented = [scores_for_prediction(s, method, p) for s, p in zip(scores, preds)]
        # ERASER comprehensiveness / sufficiency.
        faith = faithfulness(predictor, token_lists, oriented, preds=preds)
        # Agreement with human rationales (only posts that have one).
        plaus = plausibility([s for s, g in zip(scores, gold) if g is not None], [g for g in gold if g is not None])
        # Keep everything.
        all_scores[method], per_post_faith[method] = scores, faith["per_post"]
        method_metrics[method] = {
            "faithfulness": {
                "all": {k: faith[k] for k in ("comprehensiveness", "sufficiency", "per_bin", "n_posts")},
                "predicted_hof": {"comprehensiveness": subset_mean(faith["per_post"], is_hof, "comprehensiveness"),
                                  "sufficiency": subset_mean(faith["per_post"], is_hof, "sufficiency"),
                                  "n_posts": int(is_hof.sum())},
            },
            "plausibility": plaus,
            "seconds": round(time.time() - t0, 1),
        }
        # One progress line per method (token F1 is "–" without human rationales).
        print(f"[explain] {method:9s} | comp {faith['comprehensiveness']:.4f} | suff {faith['sufficiency']:.4f} | "
              f"token F1 {fmt(plaus['token_f1'])} | {method_metrics[method]['seconds']}s")

    # ===== 5. save =============================================================
    # Everything about this explanation run.
    metrics = {
        "run_name": run.name,                                         # folder name of the run
        "model": run.model_type,                                      # cnn / bilstm / transformer
        "encoder": run.cfg["model"].get("name"),                      # MuRIL / XLM-R name (None for baselines)
        "lambda": float((run.cfg.get("loss") or {}).get("lambda", 0.0)),  # rationale-loss weight (0 = vanilla)
        "dataset": args.dataset,                                      # which posts were explained
        "split": args.split,                                          # train / val / test
        "n_posts": len(records),                                      # how many posts
        "bins": list(ERASER_BINS),                                    # ERASER top-k fractions
        "lime_samples": args.lime_samples,                            # LIME perturbations per post
        "seed": args.seed,                                            # sampling / LIME / random seed
        "classification": compute_metrics(labels, preds.tolist()),    # accuracy etc. on these posts
        "methods": method_metrics,                                    # faithfulness + plausibility per method
    }
    save_json(metrics, out_dir / "metrics.json")
    # One line per post.
    posts = []
    for i, rec in enumerate(records):
        posts.append({
            "id": rec["id"], "text": rec["text"], "tokens": rec["tokens"],          # the post
            "label": int(rec["label"]), "pred": int(preds[i]),                      # gold / predicted
            "prob_hof": round(float(probs[i, 1]), 4),                               # P(HOF)
            "gold_rationale": gold[i],                                              # human mask or None
            "scores": {m: [round(float(x), 5) for x in all_scores[m][i]] for m in methods},  # per word
            "faithfulness": {m: per_post_faith[m][i] for m in methods},             # per post
        })
    # Write explanations.jsonl (UTF-8, Hindi readable).
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "explanations.jsonl").open("w", encoding="utf-8") as f:
        for post in posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    # The HTML report.
    if methods and posts:
        write_report(out_dir / "report.html", title=f"{run.name} · {args.dataset} {args.split}",
                     metrics=metrics, posts=posts, methods=methods, n_posts=args.html_posts)
    # Where everything went.
    print(f"[explain] saved to {out_dir}")
    # Ready-made rows for the README table: run | method | comp | suff | token F1 | AUPRC.
    for method in methods:
        # This method's faithfulness (all posts) and plausibility.
        f_all, plaus = method_metrics[method]["faithfulness"]["all"], method_metrics[method]["plausibility"]
        # One markdown row.
        print(f"| {run.name} | {method} | {fmt(f_all['comprehensiveness'])} | {fmt(f_all['sufficiency'])} | "
              f"{fmt(plaus['token_f1'])} | {fmt(plaus['auprc'])} |")
    # Return the metrics (the end-to-end test checks them).
    return metrics


# Run main() only when executed as a script (not when imported by a test).
if __name__ == "__main__":
    main()
