"""Run one config with several seeds and report mean +- std (issue #3: "3 seeds").

Usage (from the repo root):

    python scripts/run_seeds.py --config configs/muril_vanilla.yaml
    python scripts/run_seeds.py --config configs/muril_rationale.yaml --set loss.lambda=0.1
    python scripts/run_seeds.py --config configs/xlmr_vanilla.yaml --seeds 42 43 44

Each seed is a normal ``scripts/train.py`` run in its own folder:
    results/<run_name>[_lam<lambda>]_seed<seed>/      (best.pt, metrics.json, ...)
and the summary goes to:
    results/<run_name>[_lam<lambda>]_summary.json

Q&A
---
Q: Why several seeds?
A: Fine-tuning transformers on a few thousand posts is noisy: the random init of the
   classifier layer, dropout and the batch order can move macro-F1 by 1-2 points. A
   difference between two models only means something if it is larger than this noise.
   Reporting mean +- standard deviation over 3 seeds shows how big the noise is.

Q: Which standard deviation?
A: The sample standard deviation (divide by n - 1, ``statistics.stdev``), because the 3 runs
   are a sample of all the runs we could have made.

Q: Why is lambda put into the folder name?
A: The lambda sweep uses ONE config with different ``--set loss.lambda=...`` values; without
   it in the name every lambda would overwrite the previous one's results.
"""

# argparse: command-line options.
import argparse
# statistics: mean and sample standard deviation.
import statistics
# sys / Path: make "import train" and "import xhate" work.
import sys
from pathlib import Path

# Repo root = one folder above this file.
REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo root for "import xhate"; scripts/ for "import train".
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# train.main runs one experiment and returns its metrics dict.
import train  # noqa: E402
# Same config reader as train.py (to build the run name).
from xhate.utils import load_config, save_json  # noqa: E402


def main(argv=None):
    """Train once per seed, then print and save mean +- std of test accuracy / macro-F1."""
    # Command-line options.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="path of a YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="seeds to run")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results", help="output folder")
    args = parser.parse_args(argv)

    # Read the config once (with overrides) to build the base run name.
    cfg = load_config(args.config, overrides=args.set)
    base = cfg.get("run_name", args.config.stem)
    # Add the lambda when a run uses the rationale loss, e.g. muril_rationale_lam0.1.
    lam = (cfg.get("loss") or {}).get("lambda")
    if lam:
        base = f"{base}_lam{lam:g}"  # :g prints 0.1 as "0.1" and 10.0 as "10"

    # One metrics dict per seed.
    runs = []
    for seed in args.seeds:
        # Same overrides + this seed + its own folder name.
        overrides = list(args.set) + [f"seed={seed}", f"run_name={base}_seed{seed}"]
        print(f"\n[seeds] ===== {base} | seed {seed} =====")
        # One full train.py run; returns metrics.json's content.
        runs.append(train.main(["--config", str(args.config), "--results-dir", str(args.results_dir),
                                "--set", *overrides]))

    # Mean and std of the test scores ("std" is 0 with a single seed).
    summary = {"run_name": base, "seeds": args.seeds}
    for key in ("accuracy", "macro_f1"):
        values = [r["test"][key] for r in runs]
        summary[key] = {"values": values,
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    # Save next to the per-seed folders.
    save_json(summary, args.results_dir / f"{base}_summary.json")
    # Ready-made README row: accuracy and macro-F1 as mean +- std.
    acc, f1 = summary["accuracy"], summary["macro_f1"]
    print(f"\n[seeds] {base}: TEST accuracy {acc['mean']:.4f} +- {acc['std']:.4f} | "
          f"macro-F1 {f1['mean']:.4f} +- {f1['std']:.4f}")
    print(f"| {base} | {acc['mean']:.4f} ± {acc['std']:.4f} | {f1['mean']:.4f} ± {f1['std']:.4f} |")
    # Returned for tests.
    return summary


# Run only when executed as a script.
if __name__ == "__main__":
    main()
