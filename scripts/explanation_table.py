"""Final table: vanilla vs rationale-supervised, faithfulness and plausibility (Member 4).

After running scripts/explain.py on every run (same options for all!):

    python scripts/explanation_table.py --results-dir results --dataset hasoc
    python scripts/explanation_table.py --results-dir results --dataset hatexplain --out docs/results/explanations_hatexplain.csv

It finds every results/<run>/explain_<dataset>_<split>/metrics.json, groups the seeds of the same
configuration (muril_vanilla_seed42 / _seed43 / _seed44 -> muril_vanilla) and prints one markdown
row per configuration and method, as mean ± sample std over the seeds, ready for the README.

Q&A
---
Q: Why group by configuration and report mean ± std?
A: Member 3 found that seed-to-seed variation (std up to 0.011 macro-F1) is as large as the
   differences between configurations. The same can happen for explanation metrics, so a
   single seed could make a difference look real when it is noise.

Q: How is a configuration name found?
A: The run folder name with its "_seed<N>" suffix removed (run_seeds.py names folders
   "<run_name>[_lam<λ>]_seed<N>").

Q: Why sample std (n - 1)?
A: With only 3 seeds we estimate the spread of a population from a small sample; dividing by
   n - 1 corrects the bias of dividing by n. Member 3's table uses the same.
"""

# argparse: command-line options.
import argparse
# csv: optional CSV output.
import csv
# json: read metrics.json files.
import json
# re: strip the "_seed<N>" suffix.
import re
# sys: console encoding.
import sys
# defaultdict: group runs by configuration.
from collections import defaultdict
# Path: OS-independent paths.
from pathlib import Path

# numpy: mean and std.
import numpy as np

# The metric columns of the table: (key in our row dict, column title).
COLUMNS = [("comp", "Comprehensiveness ↑"), ("suff", "Sufficiency ↓"),
           ("comp_hof", "Comp. (pred. HOF) ↑"), ("suff_hof", "Suff. (pred. HOF) ↓"),
           ("token_f1", "Token F1 ↑"), ("auprc", "AUPRC ↑")]


def parse_args(argv=None):
    """Define and read the command-line options (argv=None -> the real command line)."""
    # Parser with this docstring as --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Folder holding the run folders.
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="folder with the run folders")
    # Which explanation outputs to read.
    parser.add_argument("--dataset", default="hasoc", choices=["hasoc", "hatexplain"], help="explained dataset")
    parser.add_argument("--split", default="test", help="explained split")
    # Optional CSV copy of the table.
    parser.add_argument("--out", type=Path, default=None, help="also write the table as CSV")
    # Read the options.
    return parser.parse_args(argv)


def config_name(run_name):
    """ "muril_vanilla_seed42" -> "muril_vanilla" (see Q&A)."""
    # Remove a trailing "_seed" + digits; other names stay as they are.
    return re.sub(r"_seed\d+$", "", run_name)


def run_rows(metrics):
    """One flat row per method from one explain metrics.json."""
    # Rows of this run.
    rows = []
    # One row per explanation method.
    for method, m in metrics["methods"].items():
        # Faithfulness on all posts / predicted-HOF posts, and plausibility.
        f_all, f_hof, plaus = m["faithfulness"]["all"], m["faithfulness"]["predicted_hof"], m["plausibility"]
        rows.append({"method": method,
                     "comp": f_all["comprehensiveness"], "suff": f_all["sufficiency"],    # ERASER, all posts
                     "comp_hof": f_hof["comprehensiveness"], "suff_hof": f_hof["sufficiency"],  # predicted HOF
                     "token_f1": plaus["token_f1"], "auprc": plaus["auprc"]})            # vs humans
    # Done.
    return rows


def mean_std(values):
    """ "0.4123 ± 0.0101" over the non-missing values ("–" if none; no ± for one value)."""
    # Drop missing values (e.g. no plausibility without rationales).
    values = [v for v in values if v is not None]
    # Nothing to summarise.
    if not values:
        return "–"
    # One seed: no spread to report.
    if len(values) == 1:
        return f"{values[0]:.4f}"
    # Mean ± sample standard deviation (ddof=1, see Q&A).
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def main(argv=None):
    """Collect every explain metrics.json, print the markdown table. Returns the table rows."""
    # Print "±" and "↑" safely on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Read the options.
    args = parse_args(argv)
    # All explain outputs for this dataset / split.
    files = sorted(args.results_dir.glob(f"*/explain_{args.dataset}_{args.split}/metrics.json"))
    # Nothing found: say what to run.
    if not files:
        raise SystemExit(f"no explain_{args.dataset}_{args.split}/metrics.json under {args.results_dir}; "
                         "run scripts/explain.py first")
    # (configuration, method) -> list of per-seed rows; plus seed counts and lambdas.
    groups, seeds, lambdas = defaultdict(list), defaultdict(int), {}
    # Read every run.
    for path in files:
        # The explain metrics of one run.
        metrics = json.loads(path.read_text(encoding="utf-8"))
        # Its configuration name (seeds grouped together).
        config = config_name(metrics["run_name"])
        # Count the seed and remember λ.
        seeds[config] += 1
        lambdas[config] = metrics.get("lambda", 0.0)
        # Add its rows to the groups.
        for row in run_rows(metrics):
            groups[(config, row["method"])].append(row)
    # Table rows: one per configuration and method.
    table = []
    for (config, method), rows in sorted(groups.items()):
        table.append({"configuration": config, "lambda": lambdas[config], "seeds": seeds[config], "method": method,
                      **{key: mean_std([r[key] for r in rows]) for key, _ in COLUMNS}})
    # Markdown header.
    header = ["Configuration", "λ", "Seeds", "Method"] + [title for _, title in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    # Markdown rows.
    for row in table:
        print("| " + " | ".join([row["configuration"], str(row["lambda"]), str(row["seeds"]), row["method"]]
                                + [row[key] for key, _ in COLUMNS]) + " |")
    # Optional CSV copy.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"[table] wrote {args.out}")
    # Return the rows (the test checks them).
    return table


# Run main() only when executed as a script (not when imported by a test).
if __name__ == "__main__":
    main()
