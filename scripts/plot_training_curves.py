"""Plot training curves (loss and validation macro-F1 per epoch) from saved runs.

Usage (from the repo root):

    python scripts/plot_training_curves.py                                   # results/ -> docs/results/
    python scripts/plot_training_curves.py --results-dir path/to/results --out-dir docs/results

It reads every ``<results-dir>/<run_name>_seed<N>/`` folder written by ``scripts/run_seeds.py``
(each holds ``history.json`` and ``metrics.json``) and writes:

    training_curves_val_macro_f1.png   validation macro-F1 per epoch, one panel per configuration,
                                       one line per seed; a ring marks the epoch kept by early stopping
    training_curves_loss.png           training (solid) and validation (dashed) loss per epoch
    training_history.csv               the numbers behind both plots (one row per run and epoch)

Q&A
---
Q: Why one panel per configuration and one line per seed?
A: Each configuration was trained with several seeds. Putting the seeds of one configuration
   together shows how much training varies just from randomness (initialisation, dropout, batch
   order). One panel per configuration keeps any single chart from having too many lines.

Q: Why are loss and macro-F1 in two separate figures?
A: They are on completely different scales. Putting both in one chart would need two y-axes,
   which is easy to misread. Two figures with one y-axis each are clearer.

Q: Why does each loss panel have its own y-axis, while the macro-F1 panels share one?
A: For rationale runs the TRAINING loss is ``CE + lambda * rationale_loss``. With lambda = 10 it
   is around 25, while plain cross-entropy is around 0.5. On a shared axis every other panel
   would look flat. Training loss is not comparable across lambdas anyway. Validation loss is
   plain cross-entropy (HASOC val has no rationales), and macro-F1 is always in [0, 1], so the
   macro-F1 panels can share an axis and be compared directly.

Q: How do you read the curves?
A: Training loss going down while validation loss goes up = overfitting (the model memorises
   the training posts). That is why ``fit`` keeps the epoch with the best validation macro-F1
   (the ringed point) rather than the last epoch.

Q: Why both colour AND marker shape per seed?
A: Some readers cannot tell certain colours apart (colour-vision deficiency), and printouts may
   be grey. A different marker per seed (circle / square / triangle) plus the legend means the
   chart does not depend on colour alone. The CSV gives the exact numbers as a table.

Q: What if a configuration is missing some seeds?
A: The panel title says how many seeds have a history (e.g. "2/3 seeds with history"). Runs
   without a ``history.json`` simply cannot be plotted.
"""

# argparse: command-line options.
import argparse
# csv: writes the table of numbers.
import csv
# json: reads history.json / metrics.json.
import json
# math: ceil() for the number of panel rows.
import math
# re: splits a folder name like "muril_vanilla_seed42" into ("muril_vanilla", 42).
import re
# Path: OS-independent file paths.
from pathlib import Path

# Repo root = one folder above this file (repo/scripts/ -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Colour + marker per seed. Colours are a colour-blind-checked categorical palette (blue, orange,
# aqua); a seed always gets the same style, so the same seed looks the same in every panel.
SEED_STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
# Text, secondary text, grid and background colours (quiet, so the data lines stand out).
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Folder names end in "_seed<number>"; this pattern captures the base name and the seed.
RUN_PATTERN = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def load_runs(results_dir):
    """Collect every run folder that has a history.json.

    Returns:
        {base_name: {seed: {"history": [...], "metrics": {...}}}}
        e.g. {"muril_vanilla": {42: {...}, 43: {...}}}
    """
    # Grouped runs, filled below.
    runs = {}
    # Every history.json one level below results_dir, in a stable (sorted) order.
    for history_path in sorted(Path(results_dir).glob("*/history.json")):
        # The folder name, e.g. "muril_vanilla_seed42".
        match = RUN_PATTERN.match(history_path.parent.name)
        # Skip folders that are not "<name>_seed<N>" (e.g. a single train.py run).
        if not match:
            continue
        # metrics.json sits next to history.json (best_epoch, lambda, encoder, ...).
        metrics_path = history_path.parent / "metrics.json"
        # Read both files as UTF-8.
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        # Group by configuration, then by seed.
        runs.setdefault(match["base"], {})[int(match["seed"])] = {"history": history, "metrics": metrics}
    # Return the grouped runs.
    return runs


def panel_title(base, seeds, n_expected):
    """Readable panel title: what was trained, plus how many seeds have a history."""
    # Any run's metrics describe the configuration (they share encoder, data and lambda).
    metrics = next(iter(seeds.values()))["metrics"]
    # Short model name from the encoder's hub name.
    encoder = str(metrics.get("encoder", ""))
    model = "XLM-R" if "xlm" in encoder.lower() else "MuRIL" if "muril" in encoder.lower() else base
    # Training data, e.g. "hasoc+hatexplain" -> "HASOC+HX".
    data = str(metrics.get("dataset", "")).replace("hatexplain", "HX").replace("hasoc", "HASOC")
    # lambda of the rationale loss (0 for vanilla / control runs).
    lam = metrics.get("lambda", 0) or 0
    # e.g. "MuRIL, HASOC+HX, λ = 0.1".
    what = f"{model}, {data}" + (f", λ = {lam:g}" if "HX" in data else "")
    # Second line: seeds with a history out of the expected number.
    return f"{what}\n{len(seeds)}/{n_expected} seeds with history"


def plot_figure(runs, order, seed_list, key, ylabel, title, out_path, dashed_key=None, share_y=True):
    """Draw one figure: a grid of panels (one per configuration), one line per seed.

    Args:
        runs: output of ``load_runs``.
        order: configuration names in panel order.
        seed_list: all seeds, so each seed always gets the same colour / marker.
        key: history field drawn as a solid line (e.g. "val_macro_f1", "train_loss").
        ylabel / title: axis label and figure title.
        out_path: PNG file to write.
        dashed_key: optional second history field drawn dashed in the same colour (e.g. "val_loss").
        share_y: True = all panels use the same y-axis (only when values are comparable).
    """
    # Imported here so that importing this module (e.g. in a test) does not open a window.
    import matplotlib
    matplotlib.use("Agg")  # "Agg" draws straight to a file; no screen needed (works on Kaggle / CI)
    import matplotlib.pyplot as plt

    # Quiet styling: grey axes and labels, light background.
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    # Grid of panels: 3 per row.
    ncols = min(3, len(order))
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows + 1.2),
                             sharex=True, sharey=share_y, squeeze=False)
    # Fixed style per seed (seed 42 -> first style, and so on).
    style = {seed: SEED_STYLES[i % len(SEED_STYLES)] for i, seed in enumerate(seed_list)}
    # Highest epoch count, for the x-axis ticks.
    max_epochs = max(len(r["history"]) for seeds in runs.values() for r in seeds.values())

    # One panel per configuration.
    for ax, base in zip(axes.flat, order):
        seeds = runs[base]
        for seed in seed_list:
            if seed not in seeds:
                continue  # this seed has no history for this configuration
            history = seeds[seed]["history"]
            color, marker = style[seed]
            epochs = [h["epoch"] for h in history]
            # Solid line (white marker edge separates overlapping points).
            ax.plot(epochs, [h[key] for h in history], color=color, lw=2, marker=marker, ms=6,
                    mec=SURFACE, mew=1.5)
            if dashed_key:
                # Dashed line with hollow markers for the second quantity.
                ax.plot(epochs, [h[dashed_key] for h in history], color=color, lw=2, ls="--",
                        marker=marker, ms=6, mfc=SURFACE, mew=1.5)
            else:
                # Ring around the epoch that early stopping kept (best validation macro-F1).
                best = seeds[seed]["metrics"].get("best_epoch")
                if best:
                    ax.plot(best, history[best - 1][key], marker="o", ms=13, mfc="none", mec=INK, mew=1.2)
        # Title, light horizontal grid, no top/right box lines.
        ax.set_title(panel_title(base, seeds, len(seed_list)), fontsize=9.5, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(range(1, max_epochs + 1))
        ax.set_ylabel(ylabel)
    # Hide unused panels (when the number of configurations is not a multiple of 3).
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    # x-label on the bottom row.
    for ax in axes[-1]:
        ax.set_xlabel("epoch")

    # Legend: one entry per seed, plus what solid / dashed / ring mean.
    handles = [plt.Line2D([], [], color=style[s][0], marker=style[s][1], lw=2, ms=6, label=f"seed {s}")
               for s in seed_list]
    if dashed_key:
        handles += [plt.Line2D([], [], color=INK2, lw=2, label="train"),
                    plt.Line2D([], [], color=INK2, lw=2, ls="--", label="validation")]
    else:
        handles.append(plt.Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK,
                                  label="epoch kept (best val)"))
    # Title at the top left, legend on its own row below it.
    fig.suptitle(title, x=0.01, ha="left", y=0.99, fontsize=11.5)
    top = 1 - 1.2 / (3.3 * nrows + 1.2)  # leave the top ~1.2 inch for title + legend
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.005, top + 0.05))
    fig.tight_layout(rect=(0, 0, 1, top))
    # Save at 200 dpi (sharp enough for the report) and free the memory.
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_history_csv(runs, order, out_path):
    """Write the per-epoch numbers as a table (one row per run and epoch)."""
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row.
        writer.writerow(["run_folder", "seed", "epoch", "train_loss", "val_loss", "val_accuracy",
                         "val_macro_f1", "best_epoch"])
        for base in order:
            for seed, run in sorted(runs[base].items()):
                for h in run["history"]:
                    writer.writerow([f"{base}_seed{seed}", seed, h["epoch"],
                                     f"{h['train_loss']:.6f}", f"{h['val_loss']:.6f}",
                                     f"{h['val_accuracy']:.6f}", f"{h['val_macro_f1']:.6f}",
                                     run["metrics"].get("best_epoch", "")])


def config_sort_key(base, seeds):
    """Panel order: HASOC-only runs first (MuRIL, then XLM-R), then HASOC+HX by increasing lambda."""
    metrics = next(iter(seeds.values()))["metrics"]
    mixed = "hatexplain" in str(metrics.get("dataset", ""))
    xlmr = "xlm" in str(metrics.get("encoder", "")).lower()
    return (mixed, xlmr, float(metrics.get("lambda", 0) or 0), base)


def main(argv=None):
    """Read the runs, write the two PNGs and the CSV. Returns the output paths (for tests)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="folder with <run_name>_seed<N>/ run folders (default: results/)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "results",
                        help="where to write the plots and CSV (default: docs/results/)")
    args = parser.parse_args(argv)

    # All runs that have a history.json.
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no <run_name>_seed<N>/history.json found under {args.results_dir}")
    # Panel order and the full list of seeds.
    order = sorted(runs, key=lambda b: config_sort_key(b, runs[b]))
    seed_list = sorted({seed for seeds in runs.values() for seed in seeds})
    # Make sure the output folder exists.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"f1": args.out_dir / "training_curves_val_macro_f1.png",
           "loss": args.out_dir / "training_curves_loss.png",
           "csv": args.out_dir / "training_history.csv"}
    # Figure 1: validation macro-F1 (comparable across panels -> shared y-axis).
    plot_figure(runs, order, seed_list, "val_macro_f1", "validation macro-F1",
                "Validation macro-F1 per epoch (runs with a saved history.json)", out["f1"])
    # Figure 2: training + validation loss (training loss NOT comparable across lambda -> own y-axis).
    plot_figure(runs, order, seed_list, "train_loss", "loss",
                "Training and validation loss per epoch (runs with a saved history.json)\n"
                "y-axis differs per panel: training loss includes the λ-weighted rationale term; "
                "validation loss is cross-entropy on HASOC",
                out["loss"], dashed_key="val_loss", share_y=False)
    # The numbers as a table.
    write_history_csv(runs, order, out["csv"])
    print(f"[plot] {sum(len(s) for s in runs.values())} runs in {len(order)} configurations -> {args.out_dir}")
    return out


# Run only when executed as a script.
if __name__ == "__main__":
    main()
