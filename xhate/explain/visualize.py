"""Highlight important words in an HTML report (Member 4).

    from xhate.explain.visualize import highlight_html, write_report
    highlight_html(["tum", "log", "gaddar"], [0.1, 0.0, 0.9])   # -> '<span ...>tum</span> ...'
    write_report("results/run/explain/report.html", title=..., metrics=..., posts=..., methods=...)

Colours:
    red   = the word pushes towards HOF (darker = more important)
    blue  = the word pushes towards NOT (only LIME has negative scores)
    underline (yellow) on the "human" row = words a human annotator marked as the rationale

Q&A
---
Q: Why HTML and not a matplotlib figure?
A: A post is a line of text; HTML can colour each word's background, wraps long posts,
   shows Devanagari with the browser's fonts, and opens on any machine without Python.
   One file can hold hundreds of posts plus the metrics table.

Q: How is the colour strength computed?
A: alpha = |score| / max |score| of that row, so the most important word of each post is
   fully coloured. Scores of different methods have different scales (attention sums to 1,
   LIME weights are probability changes), so each row is scaled on its own.
   Consequence: compare which words are DARKEST between rows, not absolute colours.

Q: Why ``html.escape``?
A: Posts contain "<user>", "<url>" and sometimes "&" or quotes. Unescaped, the browser would
   read "<user>" as an HTML tag and hide it. escape() turns "<" into "&lt;" and so on.
   (It also means a post can never inject HTML or JavaScript into the report.)

Q: What are the "good" and "bad" example sections?
A: Posts sorted by the comprehensiveness of the first method: the top ones are explanations
   whose highlighted words really drive the prediction (good examples for the report), the
   bottom ones are explanations that do not (bad examples, worth an error-analysis paragraph).
"""

# html: escape special characters in posts.
import html

# Label names for display ("NOT", "HOF").
from xhate.data.labels import LABEL_NAMES

# Page style. Colours are CSS variables so the page also works in dark mode.
_CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de; --card: #f6f8fa;
        --hof: 220, 38, 38; --not: 37, 99, 235; --gold: #d4a017; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8d96a0; --line: #30363d; --card: #161b22;
          --hof: 248, 81, 73; --not: 88, 166, 255; --gold: #e3b341; } }
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 980px; padding: 16px;
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", "Noto Sans Devanagari", sans-serif; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; overflow-x: auto; display: block; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: left; white-space: nowrap; }
.post { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 12px 0; }
.meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
.row { display: flex; gap: 8px; margin: 4px 0; } .row b { min-width: 80px; font-size: 0.85rem; }
.words { flex: 1; overflow-wrap: anywhere; }
.w { padding: 1px 3px; border-radius: 3px; margin-right: 2px; }
.gold { text-decoration: underline 3px var(--gold); text-underline-offset: 4px; }
.wrong { color: rgb(var(--hof)); font-weight: 600; }
"""


def _fmt(value, digits=4):
    """Format a metric for the table ("–" for missing values)."""
    # None (e.g. no plausibility without rationales) -> dash.
    if value is None:
        return "–"
    # Numbers with a fixed number of decimals.
    return f"{value:.{digits}f}"


def highlight_html(tokens, scores, gold=None):
    """One post as HTML spans whose background shows each word's score (see Q&A).

    Args:
        tokens: the words.
        scores: one score per word (positive = HOF / important, negative = NOT); None = no colour.
        gold: optional 0/1 human rationale; marked words are underlined.
    """
    # Largest absolute score of this row (avoid dividing by 0).
    peak = max((abs(float(s)) for s in scores), default=0.0) if scores is not None else 0.0
    # HTML pieces, one per word.
    parts = []
    # Go through the words with their position.
    for i, tok in enumerate(tokens):
        # This word's score (0 if there are no scores).
        score = float(scores[i]) if scores is not None else 0.0
        # Colour strength in [0, 1]: relative to the strongest word of the row.
        alpha = abs(score) / peak if peak > 0 else 0.0
        # Red for positive (HOF / important), blue for negative (NOT).
        colour = "var(--hof)" if score >= 0 else "var(--not)"
        # Underline words a human marked.
        cls = "w gold" if gold is not None and i < len(gold) and gold[i] else "w"
        # The span; the tooltip (title) shows the exact score.
        parts.append(f'<span class="{cls}" title="{score:.4f}" '
                     f'style="background: rgba({colour}, {alpha * 0.85:.3f})">{html.escape(tok)}</span>')
    # Words separated by spaces.
    return " ".join(parts)


def post_html(post, methods):
    """One post card: id, gold / predicted label, P(HOF), one highlighted row per method.

    ``post`` is one line of explanations.jsonl (see scripts/explain.py).
    """
    # Gold and predicted label names.
    gold_name, pred_name = LABEL_NAMES[post["label"]], LABEL_NAMES[post["pred"]]
    # Highlight wrong predictions in the header.
    pred_html = pred_name if post["pred"] == post["label"] else f'<span class="wrong">{pred_name} (wrong)</span>'
    # Per-method comprehensiveness / sufficiency, e.g. "lime: comp 0.41 / suff 0.02".
    faith = " · ".join(f"{m}: comp {_fmt(post['faithfulness'][m]['comprehensiveness'], 3)} / "
                       f"suff {_fmt(post['faithfulness'][m]['sufficiency'], 3)}"
                       for m in methods if m in post.get("faithfulness", {}))
    # Header line.
    rows = [f'<div class="meta">{html.escape(str(post["id"]))} · gold {gold_name} · predicted {pred_html} · '
            f'P(HOF) {post["prob_hof"]:.3f}<br>{faith}</div>']
    # One row per explanation method.
    for method in methods:
        rows.append(f'<div class="row"><b>{method}</b><span class="words">'
                    f'{highlight_html(post["tokens"], post["scores"][method], post.get("gold_rationale"))}</span></div>')
    # The human rationale row, if there is one.
    if post.get("gold_rationale") is not None:
        rows.append(f'<div class="row"><b>human</b><span class="words">'
                    f'{highlight_html(post["tokens"], post["gold_rationale"], post["gold_rationale"])}</span></div>')
    # The whole card.
    return f'<div class="post">{"".join(rows)}</div>'


def metrics_table_html(metrics, methods):
    """Summary table: one row per method with faithfulness and plausibility."""
    # Header row.
    out = ["<table><tr><th>method</th><th>posts</th><th>comprehensiveness ↑</th><th>sufficiency ↓</th>"
           "<th>comp. (pred. HOF)</th><th>suff. (pred. HOF)</th><th>token F1 ↑</th><th>AUPRC ↑</th>"
           "<th>plaus. posts</th></tr>"]
    # One row per method.
    for method in methods:
        # This method's numbers.
        m = metrics["methods"][method]
        # Faithfulness on all posts, on predicted-HOF posts, and plausibility.
        f_all, f_hof, plaus = m["faithfulness"]["all"], m["faithfulness"]["predicted_hof"], m["plausibility"]
        out.append(f"<tr><td>{method}</td><td>{f_all['n_posts']}</td>"
                   f"<td>{_fmt(f_all['comprehensiveness'])}</td><td>{_fmt(f_all['sufficiency'])}</td>"
                   f"<td>{_fmt(f_hof['comprehensiveness'])}</td><td>{_fmt(f_hof['sufficiency'])}</td>"
                   f"<td>{_fmt(plaus['token_f1'])}</td><td>{_fmt(plaus['auprc'])}</td>"
                   f"<td>{plaus['n_posts']}</td></tr>")
    # Close the table.
    out.append("</table>")
    return "".join(out)


def write_report(path, title, metrics, posts, methods, n_examples=10, n_posts=50):
    """Write the full HTML report: metrics table, good / bad examples, first posts.

    Args:
        path: output .html file.
        title: page title (e.g. the run name).
        metrics: the dict saved as metrics.json by scripts/explain.py.
        posts: the explained posts (lines of explanations.jsonl).
        methods: explanation methods to show, e.g. ["attention", "lime", "random"].
        n_examples: posts in each of the good / bad sections.
        n_posts: posts in the "first posts" section.
    """
    # Imported here: only needed when writing.
    from pathlib import Path
    # Rank by the first non-random method (the random baseline makes no sense to rank by).
    key_method = next((m for m in methods if m != "random"), methods[0])
    # Posts sorted by that method's comprehensiveness, highest first.
    ranked = sorted(posts, key=lambda p: p["faithfulness"][key_method]["comprehensiveness"], reverse=True)
    # The page, piece by piece.
    body = [f"<h1>{html.escape(title)}</h1>",
            f'<p class="meta">{metrics["n_posts"]} posts of {html.escape(metrics["dataset"])} / '
            f'{html.escape(metrics["split"])} · ERASER bins {metrics["bins"]} · '
            "red = towards HOF, blue = towards NOT, underline = human rationale</p>",
            "<h2>Metrics</h2>", metrics_table_html(metrics, methods),
            f"<h2>Good examples: highest {key_method} comprehensiveness</h2>",
            *(post_html(p, methods) for p in ranked[:n_examples]),
            f"<h2>Bad examples: lowest {key_method} comprehensiveness</h2>",
            *(post_html(p, methods) for p in ranked[::-1][:n_examples]),
            f"<h2>First {min(n_posts, len(posts))} posts</h2>",
            *(post_html(p, methods) for p in posts[:n_posts])]
    # Full HTML document (UTF-8 so Devanagari displays correctly).
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")
    # Make sure the folder exists and write the file.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")
