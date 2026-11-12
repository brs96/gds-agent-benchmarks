import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["Recall", "Precision", "F1"]
METRIC_KEYS = ["recall", "precision", "f1"]
COLORS = {"medium": "#5B9BD5", "high": "#2E75B6", "base": "#F4A636"}
BAR_WIDTH = 0.10

TOKEN_COLORS = {
    ("gds",      "high"):   "#2E75B6",
    ("gds",      "medium"): "#5B9BD5",
    ("baseline", "high"):   "#C0623A",
    ("baseline", "medium"): "#F4A636",
}
GROUP_LABELS = [
    ("gds",      "high",   "GDS\nhigh"),
    ("baseline", "high",   "w/o GDS\nhigh"),
    ("gds",      "medium", "GDS\nmedium"),
    ("baseline", "medium", "w/o GDS\nmedium"),
]


def load_expected_addresses(questions_file):
    with open(questions_file) as f:
        data = json.load(f)
    for entry in data:
        raw = entry.get("expected_answer", "")
        if raw.strip():
            return {a.lower() for a in raw.split()}
    return set()


def extract_predicted_addresses(text):
    marker = re.search(r"\bANSWER\b", text, re.IGNORECASE)
    text = text[marker.end():] if marker else text
    return [a.lower() for a in re.findall(r"0x[0-9a-fA-F]{40}", text)]


def compute_scores(predicted, expected):
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_set = set(predicted)
    tp = len(pred_set & expected)
    p = tp / len(pred_set)
    r = tp / len(expected)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def load_batches(path):
    """Batches of a merged results file ({"batches": [...]}); a single-run file counts as one batch."""
    with open(path) as f:
        data = json.load(f)
    return data.get("batches", [data])


def load_gds_scores_by_effort(path, expected):
    scores = defaultdict(list)
    failed = defaultdict(int)
    for batch in load_batches(path):
        for result in batch.get("raw_results", []):
            effort = result["effort"]
            final = (result.get("response_data") or {}).get("final_result", "")
            if not final or not result.get("success", False):
                scores[effort].append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
                failed[effort] += 1
            else:
                scores[effort].append(compute_scores(extract_predicted_addresses(final), expected))
    return {e: (scores[e], failed[e]) for e in scores}


def load_baseline_scores(path, expected):
    by_effort = defaultdict(list)
    failed = defaultdict(int)
    for batch in load_batches(path):
        effort = batch["benchmark_info"]["reasoning_effort"]
        for result in batch.get("raw_results", []):
            final = (result.get("response_data") or {}).get("final_result", "")
            if not final or not result.get("success", False):
                by_effort[effort].append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
                failed[effort] += 1
            else:
                by_effort[effort].append(compute_scores(extract_predicted_addresses(final), expected))
    return {e: (by_effort[e], failed[e]) for e in by_effort}


def load_runs(gds_file, baseline_file):
    runs = defaultdict(list)
    for batch in load_batches(gds_file):
        for r in batch.get("raw_results", []):
            effort = r["effort"]
            rd = r.get("response_data") or {}
            tok = rd.get("token_usage") or {}
            runs[("gds", effort)].append({
                "success":    r.get("success", False) and bool(rd.get("final_result")),
                "input":      tok.get("total_input_tokens", 0),
                "output":     tok.get("total_output_tokens", 0),
                "cached":     tok.get("cache_read_input_tokens", 0),
                "turns":      rd.get("num_turns", 0),
                "duration":   (rd.get("duration_ms") or 0) / 60000,
            })
    for batch in load_batches(baseline_file):
        effort = batch["benchmark_info"]["reasoning_effort"]
        for r in batch.get("raw_results", []):
            rd = r.get("response_data") or {}
            tok = rd.get("token_usage") or {}
            runs[("baseline", effort)].append({
                "success":    r.get("success", False) and bool(rd.get("final_result")),
                "input":      tok.get("total_input_tokens", 0),
                "output":     tok.get("total_output_tokens", 0),
                "cached":     tok.get("cache_read_input_tokens", 0),
                "turns":      rd.get("num_turns", 0),
                "duration":   (rd.get("duration_ms") or 0) / 60000,
            })
    return runs


def draw_group(ax, x, values, failed_mask, color):
    if not values:
        return
    vals = np.array(values)
    fail = np.array(failed_mask)
    ok_vals   = vals[~fail]
    fail_vals = vals[fail]
    if len(ok_vals):
        ax.scatter(np.full(len(ok_vals), x), ok_vals, color=color, s=25, zorder=4, alpha=0.8)
    if len(fail_vals):
        ax.scatter(np.full(len(fail_vals), x), fail_vals, color=color, s=40,
                   marker="x", linewidths=1.5, zorder=4, alpha=0.6)
    if len(vals):
        mean = vals.mean()
        std  = vals.std()
        ax.bar(x, mean, width=0.5, color=color, alpha=0.35, zorder=2)
        ax.errorbar(x, mean, yerr=std, color=color, linewidth=1.5, capsize=4, capthick=1.5, zorder=5)
        ax.plot(x, mean, "o", markersize=7, color=color,
                markerfacecolor="white", markeredgewidth=1.8, zorder=6)


def make_token_plot(runs, metrics, titles, output_path):
    plt.style.use("default")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans"})
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    xs = np.arange(len(GROUP_LABELS))
    for ax, metric, title in zip(axes, metrics, titles):
        ax.set_facecolor("#f9f9f9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, pad=6, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([g[2] for g in GROUP_LABELS], fontsize=8)
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for x, (arm, effort, _) in zip(xs, GROUP_LABELS):
            group = runs.get((arm, effort), [])
            values = [r[metric] for r in group]
            failed = [not r["success"] for r in group]
            draw_group(ax, x, values, failed, TOKEN_COLORS.get((arm, effort), "#aaaaaa"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{output_path}'")


def draw_series(ax, x, values, color):
    if len(values) == 0:
        return
    mn, mx = values.min(), values.max()
    mean, std, median = values.mean(), values.std(), np.median(values)

    ax.plot([x, x], [mn, mx], color=color, linewidth=1.2, zorder=2)
    for y in (mn, mx):
        ax.plot([x - 0.03, x + 0.03], [y, y], color=color, linewidth=1.2, zorder=2)

    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BAR_WIDTH / 2, mean - std), BAR_WIDTH, 2 * std,
        boxstyle="square,pad=0", linewidth=0, facecolor=color, alpha=0.4, zorder=3,
    ))
    ax.plot(x, mean, "o", markersize=9, color=color, markerfacecolor="white", markeredgewidth=1.8, zorder=5)
    ax.text(x + 0.055, mean, f"{mean:.2f}", va="center", ha="left", fontsize=8, color=color, fontweight="bold", zorder=6)
    ax.plot(x, median, "D", markersize=6, color=color, markerfacecolor="white", markeredgewidth=1.5, zorder=5)
    ax.text(x - 0.055, median, f"{median:.2f}", va="center", ha="right", fontsize=8, color=color, zorder=6)


def create_plot(effort, gds_scores, gds_failed, base_scores, base_failed, title, output_path):
    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "xtick.labelsize": 11, "ytick.labelsize": 10, "legend.fontsize": 8.5,
                         "font.family": "DejaVu Sans"})

    offsets = [-0.12, 0.12]
    gds_color = COLORS.get(effort, "#5B9BD5")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    for i, key in enumerate(METRIC_KEYS):
        draw_series(ax, i + offsets[0], np.array([s[key] for s in gds_scores]), gds_color)
        draw_series(ax, i + offsets[1], np.array([s[key] for s in base_scores]), COLORS["base"])

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_xlim(-0.35, len(METRICS) - 0.65)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color=gds_color, label="GDS agent"),
        mpatches.Patch(color=COLORS["base"], label="w/o GDS agent"),
        plt.Line2D([0], [0], color="grey", linewidth=1.2, label="Min-Max Range"),
        mpatches.Patch(color="grey", alpha=0.4, label="Mean±Std Range"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="grey", markersize=7, label="Mean"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="grey", markersize=5, label="Median"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax.set_title(title, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{output_path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gds-file", default="results_gpt-5/bybit_gds_agent.json")
    parser.add_argument("--baseline-file", default="results_gpt-5/bybit_wo_gds.json")
    parser.add_argument("--questions-file", default="questions/bybit-laundering.json")
    parser.add_argument("--model", default="GPT-5")
    parser.add_argument("--output-dir", type=Path, default=Path("results_gpt-5/bybit_analysis"))
    parser.add_argument("--output-prefix", default="bybit_comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = lambda name: str(args.output_dir / f"{args.output_prefix}_{name}.png")

    expected = load_expected_addresses(args.questions_file)
    gds_by_effort = load_gds_scores_by_effort(args.gds_file, expected)
    base_by_effort = load_baseline_scores(args.baseline_file, expected)

    for effort, (scores, failed) in sorted(gds_by_effort.items()):
        print(f"GDS agent ({effort}): {len(scores)} runs, {failed} failed")
    for effort, (scores, failed) in sorted(base_by_effort.items()):
        print(f"w/o GDS agent ({effort}): {len(scores)} runs, {failed} failed")

    if not gds_by_effort and not base_by_effort:
        print("No results found.")
        return

    for effort, (gds_scores, gds_failed) in sorted(gds_by_effort.items()):
        base_scores, base_failed = base_by_effort.get(effort, ([], 0))
        title = f"{args.model} on Bybit ({effort}): GDS agent vs w/o GDS agent"
        create_plot(effort, gds_scores, gds_failed, base_scores, base_failed, title, out(effort))

    runs = load_runs(args.gds_file, args.baseline_file)
    make_token_plot(
        runs,
        metrics=["duration", "input",  "output", "turns"],
        titles= [
            "Duration (min); X = failed run",
            "Input tokens (context re-sent each turn)",
            "Output tokens (what the model writes)",
            "Turns (model calls)",
        ],
        output_path=out("tokens_overview"),
    )
    for r in [v for group in runs.values() for v in group]:
        r["non_cached"] = max(0, r["input"] - r["cached"])
    make_token_plot(
        runs,
        metrics=["input",  "cached", "non_cached"],
        titles= [
            "Total input tokens",
            "Cached input tokens",
            "Non-cached input tokens",
        ],
        output_path=out("tokens_cache"),
    )


if __name__ == "__main__":
    main()
