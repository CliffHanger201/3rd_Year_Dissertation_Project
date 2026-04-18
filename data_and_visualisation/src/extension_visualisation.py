"""
visualise_single_instance.py
=============================
Visualisation for single-instance pre-training results.

Answers three questions
-----------------------
  Q1  How does convergence change? (fitness traces)
      §1  Fitness-trace fan plot — one sub-plot per Q-Table label (1.2, 1.3, 1.4),
          all domains overlaid, with median trajectory highlighted.

  Q2  How does the Q-Table change?
      §2  Q-Table summary heatmaps — a 2-row figure:
            Row 1: the TRAINING snapshot (qtable_<domain>_train1.pkl)
            Row 2: the EVALUATION snapshot taken after all runs on each test instance.
          Shows mean / max / min Q-value per heuristic, domain by domain.
      §3  Q-Table state-count bar chart — how many unique states were visited
          during training vs. each test instance (learning breadth indicator).

  Q4  How does heuristic usage trend across repeats?
      §4  Heuristic usage line graphs — one figure per domain.
          X-axis: repeat number (seed index 1-30).
          Y-axis: call count for that heuristic in that repeat.
          One line per heuristic, coloured distinctly.
          Separate sub-plots for each Q-Table label (test instance).

File layout expected
--------------------
  results/single_instance_pretrain_results.json
  qtables/qtable_<domain>_train1.pkl       ← training Q-Table (frozen)

Usage
-----
  python visualise_single_instance.py
"""

import json
import pickle
import os
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = "results/single_instance_pretrain_results.json"
QTABLE_DIR   = "qtables"

DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]

# Colour scheme
CMAP_TRACES    = "tab10"       # heuristic lines
ALPHA_THIN     = 0.18          # individual run opacity
ALPHA_MEDIAN   = 1.0

LABEL_COLOURS = {
    "1.2": "#4C72B0",
    "1.3": "#DD8452",
    "1.4": "#55A868",
}


# ═════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_qtable_pkl(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[warn] Q-Table not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_runs(data: dict, domain: str) -> List[dict]:
    runs = data.get(domain, [])
    return [r for r in runs if isinstance(r, dict)]


def _group_by_label(runs: List[dict]) -> Dict[str, List[dict]]:
    """Group a list of run dicts by their qtable_label field."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in runs:
        label = r.get("qtable_label", "?")
        groups[label].append(r)
    return dict(groups)


def _median_trace(traces: List[List[float]]) -> List[float]:
    if not traces:
        return []
    min_len = min(len(t) for t in traces)
    arr = np.array([t[:min_len] for t in traces])
    return np.median(arr, axis=0).tolist()


def _collect_call_counts(runs: List[dict]) -> List[List[int]]:
    """Return list-of-lists: outer = runs, inner = per-heuristic call count."""
    result = []
    for r in runs:
        cc = r.get("heuristic_call_counts")
        if isinstance(cc, list) and cc:
            result.append([int(v) for v in cc])
    return result


# ═════════════════════════════════════════════════════════════════════════════
# §1  Fitness-trace fan plots
#     One row per Q-Table label, one column per domain.
#     Answers: "How does convergence change?"
# ═════════════════════════════════════════════════════════════════════════════

def plot_fitness_traces(data: dict, labels: List[str] = None):
    """
    Grid: rows = Q-Table labels (test instances), cols = domains.
    Each cell shows all individual runs (faint) + bold median.
    Answers Q1: convergence behaviour per transfer pair.
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })
    if labels:
        plot_labels = [l for l in all_labels if l in labels]
    else:
        plot_labels = all_labels

    if not plot_labels:
        print("[§1] No fitness trace data found.")
        return

    nrows = len(plot_labels)
    ncols = len(DOMAINS)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        squeeze=False,
    )
    fig.suptitle("§1 — Fitness convergence per Q-Table transfer (training → test)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_idx, label in enumerate(plot_labels):
        colour = LABEL_COLOURS.get(label, "#888888")
        for col_idx, domain in enumerate(DOMAINS):
            ax = axes[row_idx][col_idx]
            runs = _get_runs(data, domain)
            label_runs = [r for r in runs if r.get("qtable_label") == label]

            traces = [
                list(r["fitness_trace"])
                for r in label_runs
                if isinstance(r.get("fitness_trace"), list) and r["fitness_trace"]
            ]

            if not traces:
                ax.set_visible(False)
                continue

            # Individual runs (faint)
            for tr in traces:
                ax.plot(tr, color=colour, alpha=ALPHA_THIN, linewidth=0.7)

            # Bold median
            med = _median_trace(traces)
            ax.plot(med, color=colour, linewidth=2.2,
                    label=f"Median (n={len(traces)})")

            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8, loc="upper right")

            if row_idx == 0:
                ax.set_title(domain, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Q-Table {label}\nObjective (log)", fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel("Trace step", fontsize=9)

    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# §2  Q-Table heatmaps — training vs. test
#     Answers: "How does the Q-Table change?"
# ═════════════════════════════════════════════════════════════════════════════

def _qtable_summary_from_pkl(pkl: dict) -> Tuple[np.ndarray, int]:
    """
    Returns (summary, h_count) where summary has shape (3, h_count):
      row 0 = mean, row 1 = max, row 2 = min across all visited states.
    """
    table   = pkl["table"]
    h_count = pkl["h_count"]
    if not table:
        return np.zeros((3, h_count)), h_count
    matrix  = np.vstack(list(table.values()))   # (n_states, h_count)
    summary = np.vstack([
        matrix.mean(axis=0),
        matrix.max(axis=0),
        matrix.min(axis=0),
    ])
    return summary.astype(float), h_count


def _qtable_summary_from_runs(runs: List[dict], h_count: int) -> Optional[np.ndarray]:
    """
    Reconstruct a lightweight Q-Table snapshot from logged run data.
    We use the mean, max, and min of (best_value / call_counts) as a proxy
    for Q-value magnitude per heuristic when the actual eval Q-Table is not
    saved to disk.  If heuristic_call_counts are available we normalise them
    instead — this reflects which heuristics the policy relied on most.

    Returns (3, h_count) array or None.
    """
    all_counts = _collect_call_counts(runs)
    if not all_counts:
        return None
    # Pad/trim to h_count
    padded = []
    for row in all_counts:
        r = list(row) + [0] * max(0, h_count - len(row))
        padded.append(r[:h_count])
    mat = np.array(padded, dtype=float)
    # Normalise each run so the sum = 1 (fractional usage)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = mat / row_sums
    return np.vstack([mat.mean(axis=0), mat.max(axis=0), mat.min(axis=0)])


def plot_qtable_heatmaps(data: dict):
    """
    §2 — Two-row panel per domain.
    Top row  : Q-values from the TRAINING Q-Table (pkl file).
    Bottom rows : per-heuristic usage fraction for each test instance (from run data).

    Answers: how did the policy shift when applied to test instances?
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    n_label_rows = len(all_labels)
    n_total_rows = 1 + n_label_rows   # 1 training + n test summaries
    ncols = len(DOMAINS)

    fig, axes = plt.subplots(
        n_total_rows, ncols,
        figsize=(ncols * 3.8, n_total_rows * 2.6),
        squeeze=False,
    )
    fig.suptitle("§2 — Q-Table change: training policy vs. test-instance heuristic usage",
                 fontsize=13, fontweight="bold", y=1.01)

    row_titles = ["Training (frozen Q-Table)"] + [f"Test Q-Table {l}" for l in all_labels]

    for col_idx, domain in enumerate(DOMAINS):
        # ── Row 0: training Q-Table from pkl ────────────────────────────────
        ax = axes[0][col_idx]
        pkl_path = os.path.join(QTABLE_DIR, f"qtable_{domain}_train1.pkl")
        pkl = _load_qtable_pkl(pkl_path)

        if pkl is not None:
            summary, h_count = _qtable_summary_from_pkl(pkl)
            abs_max = max(float(np.abs(summary).max()), 1e-6)
            norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.imshow(summary, aspect="auto", cmap="RdYlGn", norm=norm)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Q-value", fontsize=7)
            ax.set_xticks(range(h_count))
            ax.set_xticklabels([f"H{i}" for i in range(h_count)], fontsize=7)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Mean", "Max", "Min"], fontsize=7)
            n_states = len(pkl["table"])
            for r in range(3):
                for c in range(h_count):
                    ax.text(c, r, f"{summary[r, c]:.2f}", ha="center", va="center", fontsize=6)
            ax.set_title(f"{domain}\n({n_states} states)", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No pkl found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            h_count = 8   # fallback

        if col_idx == 0:
            ax.set_ylabel(row_titles[0], fontsize=8, labelpad=6)

        # ── Rows 1+: usage fraction derived from run data ───────────────────
        for row_idx, label in enumerate(all_labels, start=1):
            ax2 = axes[row_idx][col_idx]
            runs = [r for r in _get_runs(data, domain)
                    if r.get("qtable_label") == label]
            summary2 = _qtable_summary_from_runs(runs, h_count)

            if summary2 is not None:
                im2 = ax2.imshow(summary2, aspect="auto", cmap="YlOrRd",
                                 vmin=0, vmax=float(summary2.max()) or 1)
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Usage frac.", fontsize=7)
                ax2.set_xticks(range(h_count))
                ax2.set_xticklabels([f"H{i}" for i in range(h_count)], fontsize=7)
                ax2.set_yticks([0, 1, 2])
                ax2.set_yticklabels(["Mean", "Max", "Min"], fontsize=7)
                for r in range(3):
                    for c in range(h_count):
                        ax2.text(c, r, f"{summary2[r, c]:.2f}", ha="center", va="center", fontsize=6)
            else:
                ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes,
                         ha="center", va="center", fontsize=9, color="gray")

            if col_idx == 0:
                ax2.set_ylabel(row_titles[row_idx], fontsize=8, labelpad=6)

    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# §3  Q-Table state-count bar chart
#     Shows learning breadth: how many unique states were visited
#     during training vs. test (if eval Q-Tables are saved).
#     Answers: did the policy explore more or fewer states when transferred?
# ═════════════════════════════════════════════════════════════════════════════

def plot_qtable_state_counts(data: dict):
    """
    §3 — Horizontal bar chart of unique Q-Table states visited.
    Training (pkl) vs. call-count diversity proxy for each test label.
    """
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(len(DOMAINS) * 3.6, 4), squeeze=False)
    fig.suptitle("§3 — Q-Table state breadth: training vs. test instances",
                 fontsize=13, fontweight="bold")

    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    for col_idx, domain in enumerate(DOMAINS):
        ax = axes[0][col_idx]

        # Training state count from pkl
        pkl_path = os.path.join(QTABLE_DIR, f"qtable_{domain}_train1.pkl")
        pkl = _load_qtable_pkl(pkl_path)
        train_count = len(pkl["table"]) if pkl else 0

        bar_labels  = ["Train (frozen)"] + [f"Test {l}" for l in all_labels]
        bar_heights = [train_count]
        bar_colours = ["#4C72B0"]

        for label in all_labels:
            runs = [r for r in _get_runs(data, domain)
                    if r.get("qtable_label") == label]
            # Proxy: number of distinct call-count vectors seen ≈ state diversity
            unique_vecs = set()
            for r in runs:
                cc = r.get("heuristic_call_counts")
                if isinstance(cc, list):
                    unique_vecs.add(tuple(cc))
            bar_heights.append(len(unique_vecs))
            bar_colours.append(LABEL_COLOURS.get(label, "#888888"))

        bars = ax.barh(bar_labels, bar_heights, color=bar_colours, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, bar_heights):
            ax.text(bar.get_width() + max(bar_heights) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8)

        ax.set_title(domain, fontsize=10, fontweight="bold")
        ax.set_xlabel("State / unique-vector count", fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# §4  Heuristic usage trend lines across repeats
#     One figure per domain, sub-plots per Q-Table label.
#     X = repeat number (1-30), Y = call count for that heuristic.
#     One line per heuristic.
#     Answers: do heuristic preferences shift across seeds?
# ═════════════════════════════════════════════════════════════════════════════

def plot_heuristic_usage_trends(data: dict):
    """
    §4 — Line graphs: heuristic call counts over repeats (seeds).
    Each sub-plot = one Q-Table label (test instance).
    Each line = one heuristic.

    Shows whether the frozen policy's heuristic preferences are consistent
    across seeds or vary — and whether they shift between test instances.
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    for domain in DOMAINS:
        runs_domain = _get_runs(data, domain)
        if not runs_domain:
            continue

        # Determine h_count
        h_count = max(
            len(r["heuristic_call_counts"])
            for r in runs_domain
            if isinstance(r.get("heuristic_call_counts"), list)
        ) if any(isinstance(r.get("heuristic_call_counts"), list) for r in runs_domain) else 0

        if h_count == 0:
            continue

        ncols = len(all_labels)
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5.5, 4.5), squeeze=False)
        fig.suptitle(
            f"§4 — {domain}: heuristic usage trends across repeats (each line = one heuristic)",
            fontsize=12, fontweight="bold",
        )

        # Colour palette: one colour per heuristic
        colours = plt.get_cmap(CMAP_TRACES, h_count)

        for col_idx, label in enumerate(all_labels):
            ax = axes[0][col_idx]
            label_runs = sorted(
                [r for r in runs_domain if r.get("qtable_label") == label],
                key=lambda r: r.get("seed", 0),
            )

            if not label_runs:
                ax.set_visible(False)
                continue

            # Build matrix: (n_repeats, h_count)
            counts_matrix = []
            for r in label_runs:
                cc = r.get("heuristic_call_counts", [])
                row = [int(cc[h]) if h < len(cc) else 0 for h in range(h_count)]
                counts_matrix.append(row)

            counts_arr = np.array(counts_matrix, dtype=float)   # (n_repeats, h_count)
            x_vals = np.arange(1, len(label_runs) + 1)

            for h in range(h_count):
                y = counts_arr[:, h]
                col = colours(h)
                ax.plot(x_vals, y, color=col, linewidth=1.6, alpha=0.85,
                        label=f"H{h}", marker="o", markersize=3)

                # Trend line (thin dashed)
                if len(x_vals) > 2:
                    z = np.polyfit(x_vals, y, 1)
                    trend = np.poly1d(z)(x_vals)
                    ax.plot(x_vals, trend, color=col, linewidth=0.8,
                            linestyle="--", alpha=0.55)

            ax.set_title(f"Q-Table {label}", fontsize=10, fontweight="bold",
                         color=LABEL_COLOURS.get(label, "#333333"))
            ax.set_xlabel("Repeat (seed index)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Call count", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=max(1, h_count // 5),
                      loc="upper right", framealpha=0.7)

            # Annotate mean per heuristic at the far right
            for h in range(h_count):
                mean_val = counts_arr[:, h].mean()
                ax.annotate(f"{mean_val:.0f}",
                            xy=(x_vals[-1], counts_arr[-1, h]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=6, color=colours(h), va="center")

        plt.tight_layout()
        plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# §5  Convergence summary — final best-value distributions per label
#     Box-free: shows per-repeat final values as a strip + median line.
#     Quick visual of whether test instances are harder or easier than training.
# ═════════════════════════════════════════════════════════════════════════════

def plot_final_value_distributions(data: dict):
    """
    §5 — Strip chart of final best_value per repeat, grouped by Q-Table label.
    One panel per domain, coloured by label.
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(len(DOMAINS) * 3.8, 5), squeeze=False)
    fig.suptitle(
        "§5 — Final best-value distributions: how hard are the test instances?",
        fontsize=13, fontweight="bold",
    )

    for col_idx, domain in enumerate(DOMAINS):
        ax = axes[0][col_idx]
        runs_domain = _get_runs(data, domain)

        for j_idx, label in enumerate(all_labels):
            label_runs = [r for r in runs_domain if r.get("qtable_label") == label]
            vals = [r["best_value"] for r in label_runs
                    if isinstance(r.get("best_value"), (int, float))]
            if not vals:
                continue

            col = LABEL_COLOURS.get(label, "#888888")
            x_base = j_idx

            # Jittered strip
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.35
            ax.scatter(x_base + jitter, vals, color=col, alpha=0.55,
                       s=18, zorder=3)

            # Median line
            med = np.median(vals)
            ax.plot([x_base - 0.3, x_base + 0.3], [med, med],
                    color=col, linewidth=2.5, zorder=4,
                    label=f"Q-Table {label} (med={med:.3f})")

        ax.set_title(domain, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels([f"Q {l}" for l in all_labels], fontsize=9)
        ax.set_ylabel("Best value (lower = better)", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading results …")
    data = _load_json(RESULTS_FILE)

    print("\n§1  Fitness convergence traces …")
    plot_fitness_traces(data)

    print("\n§2  Q-Table heatmaps (training vs. test) …")
    plot_qtable_heatmaps(data)

    print("\n§3  Q-Table state-count breadth …")
    plot_qtable_state_counts(data)

    print("\n§4  Heuristic usage trends across repeats …")
    plot_heuristic_usage_trends(data)

    print("\n§5  Final best-value distributions …")
    plot_final_value_distributions(data)


if __name__ == "__main__":
    main()