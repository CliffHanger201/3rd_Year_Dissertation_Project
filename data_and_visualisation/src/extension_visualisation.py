"""
extension_visualisation.py
=============================
Visualisation for single-instance pre-training results.

Sections
--------
  §1  Fitness convergence — 4 panels per domain
        Pre-training (instance 1) + test instances 1.2, 1.3, 1.4.
        Directly shows whether the frozen policy converges on unseen instances.

  §2  Q-Table heatmaps — per domain, 2 rows × 4 columns:
        Row 1: actual Q-values (mean / max / min) loaded from pkl files.
                Train pkl and eval pkls are loaded separately.
                Eval Q-values are identical to training (freeze confirmed).
        Row 2: heuristic usage fractions derived from call counts (JSON).
                These DO differ across instances, showing behavioural change.

  §3  Q-Table state breadth — bar chart per domain
        Training state count (from pkl) vs. test-instance state counts
        (from JSON qtable_state_count).  Shows how much of the Q-Table
        is exercised on each instance.

  §4  Heuristic usage trends across repeats — 4 panels per domain
        Panel 1: Pre-training  — call counts per outer run (last pass).
        Panel 2: Test 1.2      — call counts per repeat.
        Panel 3: Test 1.3
        Panel 4: Test 1.4
        Each line = one heuristic.  Directly comparable across all 4 panels.

  §5  Final best-value distributions — strip chart per domain
        Grouped by Q-Table label; shows relative difficulty of test instances.

  §6  Improvement ratio — strip chart per domain
        Improvement ratio = (init − best) / |init|.
        Together reveal whether the frozen policy transfers effectively.

File layout expected
--------------------
  results/single_instance_pretrain_results.json
  qtables/extension/
      qtable_{domain}_train1_latest.pkl
      qtable_{domain}_eval_1.2_latest.pkl
      qtable_{domain}_eval_1.3_latest.pkl
      qtable_{domain}_eval_1.4_latest.pkl

Usage
-----
  python visualise_single_instance.py
"""

import json
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ----- Paths -----
RESULTS_FILE = "results/single_instance_pretrain_results.json"
QTABLE_DIR   = "qtables/extension"

DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]

# Individual run opacity for fan plots
ALPHA_THIN = 0.18
CMAP_TRACES = "tab10"

# Colours — one per label (train + 3 test instances)
LABEL_COLOURS = {
    "train": "#2E86AB",
    "1.2":   "#4C72B0",
    "1.3":   "#DD8452",
    "1.4":   "#55A868",
}

LABEL_DISPLAY = {
    "train": "Pre-Train (Instance 1)",
    "1.2":   "Test 1.2 (Instance 2)",
    "1.3":   "Test 1.3 (Instance 3)",
    "1.4":   "Test 1.4 (Instance 4)",
}

SAVE_DIR = "results/plots/extension"

DOMAIN_ALIASES = {
    "BinPacking": "Bin",
}

# Save helper
def _savefig(name: str) -> None:
    """Save the current figure to SAVE_DIR."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# =============================================================================
# Data helpers
# =============================================================================

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_qtable_pkl(path: str) -> Optional[dict]:
    """Load a pkl produced by _save_qtable_pkl in the runner."""
    if path is None or not os.path.exists(path):
        print(f"[warn] Q-Table pkl not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_qtable_pkl(domain: str, label: str, train_instance_id: int = 1) -> Optional[str]:
    """
    Locate the best available pkl for a given domain and label.

    label : "train" for the training Q-Table,
            "1.2" / "1.3" / "1.4" for evaluation Q-Tables.

    Search order: canonical latest → run1 fallback → any matching file.
    """
    if not os.path.exists(QTABLE_DIR):
        print(f"[warn] Q-Table directory not found: {QTABLE_DIR}")
        return None

    if label == "train":
        candidates = [
            f"qtable_{domain}_train{train_instance_id}_latest.pkl",
            f"qtable_{domain}_train{train_instance_id}_run1.pkl",
        ]
        prefix = f"qtable_{domain}_train{train_instance_id}_"
    else:
        candidates = [
            f"qtable_{domain}_eval_{label}_latest.pkl",
            f"qtable_{domain}_eval_{label}_run1.pkl",
        ]
        prefix = f"qtable_{domain}_eval_{label}_"

    for fname in candidates:
        path = os.path.join(QTABLE_DIR, fname)
        if os.path.exists(path):
            print(f"[QtableLoad:{domain}:{label}] {fname}")
            return path

    # Fallback: any matching file in directory
    try:
        matches = sorted([
            f for f in os.listdir(QTABLE_DIR)
            if f.startswith(prefix) and f.endswith(".pkl")
        ])
        if matches:
            path = os.path.join(QTABLE_DIR, matches[0])
            print(f"[QtableLoad:{domain}:{label}] fallback → {matches[0]}")
            return path
    except OSError:
        pass

    print(f"[warn] No Q-Table pkl found for {domain} label={label} in {QTABLE_DIR}")
    return None


def _get_runs(data: dict, domain: str) -> List[dict]:
    """All evaluation DomainRunResult dicts for a domain."""
    return [r for r in data.get(domain, []) if isinstance(r, dict)]


def _get_pretrain_runs(data: dict, domain: str) -> List[dict]:
    """All PreTrainRunResult dicts for a domain (from the _pretrain block)."""
    return [r for r in data.get("_pretrain", {}).get(domain, []) if isinstance(r, dict)]


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


def _qtable_summary_from_pkl(pkl: dict) -> Tuple[np.ndarray, int]:
    """
    Returns (summary, h_count) where summary has shape (3, h_count):
      row 0 = mean, row 1 = max, row 2 = min across all visited states.
    """
    table   = pkl.get("table", {})
    h_count = pkl.get("h_count", 0)
    if not table:
        return np.zeros((3, max(h_count, 1))), max(h_count, 1)
    matrix = np.vstack(list(table.values()))
    summary = np.vstack([
        matrix.mean(axis=0),
        matrix.max(axis=0),
        matrix.min(axis=0),
    ])
    return summary.astype(float), int(h_count) or matrix.shape[1]


def _usage_summary_from_runs(runs: List[dict], h_count: int) -> Optional[np.ndarray]:
    """
    Normalised heuristic usage fractions from call counts.
    Returns (3, h_count) array [mean, max, min] or None.
    """
    all_counts = _collect_call_counts(runs)
    if not all_counts:
        return None
    padded = []
    for row in all_counts:
        r = list(row) + [0] * max(0, h_count - len(row))
        padded.append(r[:h_count])
    mat = np.array(padded, dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = mat / row_sums
    return np.vstack([mat.mean(axis=0), mat.max(axis=0), mat.min(axis=0)])


def _draw_heatmap(
    ax,
    summary:    np.ndarray,
    h_count:    int,
    cmap:       str,
    norm=None,
    vmin=None,
    vmax=None,
    cb_label:   str = "",
    row_labels: List[str] = None,
    annotate:   bool = True,
    fontsize:   int = 6,
) -> None:
    """Shared helper: draw a (3, h_count) heatmap with colourbar and annotations."""
    if norm is not None:
        im = ax.imshow(summary, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(summary, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cb_label, fontsize=7)
    ax.set_xticks(range(h_count))
    ax.set_xticklabels([f"H{i}" for i in range(h_count)], fontsize=fontsize)
    ax.set_yticks(range(summary.shape[0]))
    ax.set_yticklabels(
        row_labels if row_labels else ["Mean", "Max", "Min"],
        fontsize=fontsize,
    )
    if annotate:
        for r in range(summary.shape[0]):
            for c in range(h_count):
                ax.text(c, r, f"{summary[r, c]:.2f}",
                        ha="center", va="center", fontsize=fontsize)


# =============================================================================
# §1  Fitness convergence — 4 panels per domain
# =============================================================================

def plot_fitness_traces(data: dict, filename_template="{domain}_ext_fitness") -> None:
    """
    §1 — One figure per domain, four panels:
      [Test 1.2] [Test 1.3] [Test 1.4]

    Faint lines = individual runs.  Bold line = median.
    Pre-training uses data from _pretrain JSON block (per-pass traces).
    Test instances use evaluation run data from main JSON block.

    Directly answers: does the frozen policy converge on unseen instances
    in the same way it converged during pre-training?
    """
    meta = data.get("_meta", {})
    all_panels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    for domain in DOMAINS:
        display_domain = DOMAIN_ALIASES.get(domain, domain)
        pretrain_runs = _get_pretrain_runs(data, domain)
        eval_runs     = _get_runs(data, domain)

        if not pretrain_runs and not eval_runs:
            print(f"[§1] No data for {domain}, skipping.")
            continue

        fig, axes = plt.subplots(
            1, len(all_panels),
            figsize=(len(all_panels) * 5.2, 4.8),
            squeeze=False,
        )
        fig.suptitle(
            f"{domain}: Fitness convergence  "
            f"(pre-training on instance 1  →  frozen transfer to test instances)",
            fontsize=12, fontweight="bold",
        )

        for col_idx, panel in enumerate(all_panels):
            ax     = axes[0][col_idx]
            colour = LABEL_COLOURS.get(panel, "#888888")
            title  = LABEL_DISPLAY.get(panel, panel)

            if panel == "train":
                traces = [
                    list(r["fitness_trace"])
                    for r in pretrain_runs
                    if isinstance(r.get("fitness_trace"), list) and r["fitness_trace"]
                ]
            else:
                traces = [
                    list(r["fitness_trace"])
                    for r in eval_runs
                    if r.get("qtable_label") == panel
                    and isinstance(r.get("fitness_trace"), list) and r["fitness_trace"]
                ]

            if traces:
                for tr in traces:
                    ax.plot(tr, color=colour, alpha=ALPHA_THIN, linewidth=0.8)
                med = _median_trace(traces)
                ax.plot(med, color=colour, linewidth=2.5,
                        label=f"Median (n={len(traces)})")
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8, loc="upper right")
            else:
                ax.text(0.5, 0.5,
                        "No fitness trace\ncaptured\n(check getFitnessTrace)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")

            ax.set_title(title, fontsize=10, fontweight="bold", color=colour)
            ax.set_xlabel("Trace step", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Objective value (log scale)", fontsize=9)

        plt.tight_layout()
        _savefig(filename_template.format(domain=display_domain))
        plt.show()


# =============================================================================
# §2  Q-Table heatmaps — 2 rows × 4 columns per domain
# =============================================================================

def plot_qtable_heatmaps(data: dict, filename_template="{domain}_ext_qtable") -> None:
    """
    §2 — One figure per domain.  Layout: 2 rows × (1 + n_test_labels) columns.

    Row 1 — Q-values from pkl files (mean / max / min across visited states).
      Col 0     : training Q-Table (qtable_{domain}_train{id}_latest.pkl).
      Col 1-3   : evaluation Q-Tables (frozen snapshots).
                  Q-values should be IDENTICAL to training — this is the
                  expected result of the freeze and acts as a sanity check.

    Row 2 — Heuristic usage fractions from JSON call counts.
      These DO differ across instances, showing how the same frozen policy
      selects different heuristics on different problem structures.

    Combined, the two rows answer:
      "The policy is frozen (identical Q-values, Row 1), yet it behaves
       differently (Row 2) because the state space encountered differs."
    """
    meta          = data.get("_meta", {})
    train_id      = meta.get("train_instance_id", 1)
    all_labels    = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })
    all_cols  = ["train"] + all_labels   # e.g. ["train","1.2","1.3","1.4"]
    ncols     = len(all_cols)

    for domain in DOMAINS:
        display_domain = DOMAIN_ALIASES.get(domain, domain)
        eval_runs = _get_runs(data, domain)

        fig, axes = plt.subplots(
            2, ncols,
            figsize=(ncols * 6.0, 8.0),
            squeeze=False,
        )
        fig.suptitle(
            f"{domain}: Q-Table heatmaps\n"
            f"Row 1: Q-values from pkl  |  "
            f"Row 2: Heuristic usage from call counts",
            fontsize=13, fontweight="bold", y=0.99,
        )

        h_count_fallback = 8  # used if pkl cannot be loaded

        for col_idx, label in enumerate(all_cols):
            colour = LABEL_COLOURS.get(label, "#888888")
            title  = LABEL_DISPLAY.get(label, label)

            # Row 1: Q-values from pkl
            ax_q = axes[0][col_idx]
            pkl  = _load_qtable_pkl(_find_qtable_pkl(domain, label, train_id))

            if pkl is not None:
                summary, h_count = _qtable_summary_from_pkl(pkl)
                abs_max = max(float(np.abs(summary).max()), 1e-6)
                norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                _draw_heatmap(
                    ax_q, summary, h_count,
                    cmap="RdYlGn", norm=norm,
                    cb_label="Q-value",
                    row_labels=["Mean Q", "Max Q", "Min Q"],
                )
                pretrain_runs  = _get_pretrain_runs(data, domain)
                train_counts   = [r["qtable_state_count"] for r in pretrain_runs
                                if isinstance(r.get("qtable_state_count"), int)]
                n_states_avg   = int(np.mean(train_counts)) if train_counts else len(pkl.get("table", {}))
                note = "" if label == "train" else " (frozen = same as train)"
                ax_q.set_title(
                    f"{title}\n{n_states_avg} states{note}",
                    fontsize=8, fontweight="bold", color=colour,
                )
                h_count_fallback = h_count
            else:
                ax_q.text(0.5, 0.5, "pkl not found\n(re-run pipeline)",
                          transform=ax_q.transAxes, ha="center", va="center",
                          fontsize=9, color="gray")
                ax_q.set_title(title, fontsize=8, color=colour)
                h_count = h_count_fallback

            if col_idx == 0:
                ax_q.set_ylabel("Q-values (from pkl)", fontsize=8, labelpad=6)

            # Row 2: Heuristic usage from call counts
            ax_u = axes[1][col_idx]

            if label == "train":
                # Pre-training call counts from _pretrain JSON
                pretrain_runs = _get_pretrain_runs(data, domain)
                usage_runs    = pretrain_runs
            else:
                usage_runs = [r for r in eval_runs if r.get("qtable_label") == label]

            summary2 = _usage_summary_from_runs(usage_runs, h_count)

            if summary2 is not None:
                vmax2 = float(summary2.max()) or 1.0
                _draw_heatmap(
                    ax_u, summary2, h_count,
                    cmap="YlOrRd", vmin=0, vmax=vmax2,
                    cb_label="Usage fraction",
                    row_labels=["Mean", "Max", "Min"],
                )
            else:
                ax_u.text(0.5, 0.5, "No call-count\ndata",
                          transform=ax_u.transAxes, ha="center", va="center",
                          fontsize=9, color="gray")

            if col_idx == 0:
                ax_u.set_ylabel("Heuristic usage\n(from call counts)", fontsize=8, labelpad=6)

        plt.tight_layout(rect=[0.015, 0, 1, 1.01])
        plt.subplots_adjust(wspace=0.5)
        _savefig(filename_template.format(domain=display_domain))
        plt.show()


# =============================================================================
# §3  Q-Table state breadth — bar chart per domain
# =============================================================================

def plot_qtable_state_counts(data: dict, filename_template="qtable_state_breadth") -> None:
    """
    §3 — Q-Table state breadth shown as a frequency heatmap.
    Rows = labels (train + test instances).
    Columns = distinct state count values observed across runs.
    Cell value = number of runs that produced that state count.
    """
    meta       = data.get("_meta", {})
    train_id   = meta.get("train_instance_id", 1)
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })
    all_rows = ["train"] + all_labels

    fig, axes = plt.subplots(
        2, 2,
        figsize=(8, 6),
        squeeze=False,
    )

    fig.suptitle(
        "Q-Table state breadth: frequency of state counts across runs",
        # "Cell = number of runs producing that state count  |  "
        # "Train and test rows should match — divergence = unexpected mutation.",
        fontsize=11, fontweight="bold",
    )

    for col_idx, domain in enumerate(DOMAINS):
        ax = axes[col_idx // 2][col_idx % 2]
        eval_runs = _get_runs(data, domain)

        # Collect state counts per label
        counts_per_label: Dict[str, List[int]] = {}

        pretrain_runs = _get_pretrain_runs(data, domain)
        counts_per_label["train"] = [
            r["qtable_state_count"] for r in pretrain_runs
            if isinstance(r.get("qtable_state_count"), int)
        ]
        for label in all_labels:
            label_runs = [r for r in eval_runs if r.get("qtable_label") == label]
            counts_per_label[label] = [
                r["qtable_state_count"] for r in label_runs
                if isinstance(r.get("qtable_state_count"), int)
            ]

        # Find all distinct state count values observed
        all_values = sorted({
            v for counts in counts_per_label.values() for v in counts
        })

        if not all_values:
            ax.set_visible(False)
            continue

        # Build frequency matrix: (n_rows, n_values)
        matrix = np.zeros((len(all_rows), len(all_values)), dtype=int)
        for row_idx, label in enumerate(all_rows):
            for val_idx, val in enumerate(all_values):
                matrix[row_idx, val_idx] = counts_per_label[label].count(val)

        # Draw heatmap
        n_runs = data.get("_meta", {}).get("n_runs", matrix.max())
        im = ax.imshow(matrix, aspect="auto", cmap="Blues",
                       vmin=0, vmax=n_runs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            "Run count", fontsize=7
        )

        # Annotate cells
        for r in range(len(all_rows)):
            for c in range(len(all_values)):
                val = matrix[r, c]
                if val > 0:
                    ax.text(c, r, str(val), ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if val > matrix.max() * 0.6 else "black")

        ax.set_xticks(range(len(all_values)))
        ax.set_xticklabels([str(v) for v in all_values], fontsize=9)
        ax.set_yticks(range(len(all_rows)))
        ax.set_yticklabels(
            [LABEL_DISPLAY.get(l, l) for l in all_rows],
            fontsize=8,
        )
        ax.set_xlabel("State count", fontsize=9)
        ax.set_title(domain, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 1])
    _savefig(filename_template.format())
    plt.show()


# =============================================================================
# §4  Heuristic usage trends across repeats — 4 panels per domain
# =============================================================================

def plot_heuristic_usage_trends(data: dict, filename_template="{domain}_heuristic_usage_trends") -> None:
    """
    §4 — One figure per domain, four panels (train + 3 test instances).

    All four panels share the same layout:
      X-axis : repeat number (1 → n_runs).
      Y-axis : call count per heuristic.
      Lines  : one per heuristic, distinctly coloured.
      Dashes : linear trend line.

    Pre-training panel
      Uses call counts from _pretrain JSON.  For each outer run, call counts
      are summed across all passes within that run, giving total heuristic
      usage during pre-training for that run.  This is directly comparable
      to the evaluation panels which also represent one run per x-tick.

    This directly shows:
      • Do heuristic preferences stay consistent across seeds?  (stable lines)
      • Does the same frozen policy use DIFFERENT heuristics on different
        test instances?  (different dominant lines across panels)
    """
    meta       = data.get("_meta", {})
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })
    all_panels = all_labels

    for domain in DOMAINS:
        display_domain = DOMAIN_ALIASES.get(domain, domain)
        pretrain_runs = _get_pretrain_runs(data, domain)
        eval_runs     = _get_runs(data, domain)

        # Determine h_count across all sources
        all_cc = (
            [r["heuristic_call_counts"] for r in pretrain_runs
             if isinstance(r.get("heuristic_call_counts"), list)]
            + [r["heuristic_call_counts"] for r in eval_runs
               if isinstance(r.get("heuristic_call_counts"), list)]
        )
        if not all_cc:
            continue
        h_count = max(len(cc) for cc in all_cc)

        fig, axes = plt.subplots(
            1, len(all_panels),
            figsize=(len(all_panels) * 5.5, 4.8),
            squeeze=False,
        )
        fig.suptitle(
            f"{domain}: Heuristic usage trends across repeats\n"
            f"(each line = one heuristic  |  dashed = linear trend)",
            fontsize=12, fontweight="bold",
        )

        colours = plt.get_cmap(CMAP_TRACES, h_count)

        for col_idx, panel in enumerate(all_panels):
            ax     = axes[0][col_idx]
            colour = LABEL_COLOURS.get(panel, "#888888")
            title  = LABEL_DISPLAY.get(panel, panel)

            if panel == "train":
                # Sum call counts across all passes for each run_id
                by_run: Dict[int, List[int]] = defaultdict(lambda: [0] * h_count)
                for r in pretrain_runs:
                    rid = r.get("run_id", 0)
                    cc  = r.get("heuristic_call_counts", [])
                    for h in range(h_count):
                        by_run[rid][h] += int(cc[h]) if h < len(cc) else 0
                sorted_runs  = sorted(by_run.items())
                x_vals       = np.array([item[0] for item in sorted_runs])
                counts_arr   = np.array([item[1] for item in sorted_runs], dtype=float)
            else:
                label_runs  = sorted(
                    [r for r in eval_runs if r.get("qtable_label") == panel],
                    key=lambda r: r.get("seed", 0),
                )
                if not label_runs:
                    ax.set_visible(False)
                    continue
                x_vals = np.arange(1, len(label_runs) + 1)
                counts_arr = np.array([
                    [int(r["heuristic_call_counts"][h])
                     if h < len(r.get("heuristic_call_counts", [])) else 0
                     for h in range(h_count)]
                    for r in label_runs
                ], dtype=float)

            if counts_arr.size == 0:
                ax.set_visible(False)
                continue

            for h in range(h_count):
                y   = counts_arr[:, h]
                col = colours(h)
                ax.plot(x_vals, y, color=col, linewidth=1.6, alpha=0.85,
                        label=f"H{h}", marker="o", markersize=3)
                if len(x_vals) > 2:
                    z = np.polyfit(x_vals, y, 1)
                    ax.plot(x_vals, np.poly1d(z)(x_vals),
                            color=col, linewidth=0.8, linestyle="--", alpha=0.55)

            ax.set_title(title, fontsize=10, fontweight="bold", color=colour)
            ax.set_xlabel(
                "Run number" if panel == "train" else "Repeat (seed index)",
                fontsize=9,
            )
            if col_idx == 0:
                ax.set_ylabel("Call count", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=max(1, h_count // 5),
                      loc="upper right", framealpha=0.7)

        plt.tight_layout()
        _savefig(filename_template.format(domain=display_domain))
        plt.show()


# =============================================================================
# §5  Final best-value distributions — strip chart per domain
# =============================================================================

def plot_final_value_distributions(data: dict, filename_template="domain_best_strip_chart") -> None:
    """
    §5 — Strip chart of final best_value per repeat, grouped by Q-Table label.
    One panel per domain.  Lower = better (minimisation).

    Directly answers: how hard are the test instances relative to each other?
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    fig, axes = plt.subplots(
        1, len(DOMAINS),
        figsize=(len(DOMAINS) * 3.8, 5),
        squeeze=False,
    )
    fig.suptitle(
        "Final best-value distributions per test instance\n"
        "(lower = better  |  bar = median  |  dots = individual runs)",
        fontsize=12, fontweight="bold",
    )

    for col_idx, domain in enumerate(DOMAINS):
        ax        = axes[0][col_idx]
        eval_runs = _get_runs(data, domain)

        for j_idx, label in enumerate(all_labels):
            label_runs = [r for r in eval_runs if r.get("qtable_label") == label]
            vals = [r["best_value"] for r in label_runs
                    if isinstance(r.get("best_value"), (int, float))]
            if not vals:
                continue

            col    = LABEL_COLOURS.get(label, "#888888")
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.35
            ax.scatter(j_idx + jitter, vals, color=col, alpha=0.55, s=18, zorder=3)
            med = np.median(vals)
            ax.plot([j_idx - 0.3, j_idx + 0.3], [med, med],
                    color=col, linewidth=2.5, zorder=4,
                    label=f"{LABEL_DISPLAY.get(label, label)} (med={med:.4f})")

        ax.set_title(domain, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels([LABEL_DISPLAY.get(l, l) for l in all_labels],
                           fontsize=7, rotation=15, ha="right")
        ax.set_ylabel("Best value (lower = better)", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(
            fontsize=7, loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            framealpha=0.9,
            borderpad=0.8,
            handlelength=1.5,
            handletextpad=0.5,
            labelspacing=0.4,
        )

    plt.tight_layout()
    _savefig(filename_template.format())
    plt.show()


# =============================================================================
# §6  Improvement ratio — strip chart per domain
# =============================================================================

def plot_improvement_ratio(data: dict, filename_template="domain_improvement_strip_chart") -> None:
    """
    §6 — Strip chart of improvement ratio per test instance, one panel per domain.
    improvement_ratio = (initial_value − best_value) / |initial_value|
    Values above 0 = policy improved the initial solution.
    Values near 1  = near-complete improvement.
    """
    all_labels = sorted({
        r.get("qtable_label", "")
        for domain in DOMAINS
        for r in _get_runs(data, domain)
        if r.get("qtable_label")
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        squeeze=False,
    )
    fig.suptitle(
        "Improvement ratio per test instance\n"
        "(bar = median  |  dots = individual runs  |  higher = better transfer)",
        fontsize=13, fontweight="bold",
    )

    for idx, domain in enumerate(DOMAINS):
        ax        = axes[idx // 2][idx % 2]
        eval_runs = _get_runs(data, domain)

        for j_idx, label in enumerate(all_labels):
            label_runs = [r for r in eval_runs if r.get("qtable_label") == label]
            col        = LABEL_COLOURS.get(label, "#888888")

            imp_vals = [r["improvement_ratio"] for r in label_runs
                        if isinstance(r.get("improvement_ratio"), float)]
            if imp_vals:
                jit = (np.random.rand(len(imp_vals)) - 0.5) * 0.3
                ax.scatter(j_idx + jit, imp_vals, color=col, alpha=0.55, s=20, zorder=3)
                ax.plot(
                    [j_idx - 0.3, j_idx + 0.3],
                    [np.median(imp_vals)] * 2,
                    color=col, lw=2.5, zorder=4,
                )

        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
        ax.set_title(domain, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(
            [LABEL_DISPLAY.get(l, l) for l in all_labels],
            fontsize=9, rotation=15, ha="right",
        )
        ax.set_ylabel("(init − best) / |init|", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(filename_template.format())
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    print("Loading results …")
    data = _load_json(RESULTS_FILE)

    meta = data.get("_meta", {})
    print(f"  n_runs            : {meta.get('n_runs')}")
    print(f"  train_instance_id : {meta.get('train_instance_id')}")
    print(f"  test_instance_ids : {meta.get('test_instance_ids')}")
    print(f"  n_pretrain_runs   : {meta.get('n_pretrain_runs')}")
    print(f"  Q-Table labels    : {meta.get('qtable_labels')}")

    print("\n§1  Fitness convergence (pre-training + test instances) …")
    plot_fitness_traces(data)

    print("\n§2  Q-Table heatmaps (Q-values from pkl  |  usage from call counts) …")
    plot_qtable_heatmaps(data)

    print("\n§3  Q-Table state breadth (training pkl vs. test JSON) …")
    plot_qtable_state_counts(data)

    print("\n§4  Heuristic usage trends across repeats (4 panels per domain) …")
    plot_heuristic_usage_trends(data)

    print("\n§5  Final best-value distributions …")
    plot_final_value_distributions(data)

    print("\n§6  Improvement ratio …")
    plot_improvement_ratio(data)


if __name__ == "__main__":
    main()