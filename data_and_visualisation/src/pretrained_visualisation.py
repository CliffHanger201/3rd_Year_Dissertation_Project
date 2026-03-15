"""
Visualisation for hyper-heuristic results: normal vs pretrained.

File layout expected
────────────────────
Q-Tables  (one .pkl per domain, for the pretrained HH):
    qtables/qtable_SAT.pkl
    qtables/qtable_VRP.pkl
    qtables/qtable_TSP.pkl
    qtables/qtable_BinPacking.pkl

Results JSON (multi-run format produced by run_hyflex_all_domains()):
    results/python_hh_all_domains_results.json           ← normal HH
    results/python_hh_pretrained_all_domains_results.json ← pretrained HH

Plots (in order)
────────────────
  §1  Q-Table heatmaps          — one figure per domain (pretrained only)
  §2  Fitness traces            — normal vs pretrained overlaid, one figure per domain
  §3  Heuristic call counts     — boxplots, normal vs pretrained, one figure per domain
  §4  Heuristic runtime (ms)    — boxplots, normal vs pretrained, one figure per domain
"""

import json
import pickle
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
NORMAL_FILE    = "results/python_hh_all_domains_results.json"
PRETRAINED_FILE = "results/pretrained_hh_all_domains_results.json"

QTABLE_DIR     = "qtables"
QTABLE_PATHS   = {
    domain: os.path.join(QTABLE_DIR, f"qtable_{domain}.pkl")
    for domain in ["SAT", "VRP", "TSP", "BinPacking"]
}

DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]

# Colour palette — normal=blue family, pretrained=orange family
COLOUR_NORMAL     = "#4C72B0"
COLOUR_PRETRAINED = "#DD8452"
ALPHA_TRACE       = 0.18   # individual run opacity
ALPHA_MEDIAN      = 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_qtable_pkl(path: str) -> dict:
    """Return the raw pickle dict (keys: h_count, state_bins, lr, gamma, table)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_domain_runs(data: dict, name: str) -> list:
    runs = data.get(name)
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def _median_trace(traces: list) -> list:
    """Step-wise median, truncated to the shortest trace."""
    if len(traces) < 2:
        return traces[0] if traces else []
    min_len = min(len(t) for t in traces)
    result = []
    for j in range(min_len):
        vals = sorted(t[j] for t in traces)
        m = len(vals)
        result.append(
            vals[m // 2] if m % 2 == 1
            else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
        )
    return result


def _collect_box_data(runs: list, key: str):
    """
    Returns (box_data, n_heuristics).
    box_data[h] = list of values across runs for heuristic h.
    """
    arrays = [r.get(key) for r in runs if isinstance(r.get(key), list) and r.get(key)]
    if not arrays:
        return [], 0
    n = max(len(a) for a in arrays)
    box_data = [[] for _ in range(n)]
    for a in arrays:
        for h, val in enumerate(a):
            if isinstance(val, (int, float)):
                box_data[h].append(val)
    return box_data, n


# ═════════════════════════════════════════════════════════════════════════════
# §1  Q-Table heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def _build_heatmap_matrix(table: dict, h_count: int):
    """
    Convert sparse {state_key: np.ndarray} → 2-D numpy array.

    Rows  = state keys, sorted lexicographically for reproducibility.
    Cols  = heuristic IDs (0 … h_count-1).

    Returns (matrix, row_labels).
    """
    keys = sorted(table.keys())
    if not keys:
        return np.zeros((1, h_count)), ["(empty)"]

    matrix = np.vstack([table[k] for k in keys])   # shape: (n_states, h_count)

    # Truncate / pad columns to h_count just in case
    if matrix.shape[1] < h_count:
        pad = np.zeros((matrix.shape[0], h_count - matrix.shape[1]))
        matrix = np.hstack([matrix, pad])
    elif matrix.shape[1] > h_count:
        matrix = matrix[:, :h_count]

    # Human-readable row labels: "(b0, b1, …)"
    row_labels = [str(k) for k in keys]
    return matrix.astype(float), row_labels


def plot_qtable_heatmaps():
    """§1 — One heatmap per domain for the pretrained Q-table."""
    for domain in DOMAINS:
        pkl = _load_qtable_pkl(QTABLE_PATHS[domain])
        table   = pkl["table"]
        h_count = pkl["h_count"]

        if not table:
            continue

        matrix = np.vstack(list(table.values()))  # (n_states, h_count)

        # Summarise into 3 rows instead of hundreds
        summary = np.vstack([
            matrix.mean(axis=0),
            matrix.max(axis=0),
            matrix.min(axis=0),
        ])

        fig, ax = plt.subplots(figsize=(max(6, h_count * 0.8), 3))
        abs_max = max(np.abs(summary).max(), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax.imshow(summary, aspect="auto", cmap="RdYlGn", norm=norm)
        fig.colorbar(im, ax=ax).set_label("Q-value")

        ax.set_xticks(range(h_count))
        ax.set_xticklabels([f"H{i}" for i in range(h_count)])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Mean", "Max", "Min"])
        ax.set_title(f"{domain} — Q-Table summary ({len(table)} states visited)",
                     fontweight="bold")

        # Annotate each cell
        for r in range(3):
            for c in range(h_count):
                ax.text(c, r, f"{summary[r,c]:.3f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# §2  Fitness traces — normal vs pretrained
# ═════════════════════════════════════════════════════════════════════════════

def plot_fitness_traces_comparison(normal_data: dict, pretrained_data: dict,
                                   max_overlay: int = 30):
    """
    One figure per domain.
    Blue  = normal HH runs + bold median.
    Orange = pretrained HH runs + bold median.
    """
    any_plotted = False

    for domain in DOMAINS:
        norm_runs = _get_domain_runs(normal_data,     domain)
        pre_runs  = _get_domain_runs(pretrained_data, domain)

        norm_traces = [list(r["fitness_trace"]) for r in norm_runs
                       if isinstance(r.get("fitness_trace"), list) and r["fitness_trace"]]
        pre_traces  = [list(r["fitness_trace"]) for r in pre_runs
                       if isinstance(r.get("fitness_trace"), list) and r["fitness_trace"]]

        if not norm_traces and not pre_traces:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # ── overlay individual runs ──────────────────────────────────────────
        for tr in norm_traces[:max_overlay]:
            ax.plot(tr, color=COLOUR_NORMAL,     alpha=ALPHA_TRACE, linewidth=0.8)
        for tr in pre_traces[:max_overlay]:
            ax.plot(tr, color=COLOUR_PRETRAINED, alpha=ALPHA_TRACE, linewidth=0.8)

        # ── median traces ────────────────────────────────────────────────────
        if len(norm_traces) >= 2:
            ax.plot(_median_trace(norm_traces),
                    color=COLOUR_NORMAL, linewidth=2.5,
                    label=f"Normal (median, n={len(norm_traces)})")
        elif norm_traces:
            ax.plot(norm_traces[0],
                    color=COLOUR_NORMAL, linewidth=2.5,
                    label="Normal (single run)")

        if len(pre_traces) >= 2:
            ax.plot(_median_trace(pre_traces),
                    color=COLOUR_PRETRAINED, linewidth=2.5,
                    label=f"Pretrained (median, n={len(pre_traces)})")
        elif pre_traces:
            ax.plot(pre_traces[0],
                    color=COLOUR_PRETRAINED, linewidth=2.5,
                    label="Pretrained (single run)")

        ax.set_title(f"{domain} — Fitness trace: Normal vs Pretrained",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Trace step")
        ax.set_ylabel("Objective (lower is better)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        any_plotted = True

    if not any_plotted:
        raise ValueError("No fitness_trace data found in either results file.")


# ═════════════════════════════════════════════════════════════════════════════
# §3 & §4  Heuristic usage / runtime boxplots — normal vs pretrained
# ═════════════════════════════════════════════════════════════════════════════

def plot_heuristic_boxplots_comparison(normal_data: dict, pretrained_data: dict,
                                       key: str, title: str, ylabel: str):
    """
    Side-by-side boxplots per heuristic ID.
    Left  (blue)   = normal HH.
    Right (orange) = pretrained HH.
    """
    any_plotted = False

    for domain in DOMAINS:
        norm_runs = _get_domain_runs(normal_data,     domain)
        pre_runs  = _get_domain_runs(pretrained_data, domain)

        norm_box, norm_n = _collect_box_data(norm_runs, key)
        pre_box,  pre_n  = _collect_box_data(pre_runs,  key)

        if not norm_box and not pre_box:
            continue

        n_heuristics = max(norm_n, pre_n)
        width = max(10, min(28, 0.9 * n_heuristics))
        fig, ax = plt.subplots(figsize=(width, 5))

        # ── pad shorter array with empty lists so indices align ──────────────
        while len(norm_box) < n_heuristics:
            norm_box.append([])
        while len(pre_box)  < n_heuristics:
            pre_box.append([])

        offset = 0.22   # half-gap between the two boxplots per heuristic

        # Filter out heuristics with no data for cleaner plots
        def _valid(lst):
            return [v for v in lst if isinstance(v, (int, float))]

        norm_data_clean = [_valid(norm_box[h]) for h in range(n_heuristics)]
        pre_data_clean  = [_valid(pre_box[h])  for h in range(n_heuristics)]

        positions_norm = [h - offset for h in range(n_heuristics)]
        positions_pre  = [h + offset for h in range(n_heuristics)]

        bp_norm = ax.boxplot(
            norm_data_clean,
            positions=positions_norm,
            widths=0.35,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=COLOUR_NORMAL, alpha=0.7),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=COLOUR_NORMAL),
            capprops=dict(color=COLOUR_NORMAL),
            flierprops=dict(marker="o", color=COLOUR_NORMAL,
                            markerfacecolor=COLOUR_NORMAL, alpha=0.4, markersize=4),
        )

        bp_pre = ax.boxplot(
            pre_data_clean,
            positions=positions_pre,
            widths=0.35,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=COLOUR_PRETRAINED, alpha=0.7),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=COLOUR_PRETRAINED),
            capprops=dict(color=COLOUR_PRETRAINED),
            flierprops=dict(marker="o", color=COLOUR_PRETRAINED,
                            markerfacecolor=COLOUR_PRETRAINED, alpha=0.4, markersize=4),
        )

        # Legend proxy patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOUR_NORMAL,     alpha=0.8, label="Normal HH"),
            Patch(facecolor=COLOUR_PRETRAINED, alpha=0.8, label="Pretrained HH"),
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        ax.set_xticks(range(n_heuristics))
        ax.set_xticklabels([f"H{h}" for h in range(n_heuristics)], fontsize=8)
        ax.set_xlabel("Heuristic ID", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_title(f"{domain} — {title}: Normal vs Pretrained",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()
        any_plotted = True

    if not any_plotted:
        print(f"[Boxplot] No '{key}' data found for any domain.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── load results ─────────────────────────────────────────────────────────
    print("Loading result files …")
    normal_data     = _load_json(NORMAL_FILE)
    pretrained_data = _load_json(PRETRAINED_FILE)

    # ── §1  Q-Table heatmaps (pretrained) ────────────────────────────────────
    print("\n§1  Plotting Q-Table heatmaps …")
    plot_qtable_heatmaps()

    # ── §2  Fitness traces ────────────────────────────────────────────────────
    print("\n§2  Plotting fitness traces …")
    plot_fitness_traces_comparison(
        normal_data,
        pretrained_data,
        max_overlay=30,
    )

    # ── §3  Heuristic call counts ─────────────────────────────────────────────
    print("\n§3  Plotting heuristic call-count distributions …")
    plot_heuristic_boxplots_comparison(
        normal_data, pretrained_data,
        key="heuristic_call_counts",
        title="Heuristic usage (call counts)",
        ylabel="Calls",
    )

    # ── §4  Heuristic runtimes ────────────────────────────────────────────────
    print("\n§4  Plotting heuristic runtime distributions …")
    plot_heuristic_boxplots_comparison(
        normal_data, pretrained_data,
        key="heuristic_call_times_ms",
        title="Heuristic total runtime (ms)",
        ylabel="Total time (ms)",
    )


if __name__ == "__main__":
    main()