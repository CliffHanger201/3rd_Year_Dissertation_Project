"""
visualisation.py
=====================================
Visualisation for hyper-heuristic performance across all HyFlex-style domains.

Expects the combined output file produced by run_hyflex_all_domains():
    hh_all_domains_results.json

Plots:
  1) Fitness trace (best-so-far objective) for each domain
  2) Heuristic call counts per domain
  3) Heuristic total runtime (ms) per domain
"""

import json
import os
import matplotlib.pyplot as plt

RESULT_FILE = "results/python_hh_all_domains_results.json"
DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]
SAVE_DIR = "results/plots"

DOMAIN_ALIASES = {
    "BinPacking": "Bin",
}

plt.rcParams.update({
    "font.size":        13,   # default text / tick labels
    "axes.titlesize":   15,   # subplot titles
    "axes.labelsize":   13,   # x/y axis labels
    "xtick.labelsize":  12,   # x tick numbers
    "ytick.labelsize":  12,   # y tick numbers
    "legend.fontsize":  12,   # legend entries
    "figure.titlesize": 16,   # fig.suptitle
})

def _savefig(name: str) -> None:
    """Save the current figure to SAVE_DIR, then show it."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


def _get_domain_runs(data, name):
    """
    Multi-run format:
      data[name] should be a list of run payload dicts.
    """
    runs = data.get(name)
    if not isinstance(runs, list):
        return []
    # keep only dict runs
    return [r for r in runs if isinstance(r, dict)]


def plot_fitness_traces_multi(all_data, meta, max_runs_to_overlay=None, show_median=True,
                               filename_template="{domain}_fitness_{i}"):
    """
    One figure per domain.
    - Overlays each run's fitness trace (optionally capped).
    - Optionally adds a median trace (by step index; truncates to min length).
    """
    any_plotted = False

    for i, domain in enumerate(DOMAINS, start=1):
        display_domain = DOMAIN_ALIASES.get(domain, domain)  # falls back to domain if no alias
        runs = _get_domain_runs(all_data, domain)
        if not runs:
            continue

        traces = []
        for r in runs:
            tr = r.get("fitness_trace")
            if tr and _safe_len(tr) > 0:
                traces.append(list(tr))

        if not traces:
            continue

        # Optionally cap overlays for readability
        overlay_traces = traces
        if isinstance(max_runs_to_overlay, int) and max_runs_to_overlay > 0:
            overlay_traces = traces[:max_runs_to_overlay]

        plt.figure(figsize=(10, 6))

        # Overlay traces
        for tr in overlay_traces:
            plt.plot(tr, alpha=0.25)

        # Median trace (step-wise), truncated to min length across runs
        if show_median and len(traces) >= 2:
            min_len = min(len(t) for t in traces)
            # compute median without numpy
            median_tr = []
            for j in range(min_len):
                vals = sorted(t[j] for t in traces)
                m = len(vals)
                if m % 2 == 1:
                    median_tr.append(vals[m // 2])
                else:
                    median_tr.append(0.5 * (vals[m // 2 - 1] + vals[m // 2]))
            plt.plot(median_tr, linewidth=2.5, label="Median trace")

        # Title info (meta may have base_seed/n_runs instead of seed)
        title_bits = [f"{domain} progress"]
        if isinstance(meta, dict) and meta:
            if meta.get("n_runs") is not None:
                title_bits.append(f"n_runs {meta.get('n_runs')}")
            if meta.get("base_seed") is not None:
                title_bits.append(f"base_seed {meta.get('base_seed')}")
            if meta.get("time_limit_ms") is not None:
                title_bits.append(f"{meta.get('time_limit_ms')} ms")

        plt.title(" | ".join(title_bits))
        plt.xlabel("Trace step")
        plt.ylabel("Objective (lower is better)")
        plt.grid(True)
        if show_median and len(traces) >= 2:
            plt.legend()
        plt.yscale("log")
        plt.tight_layout()
        _savefig(filename_template.format(domain=display_domain, i=1))
        plt.show()

        any_plotted = True

    if not any_plotted:
        raise ValueError("No fitness_trace found for any domain in the multi-run JSON.")


def _collect_per_heuristic_box_data(runs, key):
    """
    Returns: (box_data, n_heuristics)
    box_data is a list where box_data[h] = [value_run0, value_run1, ...] for heuristic h
    Runs with missing/short arrays are skipped for that heuristic index.
    """
    arrays = []
    for r in runs:
        arr = r.get(key)
        if isinstance(arr, list) and len(arr) > 0:
            arrays.append(arr)

    if not arrays:
        return [], 0

    n_heuristics = max(len(a) for a in arrays)
    box_data = [[] for _ in range(n_heuristics)]

    for a in arrays:
        for h, val in enumerate(a):
            # keep numeric only
            if isinstance(val, (int, float)):
                box_data[h].append(val)

    return box_data, n_heuristics


def plot_heuristic_boxplots(all_data, key: str, title: str, ylabel: str,
                             filename_template="{domain}_{key}_{i}"):
    """
    key: "heuristic_call_counts" OR "heuristic_call_times_ms"
    Boxplot distribution across runs for each heuristic ID.
    """
    any_plotted = False

    for i, domain in enumerate(DOMAINS, start=1):
        display_domain = DOMAIN_ALIASES.get(domain, domain)  # falls back to domain if no alias
        runs = _get_domain_runs(all_data, domain)
        if not runs:
            continue

        box_data, n_heuristics = _collect_per_heuristic_box_data(runs, key)
        if not box_data or n_heuristics == 0:
            continue

        # If lots of heuristics, make the figure wider
        width = max(10, min(24, 0.6 * n_heuristics))
        plt.figure(figsize=(width, 5))

        positions = list(range(n_heuristics))
        plt.boxplot(
            box_data,
            positions=positions,
            showfliers=True,
        )

        plt.title(f"{domain} — {title} (distribution across runs)")
        plt.xlabel("Heuristic ID")
        plt.ylabel(ylabel)
        plt.grid(True, axis="y")
        plt.yscale("log")
        plt.tight_layout()
        _savefig(filename_template.format(domain=display_domain, key=key, i=1))
        plt.show()

        any_plotted = True

    if not any_plotted:
        raise ValueError(f"No '{key}' arrays found for any domain in the multi-run JSON.")


def main():
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)

    meta = data.get("_meta", {})

    # 1) Fitness traces (multi-run)
    plot_fitness_traces_multi(
        data,
        meta,
        max_runs_to_overlay=30,
        show_median=True,
        filename_template="{domain}_Fitness_{i}",
    )

    # 2) Heuristic usage distributions (calls) per domain
    plot_heuristic_boxplots(
        data,
        key="heuristic_call_counts",
        title="Heuristic usage (call counts)",
        ylabel="Calls",
        filename_template="{domain}_Usage_{i}",
    )

    # 3) Heuristic runtime distributions (ms) per domain
    plot_heuristic_boxplots(
        data,
        key="heuristic_call_times_ms",
        title="Heuristic total runtime (ms)",
        ylabel="Total time (ms)",
        filename_template="{domain}_Runtime_{i}",
    )


if __name__ == "__main__":
    main()
