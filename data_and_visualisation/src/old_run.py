"""
multi_visualisation.py
======================
Compares Old Python HyperHeuristic vs New Python HyperHeuristic
across all HyFlex domains (SAT, VRP, TSP, BinPacking).

Data sources:
  - results/old_python_hh_all_domains_results.json
  - results/python_hh_all_domains_results.json

Plots produced per domain:
  1) Fitness traces  — median trace per HH
  2) Heuristic call counts  — boxplots per heuristic
  3) Heuristic call times   — boxplots per heuristic
"""

import json
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────

OLD_PYTHON_FILE = "results/old_python_hh_all_domains_results.json"
PYTHON_FILE     = "results/python_hh_all_domains_results.json"

DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]

OLD_HH = "(Old) MyHyperHeuristic"
NEW_HH = "MyHyperHeuristic"

HH_COLOURS = {
    OLD_HH: "#e6194b",
    NEW_HH: "#3cb44b",
}

# ── Data loading ────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_runs(data, domain):
    runs = data.get(domain, [])
    return [r for r in runs if isinstance(r, dict)]

# ── Computation helpers ─────────────────────────────────────────────────────

def median_trace(runs):
    traces = []
    for r in runs:
        tr = r.get("fitness_trace")
        if isinstance(tr, list) and tr:
            traces.append(tr)

    if not traces:
        return None

    min_len = min(len(t) for t in traces)
    result = []

    for i in range(min_len):
        vals = sorted(t[i] for t in traces)
        m = len(vals)
        if m % 2 == 1:
            result.append(vals[m // 2])
        else:
            result.append(0.5 * (vals[m // 2 - 1] + vals[m // 2]))

    return result


def collect_per_heuristic(runs, key):
    arrays = [r[key] for r in runs if isinstance(r.get(key), list) and r[key]]
    if not arrays:
        return [], 0

    n = max(len(a) for a in arrays)
    box_data = [[] for _ in range(n)]

    for a in arrays:
        for h, val in enumerate(a):
            if isinstance(val, (int, float)):
                box_data[h].append(val)

    return box_data, n

# ── Plot 1: Fitness traces ───────────────────────────────────────────────────

def plot_fitness_traces(old_data, new_data):
    for domain in DOMAINS:
        fig, ax = plt.subplots(figsize=(11, 6))
        any_trace = False

        # Old HH
        old_runs = get_runs(old_data, domain)
        old_trace = median_trace(old_runs)
        if old_trace:
            ax.plot(old_trace, label=f"{OLD_HH} (median)",
                    color=HH_COLOURS[OLD_HH], linewidth=2)
            any_trace = True

        # New HH
        new_runs = get_runs(new_data, domain)
        new_trace = median_trace(new_runs)
        if new_trace:
            ax.plot(new_trace, label=f"{NEW_HH} (median)",
                    color=HH_COLOURS[NEW_HH], linewidth=2, linestyle="--")
            any_trace = True

        if not any_trace:
            print(f"[fitness trace] No data for domain: {domain}")
            plt.close(fig)
            continue

        ax.set_title(f"{domain} — Fitness Trace")
        ax.set_xlabel("Trace step")
        ax.set_ylabel("Objective value (lower is better)")
        ax.legend()
        ax.grid(True)
        plt.yscale("log")

        fig.tight_layout()
        plt.show()

# ── Plot 2 & 3: Boxplots ─────────────────────────────────────────────────────

def plot_boxplots(old_data, new_data, key, title, ylabel):
    all_hhs = [OLD_HH, NEW_HH]
    n_hhs = len(all_hhs)

    for domain in DOMAINS:
        hh_boxes = {}
        max_h = 0

        # Old HH
        old_runs = get_runs(old_data, domain)
        bd, nh = collect_per_heuristic(old_runs, key)
        hh_boxes[OLD_HH] = bd
        max_h = max(max_h, nh)

        # New HH
        new_runs = get_runs(new_data, domain)
        bd, nh = collect_per_heuristic(new_runs, key)
        hh_boxes[NEW_HH] = bd
        max_h = max(max_h, nh)

        if max_h == 0:
            print(f"[{key}] No data for domain: {domain}")
            continue

        group_width = 0.8
        box_width = group_width / n_hhs
        offsets = [(i - (n_hhs - 1) / 2) * box_width for i in range(n_hhs)]

        fig_w = max(10, min(28, max_h * 1.2))
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        for hh_idx, hh_name in enumerate(all_hhs):
            bd = hh_boxes.get(hh_name, [])
            color = HH_COLOURS[hh_name]

            for h in range(max_h):
                data = bd[h] if h < len(bd) and bd[h] else [0]
                pos = h + offsets[hh_idx]

                bp = ax.boxplot(
                    data,
                    positions=[pos],
                    widths=box_width * 0.85,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                )

                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=HH_COLOURS[hh], alpha=0.7, label=hh)
            for hh in all_hhs
        ]
        ax.legend(handles=legend_handles)

        ax.set_xticks(range(max_h))
        ax.set_xticklabels([f"H{h}" for h in range(max_h)])
        ax.set_title(f"{domain} — {title}")
        ax.set_xlabel("Heuristic ID")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        plt.yscale("log")

        fig.tight_layout()
        plt.show()

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    old_data = load_json(OLD_PYTHON_FILE)
    new_data = load_json(PYTHON_FILE)

    print("=== Fitness Traces ===")
    plot_fitness_traces(old_data, new_data)

    print("=== Heuristic Call Counts ===")
    plot_boxplots(
        old_data, new_data,
        key="heuristic_call_counts",
        title="Heuristic usage (call counts)",
        ylabel="Calls",
    )

    print("=== Heuristic Call Times ===")
    plot_boxplots(
        old_data, new_data,
        key="heuristic_call_times_ms",
        title="Heuristic runtime",
        ylabel="Time (ms)",
    )


if __name__ == "__main__":
    main()