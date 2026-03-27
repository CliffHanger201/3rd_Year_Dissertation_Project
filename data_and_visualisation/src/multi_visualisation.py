"""
multi_visualisation.py
======================
Compares AdapHH, PHunter, GenHive (Java), the custom Python HyperHeuristic,
and the Pretrained Python HyperHeuristic across all HyFlex domains
(SAT, VRP, TSP, BinPacking).

Data sources:
  - results/java_hhs_all_domains_results.json        (keys: AdapHH, PHunter, GenHive)
  - results/python_hh_all_domains_results.json        (keys: SAT, VRP, TSP, BinPacking)
  - results/pretrained_hh_all_domains_results.json    (keys: SAT, VRP, TSP, BinPacking)

Plots produced per domain:
  1) Fitness traces  — median trace per HH overlaid on a single axes
  2) Heuristic call counts  — one boxplot group per HH, side-by-side per heuristic ID
  3) Heuristic call times   — same layout as (2)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ──────────────────────────────────────────────────────────────────

JAVA_FILE       = "results/java_hhs_all_domains_results.json"
PYTHON_FILE     = "results/python_hh_all_domains_results.json"
PRETRAINED_FILE = "results/pretrained_hh_all_domains_results.json"

DOMAINS     = ["SAT", "VRP", "TSP", "BinPacking"]

JAVA_HHS    = ["AdapHH", "PHunter", "GenHive"]
PYTHON_HH   = "MyHyperHeuristic"        # display label for the normal Python HH
PRETRAINED_HH = "MyHyperHeuristic (Pretrained)"  # display label for the pretrained HH

# Colours assigned to each HH (consistent across all plots)
HH_COLOURS  = {
    "AdapHH":           "#e6194b",
    "PHunter":          "#3cb44b",
    "GenHive":          "#4363d8",
    PYTHON_HH:          "#f58231",
    PRETRAINED_HH:      "#911eb4",
}

# ── Data loading helpers ─────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_python_runs(py_data, domain):
    """
    Python JSON is keyed by domain name directly.
    Returns a list of run dicts for that domain.
    """
    runs = py_data.get(domain, [])
    return [r for r in runs if isinstance(r, dict)]


def get_java_runs(java_data, hh_name, domain):
    """
    Java JSON is keyed by HH name; each entry is a run dict with a domain_name field.
    Returns only the runs matching the requested domain.
    """
    all_hh_runs = java_data.get(hh_name, [])
    return [
        r for r in all_hh_runs
        if isinstance(r, dict) and r.get("domain_name") == domain
    ]


# ── Computation helpers ──────────────────────────────────────────────────────

def median_trace(runs):
    """
    Compute a step-wise median fitness trace across a list of run dicts.
    Returns None if no valid traces are found.
    """
    traces = []
    for r in runs:
        tr = r.get("fitness_trace")
        if isinstance(tr, list) and len(tr) > 0:
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
    """
    Returns box_data: list of lists, where box_data[h] = [val_run0, val_run1, ...]
    for heuristic index h.
    """
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

def plot_fitness_traces(java_data, py_data, pretrained_data):
    """One figure per domain; median trace per HH overlaid."""
    for domain in DOMAINS:
        fig, ax = plt.subplots(figsize=(11, 6))
        any_trace = False

        # Java HHs
        for hh_name in JAVA_HHS:
            runs  = get_java_runs(java_data, hh_name, domain)
            trace = median_trace(runs)
            if trace is None:
                continue
            ax.plot(trace, label=f"{hh_name} (median)",
                    color=HH_COLOURS[hh_name], linewidth=2)
            any_trace = True

        # Normal Python HH
        py_runs  = get_python_runs(py_data, domain)
        py_trace = median_trace(py_runs)
        if py_trace is not None:
            ax.plot(py_trace, label=f"{PYTHON_HH} (median)",
                    color=HH_COLOURS[PYTHON_HH], linewidth=2, linestyle="--")
            any_trace = True

        # Pretrained Python HH
        pre_runs  = get_python_runs(pretrained_data, domain)
        pre_trace = median_trace(pre_runs)
        if pre_trace is not None:
            ax.plot(pre_trace, label=f"{PRETRAINED_HH} (median)",
                    color=HH_COLOURS[PRETRAINED_HH], linewidth=2, linestyle="--")
            any_trace = True

        if not any_trace:
            print(f"[fitness trace] No data for domain: {domain}")
            plt.close(fig)
            continue

        ax.set_title(f"{domain} — Fitness Trace (median across runs)")
        ax.set_xlabel("Trace step")
        ax.set_ylabel("Objective value (lower is better)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.yscale("log")
        plt.show()


# ── Plot 2 & 3: Boxplots ─────────────────────────────────────────────────────

def plot_boxplots(java_data, py_data, pretrained_data, key, title, ylabel):
    """
    One figure per domain.
    For each heuristic ID, draw one box per HH side-by-side.
    """
    all_hh_labels = JAVA_HHS + [PYTHON_HH, PRETRAINED_HH]
    n_hhs = len(all_hh_labels)

    for domain in DOMAINS:
        # Gather box_data and n_heuristics for every HH
        hh_boxes   = {}
        max_h      = 0

        for hh_name in JAVA_HHS:
            runs = get_java_runs(java_data, hh_name, domain)
            bd, nh = collect_per_heuristic(runs, key)
            hh_boxes[hh_name] = bd
            max_h = max(max_h, nh)

        py_runs = get_python_runs(py_data, domain)
        bd, nh  = collect_per_heuristic(py_runs, key)
        hh_boxes[PYTHON_HH] = bd
        max_h = max(max_h, nh)

        pre_runs = get_python_runs(pretrained_data, domain)
        bd, nh   = collect_per_heuristic(pre_runs, key)
        hh_boxes[PRETRAINED_HH] = bd
        max_h = max(max_h, nh)

        if max_h == 0:
            print(f"[{key}] No data for domain: {domain}")
            continue

        # Layout: n_hhs boxes per heuristic ID, small gap between groups
        group_width = 0.8
        box_width   = group_width / n_hhs
        offsets     = [(i - (n_hhs - 1) / 2) * box_width for i in range(n_hhs)]

        fig_w = max(10, min(28, max_h * 1.2))
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        for hh_idx, hh_name in enumerate(all_hh_labels):
            bd    = hh_boxes.get(hh_name, [])
            color = HH_COLOURS[hh_name]

            for h in range(max_h):
                data = bd[h] if h < len(bd) and bd[h] else [0]
                pos  = h + offsets[hh_idx]
                bp   = ax.boxplot(
                    data,
                    positions=[pos],
                    widths=box_width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="black", linewidth=1.5),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

        # Legend proxies
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1,
                           facecolor=HH_COLOURS[hh], alpha=0.7, label=hh)
            for hh in all_hh_labels
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        ax.set_xticks(range(max_h))
        ax.set_xticklabels([f"H{h}" for h in range(max_h)])
        ax.set_title(f"{domain} — {title} (distribution across runs)")
        ax.set_xlabel("Heuristic ID")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        fig.tight_layout()
        plt.yscale("log")
        plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    java_data       = load_json(JAVA_FILE)
    py_data         = load_json(PYTHON_FILE)
    pretrained_data = load_json(PRETRAINED_FILE)

    print("=== Fitness Traces ===")
    plot_fitness_traces(java_data, py_data, pretrained_data)

    print("=== Heuristic Call Counts ===")
    plot_boxplots(
        java_data, py_data, pretrained_data,
        key="heuristic_call_counts",
        title="Heuristic usage (call counts)",
        ylabel="Calls",
    )

    print("=== Heuristic Call Times (ms) ===")
    plot_boxplots(
        java_data, py_data, pretrained_data,
        key="heuristic_call_times_ms",
        title="Heuristic total runtime",
        ylabel="Total time (ms)",
    )


if __name__ == "__main__":
    main()