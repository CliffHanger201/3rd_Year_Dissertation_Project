"""
Visualisation for hyper-heuristic performance across all HyFlex-style domains.

Expects the combined output file produced by run_hyflex_all_domains():
    hh_all_domains_results.json

Plots:
  1) Fitness trace (best-so-far objective) for each domain
  2) Heuristic call counts per domain
  3) Heuristic total runtime (ms) per domain
"""

import json
import matplotlib.pyplot as plt

RESULT_FILE = "results/python_hh_all_domains_results.json"
DOMAINS = ["SAT", "VRP", "TSP", "BinPacking"]


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


def plot_fitness_traces_multi(all_data, meta, max_runs_to_overlay=None, show_median=True):
    """
    One figure per domain.
    - Overlays each run's fitness trace (optionally capped).
    - Optionally adds a median trace (by step index; truncates to min length).
    """
    any_plotted = False

    for domain in DOMAINS:
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
        for i, tr in enumerate(overlay_traces):
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

    # optional: drop heuristics that have no data at all
    # (but keep positions consistent if you prefer)
    return box_data, n_heuristics


def plot_heuristic_boxplots(all_data, key: str, title: str, ylabel: str):
    """
    key: "heuristic_call_counts" OR "heuristic_call_times_ms"
    Boxplot distribution across runs for each heuristic ID.
    """
    any_plotted = False

    for domain in DOMAINS:
        runs = _get_domain_runs(all_data, domain)
        if not runs:
            continue

        box_data, n_heuristics = _collect_per_heuristic_box_data(runs, key)
        if not box_data or n_heuristics == 0:
            continue

        # If lots of heuristics, make the figure wider
        width = max(10, min(24, 0.6 * n_heuristics))
        plt.figure(figsize=(width, 5))

        # Matplotlib wants a list of lists; positions set to heuristic IDs (0..n-1)
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
        max_runs_to_overlay=30,   # set None to overlay all
        show_median=True,
    )

    # 2) Heuristic usage distributions (calls) per domain
    plot_heuristic_boxplots(
        data,
        key="heuristic_call_counts",
        title="Heuristic usage (call counts)",
        ylabel="Calls",
    )

    # 3) Heuristic runtime distributions (ms) per domain
    plot_heuristic_boxplots(
        data,
        key="heuristic_call_times_ms",
        title="Heuristic total runtime (ms)",
        ylabel="Total time (ms)",
    )


if __name__ == "__main__":
    main()


# """
# Visualisation for the hyper-heuristic performance (HyFlex-style output)
# """

# import json
# import matplotlib.pyplot as plt


# RESULT_FILE = "hh_sat_results.json"


# def main():
#     with open(RESULT_FILE, "r") as f:
#         data = json.load(f)

#     domain = data.get("domain", "SAT")
#     instance_id = data.get("instance_id", None)
#     seed = data.get("seed", None)
#     time_limit_ms = data.get("time_limit_ms", None)

#     fitness_trace = data.get("fitness_trace", None)
#     best_value = data.get("best_value", None)

#     if not fitness_trace:
#         raise ValueError(
#             "No fitness_trace found in JSON. "
#             "Make sure your HyperHeuristic writes fitness_trace (hh.getFitnessTrace())."
#         )

#     # --- Plot 1: Best-so-far fitness trace ---
#     plt.figure(figsize=(10, 6))
#     plt.plot(fitness_trace, label="Best-so-far objective")

#     if best_value is not None:
#         plt.axhline(best_value, linestyle="--", label=f"Final best = {best_value}")

#     title_bits = [f"{domain} progress"]
#     if instance_id is not None:
#         title_bits.append(f"instance {instance_id}")
#     if seed is not None:
#         title_bits.append(f"seed {seed}")
#     if time_limit_ms is not None:
#         title_bits.append(f"{time_limit_ms} ms")

#     plt.title(" | ".join(title_bits))
#     plt.xlabel("Trace step")
#     plt.ylabel("Broken / unsatisfied clauses (lower is better)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # --- Plot 2 (optional): heuristic usage (calls) ---
#     call_counts = data.get("heuristic_call_counts", None)
#     if call_counts:
#         plt.figure(figsize=(10, 4))
#         xs = list(range(len(call_counts)))
#         plt.bar(xs, call_counts)
#         plt.title("Heuristic usage (call counts)")
#         plt.xlabel("Heuristic ID")
#         plt.ylabel("Calls")
#         plt.grid(True, axis="y")
#         plt.tight_layout()
#         plt.show()

#     # --- Plot 3 (optional): heuristic time (ms) ---
#     call_times = data.get("heuristic_call_times_ms", None)
#     if call_times:
#         plt.figure(figsize=(10, 4))
#         xs = list(range(len(call_times)))
#         plt.bar(xs, call_times)
#         plt.title("Heuristic total runtime (ms)")
#         plt.xlabel("Heuristic ID")
#         plt.ylabel("Total time (ms)")
#         plt.grid(True, axis="y")
#         plt.tight_layout()
#         plt.show()


# if __name__ == "__main__":
#     main()
