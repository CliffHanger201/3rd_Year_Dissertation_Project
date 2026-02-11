"""
Visualisation for the hyper-heuristic performance (HyFlex-style output)
"""

import json
import matplotlib.pyplot as plt


RESULT_FILE = "hh_sat_results.json"


def main():
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)

    domain = data.get("domain", "SAT")
    instance_id = data.get("instance_id", None)
    seed = data.get("seed", None)
    time_limit_ms = data.get("time_limit_ms", None)

    fitness_trace = data.get("fitness_trace", None)
    best_value = data.get("best_value", None)

    if not fitness_trace:
        raise ValueError(
            "No fitness_trace found in JSON. "
            "Make sure your HyperHeuristic writes fitness_trace (hh.getFitnessTrace())."
        )

    # --- Plot 1: Best-so-far fitness trace ---
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_trace, label="Best-so-far objective")

    if best_value is not None:
        plt.axhline(best_value, linestyle="--", label=f"Final best = {best_value}")

    title_bits = [f"{domain} progress"]
    if instance_id is not None:
        title_bits.append(f"instance {instance_id}")
    if seed is not None:
        title_bits.append(f"seed {seed}")
    if time_limit_ms is not None:
        title_bits.append(f"{time_limit_ms} ms")

    plt.title(" | ".join(title_bits))
    plt.xlabel("Trace step")
    plt.ylabel("Broken / unsatisfied clauses (lower is better)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2 (optional): heuristic usage (calls) ---
    call_counts = data.get("heuristic_call_counts", None)
    if call_counts:
        plt.figure(figsize=(10, 4))
        xs = list(range(len(call_counts)))
        plt.bar(xs, call_counts)
        plt.title("Heuristic usage (call counts)")
        plt.xlabel("Heuristic ID")
        plt.ylabel("Calls")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()

    # --- Plot 3 (optional): heuristic time (ms) ---
    call_times = data.get("heuristic_call_times_ms", None)
    if call_times:
        plt.figure(figsize=(10, 4))
        xs = list(range(len(call_times)))
        plt.bar(xs, call_times)
        plt.title("Heuristic total runtime (ms)")
        plt.xlabel("Heuristic ID")
        plt.ylabel("Total time (ms)")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
