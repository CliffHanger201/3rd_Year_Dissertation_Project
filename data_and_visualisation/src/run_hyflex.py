from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import defaultdict
import json
import os
import glob
from py4j.java_gateway import JavaGateway, GatewayParameters


# def find_jars(*jar_prefixes: str) -> List[str]:
#     """Search common locations on the OS for JARs matching the given prefixes."""
#     search_roots = [
#         os.path.expanduser("~"),           # User home directory
#         "C:/",                             # Windows root
#         "/usr",                            # Linux/Mac
#         "/opt",                            # Linux/Mac
#     ]
#     found = []
#     for root in search_roots:
#         if not os.path.exists(root):
#             continue
#         for prefix in jar_prefixes:
#             pattern = os.path.join(root, "**", f"{prefix}*.jar")
#             matches = glob.glob(pattern, recursive=True)
#             found.extend(matches)
#     # Deduplicate while preserving order
#     seen = set()
#     unique = []
#     for path in found:
#         if path not in seen:
#             seen.add(path)
#             unique.append(path)
#     return unique


@dataclass
class JavaRunResult:
    hyper_heuristic_name: str
    domain_name: str
    seed: int
    instance_id: int
    time_limit_ms: int
    memory_size: int
    init_indices: List[int]
    wall_time_s: float
    best_value: float
    best_solution_string: str | None


def run_java_hh_all_domains(
    hh_names=("AdapHH", "PHunter", "GenHive"),
    domains=("BinPacking", "TSP", "VRP", "SAT"),
    n_runs=1,
    seed=42,
    time_limit_ms=10000,
    instances=None,
    memory_size=2,
    init_indices=(0, 1),
    out_json_path="results/java_hhs_all_domains_results.json",
):
    if instances is None:
        instances = {d: 0 for d in domains}

    # # Search for slf4j JARs on the system and report findings
    # slf4j_jars = find_jars("slf4j-api", "slf4j-simple", "slf4j-nop")
    # if slf4j_jars:
    #     print("Found slf4j JARs:")
    #     for jar in slf4j_jars:
    #         print(f"  {jar}")
    #     classpath = os.pathsep.join(slf4j_jars)
    #     os.environ["CLASSPATH"] = (
    #         os.environ.get("CLASSPATH", "") + os.pathsep + classpath
    #     ).lstrip(os.pathsep)
    #     print(f"\nAdded to CLASSPATH: {classpath}\n")
    # else:
    #     print(
    #         "WARNING: No slf4j JARs found on this system.\n"
    #         "Download slf4j-api and slf4j-simple from https://www.slf4j.org/download.html\n"
    #         "and place them alongside your py4j.jar.\n"
    #     )

    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))
    runner = gateway.entry_point

    all_results = defaultdict(list)

    for run_id in range(n_runs):
        run_seed = seed + run_id
        print(f"\n==============================")
        print(f"RUN {run_id+1}/{n_runs} (seed={run_seed})")
        print(f"==============================\n")

        # Convert Python list to a Java int[] array once per run
        init_indices_list = list(init_indices)
        java_init_indices = gateway.new_array(gateway.jvm.int, len(init_indices_list))
        for i, val in enumerate(init_indices_list):
            java_init_indices[i] = val

        for hh_name in hh_names:
            for domain_name in domains:
                instance_id = instances.get(domain_name, 0)
                r = runner.runHyperHeuristic(
                    hh_name,
                    domain_name,
                    run_seed,
                    time_limit_ms,
                    instance_id,
                    memory_size,
                    java_init_indices,
                )
                result = JavaRunResult(
                    hyper_heuristic_name=str(r.getHyperHeuristicName()),
                    domain_name=str(r.getDomainName()),
                    seed=int(r.getSeed()),
                    instance_id=int(r.getInstanceId()),
                    time_limit_ms=int(r.getTimeLimitMs()),
                    memory_size=int(r.getMemorySize()),
                    init_indices=list(r.getInitIndices()),
                    wall_time_s=float(r.getWallTimeSeconds()),
                    best_value=float(r.getBestValue()),
                    best_solution_string=None if r.getBestSolutionString() is None else str(r.getBestSolutionString()),
                )
                all_results[hh_name].append(result)
                print(
                    f"[{hh_name}][{domain_name}] "
                    f"seed={run_seed} inst={instance_id} "
                    f"best={result.best_value} wall={result.wall_time_s:.3f}s"
                )

    payload = {
        hh: [asdict(x) for x in runs]
        for hh, runs in all_results.items()
    }
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {out_json_path}")
    return dict(all_results)


if __name__ == "__main__":
    run_java_hh_all_domains(
        hh_names=("AdapHH", "PHunter", "GenHive"),
        domains=("BinPacking", "TSP", "VRP", "SAT"),
        n_runs=1,
        seed=42,
        time_limit_ms=10000,
        instances=None,
        memory_size=2,
        init_indices=(0, 1),
        out_json_path="results/java_hhs_domains_results.json",
    )