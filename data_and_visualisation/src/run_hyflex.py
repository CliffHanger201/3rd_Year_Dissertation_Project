from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import defaultdict
import json

from py4j.java_gateway import JavaGateway, GatewayParameters


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
    n_runs=30,
    seed=42,
    time_limit_ms=30000,
    instances=None,
    memory_size=2,
    init_indices=(0, 1),
    out_json_path="results/java_hhs_all_domains_results.json",
):
    if instances is None:
        instances = {d: 0 for d in domains}

    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))
    runner = gateway.entry_point

    all_results = defaultdict(list)

    for run_id in range(n_runs):
        run_seed = seed + run_id
        print(f"\n==============================")
        print(f"RUN {run_id+1}/{n_runs} (seed={run_seed})")
        print(f"==============================\n")

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
                    list(init_indices),
                )

                result = JavaRunResult(
                    hyper_heuristic_name=str(r.hyperHeuristicName),
                    domain_name=str(r.domainName),
                    seed=int(r.seed),
                    instance_id=int(r.instanceId),
                    time_limit_ms=int(r.timeLimitMs),
                    memory_size=int(r.memorySize),
                    init_indices=list(r.initIndices),
                    wall_time_s=float(r.wallTimeSeconds),
                    best_value=float(r.bestValue),
                    best_solution_string=None if r.bestSolutionString is None else str(r.bestSolutionString),
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