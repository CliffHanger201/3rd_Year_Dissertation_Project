from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Sequence
from collections import defaultdict
import json
import os
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
    best_solution_string: Optional[str]
    heuristic_call_counts: List[int] = field(default_factory=list)
    heuristic_call_times_ms: List[int] = field(default_factory=list)
    fitness_trace: Optional[Any] = None


# ---------------------------------------------------------------------------
# Safe extraction helpers
# ---------------------------------------------------------------------------

def _safe_get_call_counts(java_result) -> List[int]:
    """Try common method names for heuristic call counts on the Java result object."""
    for name in ("getHeuristicCallRecord", "getHeuristicCallCounts", "getCallCounts"):
        try:
            return list(getattr(java_result, name)())
        except Exception:
            pass
    return []


def _safe_get_call_times(java_result) -> List[int]:
    """Try common method names for heuristic call times on the Java result object."""
    for name in (
        "getHeuristicCallTimes",
        "getHeuristicCallTimeRecord",
        "getheuristicCallTimeRecord",
        "getCallTimes",
    ):
        try:
            return list(getattr(java_result, name)())
        except Exception:
            pass
    return []


def _safe_get_fitness_trace(java_result) -> Optional[List[float]]:
    """Try common method names for fitness trace on the Java result object."""
    for name in ("getFitnessTrace", "getFitnessHistory", "getObjectiveTrace"):
        try:
            raw = getattr(java_result, name)()
            if raw is not None:
                return [float(v) for v in raw]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Single-run runner
# ---------------------------------------------------------------------------

def run_java_hh_domain(
    runner,
    gateway,
    hh_name: str,
    domain_name: str,
    seed: int = 42,
    time_limit_ms: int = 10000,
    instance_id: int = 0,
    memory_size: int = 2,
    init_indices: Sequence[int] = (0, 1),
) -> JavaRunResult:
    """
    Run a single Java hyper-heuristic on a single domain instance and return metrics.
    Mirrors run_hyflex_domain() from the Python runner.
    """
    # Build a Java int[] array for init_indices
    init_indices_list = list(init_indices)
    java_init_indices = gateway.new_array(gateway.jvm.int, len(init_indices_list))
    for i, val in enumerate(init_indices_list):
        java_init_indices[i] = val

    r = runner.runHyperHeuristic(
        hh_name,
        domain_name,
        seed,
        time_limit_ms,
        instance_id,
        memory_size,
        java_init_indices,
    )

    call_counts = _safe_get_call_counts(r)
    call_times_ms = _safe_get_call_times(r)
    fitness_trace = _safe_get_fitness_trace(r)

    return JavaRunResult(
        hyper_heuristic_name=str(r.getHyperHeuristicName()),
        domain_name=str(r.getDomainName()),
        seed=int(r.getSeed()),
        instance_id=int(r.getInstanceId()),
        time_limit_ms=int(r.getTimeLimitMs()),
        memory_size=int(r.getMemorySize()),
        init_indices=list(r.getInitIndices()),
        wall_time_s=float(r.getWallTimeSeconds()),
        best_value=float(r.getBestValue()),
        best_solution_string=(
            None if r.getBestSolutionString() is None
            else str(r.getBestSolutionString())
        ),
        heuristic_call_counts=call_counts,
        heuristic_call_times_ms=call_times_ms,
        fitness_trace=fitness_trace,
    )


# ---------------------------------------------------------------------------
# Multi-run runner
# ---------------------------------------------------------------------------

def run_java_hh_all_domains(
    hh_names=("AdapHH", "PHunter", "GenHive"),
    domains=("SAT", "VRP", "BinPacking", "TSP"),
    n_runs=30,
    base_seed=42,
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

    all_results: defaultdict = defaultdict(list)

    for run_id in range(n_runs):
        seed = base_seed + run_id
        print(f"\n==============================")
        print(f"RUN {run_id+1}/{n_runs} (seed={seed})")
        print(f"==============================\n")

        for hh_name in hh_names:
            for domain_name in domains:
                instance_id = instances.get(domain_name, 0)

                result = run_java_hh_domain(
                    runner=runner,
                    gateway=gateway,
                    hh_name=hh_name,
                    domain_name=domain_name,
                    seed=seed,
                    time_limit_ms=time_limit_ms,
                    instance_id=instance_id,
                    memory_size=memory_size,
                    init_indices=init_indices,
                )

                all_results[hh_name].append(result)

                print(
                    f"[{hh_name}][{domain_name}] "
                    f"seed={seed} inst={instance_id} "
                    f"best={result.best_value:.4f} wall={result.wall_time_s:.3f}s"
                )

    # Serialise all results
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    payload = {
        hh: [asdict(x) for x in runs]
        for hh, runs in all_results.items()
    }
    payload["_meta"] = {
        "n_runs": n_runs,
        "seed": seed,
        "time_limit_ms": time_limit_ms,
        "memory_size": memory_size,
        "init_indices": list(init_indices),
        "instances": instances,
    }

    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to: {out_json_path}")
    return dict(all_results)


if __name__ == "__main__":
    run_java_hh_all_domains(
        hh_names=("AdapHH", "PHunter", "GenHive"),
        domains=("SAT", "VRP", "BinPacking", "TSP"),
        n_runs=30,
        base_seed=42,
        time_limit_ms=30000,
        instances=None,
        memory_size=2,
        init_indices=(0, 1),
        out_json_path="results/java_hhs_all_domains_results.json",
    )