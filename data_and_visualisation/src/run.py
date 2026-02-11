import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from python_hyper_heuristic.src.hyperheuristic import HyperHeuristic

# Domain imports (match your SAT import style: class inside module)
from python_hyper_heuristic.domains.Python.SAT.SAT import SAT
from python_hyper_heuristic.domains.Python.VRP.VRP import VRP
from python_hyper_heuristic.domains.Python.BinPacking.BinPacking import BinPacking
from python_hyper_heuristic.domains.Python.TSP.TSP import TSP
from python_hyper_heuristic.domains.Python.FlowShop.FlowShop import FlowShop


@dataclass
class DomainRunResult:
    domain_name: str
    seed: int
    instance_id: int
    time_limit_ms: int
    memory_size: int
    init_indices: Tuple[int, ...]
    wall_time_s: float
    best_value: float
    best_solution_string: Optional[str]
    heuristic_call_counts: List[int]
    heuristic_call_times_ms: List[int]
    fitness_trace: Optional[Any]  # keep generic; HH may return list/tuple/etc.


def _safe_get_call_times(domain_obj) -> List[int]:
    # Your snippet used getheuristicCallTimeRecord() (lowercase h) — keep compatibility.
    for name in ("getheuristicCallTimeRecord", "getHeuristicCallTimeRecord"):
        if hasattr(domain_obj, name):
            try:
                return list(getattr(domain_obj, name)())
            except Exception:
                pass
    return []


def _safe_get_call_counts(domain_obj) -> List[int]:
    for name in ("getHeuristicCallRecord", "get_heuristic_call_record"):
        if hasattr(domain_obj, name):
            try:
                return list(getattr(domain_obj, name)())
            except Exception:
                pass
    return []


def _safe_get_fitness_trace(hh_obj) -> Optional[Any]:
    if hasattr(hh_obj, "getFitnessTrace"):
        try:
            return hh_obj.getFitnessTrace()
        except Exception:
            return None
    return None


def _safe_get_best_solution_string(problem) -> Optional[str]:
    """
    Prefer problem.getSolutionToString() if it exists, otherwise fall back
    to bestSolutionToString() (HyFlex-style), otherwise None.
    """
    # User asked specifically for getSolutionToString
    if hasattr(problem, "getSolutionToString"):
        try:
            return problem.getSolutionToString()
        except Exception:
            return None

    # Common HyFlex method name
    if hasattr(problem, "bestSolutionToString"):
        try:
            return problem.bestSolutionToString()
        except Exception:
            return None

    return None


def run_hyflex_domain(
    domain_cls: Type,
    domain_name: str,
    seed: int = 42,
    time_limit_ms: int = 5_000,
    instance_id: int = 0,
    memory_size: int = 2,
    init_indices: Sequence[int] = (0, 1),
) -> DomainRunResult:
    """
    Run HyperHeuristic on a single HyFlex-style domain instance and return metrics.
    Uses problem.getSolutionToString() (preferred) instead of best index.
    """
    # ------------------------------
    # Create domain + load instance
    # ------------------------------
    problem = domain_cls(seed=seed)
    problem.loadInstance(instance_id)
    problem.setMemorySize(memory_size)

    for idx in init_indices:
        problem.initialiseSolution(idx)

    # ------------------------------
    # Initialize hyper-heuristic
    # ------------------------------
    hh = HyperHeuristic(seed=seed)
    hh.setTimeLimit(time_limit_ms)
    hh.loadProblemDomain(problem)

    # ------------------------------
    # Run hyper-heuristic
    # ------------------------------
    wall_start = time.time()
    hh.run()
    wall_end = time.time()

    # ------------------------------
    # Extract results
    # ------------------------------
    best_value = problem.getBestSolutionValue()
    best_solution_str = _safe_get_best_solution_string(problem)

    fitness_trace = _safe_get_fitness_trace(hh)
    call_counts = _safe_get_call_counts(problem)
    call_times_ms = _safe_get_call_times(problem)

    return DomainRunResult(
        domain_name=domain_name,
        seed=seed,
        instance_id=instance_id,
        time_limit_ms=time_limit_ms,
        memory_size=memory_size,
        init_indices=tuple(init_indices),
        wall_time_s=wall_end - wall_start,
        best_value=best_value,
        best_solution_string=best_solution_str,
        heuristic_call_counts=call_counts,
        heuristic_call_times_ms=call_times_ms,
        fitness_trace=fitness_trace,
    )


def run_hyflex_all_domains(
    seed: int = 42,
    time_limit_ms: int = 5_000,
    instances: Optional[Dict[str, int]] = None,
    memory_size: int = 2,
    init_indices: Sequence[int] = (0, 1),
    out_json_path: str = "hh_all_domains_results.json",
) -> Dict[str, DomainRunResult]:
    """
    Run the same HyperHeuristic on SAT, VRP, BinPacking, TSP, FlowShop.
    `instances` lets you specify per-domain instance IDs, e.g. {"SAT": 0, "TSP": 3}
    """
    domain_specs = [
        ("SAT", SAT),
        ("VRP", VRP),
        ("BinPacking", BinPacking),
        ("TSP", TSP),
        ("FlowShop", FlowShop),
    ]

    if instances is None:
        instances = {name: 0 for name, _cls in domain_specs}

    results: Dict[str, DomainRunResult] = {}

    for name, cls in domain_specs:
        instance_id = instances.get(name, 0)
        res = run_hyflex_domain(
            domain_cls=cls,
            domain_name=name,
            seed=seed,
            time_limit_ms=time_limit_ms,
            instance_id=instance_id,
            memory_size=memory_size,
            init_indices=init_indices,
        )
        results[name] = res

        # ------------------------------
        # Print per-domain summary
        # ------------------------------
        print(f"=== HyFlex {name} Run Summary ===")
        print(f"Domain: {name}")
        print(f"Seed: {seed}")
        print(f"Instance ID: {instance_id}")
        print(f"Time limit (ms): {time_limit_ms}")
        print(f"Wall time (s): {res.wall_time_s:.3f}")
        print()
        print("Best objective:", res.best_value)
        if res.best_solution_string is not None:
            print("Best solution (string):")
            print(res.best_solution_string)
        else:
            print("Best solution string: <unavailable>")
        print()
        print("Heuristic call counts:", res.heuristic_call_counts)
        print("Heuristic total time (ms):", res.heuristic_call_times_ms)
        if res.heuristic_call_counts:
            print(f"Total heuristic calls: {sum(res.heuristic_call_counts)}")
        if res.heuristic_call_times_ms:
            print(f"Total heuristic time (ms): {sum(res.heuristic_call_times_ms)}")

        if res.fitness_trace is not None:
            try:
                ft_len = len(res.fitness_trace)
                print()
                print(f"Fitness trace length: {ft_len}")
                if ft_len > 0:
                    print("Fitness trace (first 10):", list(res.fitness_trace)[:10])
                    print("Fitness trace (last 10):", list(res.fitness_trace)[-10:])
            except Exception:
                pass

        print("\n")

    # ------------------------------
    # Save combined results
    # ------------------------------
    payload = {name: asdict(r) for name, r in results.items()}
    payload["_meta"] = {
        "seed": seed,
        "time_limit_ms": time_limit_ms,
        "memory_size": memory_size,
        "init_indices": list(init_indices),
        "instances": instances,
    }

    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {out_json_path}")
    return results


if __name__ == "__main__":
    # Pick per-domain instance IDs as needed.
    run_hyflex_all_domains(
        seed=42,
        time_limit_ms=5_000,
        instances={
            "SAT": 0,
            "VRP": 0,
            "BinPacking": 0,
            "TSP": 0,
            "FlowShop": 0,
        },
        memory_size=2,
        init_indices=(0, 1),
        out_json_path="hh_all_domains_results.json",
    )


# import json
# import time

# from python_hyper_heuristic.src.hyperheuristic import HyperHeuristic
# from python_hyper_heuristic.domains.Python.SAT.SAT import SAT  # expects class SAT, not module SAT.SAT


# def run_hyflex_sat(seed: int = 42,
#                   time_limit_ms: int = 5_000,
#                   instance_id: int = 0,
#                   memory_size: int = 2,
#                   init_indices=(0, 1),
#                   out_json_path: str = "hh_sat_results.json"):
#     """
#     HyFlex-style: domain loads an instance by ID, HH runs under a time limit,
#     domain tracks best solution + value.
#     """

#     # ------------------------------
#     # Create domain + load instance
#     # ------------------------------
#     sat_problem = SAT(seed=seed)
#     sat_problem.loadInstance(instance_id)

#     # HyFlex typically uses a small memory (often 2); your base class defaults to 2,
#     # but set explicitly in case you want to change it.
#     sat_problem.setMemorySize(memory_size)

#     # Initialise starting solutions in memory
#     for idx in init_indices:
#         sat_problem.initialiseSolution(idx)

#     # ------------------------------
#     # Initialize hyper-heuristic
#     # ------------------------------
#     hh = HyperHeuristic(seed=seed)
#     hh.setTimeLimit(time_limit_ms)
#     hh.loadProblemDomain(sat_problem)

#     # ------------------------------
#     # Run hyper-heuristic
#     # ------------------------------
#     wall_start = time.time()
#     hh.run()
#     wall_end = time.time()

#     # ------------------------------
#     # Extract results (HyFlex style)
#     # ------------------------------
#     best_value = sat_problem.getBestSolutionValue()
#     best_index = sat_problem.getBestSolutionIndex()

#     # Solution representation varies by domain; safest is string form
#     # (HyFlex provides solutionToString in ProblemDomain).
#     best_solution_str = None
#     if best_index is not None and best_index != -1:
#         try:
#             best_solution_str = sat_problem.solutionToString(best_index)
#         except Exception:
#             best_solution_str = None

#     # Optional trace (if your HH provides it)
#     fitness_trace = None
#     try:
#         fitness_trace = hh.getFitnessTrace()
#     except Exception:
#         fitness_trace = None

#     # Heuristic call stats (common evaluation metrics)
#     call_counts = sat_problem.getHeuristicCallRecord()
#     call_times_ms = sat_problem.getheuristicCallTimeRecord()

#     # ------------------------------
#     # Print evaluation metrics
#     # ------------------------------
#     print("=== HyFlex SAT Run Summary ===")
#     print(f"Domain: {sat_problem}")
#     print(f"Seed: {seed}")
#     print(f"Instance ID: {instance_id}")
#     print(f"Time limit (ms): {time_limit_ms}")
#     print(f"Wall time (s): {wall_end - wall_start:.3f}")
#     print()
#     print("Best objective (unsatisfied/broken clauses):", best_value)
#     print("Best solution memory index:", best_index)
#     if best_solution_str is not None:
#         print("Best solution (string):")
#         print(best_solution_str)
#     else:
#         print("Best solution string: <unavailable>")
#     print()

#     # Heuristic stats
#     print("Heuristic call counts:", call_counts)
#     print("Heuristic total time (ms):", call_times_ms)
#     total_calls = sum(call_counts)
#     total_h_ms = sum(call_times_ms)
#     print(f"Total heuristic calls: {total_calls}")
#     print(f"Total heuristic time (ms): {total_h_ms}")

#     if fitness_trace is not None:
#         print()
#         print(f"Fitness trace length: {len(fitness_trace)}")
#         if len(fitness_trace) > 0:
#             print("Fitness trace (first 10):", fitness_trace[:10])
#             print("Fitness trace (last 10):", fitness_trace[-10:])

#     # ------------------------------
#     # Save results
#     # ------------------------------
#     payload = {
#         "domain": str(sat_problem),
#         "seed": seed,
#         "instance_id": instance_id,
#         "time_limit_ms": time_limit_ms,
#         "wall_time_s": wall_end - wall_start,
#         "best_value": best_value,
#         "best_index": best_index,
#         "best_solution_string": best_solution_str,
#         "heuristic_call_counts": call_counts,
#         "heuristic_call_times_ms": call_times_ms,
#         "fitness_trace": fitness_trace,
#     }

#     with open(out_json_path, "w") as f:
#         json.dump(payload, f, indent=2)

#     print()
#     print(f"Saved results to: {out_json_path}")


# if __name__ == "__main__":
#     run_hyflex_sat(
#         seed=42,
#         time_limit_ms=5_000,
#         instance_id=0,     # choose 0..11 based on your SAT.loadInstance mapping
#         memory_size=2,
#         init_indices=(0, 1),
#         out_json_path="hh_sat_results.json",
#     )
