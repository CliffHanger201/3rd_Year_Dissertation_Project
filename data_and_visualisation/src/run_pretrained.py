"""
run_pretrained_hh.py
====================
Evaluation pipeline for PreTrainedHH across all HyFlex domains.

Pipeline
--------
For each domain:
  1. PRE-TRAIN → call pretrain_and_deploy() on instances 1-4 (training set)
  2. EVALUATE  → run the returned PreTrainedHH n_runs times on instance 0 (test set)
  3. SAVE      → write DomainRunResult records to JSON (same schema as original)

The Q-Table is built once per domain and shared across all 30 evaluation runs.
Each evaluation run continues to update the Q-Table online (§10) with a small ε,
so later runs benefit from accumulated experience — but the core warm-start comes
entirely from the pre-training phase on held-out training instances.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, DefaultDict

# ── Domain imports ─────────────────────────────────────────────────────────────
from python_hyper_heuristic.domains.Python.SAT.SAT import SAT
from python_hyper_heuristic.domains.Python.VRP.VRP import VRP
from python_hyper_heuristic.domains.Python.BinPacking.BinPacking import BinPacking
from python_hyper_heuristic.domains.Python.TSP.TSP import TSP

# ── Pre-training imports ───────────────────────────────────────────────────────
from pretrained_hyper_heuristic.src.pretrain_hyperheuristic import PreTrainedHH, HHConfig, pretrain_and_deploy, HAS_KERAS


# =============================================================================
# Result dataclass  (preserves original schema; one new field added)
# =============================================================================

@dataclass
class DomainRunResult:
    domain_name:             str
    seed:                    int
    instance_id:             int
    time_limit_ms:           int
    memory_size:             int
    init_indices:            Tuple[int, ...]
    wall_time_s:             float
    best_value:              float
    best_solution_string:    Optional[str]
    heuristic_call_counts:   List[int]
    heuristic_call_times_ms: List[int]
    fitness_trace:           Optional[Any]
    pretrained:              bool = True   # distinguishes PT runs in combined JSON


# =============================================================================
# Helpers  (unchanged from original)
# =============================================================================

def _safe_get_call_times(domain_obj) -> List[int]:
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
    for method in ("getSolutionToString", "bestSolutionToString"):
        if hasattr(problem, method):
            try:
                return getattr(problem, method)()
            except Exception:
                return None
    return None


def _init_problem(domain_cls: Type, seed: int, instance_id: int,
                  memory_size: int, init_indices: Sequence[int]):
    """Construct, load, and initialise a domain problem instance."""
    problem = domain_cls(seed=seed)
    problem.loadInstance(instance_id)
    problem.setMemorySize(memory_size)
    for idx in init_indices:
        problem.initialiseSolution(idx)
    return problem


# =============================================================================
# §1  Pre-train one domain
#     Runs pretrain_and_deploy() on train_instance_ids, returns a ready-to-use
#     PreTrainedHH whose Q-Table will be shared across all evaluation runs.
# =============================================================================

def pretrain_domain(
    domain_cls:         Type,
    domain_name:        str,
    train_instance_ids: List[int],
    seed:               int            = 42,
    memory_size:        int            = 2,
    init_indices:       Sequence[int]  = (0, 1),
    pretrain_time_ms:   int            = 30000,
    n_pretrain_runs:    int            = 5,
    use_surrogate:      bool           = False,
    qtable_save_path:   Optional[str]  = None,
    config:             Optional[HHConfig] = None,
    test_instance:      int            = 0,        # ← NEW: for guard check
) -> PreTrainedHH:
    """
    Build training problem instances and run offline collection n_pretrain_runs
    times, rebuilding problems with a fresh seed each run for genuine variation.

    Parameters
    ----------
    train_instance_ids : instance IDs for offline data collection.
                         Must NOT include the test instance (default: 0).
    pretrain_time_ms   : wall-clock budget (ms) per training instance.
    n_pretrain_runs    : how many times to run offline collection.
                         More runs = richer Q-Table coverage before deployment.
    test_instance      : held-out instance — raises an error if found in
                         train_instance_ids to prevent data leakage.

    Returns
    -------
    PreTrainedHH - warm-started and switched to online mode (small epsilon).
    """
    # Guard: test instance must never appear in training
    if test_instance in train_instance_ids:
        raise ValueError(
            f"Test instance {test_instance} must not appear in "
            f"train_instance_ids {train_instance_ids} — this would leak test data."
        )

    # Derive h_count directly from the domain so it can never mismatch
    _probe    = domain_cls(seed=seed)
    _probe.loadInstance(train_instance_ids[0])
    h_count   = _probe.getNumberOfHeuristics()
    state_dim = 4 * h_count + 3   # matches HHState.as_vector() layout

    print(f"\n[PreTrain:{domain_name}] h_count={h_count} (from domain)")
    print(f"[PreTrain:{domain_name}] Instances={train_instance_ids}, runs={n_pretrain_runs}")

    # Build PreTrainer once — Q-Table accumulates across ALL runs
    from pretrained_hyper_heuristic.src.pretrain_hyperheuristic import PreTrainer

    def _factory():
        return PreTrainedHH(config=config)

    trainer = PreTrainer(
        h_count=h_count,
        hh_factory=_factory,
        state_dim=state_dim,
        use_surrogate=use_surrogate and HAS_KERAS,
    )

    # ── Loop here: rebuild problems with a fresh seed each run ──────────────
    for run_idx in range(n_pretrain_runs):
        # Different seed each run = different starting conditions each time
        run_problems = [
            _init_problem(
                domain_cls,
                seed + run_idx * 100 + i,   # ← unique seed per run
                inst_id,
                memory_size,
                init_indices,
            )
            for i, inst_id in enumerate(train_instance_ids)
        ]
        print(f"[PreTrain:{domain_name}] Run {run_idx + 1}/{n_pretrain_runs} "
              f"(seeds {[seed + run_idx * 100 + i for i in range(len(train_instance_ids))]})")

        trainer.run_offline(run_problems, time_limit_ms=pretrain_time_ms)

    if qtable_save_path:
        trainer.save_qtable(qtable_save_path)

    hh = trainer.make_deployment_hh(config=config)
    n_states = len(hh.qtable.table) if hh.qtable else 0
    print(f"[PreTrain:{domain_name}] Done — {n_states} Q-Table state entries.")
    return hh


# =============================================================================
# §2  Single evaluation run
#     Reuses the pre-trained HH (and its shared Q-Table) on the test instance.
# =============================================================================

def run_one_eval(
    domain_cls:    Type,
    domain_name:   str,
    pretrained_hh: PreTrainedHH,
    seed:          int           = 42,
    time_limit_ms: int           = 30_000,
    instance_id:   int           = 0,
    memory_size:   int           = 2,
    init_indices:  Sequence[int] = (0, 1),
    config:        Optional[HHConfig] = None,
) -> DomainRunResult:
    """
    One evaluation episode.

    The shared Q-Table from pre-training is injected into a fresh PreTrainedHH
    shell so each run gets its own RNG seed while still benefiting from the
    warm-start.  The Q-Table is updated online (§10) — experience accumulates
    across all 30 runs, so later runs benefit from earlier ones.
    """
    problem = _init_problem(domain_cls, seed, instance_id, memory_size, init_indices)

    # eps_start=0.05 keeps exploration minimal during evaluation (§10)
    hh = PreTrainedHH(
        seed=seed,
        config=config,
        qtable=pretrained_hh.qtable,      # shared warm-started Q-Table
        eps_start=0.05,
        eps_min=0.01,
    )
    hh.tail_system = pretrained_hh.tail_system   # carry over tail memory

    hh.setTimeLimit(time_limit_ms)
    hh.loadProblemDomain(problem)

    assert hh.qtable is not None, "Q-Table was wiped by loadProblemDomain!"  # Verifying if Q-Table exists
    assert len(hh.qtable.table) > 0, "Q-Table is empty after load!"          # Verifying if Q-Table exists

    t0 = time.time()
    hh.run()
    wall_time = time.time() - t0

    return DomainRunResult(
        domain_name=domain_name,
        seed=seed,
        instance_id=instance_id,
        time_limit_ms=time_limit_ms,
        memory_size=memory_size,
        init_indices=tuple(init_indices),
        wall_time_s=wall_time,
        best_value=problem.getBestSolutionValue(),
        best_solution_string=_safe_get_best_solution_string(problem),
        heuristic_call_counts=_safe_get_call_counts(problem),
        heuristic_call_times_ms=_safe_get_call_times(problem),
        fitness_trace=_safe_get_fitness_trace(hh),
        pretrained=True,
    )


# =============================================================================
# Main loop  –  pre-train once per domain, evaluate n_runs times
# =============================================================================

def run_pretrained_all_domains(
    n_runs:             int                 = 30,
    seed:               int                 = 42,
    time_limit_ms:      int                 = 30000,
    pretrain_time_ms:   int                 = 30000,
    n_pretrain_runs:    int                 = 5,
    test_instance:      int                 = 0,
    train_instance_ids: Optional[List[int]] = None,
    memory_size:        int                 = 2,
    init_indices:       Sequence[int]       = (0, 1),
    use_surrogate:      bool                = False,
    qtable_dir:         str                 = "qtables",
    out_json_path:      str                 = "results/pretrained_hh_all_domains_results.json",
    config:             Optional[HHConfig]  = None,
) -> Dict[str, List[DomainRunResult]]:
    """
    Full pipeline: pre-train once per domain, then evaluate n_runs times.

    Parameters
    ----------
    test_instance      : instance ID held out for evaluation (never seen in pre-training).
    train_instance_ids : instance IDs for pre-training (default: [1, 2, 3, 4]).

    h_count is derived automatically per domain via getNumberOfHeuristics()
    so it can never mismatch between pre-training and evaluation.
    """
    if train_instance_ids is None:
        train_instance_ids = [1, 2, 3, 4]

    domain_specs = [
        ("SAT",        SAT),
        ("VRP",        VRP),
        ("TSP",        TSP),
        ("BinPacking", BinPacking),
    ]

    os.makedirs(qtable_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    all_results: DefaultDict[str, List[DomainRunResult]] = defaultdict(list)

    # ── Phase 1: Pre-train ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — PRE-TRAINING")
    print(f"  Training instances : {train_instance_ids}")
    print(f"  Test instance      : {test_instance}")
    print("=" * 60)

    pretrained_hhs: Dict[str, PreTrainedHH] = {}

    for name, cls in domain_specs:
        pretrained_hhs[name] = pretrain_domain(
            domain_cls=cls,
            domain_name=name,
            train_instance_ids=train_instance_ids,
            seed=seed,
            memory_size=memory_size,
            init_indices=init_indices,
            pretrain_time_ms=pretrain_time_ms,
            n_pretrain_runs=n_pretrain_runs,
            use_surrogate=use_surrogate,
            config=config,
            test_instance=test_instance,
            qtable_save_path=os.path.join(qtable_dir, f"qtable_{name}.pkl"),
        )

    # ── Phase 2: Evaluate ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — EVALUATION")
    print(f"  Runs : {n_runs}  |  Time limit : {time_limit_ms} ms  |  Instance : {test_instance}")
    print("=" * 60)

    for run_id in range(n_runs):
        seed = seed + run_id      # Revert back into base_seed later
        print(f"\n── Run {run_id + 1}/{n_runs}  (seed={seed}) ──")      # Revert back into base_seed later

        for name, cls in domain_specs:
            res = run_one_eval(
                domain_cls=cls,
                domain_name=name,
                pretrained_hh=pretrained_hhs[name],
                seed=seed,
                time_limit_ms=time_limit_ms,
                instance_id=test_instance,
                memory_size=memory_size,
                init_indices=init_indices,
                config=config,
            )
            all_results[name].append(res)
            print(f"  [{name:12s}]  best={res.best_value:.4f}  wall={res.wall_time_s:.2f}s")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        name: [asdict(r) for r in runs]
        for name, runs in all_results.items()
    }
    payload["_meta"] = {
        "n_runs":             n_runs,
        "base_seed":          seed,         # Revert back into base_seed later
        "time_limit_ms":      time_limit_ms,
        "pretrain_time_ms":   pretrain_time_ms,
        "n_pretrain_runs":    n_pretrain_runs,
        "test_instance":      test_instance,
        "train_instance_ids": train_instance_ids,
        "memory_size":        memory_size,
        "init_indices":       list(init_indices),
        "use_surrogate":      use_surrogate,
    }

    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved → {out_json_path}")
    return dict(all_results)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_pretrained_all_domains(
        n_runs=30,
        seed=42,
        time_limit_ms=30000,
        pretrain_time_ms=30000,          # budget per training instance during collection
        n_pretrain_runs=5,              # ← 5 runs × 4 instances = 20 total pre-training runs
        test_instance=0,                 # held-out, never seen during pre-training
        train_instance_ids=[1, 2, 3, 4],
        memory_size=2,
        init_indices=(0, 1),
        use_surrogate=False,             # set True if TensorFlow is installed
        qtable_dir="qtables",
        out_json_path="results/pretrained_hh_all_domains_results.json",
    )