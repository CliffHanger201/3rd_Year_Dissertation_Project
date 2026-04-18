"""

"""

import copy
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, DefaultDict

# ----- Domain Imports -----
from python_hyper_heuristic.domains.Python.SAT.SAT import SAT
from python_hyper_heuristic.domains.Python.VRP.VRP import VRP
from python_hyper_heuristic.domains.Python.TSP.TSP import TSP
from python_hyper_heuristic.domains.Python.BinPacking.BinPacking import BinPacking

# ----- Pre-training Imports -----
from pretrained_hyper_heuristic.src.pretrain_hyperheuristic import (
    PreTrainedHH,
    HHConfig,
    QTable,
    PreTrainer,
    TailSystem,
    EpsilonSchedule,
    HAS_KERAS,
)

# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class DomainRunResult:
    domain_name:             str
    seed:                    int
    train_instance_id:       int          # instance used for pre-training
    test_instance_id:        int          # instance used for this evaluation run
    qtable_label:            str          # e.g. "1.2", "1.3", "1.4"
    time_limit_ms:           int
    memory_size:             int
    init_indices:            Tuple[int, ...]
    wall_time_s:             float
    best_value:              float
    best_solution_string:    Optional[str]
    heuristic_call_counts:   List[int]
    heuristic_call_times_ms: List[int]
    fitness_trace:           Optional[Any]
    pretrained:              bool = True

# =============================================================================
# Helpers  (identical to original run_pretrained_hh.py)
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
    for method in ("getSolutionToString","bestSolutionToString"):
        if hasattr(problem, method):
            try:
                return getattr(problem, method)()
            except Exception:
                return None
    return None

def _init_problem(
    domain_cls:   Type,
    seed:         int,
    instance_id:  int,
    memory_size:  int,
    init_indices: Sequence[int],
):
    """Construct, load, and initialise a domain problem instance."""
    problem = domain_cls(seed=seed)
    problem.loadInstance(instance_id)
    problem.setMemorySize(memory_size)
    for idx in init_indices:
        problem.initialiseSolution(idx)
    return problem

# =============================================================================
# §1  Pre-train on ONE instance
#     Runs offline collection n_pretrain_runs times on a single training
#     instance, accumulating experience into one Q-Table that is then FROZEN.
# =============================================================================

def pretrain_single_instance(
    domain_cls:         Type,
    domain_name:        str,
    train_instance_id:  int,
    seed:               int            = 42,
    memory_size:        int            = 2,
    init_indices:       Sequence[int]  = (0, 1),
    pretrain_time_ms:   int            = 30_000,
    n_pretrain_runs:    int            = 5,
    use_surrogate:      bool           = False,
    qtable_save_path:   Optional[str]  = None,
    config:             Optional[HHConfig] = None,
) -> PreTrainedHH:
    """
    Train PreTrainedHH on a single domain instance.
 
    The Q-Table is accumulated over n_pretrain_runs offline collection passes,
    each using a different seed for genuine variation.  After training the
    returned PreTrainedHH is switched to online mode BUT evaluation callers
    must inject a *deep copy* of its Q-Table and disable online updates so
    that the frozen weights are never modified.
 
    Parameters
    ----------
    train_instance_id : the single instance used for offline data collection.
    n_pretrain_runs   : number of offline passes over the training instance.
 
    Returns
    -------
    PreTrainedHH whose Q-Table represents the frozen pre-trained policy.
    """
    # Probe the domain once to get h_count
    _probe = domain_cls(seed=seed)
    _probe.loadInstance(train_instance_id)
    h_count   = _probe.getNumberOfHeuristics()
    state_dim = 7 # must watch HHState.as_vector() layout

    print(f"\n[PreTrain:{domain_name}] h_count={h_count} (from domain)")
    print(f"[PreTrain:{domain_name}] Training on instances {train_instance_id}, "
          f"runs={n_pretrain_runs}")

    def factory():
        return PreTrainedHH(config=config)

    trainer = PreTrainer(
        h_count=h_count,
        hh_factory=factory,
        state_dim=state_dim,
        use_surrogate=use_surrogate and HAS_KERAS
    )

    # ----- Multiple offline passes - each run has a fresh seed for variation -----
    for run_idx in range(n_pretrain_runs):
        run_seed = seed + run_idx * 100
        problem = _init_problem(
            domain_cls, run_seed, train_instance_id, memory_size, init_indices
        )
        print(f"[PreTrain:{domain_name}] Offline pass {run_idx + 1}/{n_pretrain_runs} "
              f"(seed={run_seed}, instance={train_instance_id})")
        trainer.run_offline([problem], time_limit_ms=pretrain_time_ms)

    if qtable_save_path:
        trainer.save_qtable(qtable_save_path)

    hh = trainer.make_deployment_hh(config=config)
    n_states = len(hh.qtable.table) if hh.qtable else 0
    print(f"[PreTrain:{domain_name}] Done - {n_states} Q-Table state entries.")
    return hh

# =============================================================================
# §2  Single evaluation run with FROZEN Q-Table
#     A deep copy of the pre-trained Q-Table is injected; online Q updates are
#     disabled so the frozen weights are never modified.
#     This means results for instances 2, 3, and 4 are fully independent.
# =============================================================================

def run_one_eval_frozen(
        domain_cls:       Type,
    domain_name:      str,
    frozen_hh:        PreTrainedHH,    # the pre-trained HH (Q-Table never mutated)
    train_instance_id: int,
    test_instance_id: int,
    seed:             int            = 42,
    time_limit_ms:    int            = 30_000,
    memory_size:      int            = 2,
    init_indices:     Sequence[int]  = (0, 1),
    config:           Optional[HHConfig] = None,
    run_number:       int            = 1,
) -> DomainRunResult:
    """
    One evaluation episode with a FROZEN Q-Table.
 
    Key design decisions
    --------------------
    * The Q-Table is deep-copied from ``frozen_hh`` so the original is never
      mutated regardless of how many test instances or runs are evaluated.
    * ``qtable`` is set on the HH shell but online Q updates are suppressed by
      monkey-patching ``qtable.update`` to a no-op.  This guarantees the policy
      seen by instances 2, 3, and 4 is exactly the policy learned on instance 1.
    * ``tail_system`` is also deep-copied so per-run sliding-window state is
      fully independent across test instances.
    """
    problem = _init_problem(domain_cls, seed, test_instance_id, memory_size, init_indices)

    # --- Deep-copy the forzen Q-Table so this run can never modify it ---
    frozen_qtable_copy = copy.deepcopy(frozen_hh.qtable)

    # --- Suppress online updates
    frozen_qtable_copy.update = lambda *args, **kwargs: None # type: ignore[method-assign]

    hh = PreTrainedHH(
        seed=seed,
        config=config,
        qtable=frozen_qtable_copy,
        eps_start=0.05,
        eps_min=0.01,
        run_number=run_number
    )
    # Deep-copy tail system so histories don't bleed across runs/instances
    hh.tail_system = copy.deepcopy(frozen_hh.tail_system)

    hh.setTimeLimit(time_limit_ms)
    hh.loadProblemDomain(problem)

    assert hh.qtable is not None, "Q-Table was wiped by LoadProblemDomain!"
    assert len(hh.qtable.table) >0, "Q-Table is empty after load!"

    t0 = time.time()
    hh.run()
    wall_time = time.time() - t0

    qtable_label = f"{train_instance_id}.{test_instance_id}"

    return DomainRunResult(
        domain_name=domain_name,
        seed=seed,
        train_instance_id=train_instance_id,
        test_instance_id=test_instance_id,
        qtable_label=qtable_label,
        time_limit_ms=time_limit_ms,
        memory_size=memory_size,
        init_indices=tuple(init_indices),
        wall_time_s=wall_time,
        best_value=problem.getBestSolutionValue(),
        best_solution_string=_safe_get_best_solution_string(problem),
        heuristic_call_counts=_safe_get_call_counts(problem),
        heuristic_call_times_ms=_safe_get_call_times(problem),
        fitness_trace=_safe_get_fitness_trace(hh),
        pretrained=True
    )

# =============================================================================
# Main loop
# =============================================================================

def run_single_instance_pretrain_all_domains(
    n_runs:             int                 = 30,
    base_seed:          int                 = 42,
    time_limit_ms:      int                 = 30_000,
    pretrain_time_ms:   int                 = 30_000,
    n_pretrain_runs:    int                 = 5,
    train_instance_id:  int                 = 1,
    test_instance_ids:  Optional[List[int]] = None,
    memory_size:        int                 = 2,
    init_indices:       Sequence[int]       = (0, 1),
    use_surrogate:      bool                = False,
    qtable_dir:         str                 = "qtables/extension",
    out_json_path:      str                 = "results/single_instance_pretrain_results.json",
    config:             Optional[HHConfig]  = None,
) -> Dict[str, List[DomainRunResult]]:
    """
    Full pipeline: pre-train once per domain on ONE instance, then evaluate
    n_runs times on each of the test instances using a FROZEN Q-Table.
 
    Parameters
    ----------
    train_instance_id  : single instance used for pre-training (e.g. 1).
    test_instance_ids  : instances used for evaluation (default: [2, 3, 4]).
                         Must not overlap with train_instance_id.
 
    Q-Table label scheme
    --------------------
    Each result is labelled "<train>.<test>" (e.g. "1.2", "1.3", "1.4").
    The Q-Table trained on instance 1 is applied independently to instances
    2, 3, and 4 — results are never cross-contaminated.
 
    Output JSON schema
    ------------------
    {
      "<DomainName>": [
        { "qtable_label": "1.2", "best_value": ..., ... },
        { "qtable_label": "1.3", "best_value": ..., ... },
        ...
      ],
      "_meta": { ... }
    }
    """
    if test_instance_ids is None:
        test_instance_ids = [2, 3, 4]
    
    # Guard: training instance must not appear in test set
    overlap = set(test_instance_ids) & {train_instance_id}
    if overlap:
        raise ValueError(
            f"train_instance_id {train_instance_id} must not appear in"
            f"test_instance_ids {test_instance_ids}."
        )
    
    domain_specs: List[Tuple[str, Type]] = [
        ("SAT",        SAT),
        ("VRP",        VRP),
        ("TSP",        TSP),
        ("BinPacking", BinPacking),
    ]

    os.makedirs(qtable_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    all_results: DefaultDict[str, List[DomainRunResult]] = defaultdict(list)

    for run_id in range(n_runs):
        seed = base_seed + run_id
        print(f"\n── Run {run_id + 1}/{n_runs}  (seed={seed}) ──")

        # ----- Phase 1: Pre-train fresh for this run -----
        frozen_hhs = {}
        for name, cls in domain_specs:
            frozen_hhs[name] = pretrain_single_instance(
                domain_cls=cls,
                domain_name=name,
                train_instance_id=train_instance_id,
                seed=seed,
                memory_size=memory_size,
                init_indices=init_indices,
                pretrain_time_ms=pretrain_time_ms,
                n_pretrain_runs=n_pretrain_runs,
                use_surrogate=use_surrogate,
                config=config,
                qtable_save_path=os.path.join(
                    qtable_dir,
                    f"QTable{train_instance_id}_train_{name}_run{run_id + 1}.pkl"
                ),
            )

        # ----- Evaluate frozen Q-Table on each test instance -----
        for test_inst in test_instance_ids:
            for name, cls in domain_specs:
                res = run_one_eval_frozen(
                    domain_cls=cls,
                    domain_name=name,
                    frozen_hh=frozen_hhs[name],
                    train_instance_id=train_instance_id,
                    test_instance_id=test_inst,
                    seed=seed,
                    time_limit_ms=time_limit_ms,
                    memory_size=memory_size,
                    init_indices=init_indices,
                    config=config,
                    run_number=run_id + 1,
                )
                all_results[name].append(res)
                print(
                    f"[{name:12s}]"
                    f"qtable=QTable{train_instance_id}.{test_inst}_{run_id + 1}  "
                    f"best={res.best_value:.4f}  "
                    f"wall={res.wall_time_s:.2f}s"
                )

    # # ----- Phase 1: Pre-train once per domain -----
    # print("\n" + "=" * 70)
    # print("PHASE 1 - PRE-TRAINING (single instance per domain)")
    # print(f"  Train instance : {train_instance_id}")
    # print(f"  Test instances : {test_instance_ids}")
    # print(f"  Offline passes : {n_pretrain_runs}")
    # print("=" * 70)

    # frozen_hhs: Dict[str, PreTrainedHH] = {}

    # for name, cls in domain_specs:
    #     frozen_hhs[name] = pretrain_single_instance(
    #         domain_cls=cls,
    #         domain_name=name,
    #         train_instance_id=train_instance_id,
    #         seed=base_seed,
    #         memory_size=memory_size,
    #         init_indices=init_indices,
    #         pretrain_time_ms=pretrain_time_ms,
    #         n_pretrain_runs=n_pretrain_runs,
    #         use_surrogate=use_surrogate,
    #         config=config,
    #         qtable_save_path=os.path.join(
    #             qtable_dir, f"qtable_{name}_train{train_instance_id}.pkl"
    #         ),
    #     )

    # # ----- Phase 2: Evaluate - frozen Q-Table on each test instance -----
    # print("\n" + "=" * 70)
    # print("PHASE 2 - EVALUATION (frozen Q-Table, no online updates)")
    # print(f"  Runs per instance : {n_runs}")
    # print(f"  Time limit        : {time_limit_ms} ms")
    # print(f"  Q-Table labels  : "
    #       + ", ".join(f"{train_instance_id}.{t}" for t in test_instance_ids))
    # print("=" * 70)

    # for run_id in range(n_runs):
    #     seed = base_seed + run_id
    #     print(f"\n── Run {run_id + 1}/{n_runs}  (seed={seed}) ──")
 
    #     for test_inst in test_instance_ids:
    #         label = f"{train_instance_id}.{test_inst}"
    #         print(f"  Q-Table {label}")
 
    #         for name, cls in domain_specs:
    #             res = run_one_eval_frozen(
    #                 domain_cls=cls,
    #                 domain_name=name,
    #                 frozen_hh=frozen_hhs[name],
    #                 train_instance_id=train_instance_id,
    #                 test_instance_id=test_inst,
    #                 seed=seed,
    #                 time_limit_ms=time_limit_ms,
    #                 memory_size=memory_size,
    #                 init_indices=init_indices,
    #                 config=config,
    #                 run_number=run_id + 1,
    #             )
    #             all_results[name].append(res)
    #             print(
    #                 f"    [{name:12s}]  "
    #                 f"qtable={res.qtable_label}  "
    #                 f"best={res.best_value:.4f}  "
    #                 f"wall={res.wall_time_s:.2f}s"
    #             )
 
    # --- Save JSON ---
    payload: Dict[str, Any] = {
        name: [asdict(r) for r in runs]
        for name, runs in all_results.items()
    }
    payload["_meta"] = {
        "n_runs":             n_runs,
        "base_seed":          base_seed,
        "time_limit_ms":      time_limit_ms,
        "pretrain_time_ms":   pretrain_time_ms,
        "n_pretrain_runs":    n_pretrain_runs,
        "train_instance_id":  train_instance_id,
        "test_instance_ids":  test_instance_ids,
        "memory_size":        memory_size,
        "init_indices":       list(init_indices),
        "use_surrogate":      use_surrogate,
        "qtable_labels":      [
            f"{train_instance_id}.{t}" for t in test_instance_ids
        ],
    }
 
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)
 
    print(f"\nResults saved → {out_json_path}")
    return dict(all_results)
 
 
# =============================================================================
# Entry point
# =============================================================================
 
if __name__ == "__main__":
    run_single_instance_pretrain_all_domains(
        n_runs=10,
        base_seed=42,
        time_limit_ms=10000,
        pretrain_time_ms=10000,   # budget per offline pass on the training instance
        n_pretrain_runs=1,          # 1 passes × 1 instance = 1 total offline collections
        train_instance_id=1,        # instance used for pre-training
        test_instance_ids=[2, 3, 4],# Q-Table applied independently to each
        memory_size=2,
        init_indices=(0, 1),
        use_surrogate=True,         # set True if TensorFlow is installed
        qtable_dir="qtables/extension",
        out_json_path="results/single_instance_pretrain_results.json",
    )
