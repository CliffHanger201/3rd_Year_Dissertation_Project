"""
##############################################################################
###                           Extension Task                               ###
##############################################################################
run_pretrained_extension.py
============================================
Pipeline: pre-train on instance 1, evaluate frozen Q-Table on instances 2/3/4.

Key design
----------
* Pre-training captures per-pass fitness traces, heuristic call counts,
  and Q-Table snapshots — stored in JSON under "_pretrain".
* Training Q-Tables are saved as pkl files in a known format so the
  visualiser can load them directly.
* Evaluation Q-Tables (frozen snapshots) are also saved as pkl files.
  Their Q-values are identical to the training pkl (the freeze is working),
  but saving them gives the visualiser a consistent interface for all labels.
* All metrics are stored in the JSON for offline analysis.

Pkl naming convention
---------------------
  Training : qtable_{domain}_train{train_id}_run{run_id}.pkl
             qtable_{domain}_train{train_id}_latest.pkl     (canonical, always latest)
  Eval     : qtable_{domain}_eval_{label}_run{run_id}.pkl   (e.g. label = "1.2")
             qtable_{domain}_eval_{label}_latest.pkl

"""

import copy
import json
import math
import numpy as np
import os
import pickle
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
# Dataclasses
# =============================================================================

@dataclass
class PreTrainRunResult:
    """
    Captures everything recorded during ONE offline pre-training pass.
    One of these is produced per pass per outer run, so for n_runs=10 and
    n_pretrain_runs=5 there will be 50 PreTrainRunResults per domain.
    """
    domain_name:           str
    seed:                  int
    run_id:                int                    # outer run index (1-based)
    pass_id:               int                    # pass within this run (1-based)
    train_instance_id:     int
    time_limit_ms:         int
    wall_time_s:           float
    best_value:            float
    fitness_trace:         Optional[List[float]]
    heuristic_call_counts: List[int]              # per-heuristic call count this pass
    qtable_state_count:    int                    # unique Q-Table states after this pass
    qtable_mean_q_values:  Optional[List[float]]  # per-heuristic mean Q after this pass
    qtable_max_q_values:   Optional[List[float]]
    qtable_min_q_values:   Optional[List[float]]


@dataclass
class DomainRunResult:
    """One evaluation episode on a test instance with a FROZEN Q-Table."""
    domain_name:             str
    seed:                    int
    train_instance_id:       int
    test_instance_id:        int
    qtable_label:            str                   # e.g. "1.2", "1.3", "1.4"
    time_limit_ms:           int
    memory_size:             int
    init_indices:            Tuple[int, ...]
    wall_time_s:             float
    best_value:              float
    best_solution_string:    Optional[str]
    heuristic_call_counts:   List[int]
    heuristic_call_times_ms: List[int]
    fitness_trace:           Optional[Any]
    pretrained:              bool  = True
    # --- Solution quality metrics ---
    initial_solution_value:      Optional[float] = None   # value before any heuristic runs
    improvement_ratio:           Optional[float] = None   # (init − best) / |init|
    # --- Frozen Q-Table snapshot (same values for every test instance) ---
    qtable_state_count:          Optional[int]   = None
    qtable_mean_q_values:        Optional[List[float]] = None
    qtable_max_q_values:         Optional[List[float]] = None
    qtable_min_q_values:         Optional[List[float]] = None
    # --- Behavioural metrics ---
    epsilon_final:               Optional[float] = None   # exploration rate at end of run


# =============================================================================
# Helpers
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

def _safe_get_fitness_trace(hh_obj) -> Optional[List[float]]:
    if hasattr(hh_obj, "getFitnessTrace"):
        try:
            result = hh_obj.getFitnessTrace()
            return list(result) if result is not None else None
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

def _qtable_snapshot(
    qtable,
) -> Tuple[Optional[int], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """
    Returns (state_count, mean_q, max_q, min_q) per heuristic.
    All four values are None if the Q-Table is empty or missing.
    """
    if qtable is None or not hasattr(qtable, "table") or not qtable.table:
        return None, None, None, None
    matrix = np.vstack(list(qtable.table.values()))  # (n_states, h_count)
    return (
        len(qtable.table),
        matrix.mean(axis=0).tolist(),
        matrix.max(axis=0).tolist(),
        matrix.min(axis=0).tolist(),
    )

def _resolve_h_count(qt) -> int:
    """Determine h_count from a QTable object."""
    if hasattr(qt, "h_count") and qt.h_count:
        return int(qt.h_count)
    if qt.table:
        return len(next(iter(qt.table.values())))
    return 0

def _save_qtable_pkl(
    qtable,
    path:               str,
    run_id:             int,
    seed:               int,
    train_instance_id:  int,
    pass_id:            int = 0,
) -> None:
    """
    Save a Q-Table to disk in the canonical format expected by the visualiser.

    Pkl schema
    ----------
    {
        "table"            : dict[state_key → np.ndarray shape (h_count,)],
        "h_count"          : int,
        "run_id"           : int,
        "seed"             : int,
        "train_instance_id": int,
        "pass_id"          : int,   # 0 = final accumulated table
    }
    """
    if qtable is None or not qtable.table:
        print(f"[SaveQTable] Skipped (empty) → {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "table":             dict(qtable.table),
                "h_count":           _resolve_h_count(qtable),
                "run_id":            run_id,
                "seed":              seed,
                "train_instance_id": train_instance_id,
                "pass_id":           pass_id,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[SaveQTable] {len(qtable.table)} states, h_count={_resolve_h_count(qtable)} → {path}")


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
# Pre-training
# =============================================================================

def pretrain_single_instance(
    domain_cls:         Type,
    domain_name:        str,
    train_instance_id:  int,
    seed:               int                = 42,
    memory_size:        int                = 2,
    init_indices:       Sequence[int]      = (0, 1),
    pretrain_time_ms:   int                = 30_000,
    n_pretrain_runs:    int                = 5,
    use_surrogate:      bool               = False,
    qtable_dir:         str                = "qtables/extension",
    config:             Optional[HHConfig] = None,
    run_id:             int                = 1,
) -> Tuple[PreTrainedHH, List[PreTrainRunResult]]:
    """
    Train PreTrainedHH on a single instance, capturing per-pass snapshots.

    Each of the n_pretrain_runs offline passes produces one PreTrainRunResult
    containing the fitness trace, heuristic call counts, and Q-Table snapshot
    for that pass.  After all passes the accumulated Q-Table is frozen and
    saved to disk.

    Returns
    -------
    (frozen_hh, pretrain_results)
        frozen_hh       : PreTrainedHH with accumulated Q-Table.
        pretrain_results: list of PreTrainRunResult, one per pass.
    """
    _probe = domain_cls(seed=seed)
    _probe.loadInstance(train_instance_id)
    h_count   = _probe.getNumberOfHeuristics()
    state_dim = 7  # must match HHState.as_vector() layout

    print(f"\n[PreTrain:{domain_name}] h_count={h_count}")
    print(f"[PreTrain:{domain_name}] Instance={train_instance_id}  "
          f"passes={n_pretrain_runs}  run_id={run_id}")

    def factory():
        return PreTrainedHH(config=config)

    trainer = PreTrainer(
        h_count=h_count,
        hh_factory=factory,
        state_dim=state_dim,
        use_surrogate=use_surrogate and HAS_KERAS,
    )

    pretrain_results: List[PreTrainRunResult] = []

    # One pass at a time so we can snapshot the Q-Table after each
    for pass_idx in range(n_pretrain_runs):
        pass_seed = seed + pass_idx * 100
        problem   = _init_problem(
            domain_cls, pass_seed, train_instance_id, memory_size, init_indices
        )
        print(f"[PreTrain:{domain_name}] Pass {pass_idx + 1}/{n_pretrain_runs}  "
              f"seed={pass_seed}")

        t0 = time.time()
        trainer.run_offline([problem], time_limit_ms=pretrain_time_ms)
        wall_time = time.time() - t0

        # Q-Table snapshot after this pass (accumulated so far)
        snap_hh = trainer.make_deployment_hh(config=config)
        sc, mq, xq, nq = _qtable_snapshot(snap_hh.qtable)

        ft = None

        try:
            if hasattr(problem, "getFitnessTrace"):
                raw = problem.getFitnessTrace()
                ft  = list(raw) if raw is not None else None
        except Exception:
            ft = None

        if ft is None:
            try:
                ft = list(problem.getFitnessTrace()) \
                    if hasattr(problem, "getFitnessTrace") else None
            except Exception:
                ft = None

        if ft is None:
            # Fitness trace — try HH first, fall back to problem domain
            ft = _safe_get_fitness_trace(snap_hh)

        best_val    = float(problem.getBestSolutionValue())
        call_counts = _safe_get_call_counts(problem)

        pretrain_results.append(PreTrainRunResult(
            domain_name=domain_name,
            seed=pass_seed,
            run_id=run_id,
            pass_id=pass_idx + 1,
            train_instance_id=train_instance_id,
            time_limit_ms=pretrain_time_ms,
            wall_time_s=wall_time,
            best_value=best_val,
            fitness_trace=ft,
            heuristic_call_counts=call_counts,
            qtable_state_count=sc or 0,
            qtable_mean_q_values=mq,
            qtable_max_q_values=xq,
            qtable_min_q_values=nq,
        ))

        print(f"[PreTrain:{domain_name}] Pass {pass_idx + 1} — "
              f"best={best_val:.4f}  states={sc or 0}  wall={wall_time:.2f}s")

    # ----- Final deployment HH with fully accumulated Q-Table -----
    hh = trainer.make_deployment_hh(config=config)

    # ----- Save per-run pkl -----
    per_run_path = os.path.join(
        qtable_dir,
        f"qtable_{domain_name}_train{train_instance_id}_run{run_id}.pkl",
    )
    _save_qtable_pkl(
        hh.qtable, per_run_path, run_id, seed, train_instance_id,
        pass_id=n_pretrain_runs,
    )

    # Overwrite canonical latest pkl so visualiser always has a file
    latest_path = os.path.join(
        qtable_dir,
        f"qtable_{domain_name}_train{train_instance_id}_latest.pkl",
    )
    _save_qtable_pkl(
        hh.qtable, latest_path, run_id, seed, train_instance_id,
        pass_id=n_pretrain_runs,
    )

    n_states = len(hh.qtable.table) if hh.qtable else 0
    print(f"[PreTrain:{domain_name}] Done — {n_states} Q-Table states.")
    return hh, pretrain_results


# =============================================================================
# Evaluation (frozen Q-Table)
# =============================================================================

def run_one_eval_frozen(
    domain_cls:        Type,
    domain_name:       str,
    frozen_hh:         PreTrainedHH,
    train_instance_id: int,
    test_instance_id:  int,
    seed:              int                = 42,
    time_limit_ms:     int                = 30000,
    memory_size:       int                = 2,
    init_indices:      Sequence[int]      = (0, 1),
    config:            Optional[HHConfig] = None,
    run_number:        int                = 1,
    qtable_dir:        str                = "qtables/extension",
) -> DomainRunResult:
    """
    One evaluation episode with a FROZEN Q-Table.

    Key design decisions
    --------------------
    * The Q-Table is deep-copied so the original is never mutated.
    * Online Q updates are suppressed — the policy seen by instances 2, 3, 4
      is exactly the policy learned on instance 1.
    * The frozen Q-Table snapshot is saved to disk so the visualiser can
      load it and confirm Q-values are identical across test instances.
    * tail_system is deep-copied so sliding-window state is independent
      across runs and instances.
    """
    problem = _init_problem(domain_cls, seed, test_instance_id, memory_size, init_indices)

    # Snapshot initial value BEFORE any heuristic runs
    try:
        initial_value = float(problem.getBestSolutionValue())
    except Exception:
        initial_value = None

    # Deep-copy the frozen Q-Table — this run can never modify the original
    frozen_qtable_copy = copy.deepcopy(frozen_hh.qtable)
    frozen_qtable_copy.update = lambda *args, **kwargs: None  # suppress updates
    frozen_qtable_copy.reset_action = lambda *args, **kwargs: None
    frozen_qtable_copy.reset_state  = lambda *args, **kwargs: None

    # Prevents Q-Table for inserting unseen states
    _frozen_table    = frozen_qtable_copy.table
    _frozen_h_count  = frozen_qtable_copy.h_count
    _zero_row        = np.zeros(_frozen_h_count, dtype=np.float32)

    def frozen_get_row(key: tuple) -> np.ndarray:
        if key in _frozen_table:
            return _frozen_table[key]
        return _zero_row.copy()   # unseen state — return zeros, do NOT insert

    frozen_qtable_copy._get_row = frozen_get_row

    hh = PreTrainedHH(
        seed=seed,
        config=config,
        qtable=frozen_qtable_copy,
        eps_start=0.05,
        eps_min=0.01,
        run_number=run_number,
    )
    hh.tail_system = copy.deepcopy(frozen_hh.tail_system)
    hh.setTimeLimit(time_limit_ms)
    hh.loadProblemDomain(problem)

    assert hh.qtable is not None,       "Q-Table wiped by loadProblemDomain!"
    assert len(hh.qtable.table) > 0,    "Q-Table empty after load!"

    t0 = time.time()
    hh.run()
    wall_time = time.time() - t0

    best_value  = problem.getBestSolutionValue()
    call_counts = _safe_get_call_counts(problem)

    improvement_ratio = (
        float((initial_value - best_value) / abs(initial_value))
        if initial_value is not None and initial_value != 0
        else None
    )

    sc, mq, xq, nq = _qtable_snapshot(hh.qtable)
    eps_final = float(hh.epsilon) if hasattr(hh, "epsilon") else None

    label = f"{train_instance_id}.{test_instance_id}"

    # ----- Save frozen Q-Table snapshot for this test instance -----
    # Q-values are identical to training (freeze confirmed), but saving
    # gives the visualiser a consistent per-label pkl to load and display.
    per_run_eval_path = os.path.join(
        qtable_dir,
        f"qtable_{domain_name}_eval_{label}_run{run_number}.pkl",
    )
    _save_qtable_pkl(hh.qtable, per_run_eval_path, run_number, seed, train_instance_id)

    latest_eval_path = os.path.join(
        qtable_dir,
        f"qtable_{domain_name}_eval_{label}_latest.pkl",
    )
    _save_qtable_pkl(hh.qtable, latest_eval_path, run_number, seed, train_instance_id)

    return DomainRunResult(
        domain_name=domain_name,
        seed=seed,
        train_instance_id=train_instance_id,
        test_instance_id=test_instance_id,
        qtable_label=label,
        time_limit_ms=time_limit_ms,
        memory_size=memory_size,
        init_indices=tuple(init_indices),
        wall_time_s=wall_time,
        best_value=best_value,
        best_solution_string=_safe_get_best_solution_string(problem),
        heuristic_call_counts=call_counts,
        heuristic_call_times_ms=_safe_get_call_times(problem),
        fitness_trace=_safe_get_fitness_trace(hh),
        pretrained=True,
        initial_solution_value=initial_value,
        improvement_ratio=improvement_ratio,
        qtable_state_count=sc,
        qtable_mean_q_values=mq,
        qtable_max_q_values=xq,
        qtable_min_q_values=nq,
        epsilon_final=eps_final,
    )


# =============================================================================
# Main loop
# =============================================================================

def run_single_instance_pretrain_all_domains(
    n_runs:            int                 = 30,
    base_seed:         int                 = 42,
    time_limit_ms:     int                 = 30000,
    pretrain_time_ms:  int                 = 30000,
    n_pretrain_runs:   int                 = 5,
    train_instance_id: int                 = 1,
    test_instance_ids: Optional[List[int]] = None,
    memory_size:       int                 = 2,
    init_indices:      Sequence[int]       = (0, 1),
    use_surrogate:     bool                = False,
    qtable_dir:        str                 = "qtables/extension",
    out_json_path:     str                 = "results/single_instance_pretrain_results.json",
    config:            Optional[HHConfig]  = None,
) -> Dict[str, List[DomainRunResult]]:
    """
    Full pipeline: pre-train once per domain per outer run on ONE instance,
    then evaluate n_runs times on each test instance using a FROZEN Q-Table.

    Per-run structure
    -----------------
    For each outer run (seed):
      Phase 1 — Pre-training:
        Train on train_instance_id for all domains.
        Captures per-pass fitness traces, call counts, and Q-Table snapshots.
        Saves training pkl files.
      Phase 2 — Evaluation:
        Apply frozen Q-Table to each test instance for all domains.
        Saves eval pkl files (identical Q-values to training — freeze check).

    Output files
    ------------
    JSON : out_json_path (all results + _pretrain block + _meta)
    Pkl  : qtable_dir/
              qtable_{domain}_train{id}_run{N}.pkl
              qtable_{domain}_train{id}_latest.pkl
              qtable_{domain}_eval_{label}_run{N}.pkl
              qtable_{domain}_eval_{label}_latest.pkl
    """
    if test_instance_ids is None:
        test_instance_ids = [2, 3, 4]

    overlap = set(test_instance_ids) & {train_instance_id}
    if overlap:
        raise ValueError(
            f"train_instance_id {train_instance_id} must not appear in "
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

    all_results:          DefaultDict[str, List[DomainRunResult]]    = defaultdict(list)
    all_pretrain_results: DefaultDict[str, List[PreTrainRunResult]]  = defaultdict(list)

    for run_id in range(n_runs):
        seed = base_seed + run_id
        print(f"\n{'=' * 65}")
        print(f"  Run {run_id + 1}/{n_runs}   seed={seed}")
        print(f"{'=' * 65}")

        # Phase 1: Pre-train fresh for this run
        frozen_hhs: Dict[str, PreTrainedHH] = {}

        for name, cls in domain_specs:
            frozen_hh, pretrain_results = pretrain_single_instance(
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
                run_id=run_id + 1,
                qtable_dir=qtable_dir,
            )
            frozen_hhs[name] = frozen_hh
            all_pretrain_results[name].extend(pretrain_results)

        # Phase 2: Evaluate frozen Q-Table on each test instance
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
                    qtable_dir=qtable_dir,
                )
                all_results[name].append(res)
                print(
                    f"  [{name}]  label={res.qtable_label}  "
                    f"best={res.best_value:.4f}  "
                    f"impr={res.improvement_ratio:.3f}  "
                    f"wall={res.wall_time_s:.2f}s"
                )

    # ----- Save JSON -----
    payload: Dict[str, Any] = {
        name: [asdict(r) for r in runs]
        for name, runs in all_results.items()
    }
    payload["_pretrain"] = {
        name: [asdict(r) for r in runs]
        for name, runs in all_pretrain_results.items()
    }
    payload["_meta"] = {
        "n_runs":            n_runs,
        "base_seed":         base_seed,
        "time_limit_ms":     time_limit_ms,
        "pretrain_time_ms":  pretrain_time_ms,
        "n_pretrain_runs":   n_pretrain_runs,
        "train_instance_id": train_instance_id,
        "test_instance_ids": test_instance_ids,
        "memory_size":       memory_size,
        "init_indices":      list(init_indices),
        "use_surrogate":     use_surrogate,
        "qtable_labels":     [f"{train_instance_id}.{t}" for t in test_instance_ids],
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
        time_limit_ms=120000,
        pretrain_time_ms=120000,
        n_pretrain_runs=1,
        train_instance_id=1,
        test_instance_ids=[2, 3, 4],
        memory_size=2,
        init_indices=(0, 1),
        use_surrogate=True,
        qtable_dir="qtables/extension",
        out_json_path="results/single_instance_pretrain_results.json",
    )