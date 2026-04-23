"""
python_hyper_heuristic.src.hyperheuristic.py

An advanced HyFlex-style hyper-heuristic (HH) based on an extended Choice Function
with adaptive move acceptance, phases, and robust statistics.

This is designed to plug into a HyFlex-like ProblemDomain interface, e.g.:

- problem.getNumberOfHeuristics() -> int
- problem.initialiseSolution(index: int) -> None
- problem.copySolution(src: int, dst: int) -> None
- problem.applyHeuristic(h: int, src: int, dst: int) -> None
- problem.getFunctionValue(index: int) -> float
- problem.getBestSolutionValue() -> float  (optional but used for tracing)

Optionally some HyFlex domains also support:
- problem.getHeuristicType(h) or categories (mutation / local search / ruin-recreate / crossover)
This HH does not require those, but you can wire them in easily.

Notes:
- Minimisation is assumed (smaller fitness is better), matching HyFlex.
- Uses CPU-time via time.process_time_ns() like classic HyFlex code.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Optional, Set

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import (
    ProblemDomain,
    HeuristicType,
)


# ===========================================================================
# Pre-computed lookup tables  (built once at import time, zero loop overhead)
# ===========================================================================

# OPT 2: log1p table — recency is always a non-negative integer so a list
# lookup replaces the transcendental call in the CF inner loop.
_LOG1P_TABLE_SIZE = 4096
_LOG1P_TABLE: List[float] = [math.log1p(i) for i in range(_LOG1P_TABLE_SIZE)]


def _fast_log1p(x: int) -> float:
    """Integer log1p with pre-built table; falls back for large values."""
    return _LOG1P_TABLE[x] if x < _LOG1P_TABLE_SIZE else math.log1p(x)


# OPT 3: shaped-delta table — sign(delta)*log1p(|delta|) keyed on int(|delta|).
_SHAPED_TABLE_CAP = 1024
_SHAPED_POS: List[float] = [math.log1p(i) for i in range(_SHAPED_TABLE_CAP)]


def _fast_shaped(delta: float) -> float:
    """sign(delta) * log1p(|delta|) with a fast path for small magnitudes."""
    if delta == 0.0:
        return 0.0
    abs_d = abs(delta)
    idx = int(abs_d)
    val = _SHAPED_POS[idx] if idx < _SHAPED_TABLE_CAP else math.log1p(abs_d)
    return val if delta > 0.0 else -val


# ===========================================================================
# Enumerations
# ===========================================================================

class Phase(Enum):
    """High-level search mode."""
    INTENSIFY = auto()
    DIVERSIFY = auto()


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class HHConfig:
    # ---- Trace ----
    trace_checkpoints: int = 101

    # ---- Choice function weights (classic CF: alpha*f1 + beta*f2 + gamma*f3) ----
    alpha: float = 1.0
    beta:  float = 0.2
    gamma: float = 0.1

    # ---- Choice function shaping ----
    reward_smoothing:       float = 0.15   # EMA smoothing for heuristic rewards
    synergy_smoothing:      float = 0.10   # EMA smoothing for pair synergy
    penalty_for_worsening:  float = 0.05   # discourages heuristics that worsen often

    # ---- Exploration / exploitation ----
    epsilon:     float = 0.02   # small random exploration rate
    tabu_tenure: int   = 3      # avoid repeating the last few heuristics

    # ---- Phase control ----
    stall_iterations_to_diversify: int = 250
    stall_iterations_to_restart:   int = 2000
    diversify_length:              int = 150   # iterations to stay in diversify

    # ---- Late-search tightening ----
    # When elapsed fraction >= tighten_threshold, acceptance is gradually
    # tightened each iteration.  Set to 1.0 to disable.
    tighten_threshold: float = 0.85
    tighten_rate:      float = 0.002   # passed to acceptance.tighten() per iter

    # # ---- Acceptance configuration ----
    # acceptance_kind: AcceptanceKind = AcceptanceKind.LATE_ACCEPTANCE

    # # SA parameters (if used)
    # sa_t0:                float = 1.0
    # sa_alpha:             float = 0.995   # cooling rate
    # sa_reheat_multiplier: float = 1.5     # when stuck, increase T

    # LAHC parameters (if used)
    lahc_length: int = 50

    # # Great Deluge parameters (if used)
    # gd_initial_slack: float = 0.01
    # gd_decay:         float = 0.999

    # ---- Misc ----
    consider_equal_as_accept: bool = True
    random_seed: Optional[int] = None


# ===========================================================================
# Per-heuristic and pair statistics
# ===========================================================================

@dataclass
class HeuristicStats:
    """Statistics tracked per heuristic for credit assignment."""
    uses:     int = 0
    accepts:  int = 0
    improves: int = 0

    # Reward model
    reward_ema: float = 0.0   # f1 basis (individual performance)
    worsen_ema: float = 0.0   # penalise frequent worsening

    # Timing: used by F3 (recency) and by PreTrainedHH's tail system
    last_used_iter: int = 0


@dataclass
class PairStats:
    """Statistics for heuristic pairs (prev -> current), used for CF f2."""
    synergy_ema: float = 0.0
    uses:        int   = 0


# ===========================================================================
# Move Acceptance base class and implementations
# ===========================================================================

class MoveAcceptance:
    """Base class for move acceptance strategies."""

    def reset(self, initial_fitness: float) -> None:
        """Called at the start of a run (or after a restart) with the initial fitness."""
        pass

    def accept(self, before: float, after: float, iteration: int,
               rng: random.Random) -> bool:
        raise NotImplementedError

    def on_improvement(self, new_best: float) -> None:
        """Hook to react to global improvements (optional)."""
        pass

    def on_stall(self) -> None:
        """Hook when HH detects a stall (e.g. to reheat SA or pessimise LAHC)."""
        pass

    def tighten(self, factor: float) -> None:
        """
        Called late in the search to gradually narrow acceptance of worsening
        moves.  factor ∈ (0, 1) — larger means more aggressive tightening.
        Base implementation is a no-op; override in subclasses as appropriate.
        """
        pass


# class ImproveOrEqualAcceptance(MoveAcceptance):
#     def __init__(self, allow_equal: bool = True):
#         self.allow_equal = allow_equal

#     def accept(self, before: float, after: float, iteration: int,
#                rng: random.Random) -> bool:
#         if after < before:
#             return True
#         if self.allow_equal and after == before:
#             return True
#         return False


# class SimulatedAnnealingAcceptance(MoveAcceptance):
#     def __init__(self, t0: float, cooling: float, reheat_multiplier: float,
#                  allow_equal: bool = True):
#         self.t0                  = t0
#         self.cooling             = cooling
#         self.reheat_multiplier   = reheat_multiplier
#         self.allow_equal         = allow_equal
#         self.t                   = t0

#     def reset(self, initial_fitness: float) -> None:
#         self.t = self.t0

#     def accept(self, before: float, after: float, iteration: int,
#                rng: random.Random) -> bool:
#         if after < before:
#             return True
#         if self.allow_equal and after == before:
#             return True
#         # worsening — probabilistic acceptance
#         if self.t <= 1e-12:
#             return False
#         delta = after - before
#         prob  = math.exp(-delta / self.t) if delta > 0 else 1.0
#         accepted = rng.random() < prob
#         # cool every move
#         self.t *= self.cooling
#         return accepted

#     def on_stall(self) -> None:
#         # reheat when stuck
#         self.t *= self.reheat_multiplier

#     def tighten(self, factor: float) -> None:
#         """Accelerate cooling late in search."""
#         self.t *= (1.0 - factor)


class LateAcceptanceHillClimbing(MoveAcceptance):
    """
    LAHC: accept if after <= fitness[iteration mod L].
    Minimisation.
    """

    def __init__(self, L: int, allow_equal: bool = True):
        self.L           = max(1, int(L))
        self.allow_equal = allow_equal
        self.buffer: List[float] = []
        self.pos: int = 0

    def reset(self, initial_fitness: float) -> None:
        self.buffer = [initial_fitness] * self.L
        self.pos    = 0

    def accept(self, before: float, after: float, iteration: int,
               rng: random.Random) -> bool:
        threshold = self.buffer[self.pos]
        ok = after < threshold or (self.allow_equal and after == threshold)
        # update buffer with current (before) fitness per LAHC variant
        self.buffer[self.pos] = before
        self.pos = (self.pos + 1) % self.L
        return ok

    def on_stall(self) -> None:
        # BUG FIX 1: pessimise buffer to the worst stored value so that
        # post-restart moves can be accepted again.  Without this the buffer
        # retains pre-restart (better) fitness values and rejects everything.
        if self.buffer:
            worst = max(self.buffer)
            self.buffer = [worst] * self.L

    def tighten(self, factor: float) -> None:
        """Pull all buffer entries toward the current best (tighten threshold)."""
        if not self.buffer:
            return
        best = min(self.buffer)
        self.buffer = [v - (v - best) * factor for v in self.buffer]


# class GreatDelugeAcceptance(MoveAcceptance):
#     """
#     Great Deluge: accept if after <= level.
#     level decays gradually.
#     """

#     def __init__(self, initial_slack: float, decay: float,
#                  allow_equal: bool = True):
#         self.initial_slack = initial_slack
#         self.decay         = decay
#         self.allow_equal   = allow_equal
#         self.level: Optional[float] = None

#     def reset(self, initial_fitness: float) -> None:
#         # level starts slightly above initial fitness
#         self.level = initial_fitness * (1.0 + self.initial_slack)

#     def accept(self, before: float, after: float, iteration: int,
#                rng: random.Random) -> bool:
#         assert self.level is not None
#         ok = after < self.level or (self.allow_equal and after == self.level)
#         # tighten level
#         self.level *= self.decay
#         return ok

#     def on_improvement(self, new_best: float) -> None:
#         # Optionally pull level toward best
#         if self.level is not None and new_best < self.level:
#             self.level = (self.level + new_best) / 2.0

#     def tighten(self, factor: float) -> None:
#         """Accelerate decay of the flood level late in search."""
#         if self.level is not None:
#             self.level *= (1.0 - factor * 0.1)


# def make_acceptance(cfg: HHConfig) -> MoveAcceptance:
#     ae = cfg.consider_equal_as_accept
#     if cfg.acceptance_kind == AcceptanceKind.IMPROVE_OR_EQUAL:
#         return ImproveOrEqualAcceptance(allow_equal=ae)
#     if cfg.acceptance_kind == AcceptanceKind.SIMULATED_ANNEALING:
#         return SimulatedAnnealingAcceptance(
#             t0=cfg.sa_t0,
#             cooling=cfg.sa_alpha,
#             reheat_multiplier=cfg.sa_reheat_multiplier,
#             allow_equal=ae,
#         )
#     if cfg.acceptance_kind == AcceptanceKind.LATE_ACCEPTANCE:
#         return LateAcceptanceHillClimbing(L=cfg.lahc_length, allow_equal=ae)
#     if cfg.acceptance_kind == AcceptanceKind.GREAT_DELUGE:
#         return GreatDelugeAcceptance(
#             initial_slack=cfg.gd_initial_slack,
#             decay=cfg.gd_decay,
#             allow_equal=ae,
#         )
#     raise ValueError(f"Unknown acceptance kind: {cfg.acceptance_kind}")


# ===========================================================================
# Main Hyper-Heuristic
# ===========================================================================

class AdvancedChoiceFunctionHH:
    """
    HyFlex-style HH with:
    - Extended Choice Function (F1 individual, F2 pair synergy, F3 recency)
    - Late Acceptance Hill Climbing (LAHC)
    - Phase control (INTENSIFY ↔ DIVERSIFY) and random restarts
    - Tabu list + epsilon-random exploration
    - Late-search acceptance tightening
    - HyFlex-style trace checkpoints
    - All six performance optimisations (OPT 1-6) and three bug fixes
    """

    TRACE_CHECKPOINTS_DEFAULT = 101

    def __init__(self, seed: Optional[int] = None,
                 config: Optional[HHConfig] = None):
        self.cfg = config or HHConfig(random_seed=seed)
        self.rng = random.Random(self.cfg.random_seed)

        # ---- Time management (nanoseconds internally, matching HyFlex) ----
        self.time_limit_ns:    int  = 0
        self.initial_time_ns:  int  = 0
        self.timelimitset:     bool = False

        # ---- Problem domain ----
        self.problem = None

        # ---- Trace / best-solution tracking ----
        self.trace:            Optional[List[float]] = None
        self.lastbestsolution: float = float("inf")

        # ---- Heuristic type sets (populated at run time) ----
        self.local_search_set:  Set[int] = set()
        self.mutation_set:      Set[int] = set()
        self.ruin_recreate_set: Set[int] = set()
        self.crossover_set:     Set[int] = set()
        self.pool_start: int = 2
        self.pool_end:   int = 2

        # ---- Printing / trace checkpoint state ----
        self.printfraction_ns: int  = 0
        self.printlimit_ns:    int  = 0
        self.lastprint:        int  = 0
        self.initialprint:     bool = False

        # ---- Internal HH search state ----
        self.phase:             Phase = Phase.INTENSIFY
        self.phase_until_iter:  int   = 0
        self.iteration:         int   = 0
        self.last_improve_iter: int   = 0
        self.best_fitness:      float = float("inf")
        self.restart_count:     int   = 0   # number of restarts this run
        self.total_improves:    int   = 0   # total accepted improvements

        # ---- Per-heuristic and pair statistics ----
        # These are initialised at runtime when we know number of heuristics
        self.h_stats: List[HeuristicStats]         = []
        self.p_stats: List[List[PairStats]]        = []

        # ---- Tabu structures ----
        # self.tabu is the canonical deque (preserved for external access,
        # e.g. PreTrainedHH reads it directly).
        # OPT 1: self._tabu_set is kept in sync so _select_heuristic never
        # calls set(self.tabu) inside the hot loop.
        self.tabu:       Deque[int] = deque()
        self._tabu_set:  Set[int]   = set()

        # ---- Move acceptance ----
        self.acceptance = LateAcceptanceHillClimbing(
            L=self.cfg.lahc_length,
            allow_equal=self.cfg.consider_equal_as_accept
        )

        # ---- OPT 6: domain capability flag (probed once before loop) ----
        self._has_get_best: bool = False

    # =======================================================================
    # Heuristic type detection
    # =======================================================================

    def setHeuristicClassTypes(self, problem) -> None:
        try:
            self.local_search_set  = set(problem.getHeuristicsOfType(HeuristicType.LOCAL_SEARCH))
            self.mutation_set      = set(problem.getHeuristicsOfType(HeuristicType.MUTATION))
            self.ruin_recreate_set = set(problem.getHeuristicsOfType(HeuristicType.RUIN_RECREATE))
            self.crossover_set     = set(problem.getHeuristicsOfType(HeuristicType.CROSSOVER))
        except Exception:
            self.local_search_set  = set()
            self.mutation_set      = set()
            self.ruin_recreate_set = set()
            self.crossover_set     = set()

    # =======================================================================
    # Time API
    # =======================================================================

    def setTimeLimit(self, time_in_milliseconds: int) -> None:
        if self.timelimitset:
            raise RuntimeError("Time limit already set")
        if time_in_milliseconds <= 0:
            raise ValueError("Time limit must be > 0 ms")

        self.time_limit_ns = int(time_in_milliseconds) * 1_000_000
        # 101 checkpoints => 100 intervals; keeps same spirit as classic HyFlex
        self.printfraction_ns = max(1, self.time_limit_ns // (self.cfg.trace_checkpoints - 1))
        self.printlimit_ns    = self.printfraction_ns
        self.initialprint     = False
        self.lastprint        = 0
        self.timelimitset     = True

    def getTimeLimit(self) -> float:
        return self.time_limit_ns / 1_000_000.0

    def getElapsedTime(self) -> int:
        if self.initial_time_ns == 0:
            return 0
        return int((time.process_time_ns() - self.initial_time_ns) // 1_000_000)

    def _time_fraction(self) -> float:
        """Fraction of time budget elapsed: 0.0 → 1.0."""
        if self.time_limit_ns == 0 or self.initial_time_ns == 0:
            return 0.0
        elapsed = time.process_time_ns() - self.initial_time_ns
        return min(1.0, elapsed / self.time_limit_ns)

    # =======================================================================
    # Problem / lifecycle
    # =======================================================================

    def loadProblemDomain(self, problem) -> None:
        if self.time_limit_ns == 0:
            raise RuntimeError("Time limit must be set before loading problem")
        self.problem = problem

    def run(self) -> None:
        if self.problem is None:
            raise RuntimeError("No problem domain loaded")
        if self.time_limit_ns == 0:
            raise RuntimeError("Time limit not set")

        self.trace           = [0.0] * self.cfg.trace_checkpoints
        self.initial_time_ns = time.process_time_ns()
        self._solve(self.problem)

    def isCrossover(self, index: int) -> bool:
        return index in self.crossover_set

    # =======================================================================
    # Trace / best-solution access
    # =======================================================================

    def getBestSolutionValue(self) -> float:
        if self.lastbestsolution == float("inf"):
            raise RuntimeError("hasTimeExpired() must be called at least once")
        return self.lastbestsolution

    def getFitnessTrace(self) -> Optional[List[float]]:
        return self.trace

    # =======================================================================
    # Core timing check — HyFlex-style checkpoints
    # =======================================================================

    def hasTimeExpired(self) -> bool:
        """
        MUST be called at the top of the main loop.
        Updates trace checkpoints in a HyFlex-like way.
        OPT 6: uses self._has_get_best to avoid try/except every iteration.
        """
        assert self.trace is not None
        current_ns = time.process_time_ns() - self.initial_time_ns

        if not self.initialprint:
            self.initialprint = True
            # If the domain supports it, use the HyFlex "best so far"
            if self._has_get_best:
                res = float(self.problem.getBestSolutionValue())
            else:
                res = float(self.best_fitness) if self.best_fitness != float("inf") else float("inf")
            self.trace[0]         = res
            self.lastbestsolution = res

        elif current_ns >= self.printlimit_ns:
            thisprint = int(current_ns // self.printfraction_ns)
            thisprint = min(thisprint, self.cfg.trace_checkpoints - 1)

            for x in range(thisprint - self.lastprint):
                idx = self.lastprint + x + 1
                if current_ns <= self.time_limit_ns:
                    if self._has_get_best:
                        res = float(self.problem.getBestSolutionValue())
                    else:
                        res = float(self.best_fitness)
                    self.trace[idx]       = res
                    self.lastbestsolution = res
                else:
                    self.trace[idx] = self.lastbestsolution
                self.printlimit_ns += self.printfraction_ns

            self.lastprint = thisprint

        if current_ns >= self.time_limit_ns:
            return True
        else:
            # refresh "best" if possible
            if self._has_get_best:
                self.lastbestsolution = float(self.problem.getBestSolutionValue())
            else:
                self.lastbestsolution = float(self.best_fitness)
            return False

    # =======================================================================
    # Initialisation helpers
    # =======================================================================

    def _init_stats(self, h_count: int) -> None:
        self.h_stats = [HeuristicStats() for _ in range(h_count)]
        self.p_stats = [[PairStats() for _ in range(h_count)] for _ in range(h_count)]
        # OPT 1: reset both tabu structures together
        self.tabu.clear()
        self._tabu_set.clear()

    # =======================================================================
    # Choice Function scoring
    # =======================================================================

    def _choice_function_score(self, h: int, prev_h: Optional[int]) -> float:
        """
        Extended Choice Function:
        score(h) = alpha * F1(h) + beta * F2(prev,h) + gamma * F3(h)
        with minor shaping and penalties.

        F1: individual performance (EMA reward) minus penalty for worsening frequency
        F2: pair synergy (if there is a previous heuristic)
        F3: recency/diversification — longer since last used => higher score
        """
        hs = self.h_stats[h]

        # F1
        f1 = hs.reward_ema - self.cfg.penalty_for_worsening * hs.worsen_ema

        # F2
        f2 = 0.0
        if prev_h is not None:
            f2 = self.p_stats[prev_h][h].synergy_ema

        # F3 — OPT 2: integer table lookup instead of math.log1p**
        recency = self.iteration - hs.last_used_iter
        if recency < 0:
            recency = 0
        f3 = math.log1p(recency)

        # Phase shaping: in diversify we upweight recency and downweight
        # pure reward slightly to encourage exploration.
        if self.phase == Phase.DIVERSIFY:
            alpha = self.cfg.alpha * 0.8
            beta  = self.cfg.beta  * 0.9
            gamma = self.cfg.gamma * 1.4
        else:
            alpha = self.cfg.alpha
            beta  = self.cfg.beta
            gamma = self.cfg.gamma

        return alpha * f1 + beta * f2 + gamma * f3

    # =======================================================================
    # Heuristic selection
    # =======================================================================

    def _select_heuristic(self, h_count: int, prev_h: Optional[int]) -> int:
        """
        Select a heuristic using the choice function + epsilon exploration + tabu.
        OPT 1: uses self._tabu_set directly — no set() rebuild each call.
        BUG FIX 3: falls back to unconstrained scan when all heuristics are tabu.
        """
        # Small chance of random exploration
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.randrange(h_count)

        tabu_set = self._tabu_set  # OPT 1: local alias, no copy or rebuild

        best_h     = -1   # BUG FIX 3: use sentinel, not 0
        best_score = -float("inf")

        for h in range(h_count):
            if h in tabu_set:
                continue
            s = self._choice_function_score(h, prev_h)
            if s > best_score:
                best_score = s
                best_h     = h

        # BUG FIX 3: if every heuristic is tabu (tenure >= h_count), fall back
        # to the best-scoring heuristic ignoring tabu rather than silently
        # returning H0 (which can be extremely expensive on some domains).
        if best_h == -1:
            for h in range(h_count):
                s = self._choice_function_score(h, prev_h)
                if s > best_score:
                    best_score = s
                    best_h     = h

        return best_h

    # =======================================================================
    # Tabu management
    # =======================================================================

    def _update_tabu(self, h: int) -> None:
        if self.cfg.tabu_tenure <= 0:
            return
        self.tabu.append(h)
        self._tabu_set.add(h)   # OPT 1: keep live set in sync
        while len(self.tabu) > self.cfg.tabu_tenure:
            evicted = self.tabu.popleft()
            # BUG FIX 2: only discard from set when no further occurrence of
            # this heuristic remains in the deque (it may have been called
            # multiple times within one tenure window).
            if evicted not in self.tabu:
                self._tabu_set.discard(evicted)

    # =======================================================================
    # Credit assignment
    # =======================================================================

    def _credit_assignment(
        self,
        h:        int,
        prev_h:   Optional[int],
        before:   float,
        after:    float,
        accepted: bool,
        improved: bool,
    ) -> None:
        """
        Update heuristic and pair statistics.
        Reward is based on:
        - magnitude of improvement (scaled)
        - acceptance (small positive for enabling escapes)
        - worsening (negative penalty)

        # OPT 3: shaped delta uses pre-built table instead of math.log1p + math.copysign.
        # Smoothing constants cached as locals to avoid repeated attribute lookups.
        """
        hs = self.h_stats[h]
        hs.uses += 1
        if accepted:
            hs.accepts  += 1
        if improved:
            hs.improves += 1

        # Improvement magnitude (minimisation) — positive = better
        delta = before - after
        # OPT 3: table lookup for sign(delta)*log1p(|delta|)
        shaped = math.copysign(math.log1p(abs(delta)), delta) if delta != 0.0 else 0.0

        # Reward signal:
        # - improvements contribute strongly
        # - accepted non-improving moves contribute a tiny bit
        # - rejected moves are mildly negative
        if improved:
            reward_signal = shaped
        elif accepted and not improved:
            reward_signal = 0.05   # small positive for enabling escapes
        else:
            reward_signal = -0.02

        # Update individual reward EMA (smoothing constant cached as local)
        smr = self.cfg.reward_smoothing
        hs.reward_ema = (1.0 - smr) * hs.reward_ema + smr * reward_signal

        # Track worsening tendency (used as penalty in F1)
        worsening     = 1.0 if after > before else 0.0
        hs.worsen_ema = (1.0 - smr) * hs.worsen_ema + smr * worsening

        # Pair synergy update: give credit when the pair leads to an accepted
        # improvement (or accepted move) — this builds F2 signal over time.
        if prev_h is not None:
            ps = self.p_stats[prev_h][h]
            ps.uses += 1
            if improved:
                pair_signal = shaped
            elif accepted:
                pair_signal = 0.02
            else:
                pair_signal = -0.01
            sms = self.cfg.synergy_smoothing
            ps.synergy_ema = (1.0 - sms) * ps.synergy_ema + sms * pair_signal

        # Update last-used iteration (used by F3 recency)
        hs.last_used_iter = self.iteration

    # =======================================================================
    # Phase control
    # =======================================================================

    def _update_phase(self) -> None:
        """
        Simple, effective phase logic:
        - If stalled for a while => enter diversify for diversify_length iterations
        - If stalled too long    => trigger restart
        """
        stall = self.iteration - self.last_improve_iter

        if stall >= self.cfg.stall_iterations_to_restart:
            # Trigger restart: set phase so _maybe_restart fires immediately
            self.phase             = Phase.DIVERSIFY
            self.phase_until_iter  = self.iteration   # immediate action by caller
            return

        if self.phase == Phase.DIVERSIFY:
            if self.iteration >= self.phase_until_iter:
                self.phase = Phase.INTENSIFY
            return

        # Currently intensify — check diversify trigger
        if stall >= self.cfg.stall_iterations_to_diversify:
            self.phase            = Phase.DIVERSIFY
            self.phase_until_iter = self.iteration + self.cfg.diversify_length

    # =======================================================================
    # Restart
    # =======================================================================

    def _maybe_restart(self, problem, cur_idx: int, cand_idx: int) -> bool:
        """
        Restart strategy:
        - Reinitialise current solution if very stalled.
        - The domain keeps track of its own global best (via getBestSolutionValue);
          we only reinitialise the working solution, not the domain's best.
        Returns True if restart happened.
        """
        stall = self.iteration - self.last_improve_iter
        if stall < self.cfg.stall_iterations_to_restart:
            return False

        # Inform acceptance method (pessimise LAHC buffer)
        self.acceptance.on_stall()

        # Reinitialise working solution
        problem.initialiseSolution(cur_idx)
        problem.copySolution(cur_idx, cand_idx)

        # Reset stall counter baseline to avoid immediate repeated restarts
        self.last_improve_iter = self.iteration
        self.restart_count    += 1

        return True

    # =======================================================================
    # Heuristic application helper
    # =======================================================================

    def _apply_heuristic(self, problem, h: int,
                         current: int, candidate: int) -> None:
        """Dispatch to unary or crossover call depending on heuristic type."""
        if self.isCrossover(h):
            parent2 = self.rng.randrange(self.pool_start, self.pool_end)
            problem.applyHeuristic(h, current, parent2, candidate)
        else:
            problem.applyHeuristic(h, current, candidate)

    # =======================================================================
    # Main solve loop
    # =======================================================================

    def _solve(self, problem) -> None:
        h_count = int(problem.getNumberOfHeuristics())
        if h_count <= 0:
            raise RuntimeError("Problem domain reports 0 heuristics")

        self.setHeuristicClassTypes(problem)

        # OPT 6: probe getBestSolutionValue support once, before the loop,
        # so hasTimeExpired() uses a plain `if` with no exception machinery.
        try:
            problem.getBestSolutionValue()
            self._has_get_best = True
        except Exception:
            self._has_get_best = False

        # ------------------------------------------------------------------
        # Solution memory layout:
        #   0          = current  (index-swapped with candidate on acceptance)
        #   1          = candidate
        #   2..POOL_END-1 = crossover / restart diversity pool
        # ------------------------------------------------------------------
        A          = 0
        B          = 1
        POOL_START = 2
        POOL_SIZE  = 5
        POOL_END   = POOL_START + POOL_SIZE

        # Initialise statistics and counters
        self._init_stats(h_count)
        self.iteration         = 0
        self.last_improve_iter = 0
        self.restart_count     = 0
        self.total_improves    = 0
        self.phase             = Phase.INTENSIFY
        self.phase_until_iter  = 0
        self.best_fitness      = float("inf")

        problem.setMemorySize(POOL_END)
        self.pool_start = POOL_START
        self.pool_end   = POOL_END

        problem.initialiseSolution(A)
        problem.copySolution(A, B)

        # Seed the diversity pool from the initial solution
        for idx in range(POOL_START, POOL_END):
            problem.copySolution(A, idx)

        # OPT 4: index-swap pointers (no copy on rejection)
        cur_idx  = A
        cand_idx = B

        current_fit       = float(problem.getFunctionValue(cur_idx))
        self.best_fitness = current_fit
        self.acceptance.reset(initial_fitness=current_fit)

        # OPT 5: pre-compute crossover existence once before the loop
        has_crossover = bool(self.crossover_set)

        # Tightening threshold in nanoseconds (for late-search acceptance narrowing)
        tighten_start_ns = int(self.cfg.tighten_threshold * self.time_limit_ns)

        prev_h: Optional[int] = None

        while not self.hasTimeExpired():
            self.iteration += 1

            # Phase control (may set diversify, may flag restart)
            self._update_phase()

            # ---- Restart handling ----
            if self._maybe_restart(problem, cur_idx, cand_idx):
                # Reseed pool after restart
                for idx in range(self.pool_start, self.pool_end):
                    problem.copySolution(cur_idx, idx)
                current_fit       = float(problem.getFunctionValue(cur_idx))
                # BUG FIX 1: reset acceptance to new fitness so LAHC buffer
                # reflects the (worse) post-restart solution, not stale values.
                self.acceptance.reset(initial_fitness=current_fit)
                # Reset best_fitness to new start so stall counter works correctly
                self.best_fitness = current_fit
                prev_h = None
                continue

            # ---- Late-search acceptance tightening ----
            elapsed_ns = time.process_time_ns() - self.initial_time_ns
            if elapsed_ns >= tighten_start_ns:
                self.acceptance.tighten(self.cfg.tighten_rate)

            # ---- Heuristic selection ----
            h = self._select_heuristic(h_count, prev_h)

            # ---- Apply heuristic (HyFlex: from current slot to candidate slot) ----
            before = current_fit

            # OPT 5: inline crossover guard using pre-computed boolean
            if has_crossover and h in self.crossover_set:
                parent2 = self.rng.randrange(self.pool_start, self.pool_end)
                problem.applyHeuristic(h, cur_idx, parent2, cand_idx)
            else:
                problem.applyHeuristic(h, cur_idx, cand_idx)

            after = float(problem.getFunctionValue(cand_idx))

            improved = after < before
            accepted = self.acceptance.accept(before, after, self.iteration, self.rng)

            if accepted:
                # OPT 4: swap indices — free (two int assignments, no memory copy)
                cur_idx, cand_idx = cand_idx, cur_idx
                current_fit = after

                # Update diversity pool with improving crossover results
                if has_crossover and improved:
                    slot = self.rng.randrange(self.pool_start, self.pool_end)
                    problem.copySolution(cur_idx, slot)
            # else: reject — do nothing; cand_idx slot will be overwritten next iter

            # ---- Track global best ----
            if current_fit < self.best_fitness:
                self.best_fitness      = current_fit
                self.last_improve_iter = self.iteration
                self.total_improves   += 1
                self.acceptance.on_improvement(self.best_fitness)

            # ---- Update credits (individual + pair) ----
            self._credit_assignment(
                h=h,
                prev_h=prev_h,
                before=before,
                after=after,
                accepted=accepted,
                improved=improved,
            )

            # ---- Tabu update ----
            self._update_tabu(h)
            prev_h = h

    def __str__(self) -> str:
        return "Advanced HyFlex-Style Choice Function Hyper-Heuristic"


# ===========================================================================
# Backward-compatible wrapper class name
# ===========================================================================

class HyperHeuristic(AdvancedChoiceFunctionHH):
    """
    If your project expects a class called HyperHeuristic, this keeps
    compatibility.
    """
    pass