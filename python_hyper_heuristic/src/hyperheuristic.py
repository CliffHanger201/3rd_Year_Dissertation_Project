"""
python_hyper_heuristic.src.hyperheuristic_advanced

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

This file is intentionally written with lots of extension points and clean structure.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Deque
from collections import deque

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType

# If you want strict typing, keep this import.
# from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain


# ----------------------------
# Utility / Configuration
# ----------------------------

class Phase(Enum):
    """High-level search mode."""
    INTENSIFY = auto()
    DIVERSIFY = auto()


class AcceptanceKind(Enum):
    """Move acceptance strategy."""
    IMPROVE_OR_EQUAL = auto()
    SIMULATED_ANNEALING = auto()
    LATE_ACCEPTANCE = auto()
    GREAT_DELUGE = auto()


@dataclass
class HHConfig:
    # Trace
    trace_checkpoints: int = 101

    # Choice function weights (classic CF: alpha*f1 + beta*f2 + gamma*f3)
    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 0.1

    # Choice function shaping
    reward_smoothing: float = 0.15      # EMA smoothing for heuristic rewards
    synergy_smoothing: float = 0.10     # EMA smoothing for pair synergy
    penalty_for_worsening: float = 0.05 # discourages heuristics that worsen often

    # Exploration/exploitation
    epsilon: float = 0.02               # small random exploration rate
    tabu_tenure: int = 3                # avoid repeating the last few heuristics

    # Phase control
    stall_iterations_to_diversify: int = 250
    stall_iterations_to_restart: int = 2000
    diversify_length: int = 150         # iterations to stay in diversify once triggered

    # Acceptance configuration
    acceptance_kind: AcceptanceKind = AcceptanceKind.LATE_ACCEPTANCE

    # SA parameters (if used)
    sa_t0: float = 1.0
    sa_alpha: float = 0.995             # cooling rate
    sa_reheat_multiplier: float = 1.5   # when stuck, increase T

    # LAHC parameters (if used)
    lahc_length: int = 50

    # Great Deluge parameters (if used)
    gd_initial_slack: float = 0.01
    gd_decay: float = 0.999

    # Misc
    consider_equal_as_accept: bool = True
    random_seed: Optional[int] = None


@dataclass
class HeuristicStats:
    """Statistics tracked per heuristic for credit assignment."""
    uses: int = 0
    accepts: int = 0
    improves: int = 0

    # Reward model
    reward_ema: float = 0.0      # f1 basis (individual performance)
    worsen_ema: float = 0.0      # penalise frequent worsening

    # Timing-ish: you can add CPU time per heuristic if the domain exposes it
    last_used_iter: int = 0


@dataclass
class PairStats:
    """Statistics for heuristic pairs (prev -> current), used for CF f2."""
    synergy_ema: float = 0.0
    uses: int = 0


# ----------------------------
# Move Acceptance
# ----------------------------

class MoveAcceptance:
    """Base class for move acceptance strategies."""
    def reset(self, initial_fitness: float) -> None:
        pass

    def accept(self, before: float, after: float, iteration: int, rng: random.Random) -> bool:
        raise NotImplementedError

    def on_improvement(self, new_best: float) -> None:
        """Hook to react to global improvements (optional)."""
        pass

    def on_stall(self) -> None:
        """Hook when HH detects a stall (optional)."""
        pass


class ImproveOrEqualAcceptance(MoveAcceptance):
    def __init__(self, allow_equal: bool = True):
        self.allow_equal = allow_equal

    def accept(self, before: float, after: float, iteration: int, rng: random.Random) -> bool:
        if after < before:
            return True
        if self.allow_equal and after == before:
            return True
        return False


class SimulatedAnnealingAcceptance(MoveAcceptance):
    def __init__(self, t0: float, cooling: float, reheat_multiplier: float, allow_equal: bool = True):
        self.t0 = t0
        self.cooling = cooling
        self.reheat_multiplier = reheat_multiplier
        self.allow_equal = allow_equal
        self.t = t0

    def reset(self, initial_fitness: float) -> None:
        self.t = self.t0

    def accept(self, before: float, after: float, iteration: int, rng: random.Random) -> bool:
        if after < before:
            return True
        if self.allow_equal and after == before:
            return True
        # worsening
        if self.t <= 1e-12:
            return False
        delta = after - before
        prob = math.exp(-delta / self.t) if delta > 0 else 1.0
        accept = rng.random() < prob
        # cool every move
        self.t *= self.cooling
        return accept

    def on_stall(self) -> None:
        # reheat when stuck
        self.t *= self.reheat_multiplier


class LateAcceptanceHillClimbing(MoveAcceptance):
    """
    LAHC: accept if after <= fitness[iteration mod L]
    Minimisation.
    """
    def __init__(self, L: int, allow_equal: bool = True):
        self.L = max(1, int(L))
        self.allow_equal = allow_equal
        self.buffer: List[float] = []
        self.pos: int = 0

    def reset(self, initial_fitness: float) -> None:
        self.buffer = [initial_fitness] * self.L
        self.pos = 0

    def accept(self, before: float, after: float, iteration: int, rng: random.Random) -> bool:
        threshold = self.buffer[self.pos]
        ok = after < threshold or (self.allow_equal and after == threshold)
        # update buffer with current (before) fitness per LAHC variant
        self.buffer[self.pos] = before
        self.pos = (self.pos + 1) % self.L
        return ok


class GreatDelugeAcceptance(MoveAcceptance):
    """
    Great Deluge: accept if after <= level
    level decays gradually.
    """
    def __init__(self, initial_slack: float, decay: float, allow_equal: bool = True):
        self.initial_slack = initial_slack
        self.decay = decay
        self.allow_equal = allow_equal
        self.level = None

    def reset(self, initial_fitness: float) -> None:
        # level starts slightly above initial fitness
        self.level = initial_fitness * (1.0 + self.initial_slack)

    def accept(self, before: float, after: float, iteration: int, rng: random.Random) -> bool:
        assert self.level is not None
        ok = after < self.level or (self.allow_equal and after == self.level)
        # tighten level
        self.level *= self.decay
        return ok

    def on_improvement(self, new_best: float) -> None:
        # Optionally pull level toward best
        if self.level is not None and new_best < self.level:
            self.level = (self.level + new_best) / 2.0


def make_acceptance(cfg: HHConfig) -> MoveAcceptance:
    if cfg.acceptance_kind == AcceptanceKind.IMPROVE_OR_EQUAL:
        return ImproveOrEqualAcceptance(allow_equal=cfg.consider_equal_as_accept)
    if cfg.acceptance_kind == AcceptanceKind.SIMULATED_ANNEALING:
        return SimulatedAnnealingAcceptance(
            t0=cfg.sa_t0,
            cooling=cfg.sa_alpha,
            reheat_multiplier=cfg.sa_reheat_multiplier,
            allow_equal=cfg.consider_equal_as_accept,
        )
    if cfg.acceptance_kind == AcceptanceKind.LATE_ACCEPTANCE:
        return LateAcceptanceHillClimbing(L=cfg.lahc_length, allow_equal=cfg.consider_equal_as_accept)
    if cfg.acceptance_kind == AcceptanceKind.GREAT_DELUGE:
        return GreatDelugeAcceptance(
            initial_slack=cfg.gd_initial_slack,
            decay=cfg.gd_decay,
            allow_equal=cfg.consider_equal_as_accept,
        )
    raise ValueError(f"Unknown acceptance kind: {cfg.acceptance_kind}")


# ----------------------------
# Advanced Hyper-Heuristic
# ----------------------------

class AdvancedChoiceFunctionHH:
    """
    HyFlex-style HH with:
    - Extended Choice Function (F1 individual, F2 pair synergy, F3 recency)
    - Adaptive move acceptance (LAHC/SA/GD/etc.)
    - Phase control and restarts
    - Tabu + epsilon exploration
    - Trace checkpoints (HyFlex-style)
    """

    TRACE_CHECKPOINTS_DEFAULT = 101

    def __init__(self, seed: Optional[int] = None, config: Optional[HHConfig] = None):
        self.cfg = config or HHConfig(random_seed=seed)
        self.rng = random.Random(self.cfg.random_seed)

        # Time management (ns internally)
        self.time_limit_ns: int = 0
        self.initial_time_ns: int = 0
        self.timelimitset: bool = False

        # Problem
        self.problem = None

        # Trace
        self.trace: Optional[List[float]] = None
        self.lastbestsolution: float = float("inf")

        # Heuristic Detection
        self.local_search_set: set[int] = set()
        self.mutation_set: set[int] = set()
        self.ruin_recreate_set: set[int] = set()
        self.crossover_set: set[int] = set()
        self.pool_start: int = 2
        self.pool_end: int = 2

        # Printing/trace checkpoint logic (matches the feel of HyFlex tracing)
        self.printfraction_ns: int = 0
        self.printlimit_ns: int = 0
        self.lastprint: int = 0
        self.initialprint: bool = False

        # Internal HH state
        self.phase: Phase = Phase.INTENSIFY
        self.phase_until_iter: int = 0

        self.iteration: int = 0
        self.last_improve_iter: int = 0
        self.best_fitness: float = float("inf")

        # These are initialised at runtime when we know number of heuristics
        self.h_stats: List[HeuristicStats] = []
        self.p_stats: List[List[PairStats]] = []
        self.tabu: Deque[int] = deque()

        # Acceptance
        self.acceptance: MoveAcceptance = make_acceptance(self.cfg)

    # -------- Heuristic Types ---------
    
    def setHeuristicClassTypes(self, problem) -> None:
        try:
            self.local_search_set = set(problem.getHeuristicsOfType(HeuristicType.LOCAL_SEARCH))
            self.mutation_set = set(problem.getHeuristicsOfType(HeuristicType.MUTATION))
            self.ruin_recreate_set = set(problem.getHeuristicsOfType(HeuristicType.RUIN_RECREATE))
            self.crossover_set = set(problem.getHeuristicsOfType(HeuristicType.CROSSOVER))
        except Exception:
            self.local_search_set = set()
            self.mutation_set = set()
            self.ruin_recreate_set = set()
            self.crossover_set = set()

    # -------- Time API --------

    def setTimeLimit(self, time_in_milliseconds: int) -> None:
        if self.timelimitset:
            raise RuntimeError("Time limit already set")
        if time_in_milliseconds <= 0:
            raise ValueError("Time limit must be > 0 ms")

        self.time_limit_ns = int(time_in_milliseconds) * 1_000_000
        # 101 checkpoints => 100 intervals; keep same spirit as your original
        self.printfraction_ns = max(1, self.time_limit_ns // (self.cfg.trace_checkpoints - 1))
        self.printlimit_ns = self.printfraction_ns

        self.initialprint = False
        self.lastprint = 0
        self.timelimitset = True

    def getTimeLimit(self) -> float:
        return self.time_limit_ns / 1_000_000.0

    def getElapsedTime(self) -> int:
        if self.initial_time_ns == 0:
            return 0
        return int((time.process_time_ns() - self.initial_time_ns) // 1_000_000)

    # -------- Problem / lifecycle --------

    def loadProblemDomain(self, problem) -> None:
        if self.time_limit_ns == 0:
            raise RuntimeError("Time limit must be set before loading problem")
        self.problem = problem

    def run(self) -> None:
        if self.problem is None:
            raise RuntimeError("No problem domain loaded")
        if self.time_limit_ns == 0:
            raise RuntimeError("Time limit not set")

        self.trace = [0.0] * self.cfg.trace_checkpoints
        self.initial_time_ns = time.process_time_ns()
        self._solve(self.problem)

    def isCrossover(self, index) -> bool:
        return index in self.crossover_set

    # -------- Trace / best --------

    def getBestSolutionValue(self) -> float:
        if self.lastbestsolution == float("inf"):
            raise RuntimeError("hasTimeExpired() must be called at least once")
        return self.lastbestsolution

    def getFitnessTrace(self) -> Optional[List[float]]:
        return self.trace

    # -------- Core timing check (HyFlex-style checkpoints) --------

    def hasTimeExpired(self) -> bool:
        """
        MUST be called inside the main loop.
        Updates trace checkpoints in a HyFlex-like way.
        """
        assert self.trace is not None
        current_ns = time.process_time_ns() - self.initial_time_ns

        if not self.initialprint:
            self.initialprint = True
            # If the domain supports it, this is the HyFlex "best so far"
            try:
                res = float(self.problem.getBestSolutionValue())
            except Exception:
                # fallback: track our own best
                res = float(self.best_fitness) if self.best_fitness != float("inf") else float("inf")
            self.trace[0] = res
            self.lastbestsolution = res

        elif current_ns >= self.printlimit_ns:
            thisprint = int(current_ns // self.printfraction_ns)
            thisprint = min(thisprint, self.cfg.trace_checkpoints - 1)

            for x in range(thisprint - self.lastprint):
                idx = self.lastprint + x + 1
                if current_ns <= self.time_limit_ns:
                    try:
                        res = float(self.problem.getBestSolutionValue())
                    except Exception:
                        res = float(self.best_fitness)
                    self.trace[idx] = res
                    self.lastbestsolution = res
                else:
                    self.trace[idx] = self.lastbestsolution

                self.printlimit_ns += self.printfraction_ns

            self.lastprint = thisprint

        if current_ns >= self.time_limit_ns:
            return True
        else:
            # refresh "best" if possible
            try:
                self.lastbestsolution = float(self.problem.getBestSolutionValue())
            except Exception:
                self.lastbestsolution = float(self.best_fitness)
            return False

    # -------- HH internals --------

    def _init_stats(self, h_count: int) -> None:
        self.h_stats = [HeuristicStats() for _ in range(h_count)]
        self.p_stats = [[PairStats() for _ in range(h_count)] for __ in range(h_count)]
        self.tabu.clear()

    def _choice_function_score(self, h: int, prev_h: Optional[int]) -> float:
        """
        Extended Choice Function:
        score(h) = alpha * F1(h) + beta * F2(prev,h) + gamma * F3(h)
        with minor shaping and penalties.
        """
        hs = self.h_stats[h]

        # F1: individual performance (EMA reward) minus penalty for worsening frequency
        f1 = hs.reward_ema - self.cfg.penalty_for_worsening * hs.worsen_ema

        # F2: pair synergy (if there is a previous heuristic)
        f2 = 0.0
        if prev_h is not None:
            f2 = self.p_stats[prev_h][h].synergy_ema

        # F3: recency/diversification: longer since last used => higher
        # normalise gently to avoid runaway
        recency = max(0, self.iteration - hs.last_used_iter)
        f3 = math.log1p(recency)

        # Phase shaping: in diversify we upweight recency and downweight pure reward slightly
        if self.phase == Phase.DIVERSIFY:
            alpha = self.cfg.alpha * 0.8
            beta = self.cfg.beta * 0.9
            gamma = self.cfg.gamma * 1.4
        else:
            alpha = self.cfg.alpha
            beta = self.cfg.beta
            gamma = self.cfg.gamma

        return alpha * f1 + beta * f2 + gamma * f3

    def _select_heuristic(self, h_count: int, prev_h: Optional[int]) -> int:
        """
        Select a heuristic using the choice function + epsilon exploration + tabu.
        """
        # Small chance of random exploration (HyFlex HHs often do something like this)
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.randrange(h_count)

        # Tabu: avoid recently used heuristics to reduce cycling
        tabu_set = set(self.tabu)

        best_h = 0
        best_score = -float("inf")

        for h in range(h_count):
            if h in tabu_set:
                continue
            s = self._choice_function_score(h, prev_h)
            if s > best_score:
                best_score = s
                best_h = h

        return best_h

    def _update_tabu(self, h: int) -> None:
        if self.cfg.tabu_tenure <= 0:
            return
        self.tabu.append(h)
        while len(self.tabu) > self.cfg.tabu_tenure:
            self.tabu.popleft()

    def _ema(self, old: float, value: float, smoothing: float) -> float:
        return (1.0 - smoothing) * old + smoothing * value

    def _credit_assignment(
        self,
        h: int,
        prev_h: Optional[int],
        before: float,
        after: float,
        accepted: bool,
        improved: bool,
    ) -> None:
        """
        Update heuristic and pair stats.
        Reward is based on:
        - magnitude of improvement (scaled)
        - acceptance (small positive)
        - worsening (penalty)
        """
        hs = self.h_stats[h]
        hs.uses += 1
        if accepted:
            hs.accepts += 1
        if improved:
            hs.improves += 1

        # Improvement magnitude (minimisation)
        delta = before - after  # positive if improved
        # Scale delta to be more stable across domains
        # Use a smooth transform: sign(delta)*log(1+abs(delta))
        shaped = 0.0
        if delta != 0.0:
            shaped = math.copysign(math.log1p(abs(delta)), delta)

        # Reward signal:
        # - improvements contribute strongly
        # - accepted non-improving moves contribute a tiny bit (useful for SA/LAHC)
        # - worsening moves contribute negative
        reward_signal = 0.0
        if improved:
            reward_signal = shaped
        elif accepted and not improved:
            reward_signal = 0.05  # small positive for enabling escapes
        else:
            # rejected move is mildly negative; accepted worsening is more negative
            reward_signal = -0.02 if not accepted else -0.05

        # Update individual reward EMA
        hs.reward_ema = self._ema(hs.reward_ema, reward_signal, self.cfg.reward_smoothing)

        # Track worsening tendency
        worsening = 1.0 if after > before else 0.0
        hs.worsen_ema = self._ema(hs.worsen_ema, worsening, self.cfg.reward_smoothing)

        # Pair synergy update if prev exists:
        # give credit when the pair leads to accepted improvement (or accepted move)
        if prev_h is not None:
            ps = self.p_stats[prev_h][h]
            ps.uses += 1
            pair_signal = 0.0
            if improved:
                pair_signal = shaped
            elif accepted:
                pair_signal = 0.02
            else:
                pair_signal = -0.01
            ps.synergy_ema = self._ema(ps.synergy_ema, pair_signal, self.cfg.synergy_smoothing)

        # Update last used
        hs.last_used_iter = self.iteration

    def _update_phase(self) -> None:
        """
        Simple, effective phase logic:
        - If stalled for a while => enter diversify for diversify_length iterations
        - If stalled too long => restart
        """
        stall = self.iteration - self.last_improve_iter

        if stall >= self.cfg.stall_iterations_to_restart:
            # Trigger restart next
            self.phase = Phase.DIVERSIFY
            self.phase_until_iter = self.iteration  # immediate action by caller
            return

        if self.phase == Phase.DIVERSIFY:
            if self.iteration >= self.phase_until_iter:
                self.phase = Phase.INTENSIFY
            return

        # currently intensify
        if stall >= self.cfg.stall_iterations_to_diversify:
            self.phase = Phase.DIVERSIFY
            self.phase_until_iter = self.iteration + self.cfg.diversify_length

    def _maybe_restart(self, problem, current_idx: int, candidate_idx: int) -> bool:
        """
        Restart strategy:
        - Reinitialise current solution if very stalled.
        - Keep best found implicitly via domain (or via our own best tracking).
        Returns True if restart happened.
        """
        stall = self.iteration - self.last_improve_iter
        if stall < self.cfg.stall_iterations_to_restart:
            return False

        # Inform acceptance method (e.g., reheat SA)
        self.acceptance.on_stall()

        # Reinitialise current and candidate
        problem.initialiseSolution(current_idx)
        problem.copySolution(current_idx, candidate_idx)
        # Reset stall counter baseline to avoid immediate repeated restarts
        self.last_improve_iter = self.iteration
        return True

    # -------- Apply Heuristic Helper ---------
    def _apply_heuristic(self, problem, h : int, current : int, candidate : int) -> None:
        """Helps to determine whether unary or crossover"""
        if self.isCrossover(h):
            parent2 = self.rng.randrange(self.pool_start, self.pool_end)
            problem.applyHeuristic(h, current, parent2, candidate)
        else:
            problem.applyHeuristic(h, current, candidate)

    # -------- Main solve loop --------

    def _solve(self, problem) -> None:
        h_count = int(problem.getNumberOfHeuristics())
        if h_count <= 0:
            raise RuntimeError("Problem domain reports 0 heuristics")
        
        self.setHeuristicClassTypes(problem)

        # HyFlex solution indices
        CURRENT = 0
        CANDIDATE = 1

        POOL_START = 2
        POOL_SIZE = 5
        POOL_END = POOL_START + POOL_SIZE

        # Initialise
        self._init_stats(h_count)
        self.iteration = 0
        self.last_improve_iter = 0
        self.phase = Phase.INTENSIFY
        self.phase_until_iter = 0

        problem.setMemorySize(POOL_END)
        self.pool_start, self.pool_end = POOL_START, POOL_END

        problem.initialiseSolution(CURRENT)
        problem.copySolution(CURRENT, CANDIDATE)

        # Seed the pool
        for idx in range(POOL_START, POOL_END):
            problem.copySolution(CURRENT, idx)

        current_fit = float(problem.getFunctionValue(CURRENT))
        self.best_fitness = current_fit
        self.acceptance.reset(initial_fitness=current_fit)

        prev_h: Optional[int] = None

        while not self.hasTimeExpired():
            self.iteration += 1

            # Phase control (may set diversify, may trigger restart)
            self._update_phase()
            if self._maybe_restart(problem, CURRENT, CANDIDATE):
                # reseed pool after restart
                for idx in range(self.pool_start, self.pool_end):
                    problem.copySolution(CURRENT, idx)

                current_fit = float(problem.getFunctionValue(CURRENT))
                prev_h = None
                continue

            # Pick heuristic
            h = self._select_heuristic(h_count, prev_h)

            # Apply heuristic (HyFlex: from CURRENT to CANDIDATE)
            before = current_fit
            self._apply_heuristic(problem, h, CURRENT, CANDIDATE)
            after = float(problem.getFunctionValue(CANDIDATE))

            improved = after < before
            accepted = self.acceptance.accept(before, after, self.iteration, self.rng)

            if accepted:
                # Move accepted: commit candidate to current
                problem.copySolution(CANDIDATE, CURRENT)
                current_fit = after

                if self.crossover_set and improved:
                    slot = self.rng.randrange(self.pool_start, self.pool_end)
                    problem.copySolution(CURRENT, slot)
            else:
                # rejected: revert candidate to current
                problem.copySolution(CURRENT, CANDIDATE)

            # Track global best
            if current_fit < self.best_fitness:
                self.best_fitness = current_fit
                self.last_improve_iter = self.iteration
                self.acceptance.on_improvement(self.best_fitness)

            # Update credits (individual + pair)
            self._credit_assignment(
                h=h,
                prev_h=prev_h,
                before=before,
                after=after,
                accepted=accepted,
                improved=improved,
            )

            # Tabu update
            self._update_tabu(h)
            prev_h = h

    def __str__(self) -> str:
        return "Advanced HyFlex-Style Choice Function Hyper-Heuristic"


# ----------------------------
# Backward-compatible wrapper class name (optional)
# ----------------------------

class HyperHeuristic(AdvancedChoiceFunctionHH):
    """
    If your project expects a class called HyperHeuristic, this keeps compatibility.
    """
    pass
