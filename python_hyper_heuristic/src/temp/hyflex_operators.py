import math
import random
import numpy as np
from types import SimpleNamespace
from python_hyper_heuristic.src.operators import swarm_dynamic


# =============================================================================
# SOLUTION POOL MANAGER
# =============================================================================

class SolutionPool:
    """
    Manages HyFlex solution indices safely.
    """
    def __init__(self, domain, elite_size=1):
        self.domain = domain
        self.elite_size = elite_size
        self.max_solutions = domain.getNumberOfSolutions()
        self.free_indices = list(range(elite_size, self.max_solutions))

    def allocate(self):
        if not self.free_indices:
            raise RuntimeError("No free solution slots available")
        return self.free_indices.pop(0)

    def release(self, idx):
        if idx >= self.elite_size and idx not in self.free_indices:
            self.free_indices.append(idx)


# =============================================================================
# REPRESENTATION ADAPTERS (DOMAIN-AWARE)
# =============================================================================

class RepresentationAdapter:
    def encode(self, solution):
        raise NotImplementedError

    def decode(self, vector):
        raise NotImplementedError


class BitstringAdapter(RepresentationAdapter):
    def encode(self, solution):
        return np.array(solution, dtype=float)

    def decode(self, vector):
        return (vector > 0.5).astype(int)


class PermutationAdapter(RepresentationAdapter):
    def encode(self, solution):
        return np.array(solution, dtype=float)

    def decode(self, vector):
        perm = np.argsort(vector)
        return perm  # repair hook could be added here


# =============================================================================
# HYFLEX DOMAIN ADAPTER
# =============================================================================

class HyFlexDomainAdapter:
    def __init__(self, domain, representation_adapter, pool):
        self.domain = domain
        self.adapter = representation_adapter
        self.pool = pool

    def copy(self, src, dst):
        self.domain.copySolution(src, dst)

    def get_solution(self, idx):
        return self.domain.getSolution(idx)

    def set_solution(self, idx, solution):
        self.domain.setSolution(idx, solution)

    def fitness(self, idx):
        return self.domain.getFunctionValue(idx)

    def apply_heuristic(self, h_id, src, dst):
        self.domain.applyHeuristic(h_id, src, dst)


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

class AcceptanceCriterion:
    def accept(self, old_f, new_f):
        raise NotImplementedError


class BetterOnlyAcceptance(AcceptanceCriterion):
    def accept(self, old_f, new_f):
        return new_f < old_f


class SimulatedAnnealingAcceptance(AcceptanceCriterion):
    def __init__(self, temperature=1.0, cooling=0.995):
        self.temperature = temperature
        self.cooling = cooling

    def accept(self, old_f, new_f):
        if new_f < old_f:
            return True
        prob = math.exp((old_f - new_f) / self.temperature)
        self.temperature *= self.cooling
        return random.random() < prob


# =============================================================================
# HYFLEX OPERATORS
# =============================================================================

class HyFlexOperator:
    def apply(self, domain_adapter, src, dst):
        raise NotImplementedError


class PSOOperator(HyFlexOperator):
    def __init__(
        self,
        pop_size=5,
        acceptance=None,
        post_heuristics=None
    ):
        self.pop_size = pop_size
        self.acceptance = acceptance or BetterOnlyAcceptance()
        self.post_heuristics = post_heuristics or []

    def apply(self, da: HyFlexDomainAdapter, src, dst):
        adapter = da.adapter
        pool = da.pool

        # ----------------------------
        # Build artificial population
        # ----------------------------
        pop_indices = []
        vectors = []

        for _ in range(self.pop_size):
            idx = pool.allocate()
            da.copy(src, idx)
            pop_indices.append(idx)
            vectors.append(adapter.encode(da.get_solution(idx)))

        vectors = np.array(vectors)

        pop = SimpleNamespace(
            positions=vectors,
            velocities=np.zeros_like(vectors),
            num_agents=self.pop_size,
            num_dimensions=vectors.shape[1],
            global_best_position=vectors[0],
            particular_best_positions=vectors.copy()
        )

        swarm_dynamic(pop)

        # ----------------------------
        # Decode PSO result
        # ----------------------------
        candidate_idx = pool.allocate()
        candidate = adapter.decode(pop.positions[0])
        da.set_solution(candidate_idx, candidate)

        # ----------------------------
        # Heuristic mixing
        # ----------------------------
        for h_id in self.post_heuristics:
            tmp = pool.allocate()
            da.apply_heuristic(h_id, candidate_idx, tmp)
            pool.release(candidate_idx)
            candidate_idx = tmp

        # ----------------------------
        # Acceptance decision
        # ----------------------------
        old_f = da.fitness(dst)
        new_f = da.fitness(candidate_idx)

        if self.acceptance.accept(old_f, new_f):
            da.copy(candidate_idx, dst)

        # ----------------------------
        # Cleanup
        # ----------------------------
        pool.release(candidate_idx)
        for idx in pop_indices:
            pool.release(idx)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def apply_operator(operator_name, domain, src, dst, problem_type):
    # Choose adapter
    if problem_type in ("TSP", "VRP"):
        adapter = PermutationAdapter()
    elif problem_type == "SAT":
        adapter = BitstringAdapter()
    else:
        raise ValueError("Unsupported problem type")

    pool = SolutionPool(domain, elite_size=1)
    da = HyFlexDomainAdapter(domain, adapter, pool)

    if operator_name == "PSO":
        operator = PSOOperator(
            pop_size=5,
            acceptance=SimulatedAnnealingAcceptance(),
            post_heuristics=[0, 1]  # e.g. mutation + local search
        )
    else:
        raise ValueError("Unknown operator")

    operator.apply(da, src, dst)