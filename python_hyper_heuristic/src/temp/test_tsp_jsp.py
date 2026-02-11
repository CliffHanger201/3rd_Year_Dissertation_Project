import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from . import hyperheuristic


# -------------------------
# 1. PROBLEMS
# -------------------------

class TSPProblem:
    def __init__(self, distance_matrix):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]

    def initial_solution(self):
        sol = np.arange(self.n)
        np.random.shuffle(sol)
        return sol

    def evaluate(self, tour):
        return sum(self.D[tour[i], tour[i + 1]] for i in range(self.n - 1)) + self.D[tour[-1], tour[0]]


class JSPProblem:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_ops = sum(len(j) for j in jobs)

    def initial_solution(self):
        seq = []
        for j in range(self.num_jobs):
            for _ in range(len(self.jobs[j])):
                seq.append(j)
        np.random.shuffle(seq)
        return np.array(seq)

    def evaluate(self, perm):
        machine_time = {}
        job_time = {j: 0 for j in range(self.num_jobs)}
        op_index = {j: 0 for j in range(self.num_jobs)}

        for job in perm:
            idx = op_index[job]
            machine, ptime = self.jobs[job][idx]

            start = max(job_time[job], machine_time.get(machine, 0))
            finish = start + ptime

            job_time[job] = finish
            machine_time[machine] = finish
            op_index[job] += 1

        return max(job_time.values())


# -------------------------
# 2. OPERATORS (Permutation)
# -------------------------

def swap(sol):
    s = sol.copy()
    i, j = np.random.choice(len(s), 2, replace=False)
    s[i], s[j] = s[j], s[i]
    return s

def two_opt(sol):
    s = sol.copy()
    i, j = sorted(np.random.choice(len(s), 2, replace=False))
    s[i:j] = s[i:j][::-1]
    return s

def insert(sol):
    s = sol.copy()
    i, j = np.random.choice(len(s), 2, replace=False)
    val = s[i]
    s = np.delete(s, i)
    s = np.insert(s, j, val)
    return s

def reverse(sol):
    s = sol.copy()
    i, j = sorted(np.random.choice(len(s), 2, replace=False))
    s[i:j] = s[i:j][::-1]
    return s


# -------------------------
# 3. PERMUTATION METAHEURISTIC
# -------------------------

class PermutationMetaheuristic:
    def __init__(self, problem, operators, num_iters=100, num_agents=10, verbose=False):
        self.problem = problem
        self.operators = operators
        self.num_iters = num_iters
        self.num_agents = num_agents
        self.history = []
        self.verbose = verbose

    def run(self):
        self.solutions = [self.problem.initial_solution() for _ in range(self.num_agents)]
        self.fitness = [self.problem.evaluate(s) for s in self.solutions]
        self.best_sol = self.solutions[np.argmin(self.fitness)]
        self.best_fit = min(self.fitness)

        for t in range(self.num_iters):
            for i in range(self.num_agents):
                op_idx = random.randrange(len(self.operators))
                op = self.operators[op_idx]

                # PRINT which operator is running
                if self.verbose:
                    print(f"MH Iter {t+1}/{self.num_iters} | Agent {i+1}/{self.num_agents} | Operator: {op.__name__}")

                candidate = op(self.solutions[i])
                cand_fit = self.problem.evaluate(candidate)

                if cand_fit < self.fitness[i]:
                    self.solutions[i] = candidate
                    self.fitness[i] = cand_fit

                if cand_fit < self.best_fit:
                    self.best_fit = cand_fit
                    self.best_sol = candidate

            self.history.append(self.best_fit)

    def get_solution(self):
        return self.best_sol, self.best_fit


# -------------------------
# 4. HYPERHEURISTIC
# -------------------------

class MyHyperheuristic:
    def __init__(self, heuristic_space, problem, params=None):
        self.heuristic_space = heuristic_space
        self.problem = problem
        self.params = params or {}
        self.num_steps = self.params.get("num_steps", 100)
        self.num_replicas = self.params.get("num_replicas", 20)
        self.num_agents = self.params.get("num_agents", 10)
        self.num_iters = self.params.get("num_iters", 50)

    def evaluate_candidate(self, sequence):
        fitnesses = []
        histories = []

        for _ in range(self.num_replicas):
            mh = PermutationMetaheuristic(
                self.problem,
                [self.heuristic_space[i] for i in sequence],
                num_iters=self.num_iters,
                num_agents=self.num_agents,
                verbose=True  # <-- prints operator running
            )
            mh.run()
            _, fit = mh.get_solution()
            fitnesses.append(fit)
            histories.append(mh.history)

        return np.median(fitnesses), histories

    def solve(self):
        best_seq = None
        best_fit = float("inf")
        best_history = None

        for t in range(self.num_steps):
            seq_len = np.random.randint(1, 5)
            seq = np.random.choice(len(self.heuristic_space), seq_len, replace=True)
            fit, histories = self.evaluate_candidate(seq)

            if fit < best_fit:
                best_fit = fit
                best_seq = seq
                best_history = histories[0]  # use first replica history

        return best_seq, best_fit, best_history


# -------------------------
# 5. RUN & PLOT
# -------------------------

def run_test(problem, name):
    operators = [swap, two_opt, insert, reverse]
    hh = MyHyperheuristic(operators, problem, params={
        "num_steps": 30,
        "num_replicas": 5,
        "num_agents": 10,
        "num_iters": 50
    })

    seq, fit, history = hh.solve()

    print(f"\n{name} BEST FIT = {fit}")
    print("Best sequence:", seq)

    # Plot convergence
    plt.figure()
    best0 = history[0]
    improvement = [(best0 - f) for f in history]
    plt.plot(improvement)

    plt.title(f"{name} Convergence (Best Sequence)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()


# -------------------------
# Example TSP
# -------------------------

D = np.random.rand(10, 10)
D = (D + D.T) / 2
np.fill_diagonal(D, 0)

tsp = TSPProblem(D)
run_test(tsp, "TSP")

# -------------------------
# Example JSP
# -------------------------

jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 1)]
]

jsp = JSPProblem(jobs)
run_test(jsp, "JSP")
