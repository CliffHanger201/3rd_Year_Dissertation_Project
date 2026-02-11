"""
Docstring for python_hyper_heuristic.src.hyperheuristic
"""
import random
import time
from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain

class HyperHeuristic():
    TRACE_CHECKPOINTS = 101
    def __init__(self, seed=None):

        self.rng = random.Random(seed)
        self.timeLimit = 0                    # nanoseconds
        self.initialTime = 0
        self.problem = None

        self.printfraction = 0
        self.printlimit = 0
        self.lastprint = 0
        self.initialprint = False

        self.lastbestsolution = -1
        self.trace = None
        self.timelimitset = False

        # Choice function parameters
        self.alpha = 1.0
        self.beta = 0.1
        self.gamma = 0.0

    def HyperHeuristic(self, seed : int = 0):
        self.lastbestsolution = -1
        self.rng = random.Random(seed)
        self.timelimitset = False

    def setTimeLimit(self, time_in_milliseconds : int):
        if not self.timelimitset:
            self.timeLimit = time_in_milliseconds * 1000000 # Change to milliseconds
            self.printfraction = time_in_milliseconds * 10000
            self.printlimit = self.printfraction

            self.initialprint = False
            self.lastprint = 0
            self.timelimitset = True
        else:
            raise Exception()
    
    def getTimeLimit(self):
        return self.timeLimit/1000000

    def getElapsedTime(self):
        if self.initialTime == 0:
            return 0
        elapsed_ns = time.process_time_ns() - self.initialTime
        return elapsed_ns // 1_000_000
    
# BEST SOLUTION / TRACE

    def getBestSolutionValue(self):
        if self.lastbestsolution == -1:
            raise RuntimeError("hasTimeExpired() must be called at least once")
        return self.lastbestsolution

    def getFitnessTrace(self):
        return self.trace
    
    # --------------------------------------------------
    # Core timing check (VERY IMPORTANT)
    # --------------------------------------------------

    def hasTimeExpired(self):
        current_time = time.process_time_ns() - self.initialTime

        if not self.initialprint:
            self.initialprint = True
            res = self.problem.getBestSolutionValue()
            self.trace[0] = res
            self.lastbestsolution = res

        elif current_time >= self.printlimit:
            thisprint = int(current_time / self.printfraction)
            thisprint = min(thisprint, 100)

            for x in range(thisprint - self.lastprint):
                if current_time <= self.timeLimit:
                    res = self.problem.getBestSolutionValue()
                    self.trace[self.lastprint + x + 1] = res
                    self.lastbestsolution = res
                else:
                    self.trace[self.lastprint + x + 1] = self.lastbestsolution

                self.printlimit += self.printfraction

            self.lastprint = thisprint

        if current_time >= self.timeLimit:
            return True
        else:
            self.lastbestsolution = self.problem.getBestSolutionValue()
            return False

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def loadProblemDomain(self, problem):
        if self.timeLimit == 0:
            raise RuntimeError("Time limit must be set before loading problem")
        self.problem = problem

    def run(self):
        if self.problem is None:
            raise RuntimeError("No problem domain loaded")
        if self.timeLimit == 0:
            raise RuntimeError("Time limit not set")

        self.trace = [0.0] * self.TRACE_CHECKPOINTS
        self.initialTime = time.process_time_ns()

        self.solve(self.problem)



    def solve(self, problem: ProblemDomain):

        h_count = problem.getNumberOfHeuristics()

        # Solution indices (HyFlex convention)
        CURRENT = 0
        CANDIDATE = 1

        problem.initialiseSolution(CURRENT)
        problem.copySolution(CURRENT, CANDIDATE)

        # Statistics
        improvement = [0.0] * h_count
        times_used = [0] * h_count
        last_used = [0] * h_count

        iteration = 0

        while not self.hasTimeExpired():

            iteration += 1

            # --- Choice Function ---
            scores = []
            for h in range(h_count):
                f1 = improvement[h] / times_used[h] if times_used[h] > 0 else 0
                f2 = iteration - last_used[h]
                score = self.alpha * f1 + self.beta * f2
                scores.append(score)

            heuristic = max(range(h_count), key=lambda h: scores[h])

            before = problem.getFunctionValue(CURRENT)

            problem.applyHeuristic(heuristic, CURRENT, CANDIDATE)

            after = problem.getFunctionValue(CANDIDATE)

            # Accept if better (hill-climbing)
            if after < before:
                problem.copySolution(CANDIDATE, CURRENT)
                delta = before - after
                improvement[heuristic] += delta
            else:
                delta = 0.0

            times_used[heuristic] += 1
            last_used[heuristic] = iteration

    def __str__(self):
        return " Choice Function HyperHeuristic "