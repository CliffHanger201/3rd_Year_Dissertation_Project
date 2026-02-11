# sat_domain.py
# Converted from the provided Java SAT problem domain to fit YOUR Python ProblemDomain abstract class.

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType  # adjust import path if needed


@dataclass
class Variable:
    number: int
    state: bool = False
    age: int = 0

    def permanentflip(self) -> None:
        self.state = not self.state
        self.age = 0

    def testflip(self) -> None:
        self.state = not self.state

    def incrementAge(self) -> None:
        self.age += 1

    def clone(self) -> "Variable":
        return Variable(number=self.number, state=self.state, age=self.age)


class Clause:
    def __init__(self, numberofvariables: int, num: int):
        if numberofvariables == 0:
            raise ValueError("zero variables in this clause")
        self.number = num
        self.variablenumbers: List[int] = [0] * numberofvariables
        self.variablesigns: List[bool] = [False] * numberofvariables
        self.clausefill = 0

    def addVariable(self, n: int) -> None:
        # subtract 1 because variables are 0-indexed internally
        if n > 0:
            self.variablenumbers[self.clausefill] = n - 1
            self.variablesigns[self.clausefill] = True
        else:
            self.variablenumbers[self.clausefill] = (-n) - 1
            self.variablesigns[self.clausefill] = False
        self.clausefill += 1

    def numberOfVariables(self) -> int:
        return len(self.variablenumbers)

    def getVariableSign(self, index: int) -> bool:
        return self.variablesigns[index]

    def getVariableNumber(self, index: int) -> int:
        return self.variablenumbers[index]

    def clauseToString(self, variables: List[Variable]) -> str:
        s = "( "
        for i, vnum in enumerate(self.variablenumbers):
            if not self.variablesigns[i]:
                s += "-"
            s += f"{vnum}:{(self.variablesigns[i] == variables[vnum].state)} "
        s += ")"
        return s

    def evaluate(self, variables: List[Variable]) -> bool:
        # OR across literals
        for i, vnum in enumerate(self.variablenumbers):
            if self.variablesigns[i] == variables[vnum].state:
                return True
        return False


class Solution:
    """
    Represents an assignment. Holds Variables; knows how to evaluate itself against clauses.
    """
    def __init__(self, numberOfVariables: int, clauses: List[Clause]):
        self._num_vars = numberOfVariables
        self._clauses = clauses
        self.variables: List[Variable] = [Variable(i) for i in range(numberOfVariables)]

    def incrementAge(self) -> None:
        for v in self.variables:
            v.incrementAge()

    def numberOfBrokenClauses(self) -> int:
        broken = 0
        for c in self._clauses:
            if not c.evaluate(self.variables):
                broken += 1
        return broken

    def testFlipForBrokenClauses(self, variableToFlip: int) -> int:
        self.variables[variableToFlip].testflip()
        broken = 0
        for c in self._clauses:
            if not c.evaluate(self.variables):
                broken += 1
        self.variables[variableToFlip].testflip()
        return broken

    def testFlipForNegGain(self, variableToFlip: int) -> int:
        # satisfied before flip
        satisfied_before_numbers: Set[int] = set()
        for c in self._clauses:
            if c.evaluate(self.variables):
                satisfied_before_numbers.add(c.number)

        self.variables[variableToFlip].testflip()

        broken_after_numbers: Set[int] = set()
        for c in self._clauses:
            if not c.evaluate(self.variables):
                broken_after_numbers.add(c.number)

        self.variables[variableToFlip].testflip()

        # clauses that were satisfied before but are broken after
        return len(satisfied_before_numbers.intersection(broken_after_numbers))


class SAT(ProblemDomain):
    """
    Python conversion of the given Java SAT class, adapted to the provided ProblemDomain abstract base.
    Objective: minimize number of broken clauses.
    """

    def __init__(self, seed: int):
        # IMPORTANT: set these before super().__init__ because ProblemDomain.__init__()
        # calls getNumberOfHeuristics().
        self._mutations = [0, 1, 2, 3, 4, 5]
        self._localSearches = [7, 8]
        self._ruin_recreate = [6]
        self._crossovers = [9, 10]

        # instance state
        self.numberOfClauses: int = 0
        self.numberOfVariables: int = 0
        self.clauses: List[Clause] = []

        # best-ever (ProblemDomain also tracks best_ever_value/index, but we keep a deep copy too)
        self.bestEverSolution: Optional[Solution] = None

        # repeats controlled by depth/intensity
        self.lrepeats: int = 10
        self.mrepeats: int = 1

        super().__init__(seed)

    # -----------------------------
    # ProblemDomain-required methods
    # -----------------------------

    def getNumberOfHeuristics(self) -> int:
        return 11

    def getNumberOfInstances(self) -> int:
        return 12

    def __str__(self) -> str:
        return "SAT"

    def getHeuristicsOfType(self, heuristicType: HeuristicType):
        if heuristicType == HeuristicType.LOCAL_SEARCH:
            return list(self._localSearches)
        if heuristicType == HeuristicType.MUTATION:
            return list(self._mutations)
        if heuristicType == HeuristicType.RUIN_RECREATE:
            return list(self._ruin_recreate)
        if heuristicType == HeuristicType.CROSSOVER:
            return list(self._crossovers)
        return []

    def getHeuristicsThatUseDepthOfSearch(self):
        return list(self._localSearches)

    def getHeuristicsThatUseIntensityOfMutation(self):
        # Java returned mutations + ruin_recreate
        return list(self._mutations) + list(self._ruin_recreate)

    def setDepthOfSearch(self, depthOfSearch: float):
        super().setDepthOfSearch(depthOfSearch)
        d = self.depthOfSearch
        if d <= 0.2:
            self.lrepeats = 10
        elif d <= 0.4:
            self.lrepeats = 12
        elif d <= 0.6:
            self.lrepeats = 14
        elif d <= 0.8:
            self.lrepeats = 17
        else:
            self.lrepeats = 20

    def setIntensityOfMutation(self, intensityOfMutation: float):
        super().setIntensityOfMutation(intensityOfMutation)
        i = self.intensityOfMutation
        if i <= 0.2:
            self.mrepeats = 1
        elif i <= 0.4:
            self.mrepeats = 2
        elif i <= 0.6:
            self.mrepeats = 3
        elif i <= 0.8:
            self.mrepeats = 4
        else:
            self.mrepeats = 5

    def setMemorySize(self, size: int):
        new_mem: List[Optional[Solution]] = [None] * size
        if self.solutions:
            for i in range(min(len(self.solutions), size)):
                new_mem[i] = self.solutions[i]
        self.solutions = new_mem

        # if best index is out of range, invalidate it (safe behavior)
        if self.best_ever_index is not None and self.best_ever_index >= size:
            self.best_ever_index = None
            self.best_ever_value = float("inf")
            self.bestEverSolution = None

    def initialiseSolution(self, index: int):
        self.solutions[index] = Solution(self.numberOfVariables, self.clauses)
        s = self.solutions[index]
        assert s is not None
        for v in s.variables:
            v.state = self.rng.choice([True, False])

        val = self.evaluateObjectiveFunction(s)
        self._maybe_update_best(index, s, val)

    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int):
        src = self.solutions[solutionSourceIndex]
        if src is None:
            raise ValueError("Source solution not initialised")
        self.solutions[solutionDestinationIndex] = self.deepCopyTheSolution(src)

    # def getBestSolutionIndex(self) -> int:
    #     if self.best_ever_index is None:
    #         return -1
    #     return int(self.best_ever_index)

    def getBestSolutionValue(self) -> float:
        return float(self.best_ever_value)

    def bestSolutionToString(self) -> str:
        if self.bestEverSolution is None:
            return "No best solution recorded."
        s = self.bestEverSolution
        out = []
        out.append("Best solution Found")
        out.append(f"Objective function value: {self.getBestSolutionValue()}")
        out.append("Variables:")
        for y in range(self.numberOfVariables):
            if (y % 5) == 0:
                out.append("")
            out.append(f"{s.variables[y].number}:{s.variables[y].state}\t")
        out.append("\nClauses:")
        for x in range(self.numberOfClauses):
            if (x % 3) == 0:
                out.append("")
            out.append(self.clauses[x].clauseToString(s.variables))
        return "\n".join(out)

    def solutionToString(self, solutionIndex: int) -> str:
        s = self.solutions[solutionIndex]
        if s is None:
            raise ValueError("Solution not initialised")
        return " ".join(f"{v.number}:{v.state}" for v in s.variables)

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        s1 = self.solutions[solutionIndex1]
        s2 = self.solutions[solutionIndex2]
        if s1 is None or s2 is None:
            raise ValueError("Solution not initialised")
        for i in range(self.numberOfVariables):
            if s1.variables[i].state != s2.variables[i].state:
                return False
        return True

    def getFunctionValue(self, solutionIndex: int) -> float:
        s = self.solutions[solutionIndex]
        if s is None:
            raise ValueError("Solution not initialised")
        return self.evaluateObjectiveFunction(s)

    def applyHeuristic(
        self,
        heuristicID: int,
        solutionSourceIndex1: int,
        solutionSourceIndex2: int = None,
        solutionDestinationIndex: int = None
    ) -> float:
        """
        Supports BOTH HyFlex-style call patterns:
        - Unary:   applyHeuristic(h, src, dst)
        - Binary:  applyHeuristic(h, src1, src2, dst)
        """

        t0 = time.time()

        # Detect whether this is a 3-arg unary call:
        # HH calls applyHeuristic(h, src, dst)
        # which arrives here as (h, src, solutionSourceIndex2=dst, solutionDestinationIndex=None)
        if solutionDestinationIndex is None and solutionSourceIndex2 is not None:
            # interpret the 3rd argument as destination
            solutionDestinationIndex = solutionSourceIndex2
            solutionSourceIndex2 = None

        if solutionDestinationIndex is None:
            raise ValueError("Destination index missing")

        # Ensure destination fits memory
        if solutionDestinationIndex >= len(self.solutions):
            self.setMemorySize(solutionDestinationIndex + 1)

        src1 = self.solutions[solutionSourceIndex1]
        if src1 is None:
            raise ValueError("Source solution 1 not initialised")

        is_crossover = heuristicID in self._crossovers

        if is_crossover:
            if solutionSourceIndex2 is None:
                raise ValueError("Crossover heuristic requires 2 source solutions (src1, src2)")

            src2 = self.solutions[solutionSourceIndex2]
            if src2 is None:
                raise ValueError("Source solution 2 not initialised")

            temp1 = self.deepCopyTheSolution(src1)
            temp2 = self.deepCopyTheSolution(src2)

            if heuristicID == 9:
                self.applyHeuristic9(temp1, temp2)
            elif heuristicID == 10:
                self.applyHeuristic10(temp1, temp2)
            else:
                raise ValueError(f"Heuristic {heuristicID} is not a crossover operator")

            new_val = self.evaluateObjectiveFunction(temp1)
            self._store_to_destination(solutionDestinationIndex, temp1)

        else:
            temp = self.deepCopyTheSolution(src1)

            if temp.numberOfBrokenClauses() != 0:
                self._dispatch_unary(heuristicID, temp)

            new_val = self.evaluateObjectiveFunction(temp)
            self._store_to_destination(solutionDestinationIndex, temp)

        # bookkeeping (ms)
        elapsed_ms = int((time.time() - t0) * 1000)
        self.heuristicCallRecord[heuristicID] += 1
        self.heuristicCallTimeRecord[heuristicID] += elapsed_ms

        # update best using the destination solution
        dest_sol = self.solutions[solutionDestinationIndex]
        assert dest_sol is not None
        self._maybe_update_best(solutionDestinationIndex, dest_sol, new_val)

        return float(new_val)
    
    def _resource_root(self) -> Path:
        # SAT.py is at: .../python_hyper_heuristic/domains/Python/SAT/SAT.py
        # resources is at: .../python_hyper_heuristic/domains/resources
        here = Path(__file__).resolve()
        return here.parents[2] / "resources"   # parents[2] == .../domains

    def loadInstance(self, instanceID: int):
        # reset solution memory to default size 2 as Java did at load
        self.setMemorySize(2)

        if instanceID < 1:
            path = "sat07/crafted/Difficult/contest-02-03-04/contest02-Mat26.sat05-457.reshuffled-07.txt"
        elif instanceID < 2:
            path = "sat07/crafted/Hard/contest03/looksrandom/hidden-k3-s0-r5-n700-01-S2069048075.sat05-488.reshuffled-07.txt"
        elif instanceID < 3:
            path = "sat07/crafted/Hard/contest03/looksrandom/hidden-k3-s0-r5-n700-02-S350203913.sat05-486.reshuffled-07.txt"
        elif instanceID < 4:
            path = "sat09/crafted/parity-games/instance_n3_i3_pp.txt"
        elif instanceID < 5:
            path = "sat09/crafted/parity-games/instance_n3_i3_pp_ci_ce.txt"
        elif instanceID < 6:
            path = "sat09/crafted/parity-games/instance_n3_i4_pp_ci_ce.txt"
        elif instanceID < 7:
            path = "ms_random/highgirth/3SAT/HG-3SAT-V250-C1000-1.txt"
        elif instanceID < 8:
            path = "ms_random/highgirth/3SAT/HG-3SAT-V250-C1000-2.txt"
        elif instanceID < 9:
            path = "ms_random/highgirth/3SAT/HG-3SAT-V300-C1200-2.txt"
        elif instanceID < 10:
            path = "ms_crafted/MAXCUT/SPINGLASS/t7pm3-9999.spn.txt"
        elif instanceID < 11:
            path = "sat07/industrial/jarvisalo/eq.atree.braun.8.unsat.txt"
        elif instanceID < 12:
            path = "ms_random/highgirth/3SAT/HG-3SAT-V300-C1200-4.txt"
        else:
            raise ValueError(f"instance does not exist {instanceID}")

        filename = self._resource_root() / "data" / "sat" / path
        self._loadInstanceFromFile(filename)

        # reset best trackers
        self.best_ever_value = float("inf")
        self.best_ever_index = None
        self.bestEverSolution = None

        # after loading, rebuild heuristic repeats based on current parameters
        self.setDepthOfSearch(self.depthOfSearch)
        self.setIntensityOfMutation(self.intensityOfMutation)

    # -----------------------------
    # Internals
    # -----------------------------

    def evaluateObjectiveFunction(self, solution: Solution) -> float:
        return float(solution.numberOfBrokenClauses())

    def deepCopyTheSolution(self, solutionToCopy: Solution) -> Solution:
        newsol = Solution(self.numberOfVariables, self.clauses)
        newsol.variables = [v.clone() for v in solutionToCopy.variables]
        return newsol

    def _maybe_update_best(self, index: int, sol: Solution, val: float) -> None:
        if val < self.best_ever_value:
            self.best_ever_value = float(val)
            self.best_ever_index = index
            self.bestEverSolution = self.deepCopyTheSolution(sol)

    def _store_to_destination(self, dst: int, sol: Solution) -> None:
        if dst >= len(self.solutions):
            self.setMemorySize(dst + 1)
        self.solutions[dst] = self.deepCopyTheSolution(sol)
        # increment age as Java did when copying to destination
        self.solutions[dst].incrementAge()

    def _dispatch_unary(self, heuristicID: int, sol: Solution) -> None:
        if heuristicID == 0:
            self.applyHeuristic0(sol)
        elif heuristicID == 1:
            self.applyHeuristic1(sol)
        elif heuristicID == 2:
            self.applyHeuristic2(sol)
        elif heuristicID == 3:
            self.applyHeuristic3(sol)
        elif heuristicID == 4:
            self.applyHeuristic4(sol)
        elif heuristicID == 5:
            self.applyHeuristic5(sol)
        elif heuristicID == 6:
            self.applyHeuristic6(sol)
        elif heuristicID == 7:
            self.applyHeuristic7(sol)
        elif heuristicID == 8:
            self.applyHeuristic8(sol)
        else:
            raise ValueError(f"Heuristic {heuristicID} does not exist")

    # -----------------------------
    # Heuristic helpers (ported)
    # -----------------------------

    def getVariablesWithHighestNetGain(self, tempSolution: Solution) -> List[int]:
        numbersofbrokenclauses = [0] * self.numberOfVariables
        for x in range(self.numberOfVariables):
            numbersofbrokenclauses[x] = tempSolution.testFlipForBrokenClauses(x)

        minimum = numbersofbrokenclauses[0]
        for v in numbersofbrokenclauses:
            if v < minimum:
                minimum = v

        return [i for i, v in enumerate(numbersofbrokenclauses) if v == minimum]

    def getRandomBrokenClause(self, tempSolution: Solution) -> Optional[Clause]:
        broken = [c for c in self.clauses if not c.evaluate(tempSolution.variables)]
        if not broken:
            return None
        return broken[self.rng.randrange(len(broken))]

    def flipRandomVariableInClause(self, tempSolution: Solution, c: Clause) -> None:
        idx = self.rng.randrange(c.numberOfVariables())
        vnum = c.variablenumbers[idx]
        tempSolution.variables[vnum].permanentflip()

    def flipRandomVariableInRandomBrokenClause(self, tempSolution: Solution) -> None:
        c = self.getRandomBrokenClause(tempSolution)
        if c is None:
            return
        self.flipRandomVariableInClause(tempSolution, c)

    def getNegativeGain(self, tempSolution: Solution, variableToFlip: int) -> int:
        return tempSolution.testFlipForNegGain(variableToFlip)

    # -----------------------------
    # Heuristics 0..10 (ported)
    # -----------------------------

    def applyHeuristic0(self, sol: Solution) -> None:  # GSAT
        for _ in range(self.mrepeats):
            highest = self.getVariablesWithHighestNetGain(sol)
            i = highest[self.rng.randrange(len(highest))]
            sol.variables[i].permanentflip()

    def applyHeuristic1(self, sol: Solution) -> None:  # HSAT
        for _ in range(self.mrepeats):
            jointmins = self.getVariablesWithHighestNetGain(sol)
            largestage = sol.variables[jointmins[0]]
            for idx in jointmins:
                contender = sol.variables[idx]
                if contender.age > largestage.age:
                    largestage = contender
            largestage.permanentflip()

    def applyHeuristic2(self, sol: Solution) -> None:  # WalkSAT (no random walk)
        for _ in range(self.mrepeats):
            c = self.getRandomBrokenClause(sol)
            if c is None:
                break

            negativeGains = [0] * c.numberOfVariables()
            vars_neg0: List[int] = []
            for x in range(c.numberOfVariables()):
                vnum = c.variablenumbers[x]
                ng = self.getNegativeGain(sol, vnum)
                negativeGains[x] = ng
                if ng == 0:
                    vars_neg0.append(vnum)

            if vars_neg0:
                vnum = vars_neg0[self.rng.randrange(len(vars_neg0))]
                sol.variables[vnum].permanentflip()
            else:
                minimum = negativeGains[0]
                for ng in negativeGains[1:]:
                    if ng < minimum:
                        minimum = ng
                jointmins: List[int] = []
                for i in range(c.numberOfVariables()):
                    if negativeGains[i] == minimum:
                        jointmins.append(c.variablenumbers[i])
                vnum = jointmins[self.rng.randrange(len(jointmins))]
                sol.variables[vnum].permanentflip()

    def applyHeuristic3(self, sol: Solution) -> None:
        for _ in range(self.mrepeats):
            self.flipRandomVariableInRandomBrokenClause(sol)

    def applyHeuristic4(self, sol: Solution) -> None:
        for _ in range(self.mrepeats):
            sol.variables[self.rng.randrange(self.numberOfVariables)].permanentflip()

    def applyHeuristic5(self, sol: Solution) -> None:  # novelty
        for _ in range(self.mrepeats):
            c = self.getRandomBrokenClause(sol)
            if c is None:
                break
            self.applyNovelty(sol, c)

    def applyNovelty(self, sol: Solution, randomBrokenClause: Clause) -> None:
        p = 0.7

        numbersofbrokenclauses = [0] * randomBrokenClause.numberOfVariables()
        minimalage = 2**31 - 1

        for x in range(randomBrokenClause.numberOfVariables()):
            vnum = randomBrokenClause.variablenumbers[x]
            numbersofbrokenclauses[x] = sol.testFlipForBrokenClauses(vnum)
            if sol.variables[vnum].age < minimalage:
                minimalage = sol.variables[vnum].age

        minimum = 2**31 - 1
        secondminimum = 2**31 - 1
        for val in numbersofbrokenclauses:
            if val < minimum:
                secondminimum = minimum
                minimum = val
            elif val < secondminimum:
                secondminimum = val

        jointminimums = [i for i, val in enumerate(numbersofbrokenclauses) if val == minimum]
        i = jointminimums[self.rng.randrange(len(jointminimums))]
        chosen_vnum = randomBrokenClause.variablenumbers[i]

        if sol.variables[chosen_vnum].age == minimalage:
            # Java: if (p < rng.nextDouble()) flip it else flip second-best
            if p < self.rng.random():
                sol.variables[chosen_vnum].permanentflip()
            else:
                if randomBrokenClause.numberOfVariables() == 1:
                    sol.variables[chosen_vnum].permanentflip()
                    return

                if minimum == secondminimum:
                    while True:
                        q = jointminimums[self.rng.randrange(len(jointminimums))]
                        if q != i:
                            sol.variables[randomBrokenClause.variablenumbers[q]].permanentflip()
                            break
                else:
                    jointsecond = [q for q, val in enumerate(numbersofbrokenclauses) if val == secondminimum]
                    q = jointsecond[self.rng.randrange(len(jointsecond))]
                    sol.variables[randomBrokenClause.variablenumbers[q]].permanentflip()
        else:
            sol.variables[chosen_vnum].permanentflip()

    def applyHeuristic6(self, sol: Solution) -> None:  # ruin and recreate
        i = self.intensityOfMutation
        if i <= 0.25:
            prop = 1.0 / 5.0
        elif i <= 0.49:
            prop = 2.0 / 5.0
        elif i <= 0.75:
            prop = 3.0 / 5.0
        else:
            prop = 4.0 / 5.0

        num_vars = int(self.numberOfVariables * prop)
        chosen: Set[int] = set()
        while len(chosen) < num_vars:
            chosen.add(self.rng.randrange(self.numberOfVariables))

        for vnum in chosen:
            if self.rng.choice([True, False]):
                sol.variables[vnum].permanentflip()

    def applyHeuristic7(self, sol: Solution) -> None:
        currentres = self.evaluateObjectiveFunction(sol)
        for _ in range(self.lrepeats):
            c = self.getRandomBrokenClause(sol)
            if c is None:
                break
            idx = self.rng.randrange(c.numberOfVariables())
            vnum = c.variablenumbers[idx]
            sol.variables[vnum].testflip()
            res = self.evaluateObjectiveFunction(sol)
            sol.variables[vnum].testflip()
            if res <= currentres:
                sol.variables[vnum].permanentflip()
                currentres = res

    def applyHeuristic8(self, sol: Solution) -> None:
        currentres = self.evaluateObjectiveFunction(sol)
        for _ in range(self.lrepeats):
            vnum = self.rng.randrange(self.numberOfVariables)
            sol.variables[vnum].testflip()
            res = self.evaluateObjectiveFunction(sol)
            sol.variables[vnum].testflip()
            if res <= currentres:
                sol.variables[vnum].permanentflip()
                currentres = res

    def applyHeuristic9(self, sol1: Solution, sol2: Solution) -> None:
        # two point crossover, one child (sol1)
        cp1 = self.rng.randrange(len(sol1.variables))
        cp2 = self.rng.randrange(len(sol1.variables))
        if cp1 > cp2:
            cp1, cp2 = cp2, cp1
        for x in range(cp1, cp2):
            sol1.variables[x] = sol2.variables[x]  # matches Java object assignment

    def applyHeuristic10(self, sol1: Solution, sol2: Solution) -> None:
        # one point crossover, one child (sol1)
        cp1 = self.rng.randrange(len(sol1.variables))
        for x in range(cp1, len(sol1.variables)):
            sol1.variables[x] = sol2.variables[x]  # matches Java object assignment

    # -----------------------------
    # Instance parsing (ported)
    # -----------------------------

    def _loadInstanceFromFile(self, filename: str) -> None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"cannot find file {filename}") from e
        self._readInInstance(lines)

    def _readInInstance(self, lines: List[str]) -> None:
        # scan until line starting with 'p'
        idx = 0
        while idx < len(lines) and not lines[idx].startswith("p"):
            idx += 1
        if idx >= len(lines):
            raise ValueError("file format incorrect: missing 'p' line")

        header = lines[idx].strip()
        parts = header.split()
        self.numberOfVariables = int(parts[2])
        if len(parts) == 5:
            self.numberOfClauses = int(parts[4])
        elif len(parts) == 4:
            self.numberOfClauses = int(parts[3])
        else:
            raise ValueError("file format incorrect")

        self.clauses = []
        line_i = idx + 1
        for clause_num in range(self.numberOfClauses):
            if line_i >= len(lines):
                raise ValueError("Unexpected EOF while reading clauses")

            raw = lines[line_i].strip()
            line_i += 1

            # defensive skip blank lines
            while raw == "" and line_i < len(lines):
                raw = lines[line_i].strip()
                line_i += 1

            vars_str = raw.split()
            # last token expected to be 0 sentinel; ignore it
            c = Clause(len(vars_str) - 1, clause_num)
            for j in range(len(vars_str) - 1):
                c.addVariable(int(vars_str[j]))
            self.clauses.append(c)
