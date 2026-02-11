# Fully converted from FlowShop.FlowShop (Java) to Python.
# Assumptions (per your instruction):
#   - AbstractClasses.ProblemDomain exists and provides:
#       - rng (a Random-like object) with nextInt(n), nextDouble()
#         OR Python-style: randint/randrange/random. (See notes below.)
#       - heuristicCallRecord, heuristicCallTimeRecord arrays
#       - intensityOfMutation, depthOfSearch + getters/setters
#       - HeuristicType enum: MUTATION, RUIN_RECREATE, LOCAL_SEARCH, CROSSOVER
#   - FlowShop.Instance, FlowShop.Solution, FlowShop.BasicAlgorithms exist (converted earlier)
#
# Important: Java's rng.nextInt(n) returns [0, n). If your ProblemDomain.rng is Python's random.Random,
# replace nextInt with randrange.

from __future__ import annotations

import sys
import time
from typing import Any, List, Optional

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType  # assumed to exist
from python_hyper_heuristic.domains.Python.FlowShop.Instance import Instance
from python_hyper_heuristic.domains.Python.FlowShop.Solution import Solution
from python_hyper_heuristic.domains.Python.FlowShop.BasicAlgorithms import BasicAlgorithms


class FlowShop(ProblemDomain):
    def __init__(self, seed: int) -> None:
        self.memory: List[Optional[Solution]] = [None, None]

        super().__init__(seed)
        
        self.bestSoFar: Optional[Solution] = None
        self.probInstance: Optional[Instance] = None
        self.heuristics = BasicAlgorithms()

    # --- ProblemDomain required methods ---

    def loadInstance(self, instanceID: int) -> None:
        self.probInstance = Instance(instanceID)

    def initialiseSolution(self, targetIndex: int) -> None:
        assert self.probInstance is not None
        perm = self.generateRandomPermutation(self.probInstance.n)
        self.memory[targetIndex] = self.heuristics.neh(self.probInstance, perm)
        self.verifyBestSolution(self.memory[targetIndex])

    def setMemorySize(self, size: int) -> None:
        tempMemory: List[Optional[Solution]] = [None] * size
        if self.memory is not None:
            if len(tempMemory) <= len(self.memory):
                for i in range(len(tempMemory)):
                    tempMemory[i] = self.memory[i]
            else:
                for i in range(len(self.memory)):
                    tempMemory[i] = self.memory[i]
        self.memory = tempMemory

    def getHeuristicsOfType(self, heuristicType: "HeuristicType") -> Optional[List[int]]:
        if heuristicType == HeuristicType.MUTATION:
            return [0, 1, 2, 3, 4]
        if heuristicType == HeuristicType.RUIN_RECREATE:
            return [5, 6]
        if heuristicType == HeuristicType.LOCAL_SEARCH:
            return [7, 8, 9, 10]
        if heuristicType == HeuristicType.CROSSOVER:
            return [11, 12, 13, 14]
        return None

    def getHeuristicsThatUseDepthOfSearch(self) -> List[int]:
        return [6, 9, 10]

    def getHeuristicsThatUseIntensityOfMutation(self) -> List[int]:
        return [3, 5, 6]

    def getBestSolutionValue(self) -> float:
        assert self.bestSoFar is not None
        return float(self.bestSoFar.Cmax)

    def getFunctionValue(self, index: int) -> float:
        sol = self.memory[index]
        assert sol is not None
        return float(sol.Cmax)

    def applyHeuristic(self, llhID: int, solutionSourceIndex: int, solutionTargetIndex: int) -> float:
        start_time = time.time()

        crossovers = self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        is_crossover = False
        if crossovers is not None:
            for x in crossovers:
                if x == llhID:
                    is_crossover = True
                    break

        if is_crossover:
            self.copySolution(solutionSourceIndex, solutionTargetIndex)
        else:
            if llhID == 0:
                self.randomReinsertion(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 1:
                self.swapTwo(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 2:
                self.shuffle(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 3:
                self.shuffleSubSequence(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 4:
                self.useNEH(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 5:
                self.iteratedGreedy(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 6:
                self.deepIteratedGreedy(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 7:
                self.localSearch(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 8:
                self.fImpLocalSearch(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 9:
                self.randomLocalSearch(solutionSourceIndex, solutionTargetIndex)
            elif llhID == 10:
                self.randomFImpLocalSearch(solutionSourceIndex, solutionTargetIndex)
            else:
                print(
                    "heuristic does not exist, or the crossover index array is not set up correctly",
                    file=sys.stderr,
                )
                raise SystemExit(-1)

        # bookkeeping
        self.heuristicCallRecord[llhID] += 1
        self.heuristicCallTimeRecord[llhID] += int((time.time() - start_time) * 1000)

        sol = self.memory[solutionTargetIndex]
        assert sol is not None
        self.verifyBestSolution(sol)
        return float(sol.Cmax)

    def applyHeuristic2(
        self,
        llhID: int,
        solutionSourceIndex1: int,
        solutionSourceIndex2: int,
        solutionTargetIndex: int,
    ) -> float:
        start_time = time.time()

        if llhID == 0:
            self.randomReinsertion(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 1:
            self.swapTwo(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 2:
            self.shuffle(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 3:
            self.shuffleSubSequence(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 4:
            self.useNEH(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 5:
            self.iteratedGreedy(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 6:
            self.deepIteratedGreedy(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 7:
            self.localSearch(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 8:
            self.fImpLocalSearch(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 9:
            self.randomLocalSearch(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 10:
            self.fImpLocalSearch(solutionSourceIndex1, solutionTargetIndex)
        elif llhID == 11:
            self.ox(solutionSourceIndex1, solutionSourceIndex2, solutionTargetIndex)
        elif llhID == 12:
            self.ppx(solutionSourceIndex1, solutionSourceIndex2, solutionTargetIndex)
        elif llhID == 13:
            self.pmx(solutionSourceIndex1, solutionSourceIndex2, solutionTargetIndex)
        elif llhID == 14:
            self.oneX(solutionSourceIndex1, solutionSourceIndex2, solutionTargetIndex)
        else:
            print(
                "heuristic does not exist, or the crossover index array is not set up correctly",
                file=sys.stderr,
            )
            raise SystemExit(-1)

        self.heuristicCallRecord[llhID] += 1
        self.heuristicCallTimeRecord[llhID] += int((time.time() - start_time) * 1000)

        sol = self.memory[solutionTargetIndex]
        assert sol is not None
        self.verifyBestSolution(sol)
        return float(sol.Cmax)

    def copySolution(self, sourceIndex: int, targetIndex: int) -> None:
        src = self.memory[sourceIndex]
        assert src is not None
        self.memory[targetIndex] = src.clone()

    def getNumberOfHeuristics(self) -> int:
        return 15

    def solutionToString(self, solutionIndex: int) -> str:
        sol = self.memory[solutionIndex]
        assert sol is not None
        return str(sol)

    def __str__(self) -> str:
        return "FlowShop"

    def bestSolutionToString(self) -> str:
        assert self.bestSoFar is not None
        return str(self.bestSoFar)

    def getNumberOfInstances(self) -> int:
        return 12

    def getProblemData(self, args: str) -> Any:
        assert self.probInstance is not None
        if args == "N":
            return self.probInstance.n
        if args == "M":
            return self.probInstance.m
        if args == "SUM_P":
            return self.probInstance.getSumP()
        return self.probInstance.processingTimes

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        s1 = self.memory[solutionIndex1]
        s2 = self.memory[solutionIndex2]
        assert s1 is not None and s2 is not None

        if s1.Cmax != s2.Cmax:
            return False

        p1 = s1.permutation
        p2 = s2.permutation
        assert self.probInstance is not None
        n = self.probInstance.n
        for i in range(n):
            if p1[i] != p2[i]:
                return False
        return True

    # ---------------- MUTATION HEURISTICS ----------------

    def randomReinsertion(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None
        array = src.permutation.copy()
        i1 = self.rng.randrange(len(array))
        i2 = self.rng.randrange(len(array))
        while i2 == i1:
            i2 = self.rng.randrange(len(array))

        newArray = [0] * len(array)
        newArray[i2] = array[i1]

        if i1 < i2:
            i = 0
            count = 0
            while count < len(array):
                if i == i1:
                    count += 1
                if i == i2:
                    count -= 1
                    i += 1
                    continue
                newArray[i] = array[count]
                i += 1
                count += 1
        else:
            i = 0
            count = 0
            while count < len(array):
                if i == i2:
                    count -= 1
                    i += 1
                    continue
                newArray[i] = array[count]
                if i == i1:
                    count += 1
                i += 1
                count += 1

        Cmax = self.evaluatePermutation(newArray, self.probInstance)
        self.memory[targetIndex] = Solution(newArray, Cmax)

    def swapTwo(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        array = src.permutation.copy()
        i1 = self.rng.randrange(len(array))
        i2 = self.rng.randrange(len(array))

        # match Java's (slightly odd) swap implementation exactly
        array[i1] = src.permutation[i2]
        array[i2] = src.permutation[i1]

        Cmax = self.evaluatePermutation(array, self.probInstance)
        self.memory[targetIndex] = Solution(array, Cmax)

    def shuffle(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        array = src.permutation.copy()
        self.shufflePermutation(array)
        Cmax = self.evaluatePermutation(array, self.probInstance)
        self.memory[targetIndex] = Solution(array, Cmax)

    def useNEH(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None
        self.memory[targetIndex] = self.heuristics.neh(self.probInstance, src.permutation)

    def shuffleSubSequence(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        numbToShuffle = 2 + int(self.getIntensityOfMutation() * (self.probInstance.n - 2))

        AvailableIndices = [1] * self.probInstance.n
        count = self.probInstance.n
        IndicesToShuffle = [0] * numbToShuffle
        jobIndices = [0] * numbToShuffle

        permutation = src.permutation.copy()

        for i in range(numbToShuffle):
            randomInteger = self.rng.randrange(count)
            count -= 1
            count2 = 0
            for j in range(len(permutation)):
                if AvailableIndices[j] == 1:
                    if count2 == randomInteger:
                        IndicesToShuffle[i] = j
                        jobIndices[i] = permutation[j]
                        AvailableIndices[j] = -1
                        break
                    count2 += 1

        self.shufflePermutation(IndicesToShuffle)
        for i in range(numbToShuffle):
            permutation[IndicesToShuffle[i]] = jobIndices[i]

        Cmax = self.evaluatePermutation(permutation, self.probInstance)
        self.memory[targetIndex] = Solution(permutation, Cmax)

    # ---------------- RUIN-RECREATE HEURISTICS ----------------

    def iteratedGreedy(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        numbRemove = int(self.getIntensityOfMutation() * (self.probInstance.n - 1) * 0.5) + 1

        partialSequence = [0] * (self.probInstance.n - numbRemove)
        jobsToInsert = [0] * numbRemove
        lst = src.permutation.copy()

        for i in range(numbRemove):
            idx = self.rng.randrange(self.probInstance.n)
            while lst[idx] < 0:
                idx = self.rng.randrange(self.probInstance.n)
            jobsToInsert[i] = lst[idx]
            lst[idx] = -1

        j = 0
        for i in range(self.probInstance.n):
            if lst[i] > -1:
                partialSequence[j] = lst[i]
                j += 1

        permutation = self.heuristics.nehPartialSchedule(self.probInstance, partialSequence, jobsToInsert)
        Cmax = self.evaluatePermutation(permutation, self.probInstance)
        self.memory[targetIndex] = Solution(permutation, Cmax)

    def deepIteratedGreedy(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        numbRemove = int(self.getIntensityOfMutation() * (self.probInstance.n - 1) * 0.5) + 1
        depthOfSearch = int(self.getDepthOfSearch() * (numbRemove - 1)) + 1

        partialSequence = [0] * (self.probInstance.n - numbRemove)
        jobsToInsert = [0] * numbRemove
        lst = src.permutation.copy()

        for i in range(numbRemove):
            idx = self.rng.randrange(self.probInstance.n)
            while lst[idx] < 0:
                idx = self.rng.randrange(self.probInstance.n)
            jobsToInsert[i] = lst[idx]
            lst[idx] = -1

        j = 0
        for i in range(self.probInstance.n):
            if lst[i] > -1:
                partialSequence[j] = lst[i]
                j += 1

        permutation = self.heuristics.nehPartScheduleBT(
            self.probInstance, partialSequence, jobsToInsert, depthOfSearch
        )
        Cmax = self.evaluatePermutation(permutation, self.probInstance)
        self.memory[targetIndex] = Solution(permutation, Cmax)

    # ---------------- LOCAL SEARCH HEURISTICS ----------------

    def localSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        improved = self.heuristics.localSearch(
            self.probInstance, src.permutation, int(src.Cmax)
        )
        Cmax = self.evaluatePermutation(improved, self.probInstance)
        self.memory[targetIndex] = Solution(improved, Cmax)

    def fImpLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        improved = self.heuristics.fImpLocalSearch(
            self.probInstance, src.permutation, int(src.Cmax)
        )
        Cmax = self.evaluatePermutation(improved, self.probInstance)
        self.memory[targetIndex] = Solution(improved, Cmax)

    def randomLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        numbPasses = int(self.getDepthOfSearch() * (self.probInstance.n - 1)) + 1
        randPerm = self.generateRandomPermutation(self.probInstance.n)
        reduced = randPerm[:numbPasses]

        improved = self.heuristics.randomLocalSearch(
            self.probInstance, src.permutation, int(src.Cmax), reduced
        )
        Cmax = self.evaluatePermutation(improved, self.probInstance)
        self.memory[targetIndex] = Solution(improved, Cmax)

    def randomFImpLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        src = self.memory[sourceIndex]
        assert src is not None

        numbPasses = int(self.getDepthOfSearch() * (self.probInstance.n - 1)) + 1
        randPerm = self.generateRandomPermutation(self.probInstance.n)
        reduced = randPerm[:numbPasses]

        improved = self.heuristics.randomFImpLocalSearch(
            self.probInstance, src.permutation, int(src.Cmax), reduced
        )
        Cmax = self.evaluatePermutation(improved, self.probInstance)
        self.memory[targetIndex] = Solution(improved, Cmax)

    # ---------------- CROSSOVER HEURISTICS ----------------

    def ox(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        p1 = self.memory[sourceIndex1].permutation  # type: ignore[union-attr]
        p22 = self.memory[sourceIndex2].permutation  # type: ignore[union-attr]

        if len(p1) <= 1 or len(p22) <= 1 or len(p1) != len(p22):
            print(
                "Error in ox (order crossover) input permutation are not of the same length or one of them is of size <= 1"
            )
            raise SystemExit(0)

        p2 = p22.copy()

        n = len(p1)
        point1 = self.rng.randrange(n)
        point2 = self.rng.randrange(n)
        while point2 == point1:
            point2 = self.rng.randrange(n)
        if point2 < point1:
            point1, point2 = point2, point1
        pointsToCopy = point2 - point1

        inverseP2 = self.returnInversePermutation(p2)

        for i in range(pointsToCopy):
            job = p1[point1 + i]
            job_index = inverseP2[job]
            p2[job_index] = -1

        insertionPoint = inverseP2[p1[point1]]
        receiver = [0] * (n + pointsToCopy)

        for i in range(n + 1):
            if i < insertionPoint:
                receiver[i] = p2[i]
            if i == insertionPoint:
                for j in range(pointsToCopy):
                    receiver[i + j] = p1[point1 + j]
            if i > insertionPoint:
                receiver[i - 1 + pointsToCopy] = p2[i - 1]

        p3 = [0] * n
        counter = 0
        i = 0
        while i < n:
            job = receiver[counter]
            if job == -1:
                counter += 1
                continue
            p3[i] = job
            counter += 1
            i += 1

        Cmax = self.evaluatePermutation(p3, self.probInstance)
        self.memory[targetIndex] = Solution(p3, Cmax)

    def pmx(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        p1 = self.memory[sourceIndex1].permutation  # type: ignore[union-attr]
        p2 = self.memory[sourceIndex2].permutation  # type: ignore[union-attr]

        if len(p1) <= 1 or len(p2) <= 1 or len(p1) != len(p2):
            print(
                "Error in ox (order crossover) input permutation are not of the same length or one of them is of size <= 1"
            )
            raise SystemExit(0)

        n = len(p1)
        point1 = self.rng.randrange(n)
        point2 = self.rng.randrange(n)
        while point2 == point1:
            point2 = self.rng.randrange(n)
        if point2 < point1:
            point1, point2 = point2, point1
        pointsToCopy = point2 - point1

        p3 = [-1] * n
        jobsTaken = [0] * n

        for i in range(pointsToCopy):
            job = p1[point1 + i]
            p3[point1 + i] = job
            jobsTaken[job] = -1

        counter = 0
        for i in range(n):
            if p3[i] != -1:
                continue
            job = p2[counter]
            while jobsTaken[job] == -1:
                counter += 1
                job = p2[counter]
            counter += 1
            p3[i] = job

        Cmax = self.evaluatePermutation(p3, self.probInstance)
        self.memory[targetIndex] = Solution(p3, Cmax)

    def ppx(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        p12 = self.memory[sourceIndex1].permutation  # type: ignore[union-attr]
        p22 = self.memory[sourceIndex2].permutation  # type: ignore[union-attr]

        p1 = p12.copy()
        p2 = p22.copy()
        n = len(p1)

        inverseP1 = self.returnInversePermutation(p1)
        inverseP2 = self.returnInversePermutation(p2)

        counterP1 = 0
        counterP2 = 0
        p3 = [0] * n

        for i in range(n):
            randNumb = self.rng.randrange(2)
            if randNumb == 0:
                job = p1[counterP1]
                while job == -1:
                    counterP1 += 1
                    job = p1[counterP1]
                p3[i] = job
                p1[counterP1] = -1
                p2[inverseP2[job]] = -1
                counterP1 += 1
            else:
                job = p2[counterP2]
                while job == -1:
                    counterP2 += 1
                    job = p2[counterP2]
                p3[i] = job
                p2[counterP2] = -1
                p1[inverseP1[job]] = -1
                counterP2 += 1

        Cmax = self.evaluatePermutation(p3, self.probInstance)
        self.memory[targetIndex] = Solution(p3, Cmax)

    def oneX(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.probInstance is not None
        p1 = self.memory[sourceIndex1].permutation  # type: ignore[union-attr]
        p2 = self.memory[sourceIndex2].permutation  # type: ignore[union-attr]

        n = len(p1)
        xPoint = self.rng.randrange(n)

        inv = [0] * n
        p3 = [0] * n

        for i in range(xPoint):
            p3[i] = p1[i]
            inv[p1[i]] = 1

        count = xPoint
        for i in range(n):
            if inv[p2[i]] != 1:
                p3[count] = p2[i]
                count += 1

        Cmax = self.evaluatePermutation(p3, self.probInstance)
        self.memory[targetIndex] = Solution(p3, Cmax)

    # ---------------- UTILITY METHODS ----------------

    def shufflePermutation(self, array: List[int]) -> None:
        n = len(array)
        while n > 1:
            k = self.rng.randrange(n)
            n -= 1
            array[n], array[k] = array[k], array[n]

    def generateRandomPermutation(self, n: int) -> List[int]:
        randomPermutation = list(range(n))
        self.shufflePermutation(randomPermutation)
        return randomPermutation

    def verifyBestSolution(self, solution: Solution) -> None:
        if self.bestSoFar is None or solution.Cmax < self.bestSoFar.Cmax:
            self.bestSoFar = solution

    def evaluatePermutation(self, permutation: List[int], instance: Instance) -> int:
        processingTimes = instance.processingTimes
        n = instance.n
        m = instance.m

        releaseTimes = [0] * n

        # machine 1
        time_acc = 0
        for j in range(n):
            jobIndex = permutation[j]
            time_acc += processingTimes[jobIndex][0]
            releaseTimes[jobIndex] = time_acc

        # machines 2..m
        for i in range(1, m):
            time_acc = 0
            for j in range(n):
                jobIndex = permutation[j]
                releaseTime = releaseTimes[jobIndex]
                time_acc = (releaseTime if releaseTime >= time_acc else time_acc) + processingTimes[jobIndex][i]
                releaseTimes[jobIndex] = time_acc

        return time_acc

    def returnInversePermutation(self, p: List[int]) -> List[int]:
        inv = [0] * len(p)
        for i, job in enumerate(p):
            inv[job] = i
        return inv
