# package: travelingSalesmanProblem
# file: tsp_problem.py
#
# Pure conversion of the provided Java TSP class into Python.
# Assumptions (as you requested):
# - ProblemDomain exists (AbstractClasses.ProblemDomain) with the same API/fields:
#   - self.rng (random generator)
#   - self.depthOfSearch, self.intensityOfMutation
#   - self.heuristicCallRecord, self.heuristicCallTimeRecord
#   - methods: getIntensityOfMutation(), getDepthOfSearch() (or direct fields)
#   - HeuristicType enum-like with MUTATION/RUIN_RECREATE/LOCAL_SEARCH/CROSSOVER
# - TspInstance, TspSolution, TspBasicAlgorithms exist (you've been creating them)
# - Any remaining heuristics not included in your paste (if any) are not added.

from __future__ import annotations

import time
from typing import List, Optional

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType  # assumed to exist
from python_hyper_heuristic.domains.Python.TSP.TSPInstance import TspInstance
from python_hyper_heuristic.domains.Python.TSP.TSPBasicAlgorithms import TspBasicAlgorithms
from python_hyper_heuristic.domains.Python.TSP.TSPSolution import TspSolution


class TSP(ProblemDomain):
    # MEMBERS

    def __init__(self, seed: int):
        self.memory: List[Optional[TspSolution]] = [None, None]

        super().__init__(seed)

        self.instance: Optional[TspInstance] = None
        self.bestSoFar: Optional[TspSolution] = None
        self.algorithms: Optional[TspBasicAlgorithms] = None

    def applyHeuristicUnary(self, llhID: int, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        startTime = int(time.time() * 1000)

        isCrossover = False
        crossovers = self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        if crossovers is not None:
            for x in range(len(crossovers)):
                if crossovers[x] == llhID:
                    isCrossover = True
                    break

        if isCrossover:
            self.copySolution(solutionSourceIndex, solutionDestinationIndex)
        else:
            if llhID == 0:
                self.randomReinsertion(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 1:
                self.swapTwo(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 2:
                self.shuffle(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 3:
                self.shuffleSubSequence(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 4:
                self.nOptMove(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 5:
                self.iteratedGreedy(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 6:
                self.twoOptLocalSearch(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 7:
                self.bestImpTwoOptLocalSearch(solutionSourceIndex, solutionDestinationIndex)
            elif llhID == 8:
                self.threeOptLocalSearch(solutionSourceIndex, solutionDestinationIndex)
            else:
                raise RuntimeError(
                    "heuristic does not exist, or the crossover index array is not set up correctly"
                )

        self.heuristicCallRecord[llhID] += 1
        self.heuristicCallTimeRecord[llhID] += (int(time.time() * 1000) - startTime)

        dest = self.memory[solutionDestinationIndex]
        assert dest is not None
        self.verifyBestSolution(dest)

        assert self.algorithms is not None and self.instance is not None
        assert self.algorithms.verify_permutation(dest.permutation, self.instance.numbCities)

        return dest.cost

    def applyHeuristicCrossover(
        self,
        llhID: int,
        solutionSourceIndex1: int,
        solutionSourceIndex2: int,
        solutionDestinationIndex: int,
    ) -> float:
        startTime = int(time.time() * 1000)

        if llhID == 0:
            self.randomReinsertion(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 1:
            self.swapTwo(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 2:
            self.shuffle(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 3:
            self.shuffleSubSequence(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 4:
            self.nOptMove(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 5:
            self.iteratedGreedy(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 6:
            self.twoOptLocalSearch(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 7:
            self.bestImpTwoOptLocalSearch(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 8:
            self.threeOptLocalSearch(solutionSourceIndex1, solutionDestinationIndex)
        elif llhID == 9:
            self.ox(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
        elif llhID == 10:
            self.pmx(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
        elif llhID == 11:
            self.ppx(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
        elif llhID == 12:
            self.oneX(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
        else:
            raise RuntimeError(
                "heuristic does not exist, or the crossover index array is not set up correctly"
            )

        self.heuristicCallRecord[llhID] += 1
        self.heuristicCallTimeRecord[llhID] += (int(time.time() * 1000) - startTime)

        dest = self.memory[solutionDestinationIndex]
        assert dest is not None
        self.verifyBestSolution(dest)

        assert self.algorithms is not None and self.instance is not None
        assert self.algorithms.verify_permutation(dest.permutation, self.instance.numbCities)

        return dest.cost
    
    def applyHeuristic(
        self,
        llhID: int,
        solutionSourceIndex1: int,
        solutionSourceIndex2: int = None,
        solutionDestinationIndex: int = None,
    ) -> float:
        """
        HyFlex-compatible applyHeuristic.

        Supports:
        - Unary:     applyHeuristic(h, src, dst)
                    applyHeuristic(h, src, None, dst)
        - Crossover: applyHeuristic(h, src1, src2, dst)
        """

        # Called as applyHeuristic(h, src, dst)
        if solutionDestinationIndex is None:
            solutionDestinationIndex = solutionSourceIndex2
            solutionSourceIndex2 = None

        if solutionDestinationIndex is None:
            raise TypeError("applyHeuristic requires a destination index")

        crossovers = set(self.getHeuristicsOfType(HeuristicType.CROSSOVER) or [])
        is_crossover = llhID in crossovers

        if is_crossover:
            if solutionSourceIndex2 is None:
                # IMPORTANT: don't silently copy for crossover calls.
                raise TypeError(
                    f"Heuristic {llhID} is CROSSOVER but was called without a second parent"
                )
            return self.applyHeuristicCrossover(llhID, solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)

        return self.applyHeuristicUnary(llhID, solutionSourceIndex1, solutionDestinationIndex)

    def bestSolutionToString(self) -> str:
        assert self.bestSoFar is not None
        return str(self.bestSoFar)

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        assert self.instance is not None
        s1 = self.memory[solutionIndex1]
        s2 = self.memory[solutionIndex2]
        assert s1 is not None and s2 is not None
        for i in range(self.instance.numbCities):
            if s1.permutation[i] != s2.permutation[i]:
                return False
        return True

    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> None:
        src = self.memory[solutionSourceIndex]
        assert src is not None
        self.memory[solutionDestinationIndex] = src.clone()

    def getBestSolutionValue(self) -> float:
        assert self.bestSoFar is not None
        return self.bestSoFar.cost

    def getFunctionValue(self, solutionIndex: int) -> float:
        s = self.memory[solutionIndex]
        assert s is not None
        return s.cost

    def getHeuristicsOfType(self, heuristicType):
        if heuristicType == HeuristicType.MUTATION:
            return [0, 1, 2, 3, 4]
        if heuristicType == HeuristicType.RUIN_RECREATE:
            return [5]
        if heuristicType == HeuristicType.LOCAL_SEARCH:
            return [6, 7, 8]
        if heuristicType == HeuristicType.CROSSOVER:
            return [9, 10, 11, 12]
        return None

    def getHeuristicsThatUseDepthOfSearch(self):
        return [6, 7, 8]

    def getHeuristicsThatUseIntensityOfMutation(self):
        return [3, 4, 5]

    def getNumberOfHeuristics(self) -> int:
        return 13

    def getNumberOfInstances(self) -> int:
        return 10

    def initialiseSolution(self, i: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None
        startCity = self.rng.randrange(self.instance.numbCities)
        initialSolution = self.algorithms.greedy_heuristic(startCity)
        cost = self.algorithms.compute_cost(initialSolution)
        self.memory[i] = TspSolution(initialSolution, cost)
        self.verifyBestSolution(self.memory[i])

    def loadInstance(self, instanceID: int) -> None:
        self.instance = TspInstance(instanceID)
        self.algorithms = TspBasicAlgorithms(self.instance)

    def setMemorySize(self, size: int) -> None:
        tempMemory: List[Optional[TspSolution]] = [None] * size
        if self.memory is not None:
            if len(tempMemory) <= len(self.memory):
                for i in range(len(tempMemory)):
                    tempMemory[i] = self.memory[i]
            else:
                for i in range(len(self.memory)):
                    tempMemory[i] = self.memory[i]
        self.memory = tempMemory

    def solutionToString(self, solutionIndex: int) -> str:
        s = self.memory[solutionIndex]
        assert s is not None
        return str(s)

    def __str__(self) -> str:
        assert self.instance is not None
        return str(self.instance)

    # -----------------------
    # HEURISTICS
    # -----------------------

    # MUTATION HEURISTICS

    def randomReinsertion(self, sourceIndex: int, targetIndex: int) -> None:
        src = self.memory[sourceIndex]
        assert src is not None

        array = src.permutation.copy()
        n = len(array)
        if n < 2:
            self.memory[targetIndex] = src.clone()
            return

        i1 = self.rng.randrange(n)
        i2 = self.rng.randrange(n)
        while i2 == i1:
            i2 = self.rng.randrange(n)

        city = array.pop(i1)
        array.insert(i2, city)

        assert self.algorithms is not None
        cost = self.algorithms.compute_cost(array)
        self.memory[targetIndex] = TspSolution(array, cost)

    # def randomReinsertion(self, sourceIndex: int, targetIndex: int) -> None:
    #     array = self.memory[sourceIndex].permutation.copy()
    #     i1 = self.rng.randrange(len(array))
    #     i2 = self.rng.randrange(len(array))
    #     while i2 == i1:
    #         i2 = self.rng.randrange(len(array))

    #     newArray = [0] * len(array)
    #     newArray[i2] = array[i1]
    #     if i1 < i2:
    #         i = 0
    #         count = 0
    #         while count < len(array):
    #             if i == i1:
    #                 count += 1
    #             if i == i2:
    #                 count -= 1
    #                 i += 1
    #                 continue
    #             newArray[i] = array[count] # newArray[i]
    #             i += 1
    #             count += 1
    #     else:
    #         i = 0
    #         count = 0
    #         while count < len(array):
    #             if i == i2:
    #                 count -= 1
    #                 i += 1
    #                 continue
    #             newArray[i] = array[count]
    #             if i == i1:
    #                 count += 1
    #             i += 1
    #             count += 1

    #     assert self.algorithms is not None
    #     cost = self.algorithms.compute_cost(newArray)
    #     self.memory[targetIndex] = TspSolution(newArray, cost)

    def swapTwo(self, sourceIndex: int, targetIndex: int) -> None:
        array = self.memory[sourceIndex].permutation.copy()
        i1 = self.rng.randrange(len(array))
        i2 = self.rng.randrange(len(array))
        array[i1] = self.memory[sourceIndex].permutation[i2]
        array[i2] = self.memory[sourceIndex].permutation[i1]
        assert self.algorithms is not None
        cost = self.algorithms.compute_cost(array)
        self.memory[targetIndex] = TspSolution(array, cost)

    def shuffle(self, sourceIndex: int, targetIndex: int) -> None:
        array = self.memory[sourceIndex].permutation.copy()
        assert self.algorithms is not None
        self.algorithms.shuffle_permutation(array, self.rng)
        cost = self.algorithms.compute_cost(array)
        self.memory[targetIndex] = TspSolution(array, cost)

    def shuffleSubSequence(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        numbToShuffle = 2 + int(self.getIntensityOfMutation() * (self.instance.numbCities - 2))
        AvailableIndices = [1] * self.instance.numbCities
        count = self.instance.numbCities
        IndicesToShuffle = [0] * numbToShuffle
        jobIndices = [0] * numbToShuffle
        permutation = self.memory[sourceIndex].permutation.copy()

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

        self.algorithms.shuffle_permutation(IndicesToShuffle, self.rng)
        for i in range(numbToShuffle):
            permutation[IndicesToShuffle[i]] = jobIndices[i]

        cost = self.algorithms.compute_cost(permutation)
        self.memory[targetIndex] = TspSolution(permutation, cost)

    def nOptMove(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        N = 2
        if self.intensityOfMutation >= 0.25:
            N = 3
        if self.intensityOfMutation >= 0.5:
            N = 4
        if self.intensityOfMutation >= 0.75:
            N = 5

        tour = self.memory[sourceIndex].permutation
        newTour = tour.copy()
        for _i in range(N - 1):
            city1 = self.rng.randrange(self.instance.numbCities)
            city2 = self.rng.randrange(self.instance.numbCities)
            while city2 == city1:
                city2 = self.rng.randrange(self.instance.numbCities)
            newTour = TspBasicAlgorithms.flip(newTour, city1, city2)

        cost = self.algorithms.compute_cost(newTour)
        self.memory[targetIndex] = TspSolution(newTour, cost)

    # RUIN RECREATE HEURISTICS

    def iteratedGreedy(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        numbCities = self.instance.numbCities
        mut = self.getIntensityOfMutation()
        if mut < 0.2:
            mut = 0.1
        elif mut < 0.4:
            mut = 0.2
        elif mut < 0.6:
            mut = 0.3
        elif mut < 0.8:
            mut = 0.4
        else:
            mut = 0.5

        numbRemove = int(mut * (numbCities - 1)) + 1
        partialTour = [0] * (numbCities - numbRemove)
        citiesToInsert = [0] * numbRemove
        lst = self.memory[sourceIndex].permutation.copy()

        for i in range(numbRemove):
            id_ = self.rng.randrange(numbCities)
            while lst[id_] < 0:
                id_ = self.rng.randrange(numbCities)
            citiesToInsert[i] = lst[id_]
            lst[id_] = -1

        j = 0
        for i in range(numbCities):
            if lst[i] > -1:
                partialTour[j] = lst[i]
                j += 1

        cost = self.algorithms.compute_cost(partialTour)

        # In your Java code, greedyInsertion(partialTour, citiesToInsert, cost) returns int[]
        # In our earlier Python conversion, greedy_insertion_with_lists returns (tour, cost).
        # To keep this class runnable, we accept both forms.
        gi = self.algorithms.greedy_insertion_with_lists(partialTour, citiesToInsert, cost)
        if isinstance(gi, tuple):
            permutation = gi[0]
        else:
            permutation = gi

        cost = self.algorithms.compute_cost(permutation)
        self.memory[targetIndex] = TspSolution(permutation, cost)

    # LOCAL SEARCH HEURISTICS

    def twoOptLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        maxiterations = 10
        if self.depthOfSearch < 0.2:
            maxiterations = 10
        elif self.depthOfSearch < 0.4:
            maxiterations = 20
        elif self.depthOfSearch < 0.6:
            maxiterations = 30
        elif self.depthOfSearch < 0.8:
            maxiterations = 40
        else:
            maxiterations = 50

        improvedTour = self.algorithms.two_opt_first_improvement(
            self.memory[sourceIndex].permutation, self.instance, maxiterations
        )
        cost = self.algorithms.compute_cost(improvedTour)
        self.memory[targetIndex] = TspSolution(improvedTour, cost)

    def bestImpTwoOptLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        maxiterations = 10
        if self.depthOfSearch < 0.2:
            maxiterations = 10
        elif self.depthOfSearch < 0.4:
            maxiterations = 20
        elif self.depthOfSearch < 0.6:
            maxiterations = 30
        elif self.depthOfSearch < 0.8:
            maxiterations = 40
        else:
            maxiterations = 50

        improvedTour = self.algorithms.two_opt_best_improvement(
            self.memory[sourceIndex].permutation, self.instance, maxiterations
        )
        cost = self.algorithms.compute_cost(improvedTour)
        self.memory[targetIndex] = TspSolution(improvedTour, cost)

    def threeOptLocalSearch(self, sourceIndex: int, targetIndex: int) -> None:
        assert self.instance is not None
        assert self.algorithms is not None

        maxiterations = 10
        if self.depthOfSearch < 0.2:
            maxiterations = 10
        elif self.depthOfSearch < 0.4:
            maxiterations = 20
        elif self.depthOfSearch < 0.6:
            maxiterations = 30
        elif self.depthOfSearch < 0.8:
            maxiterations = 40
        else:
            maxiterations = 50

        improvedTour = self.algorithms.three_opt(
            self.memory[sourceIndex].permutation, self.instance, maxiterations
        )
        cost = self.algorithms.compute_cost(improvedTour)
        self.memory[targetIndex] = TspSolution(improvedTour, cost)

    # CROSSOVER HEURISTICS

    def ox(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.algorithms is not None

        p1 = self.memory[sourceIndex1].permutation
        p22 = self.memory[sourceIndex2].permutation
        if len(p1) <= 1 or len(p22) <= 1 or len(p1) != len(p22):
            raise RuntimeError(
                "Error in ox (order crossover) input permutation are not of the same length or one of them is of size <= 1"
            )
        p2 = p22.copy()

        # STEP 1: selecting two crossing points
        n = len(p1)
        point1 = self.rng.randrange(n)
        point2 = self.rng.randrange(n)
        while point2 == point1:
            point2 = self.rng.randrange(n)
        if point2 < point1:
            temp = point1
            point1 = point2
            point2 = temp
        pointsToCopy = point2 - point1

        # STEP 2: remove elements from p1 that lie between point1 and point2
        inverseP2 = self.algorithms.inverse_permutation(p2)
        for i in range(pointsToCopy):
            job = p1[point1 + i]
            job_index = inverseP2[job]
            p2[job_index] = -1

        # STEP 3: copy segment from p1 into receiver
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

        # STEP 4: remove repeated elements (-1) from receiver
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

        cost = self.algorithms.compute_cost(p3)
        self.memory[targetIndex] = TspSolution(p3, cost)

    def pmx(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.algorithms is not None

        p1 = self.memory[sourceIndex1].permutation
        p2 = self.memory[sourceIndex2].permutation
        if len(p1) <= 1 or len(p2) <= 1 or len(p1) != len(p2):
            raise RuntimeError(
                "Error in ox (order crossover) input permutation are not of the same length or one of them is of size <= 1"
            )

        # STEP 1: selecting two crossing points
        n = len(p1)
        point1 = self.rng.randrange(n)
        point2 = self.rng.randrange(n)
        while point2 == point1:
            point2 = self.rng.randrange(n)
        if point2 < point1:
            temp = point1
            point1 = point2
            point2 = temp
        pointsToCopy = point2 - point1

        # STEP 2: copy segment
        p3 = [-1] * n
        jobsTaken = [0] * n
        for i in range(pointsToCopy):
            job = p1[point1 + i]
            p3[point1 + i] = job
            jobsTaken[job] = -1

        # STEP 3: fill rest in order of p2
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

        cost = self.algorithms.compute_cost(p3)
        self.memory[targetIndex] = TspSolution(p3, cost)

    def ppx(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.algorithms is not None

        p12 = self.memory[sourceIndex1].permutation
        p22 = self.memory[sourceIndex2].permutation
        p1 = p12.copy()
        p2 = p22.copy()
        n = len(p1)
        inverseP1 = self.algorithms.inverse_permutation(p1)
        inverseP2 = self.algorithms.inverse_permutation(p2)

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

        Cost = self.algorithms.compute_cost(p3)
        self.memory[targetIndex] = TspSolution(p3, Cost)

    def oneX(self, sourceIndex1: int, sourceIndex2: int, targetIndex: int) -> None:
        assert self.algorithms is not None

        p1 = self.memory[sourceIndex1].permutation
        p2 = self.memory[sourceIndex2].permutation

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

        cost = self.algorithms.compute_cost(p3)
        self.memory[targetIndex] = TspSolution(p3, cost)

    def verifyBestSolution(self, solution: TspSolution) -> None:
        if self.bestSoFar is None or solution.cost < self.bestSoFar.cost:
            self.bestSoFar = solution
