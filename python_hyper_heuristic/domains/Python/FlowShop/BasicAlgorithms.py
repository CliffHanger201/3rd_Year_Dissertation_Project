# Fully converted from FlowShop.BasicAlgorithms (Java) to Python.
# Assumptions (per your instruction): Instance and Solution already exist (or will be created).
# Expected Instance fields:
#   - n: number of jobs
#   - m: number of machines
#   - processingTimes: 2D list/array, shape [n][m]
# Expected Solution constructor signature:
#   - Solution(permutation: list[int], cmax: int)

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import bisect

from python_hyper_heuristic.domains.Python.FlowShop.Instance import Instance
from python_hyper_heuristic.domains.Python.FlowShop.Solution import Solution


class BasicAlgorithms:
    def __init__(self) -> None:
        self.Cmax: int = 0

    # ---- Public API (as in Java) ----

    def neh(self, instance: "Instance", ordering: List[int] | None = None) -> "Solution":
        if not ordering:
            ordering = self.generateLPTSequence(instance)
        partial_schedule: List[int] = [ordering[0]]
        for j in range(1, instance.n):
            partial_schedule = self.insert(partial_schedule, ordering[j], instance)
        return Solution(partial_schedule, self.Cmax)

    def nehReturnCmax(self, instance: "Instance") -> int:
        self.neh(instance)
        return self.Cmax

    def nehBT(self, instance: "Instance", backTrackLevel: int) -> "Solution":
        initialPermutation = self.generateLPTSequence(instance)
        partialSchedule: List[int] = [initialPermutation[0]]
        for j in range(1, backTrackLevel):
            partialSchedule = self.insert(partialSchedule, initialPermutation[j], instance)

        jobsToInsert = list(initialPermutation[backTrackLevel:])
        schedule = self.nehPartScheduleBT(instance, partialSchedule, jobsToInsert, 3)
        return Solution(schedule, self.Cmax)

    def nehPartialSchedule(
        self, instance: "Instance", partialSchedule: List[int], jobsToInsert: List[int]
    ) -> List[int]:
        newSchedule = partialSchedule
        for job in jobsToInsert:
            newSchedule = self.insert(newSchedule, job, instance)
        return newSchedule

    def nehPartScheduleBT(
        self,
        problem: "Instance",
        partialSchedule: List[int],
        jobsToInsert: List[int],
        depthOfSearch: int,
    ) -> List[int]:
        # inserting first job
        initialPartialSchedules = self.insert_k(partialSchedule, jobsToInsert[0], problem, depthOfSearch)

        for i in range(1, len(jobsToInsert)):
            newPartialSchedules: List[BasicAlgorithms.PartialSchedule] = []

            for schedule in initialPartialSchedules:
                schedule.insertJob()

            jobToInsert = jobsToInsert[i]
            for j in range(depthOfSearch):
                newPartialSchedules.extend(
                    self.insert_k(
                        initialPartialSchedules[j].partialSchedule,
                        jobToInsert,
                        problem,
                        depthOfSearch,
                    )
                )

            newPartialSchedules.sort()
            initialPartialSchedules = newPartialSchedules[:depthOfSearch]

        initialPartialSchedules[0].insertJob()
        return initialPartialSchedules[0].partialSchedule

    def fImpLocalSearch(
        self, problem: "Instance", initialSchedule: List[int], initialCmax: int
    ) -> List[int]:
        bestCmax = initialCmax
        improvement = True
        bestSchedule = initialSchedule

        while improvement:
            improvement = False
            for i in range(len(initialSchedule)):
                newSchedule = self.fImpLocalSearchPass(problem, bestSchedule, i, bestCmax)
                newCmax = self.evaluatePermutation(newSchedule, problem)
                if newCmax < bestCmax:
                    bestSchedule = newSchedule
                    bestCmax = newCmax
                    improvement = True
                    break
        return bestSchedule

    def fImpLocalSearchPass(
        self,
        problem: "Instance",
        initialSchedule: List[int],
        indexToReinsert: int,
        initialCmax: int,
    ) -> List[int]:
        jobToInsert = initialSchedule[indexToReinsert]
        partialSchedule = self.removeIndex(initialSchedule, indexToReinsert)
        L = len(partialSchedule)

        e = self.calculate_e(L, problem.m, partialSchedule, problem.processingTimes)
        q = self.calculate_q(L, problem.m, partialSchedule, problem.processingTimes)
        f = self.calculate_f(jobToInsert, L, partialSchedule, e, problem.m, problem.processingTimes)

        for i in range(L + 1):
            newCmax = self.calculatePartialCmax(i, problem.m, q, f, L)
            if newCmax < initialCmax:
                return self.insertJob(partialSchedule, jobToInsert, i)
        return initialSchedule

    def localSearch(
        self, problem: "Instance", initialSchedule: List[int], initialCmax: int
    ) -> List[int]:
        bestCmax = initialCmax
        improvement = True
        bestSchedule = initialSchedule

        while improvement:
            improvement = False
            for i in range(len(initialSchedule)):
                newSchedule = self.localSearchPass(problem, bestSchedule, i, bestCmax)
                newCmax = self.evaluatePermutation(newSchedule, problem)
                if newCmax < bestCmax:
                    bestSchedule = newSchedule
                    bestCmax = newCmax
                    improvement = True
                    break
        return bestSchedule

    def localSearchPass(
        self,
        problem: "Instance",
        initialSchedule: List[int],
        indexToReinsert: int,
        initialCmax: int,
    ) -> List[int]:
        jobToInsert = initialSchedule[indexToReinsert]
        partialSchedule = self.removeIndex(initialSchedule, indexToReinsert)
        L = len(partialSchedule)

        e = self.calculate_e(L, problem.m, partialSchedule, problem.processingTimes)
        q = self.calculate_q(L, problem.m, partialSchedule, problem.processingTimes)
        f = self.calculate_f(jobToInsert, L, partialSchedule, e, problem.m, problem.processingTimes)

        minCmax = initialCmax
        position = 0
        foundImprovement = False

        for i in range(L + 1):
            newCmax = self.calculatePartialCmax(i, problem.m, q, f, L)
            if newCmax < minCmax:
                minCmax = newCmax
                position = i
                foundImprovement = True

        if foundImprovement:
            return self.insertJob(partialSchedule, jobToInsert, position)
        return initialSchedule

    def randomFImpLocalSearch(
        self,
        problem: "Instance",
        initialSchedule: List[int],
        initialCmax: int,
        jobIndices: List[int],
    ) -> List[int]:
        bestCmax = initialCmax
        bestSchedule = initialSchedule

        for idx in jobIndices:
            newSchedule = self.fImpLocalSearchPass(problem, bestSchedule, idx, bestCmax)
            newCmax = self.evaluatePermutation(newSchedule, problem)
            if newCmax <= bestCmax:
                bestSchedule = newSchedule
                bestCmax = newCmax
        return bestSchedule

    def randomLocalSearch(
        self,
        problem: "Instance",
        initialSchedule: List[int],
        initialCmax: int,
        jobIndices: List[int],
    ) -> List[int]:
        bestCmax = initialCmax
        bestSchedule = initialSchedule

        for idx in jobIndices:
            newSchedule = self.localSearchPass(problem, bestSchedule, idx, bestCmax)
            newCmax = self.evaluatePermutation(newSchedule, problem)
            if newCmax <= bestCmax:
                bestSchedule = newSchedule
                bestCmax = newCmax
        return bestSchedule

    # ---- Core NEH evaluation helpers ----

    def evaluatePermutation(self, permutation: List[int], instance: "Instance") -> int:
        processingTimes = instance.processingTimes
        n = instance.n
        m = instance.m

        releaseTimes = [0] * n

        # machine 1
        time = 0
        for j in range(n):
            jobIndex = permutation[j]
            time += processingTimes[jobIndex][0]
            releaseTimes[jobIndex] = time

        # machines 2..m
        for i in range(1, m):
            time = 0
            for j in range(n):
                jobIndex = permutation[j]
                releaseTime = releaseTimes[jobIndex]
                time = (releaseTime if releaseTime >= time else time) + processingTimes[jobIndex][i]
                releaseTimes[jobIndex] = time

        return time

    def calculate_e(
        self,
        partialScheduleLength: int,
        m: int,
        partialSchedule: List[int],
        processingTimes: List[List[int]],
    ) -> List[List[int]]:
        e = [[0] * m for _ in range(len(partialSchedule))]
        for j in range(partialScheduleLength):
            jobIndex = partialSchedule[j]
            for k in range(m):
                if j == 0 and k == 0:
                    e[j][k] = processingTimes[jobIndex][k]
                elif j == 0:
                    e[j][k] = e[j][k - 1] + processingTimes[jobIndex][k]
                elif k == 0:
                    e[j][k] = e[j - 1][k] + processingTimes[jobIndex][k]
                else:
                    e[j][k] = max(e[j][k - 1], e[j - 1][k]) + processingTimes[jobIndex][k]
        return e

    def calculate_f(
        self,
        jobIndex: int,
        partialScheduleLength: int,
        partialSchedule: List[int],
        e: List[List[int]],
        m: int,
        processingTimes: List[List[int]],
    ) -> List[List[int]]:
        f = [[0] * m for _ in range(len(partialSchedule) + 1)]
        for j in range(partialScheduleLength + 1):
            for k in range(m):
                if j == 0 and k == 0:
                    f[j][k] = processingTimes[jobIndex][k]
                elif j == 0:
                    f[j][k] = f[j][k - 1] + processingTimes[jobIndex][k]
                elif k == 0:
                    f[j][k] = e[j - 1][k] + processingTimes[jobIndex][k]
                else:
                    f[j][k] = max(f[j][k - 1], e[j - 1][k]) + processingTimes[jobIndex][k]
        return f

    def calculate_q(
        self,
        partialScheduleLength: int,
        m: int,
        partialSchedule: List[int],
        processingTimes: List[List[int]],
    ) -> List[List[int]]:
        q = [[0] * m for _ in range(partialScheduleLength)]
        for j in range(partialScheduleLength - 1, -1, -1):
            jobIndex = partialSchedule[j]
            for k in range(m - 1, -1, -1):
                if j == partialScheduleLength - 1 and k == m - 1:
                    q[j][k] = processingTimes[jobIndex][k]
                elif j == partialScheduleLength - 1:
                    q[j][k] = q[j][k + 1] + processingTimes[jobIndex][k]
                elif k == m - 1:
                    q[j][k] = q[j + 1][k] + processingTimes[jobIndex][k]
                else:
                    q[j][k] = max(q[j + 1][k], q[j][k + 1]) + processingTimes[jobIndex][k]
        return q

    def calculatePartialCmax(
        self,
        position: int,
        m: int,
        q: List[List[int]],
        f: List[List[int]],
        partialScheduleLength: int,
    ) -> int:
        M = 0
        if position == partialScheduleLength:
            M = f[position][m - 1]
        else:
            for k in range(m):
                M = max(M, f[position][k] + q[position][k])
        return M

    # ---- LPT sequence and insertion ----

    def generateLPTSequence(self, instance: "Instance") -> List[int]:
        m = instance.m
        n = instance.n
        processingTimes = instance.processingTimes

        sumProcTimes = [0] * n
        for i in range(n):
            s = 0
            for j in range(m):
                s += processingTimes[i][j]
            sumProcTimes[i] = s

        jobs = [self.Job(i, sumProcTimes[i]) for i in range(n)]
        jobs.sort()  # ascending by compareTo => descending sumproctimes
        return [job.index for job in jobs]

    def insert(self, partialSchedule: List[int], jobToInsert: int, instance: "Instance") -> List[int]:
        m = instance.m
        processingTimes = instance.processingTimes
        L = len(partialSchedule)

        e = self.calculate_e(L, m, partialSchedule, processingTimes)
        q = self.calculate_q(L, m, partialSchedule, processingTimes)
        f = self.calculate_f(jobToInsert, L, partialSchedule, e, m, processingTimes)

        minCmax = self.calculatePartialCmax(0, m, q, f, L)
        position = 0

        for i in range(1, L + 1):
            newCmax = self.calculatePartialCmax(i, m, q, f, L)
            if newCmax < minCmax:
                minCmax = newCmax
                position = i

        self.Cmax = minCmax
        return self.insertJob(partialSchedule, jobToInsert, position)

    def insert_k(
        self,
        partialSchedule: List[int],
        jobToInsert: int,
        problem: "Instance",
        depthOfSearch: int,
    ) -> List["BasicAlgorithms.PartialSchedule"]:
        L = len(partialSchedule)
        e = self.calculate_e(L, problem.m, partialSchedule, problem.processingTimes)
        q = self.calculate_q(L, problem.m, partialSchedule, problem.processingTimes)
        f = self.calculate_f(jobToInsert, L, partialSchedule, e, problem.m, problem.processingTimes)

        partialSchedules: List[BasicAlgorithms.PartialSchedule] = []
        for i in range(L + 1):
            newCmax = self.calculatePartialCmax(i, problem.m, q, f, L)
            partialSchedules.append(self.PartialSchedule(partialSchedule, jobToInsert, i, newCmax))

        partialSchedules.sort()
        return partialSchedules[:depthOfSearch]

    # ---- Small sequence utilities ----

    def insertJob(self, initialSequence: List[int], jobToInsert: int, placeToInsert: int) -> List[int]:
        newSequence = [0] * (len(initialSequence) + 1)
        counter = 0
        newSequence[placeToInsert] = jobToInsert
        for i in range(len(newSequence)):
            if i == placeToInsert:
                continue
            newSequence[i] = initialSequence[counter]
            counter += 1
        return newSequence

    def removeIndex(self, initialSequence: List[int], indexToRemove: int) -> List[int]:
        return [initialSequence[i] for i in range(len(initialSequence)) if i != indexToRemove]

    # ---- Inner classes ----

    @dataclass(order=True)
    class Job:
        # Java compareTo: return (-this.sumproctimes + o.sumproctimes)
        # => sorts descending by sumproctimes when using natural ascending sort.
        sort_key: int
        index: int

        def __init__(self, index: int, sumproctimes: int) -> None:
            self.index = index
            self.sort_key = -sumproctimes  # negative so ascending sort == descending sumproctimes

    @dataclass(order=True)
    class PartialSchedule:
        Cmax: int
        partialSchedule: List[int]
        jobToInsert: int
        placeToInsert: int

        def __init__(
            self,
            partialSchedule: List[int],
            jobToInsert: int,
            placeToInsert: int,
            Cmax: int,
        ) -> None:
            self.partialSchedule = partialSchedule
            self.jobToInsert = jobToInsert
            self.placeToInsert = placeToInsert
            self.Cmax = Cmax

        def insertJob(self) -> None:
            newSchedule = [0] * (len(self.partialSchedule) + 1)
            counter = 0
            newSchedule[self.placeToInsert] = self.jobToInsert
            for i in range(len(newSchedule)):
                if i == self.placeToInsert:
                    continue
                newSchedule[i] = self.partialSchedule[counter]
                counter += 1
            self.partialSchedule = newSchedule
