# package VRP

import math
import time
from typing import List, Optional

# Assume these exist (or will be created) elsewhere in your project:
# from AbstractClasses.ProblemDomain import ProblemDomain, HeuristicType
# from VRP.Instance import Instance
# from VRP.Solution import Solution
# from VRP.Route import Route
# from VRP.RouteItem import RouteItem
# from VRP.Location import Location

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType
from python_hyper_heuristic.domains.Python.VRP.Instance import Instance
from python_hyper_heuristic.domains.Python.VRP.Solution import Solution
from python_hyper_heuristic.domains.Python.VRP.Route import Route
from python_hyper_heuristic.domains.Python.VRP.RouteItem import RouteItem
from python_hyper_heuristic.domains.Python.VRP.Location import Location


class VRP(ProblemDomain):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.instance: Optional[Instance] = None
        self.solutions: Optional[List[Optional[Solution]]] = None

        self.bestSolutionValue: float = float("inf")
        self.bestSolution: Solution = Solution()

        self.mutations = [0, 1, 7]
        self.ruinRecreates = [2, 3]
        self.localSearches = [4, 8, 9]
        self.crossovers = [5, 6]

    # ---------------- ProblemDomain overrides ----------------

    def getHeuristicsOfType(self, heuristicType):
        if heuristicType == HeuristicType.LOCAL_SEARCH:
            return self.localSearches
        if heuristicType == HeuristicType.RUIN_RECREATE:
            return self.ruinRecreates
        if heuristicType == HeuristicType.MUTATION:
            return self.mutations
        if heuristicType == HeuristicType.CROSSOVER:
            return self.crossovers
        return None

    def getHeuristicsThatUseIntensityOfMutation(self):
        return [0, 1, 2, 3]

    def getHeuristicsThatUseDepthOfSearch(self):
        return [4, 8, 9]

    def loadInstance(self, instanceID: int):
        self.instance = Instance(instanceID)

    def setMemorySize(self, size: int):
        newSolutionMemory: List[Optional[Solution]] = [None] * size
        if self.solutions is not None:
            for x in range(len(self.solutions)):
                if x < size:
                    newSolutionMemory[x] = self.solutions[x]
        self.solutions = newSolutionMemory

    def initialiseSolution(self, index: int):
        if self.instance is None:
            raise RuntimeError("Instance not loaded. Call loadInstance() first.")
        if self.solutions is None:
            raise RuntimeError("Memory not set. Call setMemorySize() first.")
        self.solutions[index] = self.constructiveHeuristic(self.instance)
        self.getFunctionValue(index)

    def getNumberOfHeuristics(self) -> int:
        return 10

    def applyHeuristic(self, heuristicID: int, solutionSourceIndex: int,
                      solutionDestinationIndex: int) -> float:
        start_ms = int(time.time() * 1000)

        # detect crossover
        isCrossover = False
        crossovers = self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        for h in crossovers:
            if h == heuristicID:
                isCrossover = True
                break

        if self.solutions is None:
            raise RuntimeError("Memory not set. Call setMemorySize() first.")

        if isCrossover:
            self.solutions[solutionDestinationIndex] = self.solutions[solutionSourceIndex].copySolution()
            return self.getFunctionValue(solutionDestinationIndex)

        score = 0.0
        if heuristicID == 0:
            score = self.twoOpt(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 1:
            score = self.orOpt(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 2:
            score = self.locRR(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 3:
            score = self.timeRR(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 4:
            score = self.shift(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 7:
            score = self.shiftMutate(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 8:
            score = self.twoOptStar(solutionSourceIndex, solutionDestinationIndex)
        elif heuristicID == 9:
            score = self.GENI(solutionSourceIndex, solutionDestinationIndex)
        else:
            print(f"Heuristic {heuristicID} does not exist")
            return 0.0

        # records exist in ProblemDomain (assumed)
        self.heuristicCallRecord[heuristicID] += 1
        self.heuristicCallTimeRecord[heuristicID] += int(int(time.time() * 1000) - start_ms)
        return score

    def applyHeuristic4(self, heuristicID: int, solutionSourceIndex1: int,
                        solutionSourceIndex2: int, solutionDestinationIndex: int) -> float:
        """
        Java overload: applyHeuristic(int heuristicID, int src1, int src2, int dest)
        Named applyHeuristic4 to avoid clobbering the 3-arg version in Python.
        """
        start_ms = int(time.time() * 1000)

        # detect crossover
        isCrossover = False
        crossovers = self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        for h in crossovers:
            if h == heuristicID:
                isCrossover = True
                break

        if self.solutions is None:
            raise RuntimeError("Memory not set. Call setMemorySize() first.")

        if isCrossover:
            score = 0.0
            if heuristicID == 5:
                score = self.combine(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
            elif heuristicID == 6:
                score = self.combineLong(solutionSourceIndex1, solutionSourceIndex2, solutionDestinationIndex)
            else:
                print(f"Heuristic {heuristicID} does not exist")
                return 0.0

            self.heuristicCallRecord[heuristicID] += 1
            self.heuristicCallTimeRecord[heuristicID] += int(int(time.time() * 1000) - start_ms)
            return score

        score = 0.0
        if heuristicID == 0:
            score = self.twoOpt(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 1:
            score = self.orOpt(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 2:
            score = self.locRR(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 3:
            score = self.timeRR(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 4:
            score = self.shift(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 7:
            score = self.shiftMutate(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 8:
            score = self.twoOptStar(solutionSourceIndex1, solutionDestinationIndex)
        elif heuristicID == 9:
            score = self.GENI(solutionSourceIndex1, solutionDestinationIndex)
        else:
            print(f"Heuristic {heuristicID} does not exist")
            return 0.0

        self.heuristicCallRecord[heuristicID] += 1
        self.heuristicCallTimeRecord[heuristicID] += int(int(time.time() * 1000) - start_ms)
        return score

    def copySolution(self, solutionSourceIndex: int, solutionDestinationIndex: int):
        self.solutions[solutionDestinationIndex] = self.solutions[solutionSourceIndex].copySolution()

    def __str__(self):
        return "Vehicle Routing"

    def getNumberOfInstances(self) -> int:
        return 56

    def bestSolutionToString(self) -> str:
        return self.printToString(self.bestSolution)
    
    def getBestSolutionIndex(self) -> int:
        if self.best_ever_index is None:
            return -1
        return int(self.best_ever_index)

    def getBestSolutionValue(self) -> float:
        return self.bestSolutionValue

    def solutionToString(self, solutionIndex: int) -> str:
        return self.printToString(self.solutions[solutionIndex])

    def getFunctionValue(self, solutionIndex: int) -> float:
        routes = self.solutions[solutionIndex].getRoutes()
        value = self.calcFunction(routes)
        if value < self.bestSolutionValue:
            self.bestSolutionValue = value
            self.bestSolution = self.solutions[solutionIndex].copySolution()
        return value

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        rs1 = self.solutions[solutionIndex1].getRoutes()
        rs2 = self.solutions[solutionIndex2].getRoutes()
        for i in range(len(rs1)):
            if not rs1[i].compareRoute(rs2[i]):
                return False
        return True

    # ---------------- Core construction + utilities ----------------

    def constructiveHeuristic(self, i: Instance) -> Solution:
        locations: List[Location] = []
        tempLocs = i.getDemands()
        for j in range(len(tempLocs)):
            locations.append(tempLocs[j].copyLocation())

        routes: List[Route] = []
        depot = locations[0]
        locations.pop(0)

        numRoutes = 1
        route = Route(depot, (numRoutes - 1), 0)
        routes.append(route)

        while len(locations) > 0:
            bestIndex = -1
            bestScore = 1000000.0
            bestTimeMinusReady = 0.0

            for j in range(len(locations)):
                if (routes[numRoutes - 1].getVolume() + locations[j].getDemand()) < i.getVehicleCapacity():
                    diff1 = (
                        routes[numRoutes - 1].getLast().getCurrLocation().getDueDate()
                        - (
                            locations[j].getReadyTime()
                            + locations[j].getServiceTime()
                            + self.calcDistance(locations[j], routes[numRoutes - 1].getLast().getCurrLocation())
                        )
                    )
                    if diff1 > 0:
                        lastStop = routes[numRoutes - 1].getLast().getPrev()
                        prevLocation = lastStop.getCurrLocation()
                        dist = self.calcDistance(prevLocation, locations[j])
                        due = locations[j].getDueDate()
                        lastTime = lastStop.getTimeArrived()
                        timeDiff = due - (lastTime + prevLocation.getServiceTime() + dist)
                        if timeDiff >= 0:
                            readyDueDiff = due - locations[j].getReadyTime()
                            if diff1 >= (readyDueDiff - timeDiff):
                                score = (dist + (due - lastTime)) * self.rng.random()
                                if score < bestScore:
                                    bestIndex = j
                                    bestScore = score
                                    bestTimeMinusReady = timeDiff - readyDueDiff

                if j == (len(locations) - 1):
                    if bestIndex >= 0:
                        lastStop = routes[numRoutes - 1].getLast().getPrev()
                        prevLocation = lastStop.getCurrLocation()

                        if bestTimeMinusReady > 0:
                            lastStop.setWaitingTime(bestTimeMinusReady)
                        else:
                            lastStop.setWaitingTime(0)

                        routes[numRoutes - 1].addPenultimate(
                            locations[bestIndex],
                            lastStop.getTimeArrived()
                            + prevLocation.getServiceTime()
                            + lastStop.getWaitingTime()
                            + self.calcDistance(prevLocation, locations[bestIndex]),
                        )
                        locations.pop(bestIndex)
                        break

                    numRoutes += 1
                    route2 = Route(depot, (numRoutes - 1), 0)
                    routes.append(route2)

        return Solution(routes)

    def countLocs(self, rs: List[Route]) -> int:
        numLocs = 0
        for r in rs:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                numLocs += 1
        return numLocs

    def calcDistance(self, l1: Location, l2: Location) -> float:
        xdiff = abs(l1.getXCoord() - l2.getXCoord())
        ydiff = abs(l1.getYCoord() - l2.getYCoord())
        return math.sqrt((xdiff * xdiff) + (ydiff * ydiff))

    def printToString(self, s: Solution) -> str:
        rs = s.getRoutes()
        nL = "\n"
        printed = ""
        for i in range(len(rs)):
            printed += f"Route {i}{nL}"
            rI = rs[i].getFirst()
            while rI is not None:
                printed += f"Location{rI.getCurrLocation().getId()} visited at {rI.getTimeArrived()}{nL}"
                rI = rI.getNext()
        return printed

    def calcFunction(self, rs: List[Route]) -> float:
        routes = rs
        numRs = len(routes)
        distance = 0.0
        for r in routes:
            rItem = r.getFirst()
            while rItem.getNext() is not None:
                distance += self.calcDistance(rItem.getCurrLocation(), rItem.getNext().getCurrLocation())
                rItem = rItem.getNext()
        return (1000 * numRs) + distance

    # ---------------- Helpers mirroring Java ----------------

    def deleteUnwantedRoutes(self, rs: List[Route]) -> List[Route]:
        routesToDelete: List[Route] = []
        for r in rs:
            if r.sizeOfRoute() <= 2:
                routesToDelete.append(r)
        for r in routesToDelete:
            rs.remove(r)
        return rs

    def containsID(self, iD: int, ids: List[int]) -> bool:
        for i in ids:
            if i == iD:
                return True
        return False

    def useableRoute(self, r: Route, ls: List[int]) -> bool:
        ri = r.getFirst()
        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            for i in ls:
                if i == ri.getCurrLocation().getId():
                    return False
        return True

    def reOptimise(self, r: Route) -> Route:
        ri = r.getFirst()
        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            prev = ri.getPrev()
            diff = ri.getCurrLocation().getDueDate() - (
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )
            readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
            if diff > readyDueDiff:
                prev.setWaitingTime(diff - readyDueDiff)
            else:
                prev.setWaitingTime(0)
            ri.setTimeArrived(
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + prev.getWaitingTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )
        return r

    def appears(self, arr: List[int], ref: int) -> bool:
        for i in arr:
            if i == ref:
                return True
        return False

    def duplicates(self, rs: List[Route]) -> bool:
        ids: List[int] = []
        for r in rs:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                if ri.getCurrLocation().getId() in ids:
                    print(ri.getCurrLocation().getId())
                    return True
                ids.append(ri.getCurrLocation().getId())
        return False

    def traverseBack(self, s: Solution):
        rs = s.getRoutes()
        for r in rs:
            ri = r.getLast()
            while True:
                ri = ri.getPrev()
                if ri is None:
                    break
                print(ri.getCurrLocation().getId())

    # ---------------- Insertion + feasibility ----------------

    def insertLocIntoRoute(self, rs: List[Route], loc: Location) -> List[Route]:
        if self.instance is None:
            raise RuntimeError("Instance not loaded.")

        bestRouteNum = -1
        bestRouteElemPosition = 0
        bestWaitingTime = 1000000.0

        for i in range(len(rs)):
            routeElemPosition = 0
            ri = rs[i].getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                routeElemPosition += 1
                if self.checkFeasibility(rs[i], routeElemPosition, loc):
                    prev = ri.getPrev()
                    timeDiff = loc.getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(loc, prev.getCurrLocation())
                    )
                    readyDueDiff = loc.getDueDate() - loc.getReadyTime()
                    if timeDiff > readyDueDiff:
                        cand = (timeDiff - readyDueDiff + self.calcDistance(loc, prev.getCurrLocation()))
                        if bestWaitingTime > cand:
                            bestWaitingTime = cand
                            bestRouteNum = i
                            bestRouteElemPosition = routeElemPosition
                    else:
                        cand = self.calcDistance(loc, prev.getCurrLocation())
                        if bestWaitingTime > cand:
                            bestWaitingTime = cand
                            bestRouteNum = i
                            bestRouteElemPosition = routeElemPosition

        if bestRouteNum == -1:
            if len(rs) == 0:
                newR = Route(self.instance.getDepot(), 0, 0)
            else:
                newR = Route(rs[0].getFirst().getCurrLocation(), len(rs), 0)

            ri = newR.getFirst()
            timeDiff = loc.getDueDate() - (
                ri.getTimeArrived() + ri.getCurrLocation().getServiceTime() + self.calcDistance(loc, ri.getCurrLocation())
            )
            readyDueDiff = loc.getDueDate() - loc.getReadyTime()
            if timeDiff > readyDueDiff:
                ri.setWaitingTime(timeDiff - readyDueDiff)
            else:
                ri.setWaitingTime(0)

            newRI = RouteItem(
                loc,
                ri,
                ri.getNext(),
                ri.getTimeArrived()
                + ri.getCurrLocation().getServiceTime()
                + ri.getWaitingTime()
                + self.calcDistance(loc, ri.getCurrLocation()),
            )
            ri.getNext().setPrev(newRI)
            ri.setNext(newRI)
            rs.append(newR)
            return rs

        currR = rs[bestRouteNum]
        ri = currR.getFirst()
        for _ in range(bestRouteElemPosition):
            ri = ri.getNext()

        prev = ri.getPrev()
        timeDiff = loc.getDueDate() - (
            prev.getTimeArrived()
            + prev.getCurrLocation().getServiceTime()
            + self.calcDistance(loc, prev.getCurrLocation())
        )
        readyDueDiff = loc.getDueDate() - loc.getReadyTime()
        if timeDiff > readyDueDiff:
            prev.setWaitingTime(timeDiff - readyDueDiff)
        else:
            prev.setWaitingTime(0)

        newRI = RouteItem(
            loc,
            prev,
            prev.getNext(),
            prev.getTimeArrived()
            + prev.getCurrLocation().getServiceTime()
            + prev.getWaitingTime()
            + self.calcDistance(loc, prev.getCurrLocation()),
        )
        prev.getNext().setPrev(newRI)
        prev.setNext(newRI)

        ri = newRI
        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            prev = ri.getPrev()
            diff = ri.getCurrLocation().getDueDate() - (
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )
            readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
            if diff > readyDueDiff:
                prev.setWaitingTime(diff - readyDueDiff)
            else:
                prev.setWaitingTime(0)
            ri.setTimeArrived(
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + prev.getWaitingTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )

        return rs

    def checkFeasibility(self, r: Route, position: int, l: Location) -> bool:
        if self.instance is None:
            raise RuntimeError("Instance not loaded.")

        if r.calcVolume() + l.getDemand() > self.instance.getVehicleCapacity():
            return False

        route = r.copyRoute()
        ri = route.getFirst()
        for _ in range(position):
            ri = ri.getNext()

        prev = ri.getPrev()
        diff = l.getDueDate() - (
            prev.getTimeArrived()
            + prev.getCurrLocation().getServiceTime()
            + self.calcDistance(l, prev.getCurrLocation())
        )
        if diff < 0:
            return False

        readyDueDiff = l.getDueDate() - l.getReadyTime()
        if diff > readyDueDiff:
            prev.setWaitingTime(diff - readyDueDiff)
        else:
            prev.setWaitingTime(0)

        newRI = RouteItem(
            l,
            prev,
            prev.getNext(),
            prev.getTimeArrived()
            + prev.getCurrLocation().getServiceTime()
            + prev.getWaitingTime()
            + self.calcDistance(l, prev.getCurrLocation()),
        )
        prev.getNext().setPrev(newRI)
        prev.setNext(newRI)

        ri = newRI
        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            prev = ri.getPrev()
            diff = ri.getCurrLocation().getDueDate() - (
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )
            readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
            if diff < 0:
                return False
            if diff > readyDueDiff:
                prev.setWaitingTime(diff - readyDueDiff)
            else:
                prev.setWaitingTime(0)
            ri.setTimeArrived(
                prev.getTimeArrived()
                + prev.getCurrLocation().getServiceTime()
                + prev.getWaitingTime()
                + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
            )

        ri_last = route.getLast()
        if ri_last.getCurrLocation().getDueDate() < (
            ri_last.getPrev().getTimeArrived()
            + ri_last.getPrev().getCurrLocation().getServiceTime()
            + self.calcDistance(ri_last.getPrev().getCurrLocation(), ri_last.getCurrLocation())
        ):
            return False

        return True

    # ---------------- Heuristics converted from the provided Java ----------------
    # NOTE: These methods are long; this conversion keeps the original control flow and pointer edits.

    def twoOpt(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numRoutesToBeMutated = int(self.intensityOfMutation * len(rs))
        routesToBeMutated = [-1] * numRoutesToBeMutated

        for i in range(numRoutesToBeMutated):
            ref = self.rng.randrange(len(rs))
            while self.appears(routesToBeMutated, ref):
                ref = self.rng.randrange(len(rs))
            routesToBeMutated[i] = ref

        for i in range(numRoutesToBeMutated):
            routeToMutate = rs[routesToBeMutated[i]]
            if routeToMutate.sizeOfRoute() >= 4:
                if routeToMutate.sizeOfRoute() == 4:
                    startRI = 0
                else:
                    startRI = self.rng.randrange(routeToMutate.sizeOfRoute() - 4)

                r2 = routeToMutate.copyRoute()
                ri = r2.getFirst()
                for _ in range(startRI):
                    ri = ri.getNext()

                tRI = ri.getNext()
                ri.setNext(ri.getNext().getNext())
                ri.getNext().setPrev(ri)

                tRI.setNext(ri.getNext().getNext())
                tRI.setPrev(ri.getNext())
                ri.getNext().setNext(tRI)
                tRI.getNext().setPrev(tRI)

                feasible = True
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    if diff >= 0:
                        readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                        if diff > readyDueDiff:
                            prev.setWaitingTime(diff - readyDueDiff)
                        else:
                            prev.setWaitingTime(0)
                        ri.setTimeArrived(
                            prev.getTimeArrived()
                            + prev.getCurrLocation().getServiceTime()
                            + prev.getWaitingTime()
                            + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                        )
                    else:
                        feasible = False
                        break

                rL = r2.getLast()
                depotDiff = rL.getCurrLocation().getDueDate() - (
                    rL.getPrev().getTimeArrived()
                    + rL.getPrev().getCurrLocation().getServiceTime()
                    + self.calcDistance(rL.getCurrLocation(), rL.getPrev().getCurrLocation())
                )
                if depotDiff < 0:
                    feasible = False

                if feasible:
                    rs[routesToBeMutated[i]] = r2

        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def orOpt(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numRoutesToBeMutated = int(self.intensityOfMutation * len(rs))
        routesToBeMutated = [-1] * numRoutesToBeMutated

        for i in range(numRoutesToBeMutated):
            ref = self.rng.randrange(len(rs))
            while self.appears(routesToBeMutated, ref):
                ref = self.rng.randrange(len(rs))
            routesToBeMutated[i] = ref

        for i in range(numRoutesToBeMutated):
            routeToMutate = rs[routesToBeMutated[i]]
            if routeToMutate.sizeOfRoute() >= 6:
                if routeToMutate.sizeOfRoute() == 6:
                    startRI = 0
                else:
                    startRI = self.rng.randrange(routeToMutate.sizeOfRoute() - 6)

                r2 = routeToMutate.copyRoute()
                ri = r2.getFirst()
                for _ in range(startRI):
                    ri = ri.getNext()

                tRI = ri.getNext()
                tRI2 = ri

                ri.setNext(ri.getNext().getNext().getNext())
                ri.getNext().setPrev(ri)

                ri = ri.getNext().getNext()
                ri.getNext().setPrev(tRI.getNext())
                tRI.getNext().setNext(ri.getNext())
                tRI.setPrev(ri)
                ri.setNext(tRI)
                ri = tRI2

                feasible = True
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    if diff >= 0:
                        readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                        if diff > readyDueDiff:
                            prev.setWaitingTime(diff - readyDueDiff)
                        else:
                            prev.setWaitingTime(0)
                        ri.setTimeArrived(
                            prev.getTimeArrived()
                            + prev.getCurrLocation().getServiceTime()
                            + prev.getWaitingTime()
                            + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                        )
                    else:
                        feasible = False
                        break

                rL = r2.getLast()
                depotDiff = rL.getCurrLocation().getDueDate() - (
                    rL.getPrev().getTimeArrived()
                    + rL.getPrev().getCurrLocation().getServiceTime()
                    + self.calcDistance(rL.getCurrLocation(), rL.getPrev().getCurrLocation())
                )
                if depotDiff < 0:
                    feasible = False

                if feasible:
                    rs[routesToBeMutated[i]] = r2

        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def shiftMutate(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        copyS2 = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numRoutesToUse = int(self.intensityOfMutation * len(rs))
        if numRoutesToUse < 1:
            numRoutesToUse = 1

        routesToUse = [-1] * numRoutesToUse
        for i in range(numRoutesToUse):
            r = self.rng.randrange(numRoutesToUse)
            while self.appears(routesToUse, r):
                r = self.rng.randrange(numRoutesToUse)
            routesToUse[i] = rs[r].getId()

        for i in range(numRoutesToUse):
            routes = copyS2.getRoutes()
            there = False
            r = routes[0]
            for m in range(len(routes)):
                if routes[m].getId() == routesToUse[i]:
                    r = routes[m]
                    there = True
            if not there:
                continue

            bestPos = 1
            greatestDistance = 0.0
            pos = 0
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                pos += 1
                dist = (
                    self.calcDistance(ri.getPrev().getCurrLocation(), ri.getCurrLocation())
                    + self.calcDistance(ri.getNext().getCurrLocation(), ri.getCurrLocation())
                ) * self.rng.random()
                if dist > greatestDistance:
                    greatestDistance = dist
                    bestPos = pos

            ri = r.getFirst()
            for _ in range(bestPos):
                ri = ri.getNext()

            locToInsert = ri.getCurrLocation()
            ri.getPrev().setNext(ri.getNext())
            ri.getNext().setPrev(ri.getPrev())
            ri = ri.getPrev()

            if r.sizeOfRoute() <= 2:
                routes.remove(r)
            else:
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                    if diff > readyDueDiff:
                        prev.setWaitingTime(diff - readyDueDiff)
                    else:
                        prev.setWaitingTime(0)
                    ri.setTimeArrived(
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + prev.getWaitingTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )

            routes = self.insertLocIntoRoute(routes, locToInsert)
            copyS.setRoutes(routes)
            copyS2.setRoutes(routes)
            copyS = copyS.copySolution()

        copyS.setRoutes(self.deleteUnwantedRoutes(copyS.getRoutes()))
        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def locRR(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        rChoice = rs[self.rng.randrange(len(rs))]
        routePos = self.rng.randrange(rChoice.sizeOfRoute() - 1)
        rii = rChoice.getFirst()
        for _ in range(routePos):
            rii = rii.getNext()

        baseLocation = rii.getCurrLocation()

        furthest = 0.0
        for r in rs:
            rii = r.getFirst()
            while rii is not None:
                furthest = max(furthest, self.calcDistance(rii.getCurrLocation(), baseLocation))
                rii = rii.getNext()

        distanceWindow = self.intensityOfMutation * (3 * (furthest / 4))
        locs: List[Location] = []
        routesToDelete: List[Route] = []

        for r in rs:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                dist = self.calcDistance(ri.getCurrLocation(), baseLocation)
                if dist < distanceWindow:
                    locs.append(ri.getCurrLocation())
                    ri.getPrev().setNext(ri.getNext())
                    ri.getNext().setPrev(ri.getPrev())

            if r.sizeOfRoute() <= 2:
                routesToDelete.append(r)
            else:
                ri = r.getFirst()
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                    if diff > readyDueDiff:
                        prev.setWaitingTime(diff - readyDueDiff)
                    else:
                        prev.setWaitingTime(0)
                    ri.setTimeArrived(
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + prev.getWaitingTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )

        for r in routesToDelete:
            rs.remove(r)

        for l in locs:
            rs = self.insertLocIntoRoute(rs, l)

        copyS.setRoutes(self.deleteUnwantedRoutes(copyS.getRoutes()))
        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def timeRR(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        timeRef = self.rng.randrange(rs[0].getLast().getCurrLocation().getDueDate())
        timeWindow = self.intensityOfMutation * (rs[0].getLast().getCurrLocation().getDueDate() / 2)

        locs: List[Location] = []
        routesToDelete: List[Route] = []

        for r in rs:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                timeDiff = abs(ri.getTimeArrived() - timeRef)
                if timeDiff < timeWindow:
                    locs.append(ri.getCurrLocation())
                    ri.getPrev().setNext(ri.getNext())
                    ri.getNext().setPrev(ri.getPrev())

            if r.sizeOfRoute() <= 2:
                routesToDelete.append(r)
            else:
                ri = r.getFirst()
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                    if diff > readyDueDiff:
                        prev.setWaitingTime(diff - readyDueDiff)
                    else:
                        prev.setWaitingTime(0)
                    ri.setTimeArrived(
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + prev.getWaitingTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )

        for r in routesToDelete:
            rs.remove(r)

        for l in locs:
            rs = self.insertLocIntoRoute(rs, l)

        copyS.setRoutes(self.deleteUnwantedRoutes(copyS.getRoutes()))
        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def shift(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        copyS2 = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numRoutesToUse = int(self.depthOfSearch * len(rs))
        if numRoutesToUse < 1:
            numRoutesToUse = 1

        routesToUse = [-1] * numRoutesToUse
        for i in range(numRoutesToUse):
            r = self.rng.randrange(numRoutesToUse)
            while self.appears(routesToUse, r):
                r = self.rng.randrange(numRoutesToUse)
            routesToUse[i] = rs[r].getId()

        for i in range(numRoutesToUse):
            routes = copyS2.getRoutes()
            there = False
            r = routes[0]
            for m in range(len(routes)):
                if routes[m].getId() == routesToUse[i]:
                    r = routes[m]
                    there = True
            if not there:
                continue

            firstFunc = self.calcFunction(routes)

            bestPos = 1
            greatestDistance = 0.0
            pos = 0
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                pos += 1
                dist = (
                    self.calcDistance(ri.getPrev().getCurrLocation(), ri.getCurrLocation())
                    + self.calcDistance(ri.getNext().getCurrLocation(), ri.getCurrLocation())
                ) * self.rng.random()
                if dist > greatestDistance:
                    greatestDistance = dist
                    bestPos = pos

            ri = r.getFirst()
            for _ in range(bestPos):
                ri = ri.getNext()

            locToInsert = ri.getCurrLocation()
            ri.getPrev().setNext(ri.getNext())
            ri.getNext().setPrev(ri.getPrev())
            ri = ri.getPrev()

            if r.sizeOfRoute() <= 2:
                routes.remove(r)
            else:
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    prev = ri.getPrev()
                    diff = ri.getCurrLocation().getDueDate() - (
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )
                    readyDueDiff = ri.getCurrLocation().getDueDate() - ri.getCurrLocation().getReadyTime()
                    if diff > readyDueDiff:
                        prev.setWaitingTime(diff - readyDueDiff)
                    else:
                        prev.setWaitingTime(0)
                    ri.setTimeArrived(
                        prev.getTimeArrived()
                        + prev.getCurrLocation().getServiceTime()
                        + prev.getWaitingTime()
                        + self.calcDistance(ri.getCurrLocation(), prev.getCurrLocation())
                    )

            routes = self.insertLocIntoRoute(routes, locToInsert)
            newFunc = self.calcFunction(routes)

            if newFunc >= firstFunc:
                copyS2.setRoutes(copyS.copySolution().getRoutes())
            else:
                copyS.setRoutes(routes)
                copyS2.setRoutes(routes)
                copyS = copyS.copySolution()

        copyS.setRoutes(self.deleteUnwantedRoutes(copyS.getRoutes()))
        self.solutions[solutionDestinationIndex] = copyS
        return self.getFunctionValue(solutionDestinationIndex)

    def interchange(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        copyS2 = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        # Get two routes for interchange
        r1 = rs[self.rng.randrange(len(rs))]
        while r1.sizeOfRoute() <= 2:
            r1 = rs[self.rng.randrange(len(rs))]

        r2 = rs[self.rng.randrange(len(rs))]
        # Ensure routes are different
        while (r1 is r2) or (r2.sizeOfRoute() <= 2):
            r2 = rs[self.rng.randrange(len(rs))]

        # Get two locations to interchange based upon a time and distance matrix
        ri = r1.getFirst()
        tdScore = 1000000.0
        ri1 = r1.getFirst().getNext()
        ri2 = r2.getFirst().getNext()

        for _ in range(r1.sizeOfRoute() - 2):
            ri = ri.getNext()
            rij = r2.getFirst()
            for _ in range(r2.sizeOfRoute() - 2):
                rij = rij.getNext()
                matrix = self.calcDistance(ri.getCurrLocation(), rij.getCurrLocation()) + abs(
                    ri.getCurrLocation().getDueDate() - rij.getCurrLocation().getDueDate()
                )
                if matrix < tdScore:
                    tdScore = matrix
                    ri1 = ri
                    ri2 = rij

        # Remove locations and fix pointers
        loc1 = ri1.getCurrLocation()
        loc2 = ri2.getCurrLocation()

        ri1.getPrev().setNext(ri1.getNext())
        ri1.getNext().setPrev(ri1.getPrev())

        ri2.getPrev().setNext(ri2.getNext())
        ri2.getNext().setPrev(ri2.getPrev())

        r1 = self.reOptimise(r1)
        r2 = self.reOptimise(r2)

        # Insert location 1 in the best possible place in route 2, or create new route if necessary
        score = 1000000.0
        bestPos = -1
        ri = r2.getFirst()
        routeElemPosition = 0

        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            routeElemPosition += 1
            if self.checkFeasibility(r2, routeElemPosition, loc1):
                matrix = self.calcDistance(ri.getPrev().getCurrLocation(), loc1) + abs(
                    ri.getPrev().getCurrLocation().getDueDate() - loc1.getDueDate()
                )
                if matrix < score:
                    score = matrix
                    bestPos = routeElemPosition

        if bestPos != -1:
            ri = r2.getFirst()
            for _ in range(routeElemPosition):
                ri = ri.getNext()
            newRI = RouteItem(loc1, ri.getPrev(), ri, 0)
            ri.getPrev().setNext(newRI)
            ri.setPrev(newRI)
            r2 = self.reOptimise(r2)
        else:
            newR = Route(r2.getFirst().getCurrLocation(), len(rs), 0)
            ri = newR.getFirst()
            newRI = RouteItem(loc1, ri, ri.getNext(), 0)
            ri.getNext().setPrev(newRI)
            ri.setNext(newRI)
            newR = self.reOptimise(newR)
            rs.append(newR)

        # Insert location 2 in the best possible place in route 1, or create new route if necessary
        score = 1000000.0
        bestPos = -1
        ri = r1.getFirst()
        routeElemPosition = 0

        while True:
            ri = ri.getNext()
            if ri.getNext() is None:
                break
            routeElemPosition += 1
            if self.checkFeasibility(r1, routeElemPosition, loc2):
                matrix = self.calcDistance(ri.getPrev().getCurrLocation(), loc2) + abs(
                    ri.getPrev().getCurrLocation().getDueDate() - loc2.getDueDate()
                )
                if matrix < score:
                    score = matrix
                    bestPos = routeElemPosition

        if bestPos != -1:
            ri = r1.getFirst()
            for _ in range(routeElemPosition):
                ri = ri.getNext()
            newRI = RouteItem(loc2, ri.getPrev(), ri, 0)
            ri.getPrev().setNext(newRI)
            ri.setPrev(newRI)
            r1 = self.reOptimise(r1)
        else:
            newR = Route(r1.getFirst().getCurrLocation(), len(rs), 0)
            ri = newR.getFirst()
            # NOTE: This matches your Java exactly (it inserts loc1 here, not loc2)
            newRI = RouteItem(loc1, ri, ri.getNext(), 0)
            ri.getNext().setPrev(newRI)
            ri.setNext(newRI)
            newR = self.reOptimise(newR)
            rs.append(newR)

        oldFunc = self.calcFunction(copyS2.getRoutes())
        copyS.setRoutes(self.deleteUnwantedRoutes(copyS.getRoutes()))
        newFunc = self.calcFunction(copyS.getRoutes())

        if newFunc < oldFunc:
            self.solutions[solutionDestinationIndex] = copyS
            if newFunc < self.bestSolutionValue:
                self.bestSolutionValue = newFunc
                self.bestSolution = copyS.copySolution()
            return newFunc
        else:
            self.solutions[solutionDestinationIndex] = copyS2
            return oldFunc


    def twoOptStar(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numTimesToPerform = 1
        if int(self.depthOfSearch * len(rs)) != 0:
            numTimesToPerform = self.rng.randrange(int(self.depthOfSearch * len(rs)))

        if numTimesToPerform == 0:
            numTimesToPerform += 1

        for _ in range(numTimesToPerform):
            copyS2 = copyS.copySolution()
            routes = copyS2.getRoutes()

            r1 = routes[self.rng.randrange(len(routes))]
            r2 = routes[self.rng.randrange(len(routes))]
            while r1 is r2:
                r2 = routes[self.rng.randrange(len(routes))]

            bestScore = float("inf")
            bestR1Pos = -1
            bestR2Pos = -1

            currR1Pos = 0
            ri = r1.getFirst()
            while True:
                ri = ri.getNext()
                if ri is None:
                    break
                currR1Pos += 1

                currR2Pos = 0
                ri2 = r2.getFirst()
                while True:
                    ri2 = ri2.getNext()
                    if ri2 is None:
                        break
                    currR2Pos += 1

                    score = self.feasibilityAndScore(r1, r2, currR1Pos, currR2Pos)
                    if score <= bestScore:
                        bestR1Pos = currR1Pos
                        bestR2Pos = currR2Pos
                        bestScore = score

            ri = r1.getFirst()
            ri2 = r2.getFirst()
            for _ in range(bestR1Pos):
                ri = ri.getNext()
            for _ in range(bestR2Pos):
                ri2 = ri2.getNext()

            # Fix pointers
            ri.getPrev().setNext(ri2)
            ri2.getPrev().setNext(ri)
            riTemp = ri.getPrev()
            ri.setPrev(ri2.getPrev())
            ri2.setPrev(riTemp)

            # Optimise
            r1 = self.reOptimise(r1)
            r2 = self.reOptimise(r2)

            routes = self.deleteUnwantedRoutes(routes)

            oldFunc = self.calcFunction(copyS.getRoutes())
            newFunc = self.calcFunction(routes)
            if newFunc <= oldFunc:
                copyS.setRoutes(routes)

        self.solutions[solutionDestinationIndex] = copyS.copySolution()
        func = self.getFunctionValue(solutionDestinationIndex)
        return func


    def feasibilityAndScore(self, r1, r2, r1Pos: int, r2Pos: int) -> float:
        r11 = r1.copyRoute()
        r22 = r2.copyRoute()

        ri = r11.getFirst()
        ri2 = r22.getFirst()

        for _ in range(r1Pos):
            ri = ri.getNext()
        for _ in range(r2Pos):
            ri2 = ri2.getNext()

        # Fix pointers
        ri.getPrev().setNext(ri2)
        ri2.getPrev().setNext(ri)
        riTemp = ri.getPrev()
        ri.setPrev(ri2.getPrev())
        ri2.setPrev(riTemp)

        # Optimise
        r11 = self.reOptimise(r11)
        r22 = self.reOptimise(r22)

        # Check feasibility and calculate score
        volume = 0.0
        score = 0.0

        ri = r11.getFirst()
        ri2 = r22.getFirst()

        while True:
            ri = ri.getNext()
            if ri is None:
                break
            volume += ri.getCurrLocation().getDemand()
            if ri.getTimeArrived() > ri.getCurrLocation().getDueDate():
                return float("inf")
            score += self.calcDistance(ri.getPrev().getCurrLocation(), ri.getCurrLocation())

        if self.instance is not None and volume > self.instance.getVehicleCapacity():
            return float("inf")

        volume = 0.0
        while True:
            ri2 = ri2.getNext()
            if ri2 is None:
                break
            volume += ri2.getCurrLocation().getDemand()
            if ri2.getTimeArrived() > ri2.getCurrLocation().getDueDate():
                return float("inf")
            score += self.calcDistance(ri2.getPrev().getCurrLocation(), ri2.getCurrLocation())

        if self.instance is not None and volume > self.instance.getVehicleCapacity():
            return float("inf")

        if (r11.sizeOfRoute() <= 2) or (r22.sizeOfRoute() <= 2):
            score -= 1000

        return score


    def GENI(self, solutionSourceIndex: int, solutionDestinationIndex: int) -> float:
        copyS = self.solutions[solutionSourceIndex].copySolution()
        rs = copyS.getRoutes()

        numTimesToPerform = 1
        if int(self.depthOfSearch * len(rs)) != 0:
            numTimesToPerform = self.rng.randrange(int(self.depthOfSearch * len(rs)))

        if numTimesToPerform == 0:
            numTimesToPerform += 1

        for _ in range(numTimesToPerform):
            copyS2 = copyS.copySolution()
            routes = copyS2.getRoutes()

            r1 = routes[self.rng.randrange(len(routes))]
            while r1.sizeOfRoute() <= 3:
                r1 = routes[self.rng.randrange(len(routes))]

            r2 = routes[self.rng.randrange(len(routes))]
            while (r1 is r2) or (r2.sizeOfRoute() <= 3):
                r2 = routes[self.rng.randrange(len(routes))]

            # First, pick location most "out of place" in route 1
            worstRouteItem = r1.getFirst()
            worstScore = 0.0
            ri = r1.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                tempScore = (
                    (ri.getCurrLocation().getReadyTime() - ri.getPrev().getCurrLocation().getReadyTime())
                    + (ri.getNext().getCurrLocation().getDueDate() - ri.getCurrLocation().getDueDate())
                    + self.calcDistance(ri.getCurrLocation(), ri.getPrev().getCurrLocation())
                    + self.calcDistance(ri.getCurrLocation(), ri.getNext().getCurrLocation())
                )
                if tempScore > worstScore:
                    worstScore = tempScore
                    worstRouteItem = ri

            # Remove and fix pointers
            worstRouteItem.getPrev().setNext(worstRouteItem.getNext())
            worstRouteItem.getNext().setPrev(worstRouteItem.getPrev())

            # Find two closest locations
            firstClosestScore = float("inf")
            secondClosestScore = float("inf")
            firstRI = r2.getFirst()
            secondRI = r2.getFirst()

            ri2 = r2.getFirst()
            while True:
                ri2 = ri2.getNext()
                if ri2.getNext() is None:
                    break
                score = abs(worstRouteItem.getCurrLocation().getDueDate() - ri2.getCurrLocation().getDueDate()) + \
                        self.calcDistance(worstRouteItem.getCurrLocation(), ri2.getCurrLocation())

                if score < firstClosestScore:
                    secondClosestScore = firstClosestScore
                    secondRI = firstRI
                    firstClosestScore = score
                    firstRI = ri2
                elif score < secondClosestScore:
                    secondClosestScore = score
                    secondRI = ri2

            # Fix pointers
            earlyRI = firstRI
            lateRI = secondRI
            if secondRI.getCurrLocation().getDueDate() < firstRI.getCurrLocation().getDueDate():
                earlyRI = secondRI
                lateRI = firstRI

            lateRI.getNext().setPrev(lateRI.getPrev())
            lateRI.getPrev().setNext(lateRI.getNext())

            lateRI.setNext(earlyRI.getNext())
            earlyRI.getNext().setPrev(lateRI)

            lateRI.setPrev(worstRouteItem)
            worstRouteItem.setNext(lateRI)

            earlyRI.setNext(worstRouteItem)
            worstRouteItem.setPrev(earlyRI)

            # Reoptimise
            r1 = self.reOptimise(r1)
            r2 = self.reOptimise(r2)

            # Test feasibility
            feasible = True
            volume = 0.0
            ri = r2.getFirst()
            while True:
                ri = ri.getNext()
                if ri is None:
                    break
                volume += ri.getCurrLocation().getDemand()
                if ri.getTimeArrived() > ri.getCurrLocation().getDueDate():
                    feasible = False

            if self.instance is not None and volume > self.instance.getVehicleCapacity():
                feasible = False

            if feasible:
                oldFunc = self.calcFunction(copyS.getRoutes())
                newFunc = self.calcFunction(routes)
                if newFunc <= oldFunc:
                    copyS.setRoutes(routes)

        self.solutions[solutionDestinationIndex] = copyS.copySolution()
        func = self.getFunctionValue(solutionDestinationIndex)
        return func


    def combine(self, solutionSourceIndex1: int, solutionSourceIndex2: int, solutionDestinationIndex: int) -> float:
        # Get copies of source solutions and routes
        copyS1 = self.solutions[solutionSourceIndex1].copySolution()
        copyS2 = self.solutions[solutionSourceIndex2].copySolution()
        rs1 = copyS1.getRoutes()
        rs2 = copyS2.getRoutes()

        # List of all locations, for reference
        locations: List[int] = []
        for r in rs1:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                locations.append(ri.getCurrLocation().getId())

        # Choose rough percentage of routes to keep from chosen solution (between 25-75%)
        rand = self.rng.randrange(50)
        perc = (75 - rand) / 100.0

        # Choose which solution to take routes from
        rand = self.rng.randrange(2)
        if rand == 0:
            chosenOnes = rs1
            others = rs2
        else:
            chosenOnes = rs2
            others = rs1

        # New solution
        newRoutes: List[Route] = []
        addedLocations: List[int] = []

        # Based on percentage above, choose some routes to be included in the new solution
        for r in chosenOnes:
            if self.rng.random() < perc:
                newRoutes.append(r)
                ri = r.getFirst()
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    addedLocations.append(ri.getCurrLocation().getId())

        # Add any routes from the other routes that don't have any conflicting locations
        for r in others:
            if self.useableRoute(r, addedLocations):
                newRoutes.append(r)
                ri = r.getFirst()
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    addedLocations.append(ri.getCurrLocation().getId())

        # Insert remaining locations into solution in the order they appear in others
        for r in others:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                if not self.containsID(ri.getCurrLocation().getId(), addedLocations):
                    addedLocations.append(ri.getCurrLocation().getId())
                    newRoutes = self.insertLocIntoRoute(newRoutes, ri.getCurrLocation())

        # Copy new solutions to destination place
        newSolution = Solution(newRoutes)
        self.solutions[solutionDestinationIndex] = newSolution
        func = self.getFunctionValue(solutionDestinationIndex)
        return func

    def combineLong(self, solutionSourceIndex1: int, solutionSourceIndex2: int, solutionDestinationIndex: int) -> float:
        # Get copies of source solutions and routes
        copyS1 = self.solutions[solutionSourceIndex1].copySolution()
        copyS2 = self.solutions[solutionSourceIndex2].copySolution()
        rs1 = copyS1.getRoutes()
        rs2 = copyS2.getRoutes()

        # List of locations added to new solution
        addedLocations: list[int] = []

        # New list of routes
        newRoutes: list["Route"] = []

        # Ordered list of routes by length
        orderedRoutes: list["Route"] = []

        # Insert routes from first set (keeping orderedRoutes sorted by descending sizeOfRoute())
        for r in rs1:
            for i in range(len(orderedRoutes)):
                if r.sizeOfRoute() > orderedRoutes[i].sizeOfRoute():
                    orderedRoutes.insert(i, r)
                    break
                if i == (len(orderedRoutes) - 1):
                    orderedRoutes.append(r)
                    break
            if len(orderedRoutes) == 0:
                orderedRoutes.append(r)

        # Insert routes from second set
        for r in rs2:
            for i in range(len(orderedRoutes)):
                if r.sizeOfRoute() > orderedRoutes[i].sizeOfRoute():
                    orderedRoutes.insert(i, r)
                    break
                if i == (len(orderedRoutes) - 1):
                    orderedRoutes.append(r)
                    break
            if len(orderedRoutes) == 0:
                orderedRoutes.append(r)

        # Pick non-conflicting routes to include in solution, searching from largest to smallest
        for r in orderedRoutes:
            if self.useableRoute(r, addedLocations):
                newRoutes.append(r)
                ri = r.getFirst()
                while True:
                    ri = ri.getNext()
                    if ri.getNext() is None:
                        break
                    addedLocations.append(ri.getCurrLocation().getId())

        # Pick route set at random to get remaining location order from
        choose = self.rng.randrange(2)
        remainingLocsRoutes = rs1
        if choose == 0:
            remainingLocsRoutes = rs2

        # Insert remaining locations into new routes
        for r in remainingLocsRoutes:
            ri = r.getFirst()
            while True:
                ri = ri.getNext()
                if ri.getNext() is None:
                    break
                if not self.containsID(ri.getCurrLocation().getId(), addedLocations):
                    addedLocations.append(ri.getCurrLocation().getId())
                    newRoutes = self.insertLocIntoRoute(newRoutes, ri.getCurrLocation())

        # Copy new solutions to destination place
        newSolution = Solution(newRoutes)
        self.solutions[solutionDestinationIndex] = newSolution
        func = self.getFunctionValue(solutionDestinationIndex)
        return func

    @staticmethod
    def main(args=None) -> None:
        vrp = VRP(4234)
        vrp.loadInstance(0)
        vrp.setMemorySize(2)
        vrp.initialiseSolution(0)
        print(vrp.duplicates(vrp.solutions[0].getRoutes()))
        print(vrp.calcFunction(vrp.solutions[0].getRoutes()))
        vrp.setIntensityOfMutation(1)
        vrp.setDepthOfSearch(1)

        for _ in range(1):
            print(vrp.twoOptStar(0, 1))
            print(vrp.duplicates(vrp.solutions[0].getRoutes()))
            print(vrp.duplicates(vrp.solutions[1].getRoutes()))
            print(vrp.shift(1, 0))
            print(vrp.interchange(0, 1))
            print(vrp.GENI(1, 0))

        for j in range(1):
            if j % 2 == 0:
                print(vrp.locRR(0, 0))
            else:
                print(vrp.timeRR(0, 0))

            for _ in range(1):
                print(vrp.twoOptStar(0, 1))
                print(vrp.shift(1, 0))
                print(vrp.interchange(0, 1))
                print(vrp.GENI(1, 0))

        print(vrp.bestSolutionValue)
        print(len(vrp.bestSolution.getRoutes()))
