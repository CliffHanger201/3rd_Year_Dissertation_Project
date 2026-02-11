from abc import ABC, abstractmethod
from enum import Enum
import random
import time


class HeuristicType(Enum):
    """
    An enumeration of the different types of low-level heuristics.
    """
    MUTATION = 1
    CROSSOVER = 2
    RUIN_RECREATE = 3
    LOCAL_SEARCH = 4
    OTHER = 5


class ProblemDomain(ABC):
    """
    Problem domain is an abstract class containing methods
    for applying heuristics and managing solutions.
    """

    def __init__(self, seed: int):
        """
        Creates a new problem domain and initialises the RNG.
        If seed == -1, system time is used.
        Sets solution memory size to 2.
        """
        if seed == -1:
            self.rng = random.Random()
        else:
            self.rng = random.Random(seed)

        self.depthOfSearch = 0.2
        self.intensityOfMutation = 0.2

        # solution memory
        self.solutions = []

        # track best solution properly
        self.best_ever_value = float("inf")
        self.best_ever_index = None

        # These depend on getNumberOfHeuristics(), which must be
        # implemented by subclasses
        n_heuristics = self.getNumberOfHeuristics()
        self.heuristicCallRecord = [0] * n_heuristics
        self.heuristicCallTimeRecord = [0] * n_heuristics

        self.setMemorySize(2)
        self.setDepthOfSearch(self.depthOfSearch)
        self.setIntensityOfMutation(self.intensityOfMutation)

    # -----------------------------
    # Heuristic bookkeeping
    # -----------------------------

    def getHeuristicCallRecord(self):
        """
        Returns how many times each heuristic has been called.
        """
        return self.heuristicCallRecord

    def getheuristicCallTimeRecord(self):
        """
        Returns total runtime (ms) for each heuristic.
        """
        return self.heuristicCallTimeRecord

    # -----------------------------
    # Parameter setters / getters
    # -----------------------------

    def setDepthOfSearch(self, depthOfSearch: float):
        """
        Sets depth of search in range [0, 1].
        """
        self.depthOfSearch = max(0.0, min(1.0, depthOfSearch))

    def setIntensityOfMutation(self, intensityOfMutation: float):
        """
        Sets intensity of mutation in range [0, 1].
        """
        self.intensityOfMutation = max(0.0, min(1.0, intensityOfMutation))

    def getDepthOfSearch(self) -> float:
        return self.depthOfSearch

    def getIntensityOfMutation(self) -> float:
        return self.intensityOfMutation

    # -----------------------------
    # Abstract API
    # -----------------------------

    @abstractmethod
    def getHeuristicsOfType(self, heuristicType: HeuristicType):
        """
        Returns a list of heuristic IDs of the specified type.
        """
        pass

    @abstractmethod
    def getHeuristicsThatUseIntensityOfMutation(self):
        """
        Returns heuristic IDs that use intensityOfMutation.
        """
        pass

    @abstractmethod
    def getHeuristicsThatUseDepthOfSearch(self):
        """
        Returns heuristic IDs that use depthOfSearch.
        """
        pass

    @abstractmethod
    def loadInstance(self, instanceID: int):
        """
        Loads the problem instance specified by instanceID.
        """
        pass

    @abstractmethod
    def setMemorySize(self, size: int):
        """
        Sets the size of the solution memory.
        """
        pass

    @abstractmethod
    def initialiseSolution(self, index: int):
        """
        Creates an initial solution at the given memory index.
        """
        pass

    @abstractmethod
    def getNumberOfHeuristics(self) -> int:
        """
        Returns the number of heuristics available.
        """
        pass

    @abstractmethod
    def applyHeuristic(self, heuristicID: int,
                       solutionSourceIndex1: int,
                       solutionSourceIndex2: int = None,
                       solutionDestinationIndex: int = None) -> float:
        """
        Applies a heuristic and returns the objective value
        of the resulting solution.
        """
        pass

    @abstractmethod
    def copySolution(self, solutionSourceIndex: int,
                     solutionDestinationIndex: int):
        """
        Copies a solution from one memory index to another.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns the name of the problem domain.
        """
        pass

    @abstractmethod
    def getNumberOfInstances(self) -> int:
        """
        Returns the number of available instances.
        """
        pass

    @abstractmethod
    def bestSolutionToString(self) -> str:
        """
        Returns a string representation of the best solution found.
        """
        pass

    # @abstractmethod
    # def getBestSolutionIndex(self) -> int:
    #     """
    #     Returns the objective index of the best solution found.
    #     """
    #     pass

    @abstractmethod
    def getBestSolutionValue(self) -> float:
        """
        Returns the objective value of the best solution found.
        """
        pass

    @abstractmethod
    def solutionToString(self, solutionIndex: int) -> str:
        """
        Returns a string representation of a solution in memory.
        """
        pass

    @abstractmethod
    def getFunctionValue(self, solutionIndex: int) -> float:
        """
        Returns the objective value of a solution.
        """
        pass

    @abstractmethod
    def compareSolutions(self, solutionIndex1: int,
                         solutionIndex2: int) -> bool:
        """
        Compares two solutions structurally (not just fitness).
        """
        pass
