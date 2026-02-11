# package: travelingSalesmanProblem
# file: tsp_basic_algorithms.py

from __future__ import annotations

import random

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from python_hyper_heuristic.domains.Python.TSP.TSPInstance import TspInstance
from python_hyper_heuristic.domains.Python.TSP.TSPDataStructure import TspDataStructure


class TspBasicAlgorithms:
    def __init__(self, instance: "TspInstance"):
        self.instance = instance
        self.ds: Optional["TspDataStructure"] = None

    # -----------------------
    # RANDOM PERMUTATIONS
    # -----------------------

    def generate_random_permutation(self, n: int, rng: random.Random) -> List[int]:
        """Generates a random permutation with n elements, starting at 0."""
        perm = list(range(n))
        self.shuffle_permutation(perm, rng)
        return perm

    def shuffle_permutation(self, array: List[int], rng: random.Random) -> None:
        """In-place Fisher–Yates shuffle."""
        n = len(array)
        while n > 1:
            k = rng.randrange(n)  # 0 <= k < n
            n -= 1
            array[n], array[k] = array[k], array[n]

    # -----------------------
    # COSTS
    # -----------------------

    def compute_cost(self, permutation: Sequence[int]) -> float:
        """
        Returns the cost of the tour represented by the permutation.
        Tour may be partial (not including all cities).
        """
        if not permutation:
            return 0.0

        s = 0.0
        for i in range(1, len(permutation)):
            s += self.instance.getDistance(permutation[i - 1], permutation[i])
        s += self.instance.getDistance(permutation[0], permutation[-1])
        return s

    def incremental_insertion_cost(self, tour: Sequence[int], initial_cost: float, index1: int, index2: int) -> float:
        return initial_cost + self.reinsertion_cost(tour, index1, index2)

    def incremental_cost_flip(self, tour: Sequence[int], initial_cost: float, index1: int, index2: int) -> float:
        return initial_cost + self.flip_cost(tour, index1, index2)

    def incremental_cost_swap(self, tour: Sequence[int], initial_cost: float, index1: int, index2: int) -> float:
        return initial_cost + self.swap_cost(tour, index1, index2)

    def reinsertion_cost(self, tour: Sequence[int], index1: int, index2: int) -> float:
        n = self.instance.numbCities
        city1 = tour[index1]
        city2 = tour[index2]

        prev1 = tour[index1 - 1] if index1 != 0 else tour[n - 1]
        next1 = tour[index1 + 1] if index1 != (n - 1) else tour[0]
        prev2 = tour[index2 - 1] if index2 != 0 else tour[n - 1]

        current_cost = (
            self.instance.getDistance(city1, prev1)
            + self.instance.getDistance(city1, next1)
            + self.instance.getDistance(city2, prev2)
        )
        new_cost = (
            self.instance.getDistance(prev2, city1)
            + self.instance.getDistance(city1, city2)
            + self.instance.getDistance(prev1, next1)
        )
        return new_cost - current_cost

    def insertion_cost(self, tour: Sequence[int], city1: int, index: int) -> float:
        if index == 0 or index == len(tour):
            return self.instance.getDistance(city1, tour[0]) + self.instance.getDistance(city1, tour[-1])

        return self.instance.getDistance(city1, tour[index - 1]) + self.instance.getDistance(city1, tour[index])

    def flip_cost(self, tour: Sequence[int], index1: int, index2: int) -> float:
        n = self.instance.numbCities
        city1 = tour[index1]
        city2 = tour[index2]

        prev1 = tour[index1 - 1] if index1 != 0 else tour[n - 1]
        next2 = tour[index2 + 1] if index2 != (n - 1) else tour[0]

        if prev1 == city2 or next2 == city1:
            return 0.0

        current_cost = self.instance.getDistance(city1, prev1) + self.instance.getDistance(city2, next2)
        new_cost = self.instance.getDistance(city1, next2) + self.instance.getDistance(city2, prev1)
        return new_cost - current_cost

    def flip_cost_ds(self, ds: "TspDataStructure", city1: int, city2: int) -> float:
        prev1 = ds.prev(city1)
        next2 = ds.next(city2)
        if prev1 == city2 or next2 == city1:
            return 0.0

        current_cost = self.instance.getDistance(city1, prev1) + self.instance.getDistance(city2, next2)
        new_cost = self.instance.getDistance(city1, next2) + self.instance.getDistance(city2, prev1)
        return new_cost - current_cost

    def swap_cost(self, tour: Sequence[int], index1: int, index2: int) -> float:
        n = self.instance.numbCities
        city1 = tour[index1]
        city2 = tour[index2]

        prev1 = tour[index1 - 1] if index1 != 0 else tour[n - 1]
        prev2 = tour[index2 - 1] if index2 != 0 else tour[n - 1]
        next1 = tour[index1 + 1] if index1 != (n - 1) else tour[0]
        next2 = tour[index2 + 1] if index2 != (n - 1) else tour[0]

        current_cost = (
            self.instance.getDistance(prev1, city1)
            + self.instance.getDistance(city1, next1)
            + self.instance.getDistance(prev2, city2)
            + self.instance.getDistance(city2, next2)
        )

        new_cost = (
            self.instance.getDistance(prev2, city1)
            + self.instance.getDistance(city1, next2)
            + self.instance.getDistance(prev1, city2)
            + self.instance.getDistance(city2, next1)
        )
        return new_cost - current_cost

    # -----------------------
    # ALGORITHMS
    # -----------------------

    @staticmethod
    def flip(permutation: Sequence[int], a: int, b: int) -> List[int]:
        """
        Reverses a segment of the permutation similarly to the Java code.
        NOTE: the Java logic has a wrap-like adjustment when a > b; we replicate it.
        """
        if a > b:
            temp = a
            a = b + 1
            b = temp - 1

        if a == b:
            return list(permutation)

        new_perm = list(permutation)
        # reverse segment [a..b]
        new_perm[a : b + 1] = reversed(new_perm[a : b + 1])
        return new_perm

    def greedy_insertion(self, cost: float = 0.0) -> Tuple[List[int], float]:
        """
        Builds a tour by greedy insertion starting from [0, 1].
        Returns (tour, cost) since Python floats are immutable (unlike Java's boxed Double usage here).
        """
        tour: List[int] = [0, 1]
        to_insert = list(range(2, self.instance.numbCities))
        cost += 2 * self.instance.getDistance(0, 1)
        return self.greedy_insertion_with_lists(tour, to_insert, cost)

    def greedy_insertion_with_lists(self, tour: List[int], to_insert: Sequence[int], cost: float) -> Tuple[List[int], float]:
        for city in to_insert:
            best = self.insertion_cost(tour, city, 0)
            best_index = 0
            for j in range(1, len(tour) + 1):
                temp = self.insertion_cost(tour, city, j)
                if temp < best:
                    best = temp
                    best_index = j
            cost += best
            tour = self.insert(tour, city, best_index)
        return tour, cost

    def greedy_heuristic(self, start_city: int) -> List[int]:
        numb_cities = self.instance.numbCities
        tour = [0] * numb_cities
        included = [False] * numb_cities

        tour[0] = start_city
        included[start_city] = True

        for i in range(1, numb_cities):
            best_dist = float("inf")
            best_city = 0
            for j in range(numb_cities):
                if included[j]:
                    continue
                d = self.instance.getDistance(tour[i - 1], j)
                if d < best_dist:
                    best_dist = d
                    best_city = j
            tour[i] = best_city
            included[best_city] = True

        return tour

    # -----------------------
    # LOCAL SEARCH
    # -----------------------

    def two_opt_best_improvement(self, tour: Sequence[int], instance: "TspInstance", maxiterations: int) -> List[int]:
        ds = TspDataStructure.create(list(tour))
        numb_cities = instance.numbCities

        nearest = instance.nearestCities
        N = len(nearest[0]) if nearest and nearest[0] else 0

        improvement = True
        completed = 0
        while improvement and (completed < maxiterations):
            completed += 1
            improvement = False
            for i in range(numb_cities):
                best_move = -1
                best_cost = 0.0
                for j in range(N):
                    fc = self.flip_cost_ds(ds, i, nearest[i][j])
                    if fc < best_cost and fc < 0.0:
                        best_move = nearest[i][j]
                        best_cost = fc
                if best_move != -1:
                    ds.flip(i, best_move)
                    improvement = True

        return ds.returnTour(tour[0])

    def two_opt_first_improvement(self, tour: Sequence[int], instance: "TspInstance", maxiterations: int) -> List[int]:
        ds = TspDataStructure.create(list(tour))
        numb_cities = instance.numbCities

        nearest = instance.nearestCities
        N = len(nearest[0]) if nearest and nearest[0] else 0

        improvement = True
        completed = 0
        while improvement and (completed < maxiterations):
            completed += 1
            improvement = False
            for i in range(numb_cities):
                for j in range(N):
                    fc = self.flip_cost_ds(ds, i, nearest[i][j])
                    if fc < 0.0:
                        ds.flip(i, nearest[i][j])
                        improvement = True

        return ds.returnTour(tour[0])

    def three_opt(self, tour: Sequence[int], instance: "TspInstance", maxiterations: int) -> List[int]:
        numb_cities = instance.numbCities
        ds = TspDataStructure.create(list(tour))

        nearest = instance.nearestCities
        N = len(nearest[0]) if nearest and nearest[0] else 0

        global_improvement = True
        completed = 0
        while global_improvement and (completed < maxiterations):
            completed += 1
            global_improvement = False

            for i in range(numb_cities):
                local_improvement = False
                i_prev = ds.prev(i)

                for j2 in range(N):
                    j = nearest[i][j2]

                    if j == i_prev or j == ds.prev(i_prev):
                        continue

                    cost = self.flip_cost_ds(ds, i, j)
                    ds.flip(i, j)

                    j_next = ds.next(j)

                    # loop nearest cities to j_next
                    for k in range(N):
                        next_city = nearest[j_next][k]
                        if ds.sequence(i_prev, next_city, j_next) or next_city == i_prev:
                            continue
                        if cost + self.flip_cost_ds(ds, j_next, next_city) < 0:
                            ds.flip(j_next, next_city)
                            local_improvement = True
                            global_improvement = True
                            break

                    if local_improvement:
                        break

                    # loop nearest cities to i_prev
                    for k in range(N):
                        prev_city = nearest[i_prev][k]
                        if ds.sequence(i_prev, prev_city, j_next) or prev_city == j_next:
                            continue
                        if cost + self.flip_cost_ds(ds, i_prev, prev_city) < 0:
                            ds.flip(i_prev, prev_city)
                            local_improvement = True
                            global_improvement = True
                            break

                    if local_improvement:
                        break

                    ds.flip(j, i)  # undo

        return ds.returnTour(tour[0])

    # -----------------------
    # UTILITIES
    # -----------------------

    def inverse_permutation(self, permutation: Sequence[int]) -> List[int]:
        inv = [0] * len(permutation)
        for i, v in enumerate(permutation):
            inv[v] = i
        return inv

    def insert(self, initial_tour: Sequence[int], x: int, index: int) -> List[int]:
        new_tour = list(initial_tour)
        new_tour.insert(index, x)
        return new_tour

    def verify_permutation(self, p: Sequence[int], n: int) -> bool:
        if len(p) != n:
            return False
        included = [False] * n
        for v in p:
            if v < 0 or v >= n:
                return False
            if included[v]:
                return False
            included[v] = True
        return all(included)

    def tour_to_string(self, tour: Sequence[int]) -> str:
        return " ".join(str(x) for x in tour) + (" " if tour else "")

