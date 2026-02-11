# package: travelingSalesmanProblem
# file: tsp_instance.py

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, TextIO


class TspInstance:
    """
    Python conversion of the Java TspInstance.

    Notes on behavior parity:
    - Keeps N fixed to 8 by default (matches "DO NOT CHANGE" comment).
    - Tries two load strategies for .tsp and NearestCities files:
        1) filesystem path with backslashes: data\\tsp\\...
        2) fallback path with forward slashes: data/tsp/...
      In Java, the fallback uses classpath resources. In Python, unless you're packaging
      data as package resources, that doesn't exist automatically; here we attempt a
      plain filesystem open of the fallback path.
    """

    instanceNames = [
        "pr299", "pr439", "rat575", "u724",
        "rat783", "pcb1173", "d1291", "u2152",
        "usa13509", "d18512",
    ]
    folder_path = Path(__file__).resolve().parents[2] / "resources"

    def __init__(self, number: int):
        self.numbCities: int = 0
        self.coordinates: List[List[float]] = []
        self.name: str = self.instanceNames[number]

        # For efficiency: N nearest cities per city
        self.N: int = 8  # DO NOT CHANGE
        self.nearestCities: Optional[List[List[int]]] = None
        self.D: Optional[List[List[float]]] = None

        tsp_path = self.folder_path / "data" / "tsp" / f"{self.name}.tsp"

        if not tsp_path.exists():
            raise FileNotFoundError(f"TSP file not found: {tsp_path}")

        with open(tsp_path, "r", encoding="utf-8", errors="replace") as f:
            self._load_data(f)

        # # Load .tsp instance data
        # tsp_file1 = os.path.join(self.folder_path, "data", "tsp", f"{self.name}.tsp")  # platform-safe
        # # Keep the Java-looking path attempt as well
        # tsp_file_javaish = f"data\\tsp\\{self.name}.tsp"
        # tsp_file2 = f"data/tsp/{self.name}.tsp"

        # last_exc: Optional[BaseException] = None
        # for path in (tsp_file_javaish, tsp_file1, tsp_file2):
        #     try:
        #         with open(path, "r", encoding="utf-8", errors="replace") as f:
        #             self._load_data(f)
        #         last_exc = None
        #         break
        #     except Exception as ex:
        #         last_exc = ex

        # if last_exc is not None:
        #     raise FileNotFoundError(f"problem when opening file for instance {self.name}") from last_exc

        # Load nearest-cities table
        self.load_nearest_cities()

    # -----------------------
    # STATIC SAVE HELPERS
    # -----------------------

    @staticmethod
    def save_nearest(nearest: List[List[int]], file_name: str) -> None:
        lines: List[str] = []
        for row in nearest:
            lines.append(" ".join(str(x) for x in row))
        TspInstance.save(file_name, "\n".join(lines))

    @staticmethod
    def save(file_name: str, data: str) -> None:
        with open(file_name, "w", encoding="utf-8") as writer:
            writer.write(data)

    # -----------------------
    # NEAREST CALCULATION
    # -----------------------

    def calculate_nearest(self, N: int) -> None:
        """
        Calculate matrices with the N closest cities to each city
        and their corresponding distances.
        """
        self.nearestCities = [[0] * N for _ in range(self.numbCities)]
        self.D = [[float("inf")] * N for _ in range(self.numbCities)]

        for i in range(self.numbCities):
            # Java used Arrays.fill(D[i], Integer.MAX_VALUE)
            # We'll keep +inf
            for j in range(self.numbCities):
                if i == j:
                    continue
                cost = self.getDistance(i, j)
                max_idx = self.get_max(self.D[i])
                if cost < self.D[i][max_idx]:
                    self.D[i][max_idx] = cost
                    self.nearestCities[i][max_idx] = j

    @staticmethod
    def get_max(array: List[float]) -> int:
        max_val = -float("inf")
        max_idx = 0
        for i, v in enumerate(array):
            if v > max_val:
                max_val = v
                max_idx = i
        return max_idx

    def is_nearest(self, city_index: int, candidate: int) -> bool:
        if self.nearestCities is None:
            return False
        # Java bug/oddity: looped i < nearestCities.length but indexed nearestCities[cityIndex][i]
        # That only works when N == numbCities, which it isn't. We'll implement the intended logic:
        return candidate in self.nearestCities[city_index]

    def get_distance_to_nearest(self, city_index: int, nth_nearest: int) -> float:
        if self.D is None:
            raise ValueError("Nearest-distance matrix D is not computed/loaded.")
        return self.D[city_index][nth_nearest]

    # -----------------------
    # LOADING FILES
    # -----------------------

    def load_nearest_cities(self) -> None:
        tsp_path = self.folder_path / "data" / "tsp" / f"{self.name}NearestCities.txt"

        if not tsp_path.exists():
            raise FileNotFoundError(f"TSP file not found: {tsp_path}")

        with open(tsp_path, "r", encoding="utf-8", errors="replace") as f:
            self._read_table(f)

        # file1 = f"data\\tsp\\{self.name}NearestCities.txt"
        # file2 = os.path.join("data", "tsp", f"{self.name}NearestCities.txt")
        # file3 = f"data/tsp/{self.name}NearestCities.txt"

        # last_exc: Optional[BaseException] = None
        # for path in (file1, file2, file3):
        #     try:
        #         with open(path, "r", encoding="utf-8", errors="replace") as f:
        #             self._read_table(f)
        #         last_exc = None
        #         break
        #     except Exception as ex:
        #         last_exc = ex

        # if last_exc is not None:
        #     raise FileNotFoundError(f"problem when opening file {file1} (or fallbacks)") from last_exc

    def _read_table(self, f: TextIO) -> None:
        self.nearestCities = [[0] * self.N for _ in range(self.numbCities)]
        for i in range(self.numbCities):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading nearest cities table.")
            parts = line.split()
            if len(parts) < self.N:
                raise ValueError(f"Nearest table line {i} has too few entries: {len(parts)} < {self.N}")
            for j in range(self.N):
                self.nearestCities[i][j] = int(parts[j])

    def _load_data(self, f: TextIO) -> None:
        # ignore first three lines
        for _ in range(3):
            _ = f.readline()

        # 4th line gives the number of cities
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF reading DIMENSION line.")
        tok = line.split()
        # Java: tok.nextToken(); tok.nextToken(); numbCities = nextToken()
        if len(tok) < 3:
            raise ValueError(f"Bad DIMENSION line: {line!r}")
        self.numbCities = int(tok[2])

        # skip until NODE_COORD_SECTION
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF before NODE_COORD_SECTION.")
        while line.strip() != "NODE_COORD_SECTION":
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF searching for NODE_COORD_SECTION.")

        # retrieve coordinates
        self.coordinates = [[0.0, 0.0] for _ in range(self.numbCities)]
        for i in range(self.numbCities):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading coordinates.")
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad coordinate line: {line!r}")
            # parts[0] is the city id (ignored like Java)
            self.coordinates[i][0] = float(parts[1])
            self.coordinates[i][1] = float(parts[2])

    # -----------------------
    # REPRESENTATION / GETTERS
    # -----------------------

    def __str__(self) -> str:
        return "".join(f"{xy[0]} {xy[1]}\n" for xy in self.coordinates)

    def getNumbCities(self) -> int:
        return self.numbCities

    def getCoordinates(self) -> List[List[float]]:
        return self.coordinates

    # -----------------------
    # DISTANCES
    # -----------------------

    def getDistance(self, city1: int, city2: int) -> float:
        x1, y1 = self.coordinates[city1]
        x2, y2 = self.coordinates[city2]
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
