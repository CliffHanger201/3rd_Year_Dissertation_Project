from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import os
import sys


class Instance:
    def __init__(self, insnumber: int) -> None:
        self.n: int = 0  # number of jobs
        self.m: int = 0  # number of machines
        self.processingTimes: List[List[int]] = []  # [job][machine]

        number = self._map_instance_number(insnumber)

        # Expected dims from the folder group (e.g., 100x20)
        exp_n, exp_m = self._dims_from_number(number)

        file_name = self.returnNameForFolder(number)
        raw = self.openFileFromFolder(file_name)

        # Parse as flat ints and reshape to m rows x n cols, then transpose to n x m
        matrix_m_by_n = self.openDataAsInt_from_text(raw, exp_n=exp_n, exp_m=exp_m, file=file_name)
        self.processingTimes = self.transposeMatrix(matrix_m_by_n)  # now n x m

        self.n = len(self.processingTimes)
        self.m = len(self.processingTimes[0]) if self.n > 0 else 0

        if self.n != exp_n or self.m != exp_m:
            raise ValueError(
                f"Loaded instance shape mismatch for {file_name}: "
                f"got n={self.n}, m={self.m}, expected n={exp_n}, m={exp_m}"
            )

    # --------------------
    # Mapping + dimensions
    # --------------------

    def _map_instance_number(self, insnumber: int) -> int:
        mapping = {
            0: 80, 1: 81, 2: 82, 3: 83, 4: 84,
            5: 91, 6: 92,
            7: 110, 8: 111, 9: 113,
            10: 100, 11: 112,
        }
        if insnumber not in mapping:
            print(f"instance does not exist {insnumber}", file=sys.stderr)
            raise SystemExit(-1)
        return mapping[insnumber]

    def _dims_from_number(self, number: int) -> Tuple[int, int]:
        a = number // 10
        if a == 0:  return (20, 5)
        if a == 1:  return (20, 10)
        if a == 2:  return (20, 20)
        if a == 3:  return (50, 5)
        if a == 4:  return (50, 10)
        if a == 5:  return (50, 20)
        if a == 6:  return (100, 5)
        if a == 7:  return (100, 10)
        if a == 8:  return (100, 20)
        if a == 9:  return (200, 10)
        if a == 10: return (200, 20)
        if a == 11: return (500, 20)
        raise ValueError(f"Unknown instance group for number={number}")

    # --------------------
    # Getters
    # --------------------

    def getM(self) -> int:
        return self.m

    def getN(self) -> int:
        return self.n

    def getSumP(self) -> int:
        return sum(sum(row) for row in self.processingTimes)

    def getProcTimes(self) -> List[List[int]]:
        return self.processingTimes

    def __str__(self) -> str:
        lines = ["Processing times:"]
        for i in range(self.n):
            lines.append(" ".join(str(self.processingTimes[i][j]) for j in range(self.m)))
        return "\n".join(lines) + "\n"

    # --------------------
    # File + parsing
    # --------------------

    def _find_resources_dir(self) -> Path:
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            cand = parent / "resources"
            if cand.is_dir():
                return cand
        raise FileNotFoundError("Could not locate a 'resources' directory above Instance.py")

    def openFileFromFolder(self, file: str) -> str:
        base_dir = self._find_resources_dir()
        path = base_dir / file
        if not path.exists():
            raise FileNotFoundError(f"Flowshop instance not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def openDataAsInt_from_text(self, text: str, exp_n: int, exp_m: int, file: str) -> List[List[int]]:
        # Extract all ints ignoring line boundaries
        ints: List[int] = []
        for tok in text.split():
            try:
                ints.append(int(tok))
            except ValueError:
                # ignore stray tokens (shouldn't happen in these datasets)
                pass

        if not ints:
            raise ValueError(f"No numeric data found in {file}")

        need = exp_n * exp_m

        # Some datasets include header "n m" at the start. Skip it if present.
        start = 0
        if len(ints) >= 2 and ints[0] == exp_n and ints[1] == exp_m:
            start = 2

        data = ints[start:]
        if len(data) < need:
            raise ValueError(f"Not enough ints in {file}: need {need}, got {len(data)} after header-skip")

        data = data[:need]  # ignore anything after the processing times

        # Build matrix as m rows × n columns (machine-major), matching the Java transpose step
        rows: List[List[int]] = []
        idx = 0
        for _ in range(exp_m):
            rows.append(data[idx: idx + exp_n])
            idx += exp_n
        return rows

    # --------------------
    # Name mapping
    # --------------------

    def returnNameForFolder(self, number: int) -> str:
        fileName = os.path.join("data", "flowshop")
        a = number // 10
        b = number % 10

        if a == 0:
            fileName = os.path.join(fileName, "20x5")
        elif a == 1:
            fileName = os.path.join(fileName, "20x10")
        elif a == 2:
            fileName = os.path.join(fileName, "20x20")
        elif a == 3:
            fileName = os.path.join(fileName, "50x5")
        elif a == 4:
            fileName = os.path.join(fileName, "50x10")
        elif a == 5:
            fileName = os.path.join(fileName, "50x20")
        elif a == 6:
            fileName = os.path.join(fileName, "100x5")
        elif a == 7:
            fileName = os.path.join(fileName, "100x10")
        elif a == 8:
            fileName = os.path.join(fileName, "100x20")
        elif a == 9:
            fileName = os.path.join(fileName, "200x10")
        elif a == 10:
            fileName = os.path.join(fileName, "200x20")
        elif a == 11:
            fileName = os.path.join(fileName, "500x20")

        return os.path.join(fileName, f"{b + 1}.txt")

    # --------------------
    # Matrix helper
    # --------------------

    def transposeMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix or not matrix[0]:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        newMatrix = [[0] * rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                newMatrix[j][i] = matrix[i][j]
        return newMatrix
