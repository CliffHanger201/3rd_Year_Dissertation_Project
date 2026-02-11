# Fully converted from FlowShop.Instance (Java) to Python.
# Assumes your project layout will provide the data files similarly.
# This class loads a flowshop instance, trying:
#   1) "folder" path (Windows-style) like data\flowshop\...
#   2) "jar" path (resource-style) like data/flowshop/...
#
# In Python, there is no "jar". The closest equivalent is package resources.
# So: openFileFromJar() tries importlib.resources first, then falls back to
# trying to open the path relative to the current working directory.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
import sys
import importlib.resources as pkg_resources


class Instance:
    def __init__(self, insnumber: int) -> None:
        self.n: int = 0  # number of jobs
        self.m: int = 0  # number of machines
        self.processingTimes: List[List[int]] = []  # [j][k]

        number = insnumber

        # Mapping exactly as in Java
        if insnumber == 0:
            number = 80
        elif insnumber == 1:
            number = 81
        elif insnumber == 2:
            number = 82
        elif insnumber == 3:
            number = 83
        elif insnumber == 4:
            number = 84
        elif insnumber == 5:
            number = 91
        elif insnumber == 6:
            number = 92
        elif insnumber == 7:
            number = 110
        elif insnumber == 8:
            number = 111
        elif insnumber == 9:
            number = 113
        elif insnumber == 10:
            number = 100
        elif insnumber == 11:
            number = 112
        else:
            print(f"instance does not exist {insnumber}", file=sys.stderr)
            raise SystemExit(-1)

        # Try folder first, then "jar resources"
        try:
            fileName = self.returnNameForFolder(number)
            data = self.openDataAsInt(fileName, fromFolder=True)
            self.processingTimes = self.transposeMatrix(data)
            self.n = len(self.processingTimes)
            self.m = len(self.processingTimes[0]) if self.n > 0 else 0
        except Exception as ex:
            # try:
            #     fileName = self.returnNameForJar(number)
            #     data = self.openDataAsInt(fileName, fromFolder=False)
            #     self.processingTimes = self.transposeMatrix(data)
            #     self.n = len(self.processingTimes)
            #     self.m = len(self.processingTimes[0]) if self.n > 0 else 0
            # except Exception as ex2:
            print(f"Could not open file from Folder: {ex}")
            #     # print(f"Could not open file from Jar: {ex2}")
            #     raise SystemExit(0)

    # --- getters (kept to match Java style) ---

    def getM(self) -> int:
        return self.m

    def getN(self) -> int:
        return self.n

    def getSumP(self) -> int:
        s = 0
        for i in range(self.n):
            for j in range(self.m):
                s += self.processingTimes[i][j]
        return s

    def getProcTimes(self) -> List[List[int]]:
        return self.processingTimes

    def __str__(self) -> str:
        lines = ["Processing times:"]
        for i in range(self.n):
            lines.append(" ".join(str(self.processingTimes[i][j]) for j in range(self.m)))
        return "\n".join(lines) + "\n"

    # --- file + parsing helpers ---

    def openDataAsInt(self, file: str) -> List[List[int]]:
        data = self.openFileFromFolder(file)

        rows: List[List[int]] = []
        # Java uses StringTokenizer; here we split whitespace.
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append([int(tok) for tok in line.split()])

        if not rows:
            raise ValueError(f"No data found in {file}")

        # In Java they determine number of columns from first line; we can enforce rectangular
        cols = len(rows[0])
        for r in rows:
            if len(r) != cols:
                raise ValueError(f"Non-rectangular data in {file}: expected {cols} cols, got {len(r)}")

        return rows

    def openFileFromFolder(self, file: str) -> str:
        base_dir = Path(__file__).resolve().parents[2] / "resources"
        path = base_dir / file
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # def openFileFromJar(self, file: str) -> str:
    #     """
    #     Java reads from classloader resources. In Python we try importlib.resources.
    #     Since we don't know your exact package structure, we attempt a few reasonable options:
    #       - try reading `file` as a package resource relative to the FlowShop package
    #       - if that fails, try opening it as a normal path (POSIX-style) from CWD
    #     """
    #     # 1) Try importlib.resources under this package name "FlowShop"
    #     try:
    #         # file like "data/flowshop/20x5/1.txt" -> treat as relative path under package
    #         # importlib.resources expects a traversable; we use files("FlowShop") / file
    #         base = pkg_resources.files(__package__ or "FlowShop")
    #         target = base.joinpath(file)
    #         return target.read_text(encoding="utf-8")
    #     except Exception:
    #         pass

    #     # 2) Fallback: try open relative to working directory
    #     with open(file, "r", encoding="utf-8") as f:
    #         return f.read()

    # --- name mapping helpers ---

    def returnNameForFolder(self, number: int) -> str:
        # Java uses "data\\flowshop\\"
        fileName = os.path.join("data", "flowshop")
        a = number // 10
        b = number % 10

        # exactly the same switch mapping
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


    # def returnNameForJar(self, number: int) -> str:
    #     # Java uses "data/flowshop/"
    #     fileName = "data/flowshop/"
    #     a = number // 10
    #     b = number % 10

    #     if a == 0:
    #         fileName += "20x5/"
    #     elif a == 1:
    #         fileName += "20x10/"
    #     elif a == 2:
    #         fileName += "20x20/"
    #     elif a == 3:
    #         fileName += "50x5/"
    #     elif a == 4:
    #         fileName += "50x10/"
    #     elif a == 5:
    #         fileName += "50x20/"
    #     elif a == 6:
    #         fileName += "100x5/"
    #     elif a == 7:
    #         fileName += "100x10/"
    #     elif a == 8:
    #         fileName += "100x20/"
    #     elif a == 9:
    #         fileName += "200x10/"
    #     elif a == 10:
    #         fileName += "200x20/"
    #     elif a == 11:
    #         fileName += "500x20/"

    #     return fileName + f"{b + 1}.txt"

    # --- matrix helper ---

    def transposeMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        # Java: new int[matrix[0].length][matrix.length]
        if not matrix or not matrix[0]:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        newMatrix = [[0] * rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                newMatrix[j][i] = matrix[i][j]
        return newMatrix
