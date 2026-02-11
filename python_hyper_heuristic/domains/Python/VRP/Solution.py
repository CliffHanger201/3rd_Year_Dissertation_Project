# package VRP

# Assumes Route exists (or will be created) with:
# - copyRoute()

from typing import List

from python_hyper_heuristic.domains.Python.VRP.Route import Route


class Solution:
    def __init__(self, rs: List["Route"] | None = None):
        if rs is None:
            self.routes: List["Route"] = []
        else:
            self.routes: List["Route"] = rs

    def setRoutes(self, routes: List["Route"]) -> None:
        self.routes = routes

    def getRoutes(self) -> List["Route"]:
        return self.routes

    def copySolution(self) -> "Solution":
        newRoutes: List["Route"] = []
        for r in self.routes:
            newRoutes.append(r.copyRoute())
        return Solution(newRoutes)
