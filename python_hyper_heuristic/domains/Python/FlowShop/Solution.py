# Fully converted from FlowShop.Solution (Java) to Python.
# Assumes this lives in your FlowShop package/module.

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import copy


@dataclass
class Solution:
    permutation: List[int]
    Cmax: float

    def clone(self) -> "Solution":
        # Java's int[].clone() is a shallow copy of the array;
        # in Python we copy the list.
        return Solution(self.permutation.copy(), self.Cmax)

    def __str__(self) -> str:
        parts = [f"Cmax = {self.Cmax}\n"]
        parts.append("".join(f" {x}" for x in self.permutation))
        return "".join(parts)
