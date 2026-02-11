# package BinPacking

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Piece:
    size: float = 0.0
    number: int = 0

    def getSize(self) -> float:
        return self.size

    def getNumber(self) -> int:
        # Java returns double but it's actually an int field; keep it as int in Python.
        return self.number

    def clone(self) -> "Piece":
        return Piece(self.size, self.number)

    # Java equals compares only by number
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Piece):
            return False
        return self.number == other.number

    # Comparable: Java compares by size (descending)
    def __lt__(self, other: "Piece") -> bool:
        # For Python sorting ascending, define lt so larger size comes first
        return self.size > other.size
