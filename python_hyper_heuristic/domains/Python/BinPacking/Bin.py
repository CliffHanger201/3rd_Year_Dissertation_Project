# package BinPacking
# Assumes Piece exists with: getSize(), getNumber(), clone()

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from python_hyper_heuristic.domains.Python.BinPacking.Piece import Piece


@dataclass
class Bin:
    pieces_in_this_bin: List["Piece"] = field(default_factory=list)

    def addPiece(self, p: "Piece") -> None:
        self.pieces_in_this_bin.append(p)

    def getFullness(self) -> float:
        fullness = 0.0
        for p in self.pieces_in_this_bin:
            fullness += p.getSize()
        return fullness

    def numberOfPiecesInThisBin(self) -> int:
        return len(self.pieces_in_this_bin)

    def addToString(self, s: str) -> str:
        s += f"[{self.getFullness()}]  ["
        for p in self.pieces_in_this_bin:
            s += f"{p.getSize()}, "
        s += "] \n"
        return s

    # Hyflex: boolean contains(int num)
    def containsNumber(self, num: int) -> bool:
        for item in self.pieces_in_this_bin:
            if item.getNumber() == num:
                return True
        return False

    def copyPieceNumbers(self, v: List[int]) -> None:
        for p in self.pieces_in_this_bin:
            v.append(int(p.getNumber()))

    # Hyflex: int contains(Piece p)
    def indexOfPiece(self, p: "Piece") -> int:
        # Java version matches by piece number, not object identity
        idx = -1
        for u, item in enumerate(self.pieces_in_this_bin):
            if item.getNumber() == p.getNumber():
                idx = u
        return idx

    def clone(self) -> "Bin":
        copy_bin = Bin()
        for p in self.pieces_in_this_bin:
            copy_bin.addPiece(p.clone())
        return copy_bin

    # Hyflex: Piece removePiece(Piece p)
    def removePiece(self, p: "Piece") -> "Piece":
        # Java removes by indexOf(p) using equality; in Python list.remove uses equality too.
        # If Piece equality isn't defined, this may fail; fallback to number-match.
        try:
            self.pieces_in_this_bin.remove(p)
            return p
        except ValueError:
            idx = self.indexOfPiece(p)
            if idx == -1:
                raise ValueError("Piece not found in bin") from None
            return self.pieces_in_this_bin.pop(idx)

    def getPieceSize(self, index: int) -> float:
        return self.pieces_in_this_bin[index].getSize()

    def removeTwoPieces(self, a: int, b: int) -> List["Piece"]:
        # Remove the two pieces at indices a and b (original Java grabs both first, then removes)
        p1 = self.pieces_in_this_bin[a]
        p2 = self.pieces_in_this_bin[b]
        removed1 = self.removePiece(p1)
        removed2 = self.removePiece(p2)
        return [removed1, removed2]

    # Hyflex: Piece removePiece(int index)
    def removePieceAt(self, index: int) -> "Piece":
        return self.pieces_in_this_bin.pop(index)

    # Comparable: Java compares by fullness (descending)
    def __lt__(self, other: "Bin") -> bool:
        # For sorting ascending in Python, define lt so "more full" is "smaller"
        return self.getFullness() > other.getFullness()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bin):
            return False
        return self.getFullness() == other.getFullness()

    def print(self) -> None:
        # matches Java output style closely
        print(f"[{self.getFullness()} ", end="")
        for p in self.pieces_in_this_bin:
            print(f"{p.getNumber()},", end="")
        print("] ", end="")
