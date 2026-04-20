from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from python_hyper_heuristic.domains.Python.AbstractProblem.ProblemDomain import ProblemDomain, HeuristicType
from python_hyper_heuristic.domains.Python.BinPacking.Bin import Bin
from python_hyper_heuristic.domains.Python.BinPacking.Piece import Piece


# ---- BinPacking domain ----
class BinPacking(ProblemDomain):
    HIGHEST_FIRST = True
    LOWEST_FIRST = False
    defaultmemorysize = 2

    def __init__(self, seed: int):
        self.solutionMemory: List[Optional[BinPacking.Solution]] = [None] * self.defaultmemorysize
        
        super().__init__(seed)

        self.mutations = [0, 3, 5]
        self.ruinRecreates = [1, 2]
        self.localSearches = [4, 6]
        self.crossovers = [7]

        # self.solutionMemory: List[Optional[BinPacking.Solution]] = []
        self.bestEverSolution: Optional[List[Bin]] = None
        self.bestEverObjectiveFunction: float = float("inf")
        self.bestEverNumberOfBins: float = float("inf")

        self.capacity: float = 0.0
        self.totalpiecesize: float = 0.0
        self.numberOfPieces: int = 0
        self.pieces: List[Piece] = []

        self.lrepeats: int = 10
        self.mrepeats: int = 1

    # --- parameter mapping (matches Java logic) ---
    def setDepthOfSearch(self, depthOfSearch: float) -> None:
        super().setDepthOfSearch(depthOfSearch)
        if depthOfSearch <= 0.2:
            self.lrepeats = 10
        elif depthOfSearch <= 0.4:
            self.lrepeats = 12
        elif depthOfSearch <= 0.6:
            self.lrepeats = 14
        elif depthOfSearch <= 0.8:
            self.lrepeats = 17
        else:
            self.lrepeats = 20

    def setIntensityOfMutation(self, intensityOfMutation: float) -> None:
        super().setIntensityOfMutation(intensityOfMutation)
        if intensityOfMutation <= 0.2:
            self.mrepeats = 1
        elif intensityOfMutation <= 0.4:
            self.mrepeats = 2
        elif intensityOfMutation <= 0.6:
            self.mrepeats = 3
        elif intensityOfMutation <= 0.8:
            self.mrepeats = 4
        else:
            self.mrepeats = 5

    def getHeuristicsThatUseDepthOfSearch(self):
        return self.localSearches

    def getHeuristicsThatUseIntensityOfMutation(self):
        return self.mutations + self.ruinRecreates

    # --- instance loading ---
    def _loadInstance_from_filename(self, filename: str) -> None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            self._readInInstance_lines(lines)
        except FileNotFoundError:
            # In Java they fallback to classpath resource stream; in Python you can adapt this
            raise FileNotFoundError(f"cannot find file {filename}")

    def _readInInstance_lines(self, lines: List[str]) -> None:
        if len(lines) < 3:
            raise ValueError("Instance file too short")

        self.totalpiecesize = 0.0
        header = lines[2].strip().split()
        if len(header) < 2:
            raise ValueError(f"Header line does not contain capacity and nPieces: {lines[2]!r}")

        self.capacity = float(header[0]) * 10.0
        self.numberOfPieces = int(header[1])
        self.pieces = []
        start = 3
        end = start + self.numberOfPieces
        if len(lines) < end:
            raise ValueError(f"Expected {self.numberOfPieces} piece lines, found only {len(lines) - start}")

        for piece_idx in range(self.numberOfPieces):
            size_line = lines[start + piece_idx].strip()
            piecesize = float(size_line) * 10.0
            self.totalpiecesize += piecesize
            self.pieces.append(Piece(piecesize, piece_idx))
    
    # def _readInInstance_lines(self, lines: List[str]) -> None:
    #     # Java:
    #     # readLine(); readLine(); l = readLine(); capacity = split[1]*10; numberOfPieces = split[2]
    #     # then read piece sizes each line *10
    #     if len(lines) < 3:
    #         raise ValueError("Instance file too short")

    #     self.totalpiecesize = 0.0
    #     header3 = lines[2].strip().split()
    #     self.capacity = float(header3[1]) * 10.0
    #     print("LINE2 RAW:", repr(lines[2]))
    #     print("LINE2 TOKS:", lines[2].strip().split())
    #     self.numberOfPieces = int(header3[2])
    #     self.pieces = []
    #     for piece_idx in range(self.numberOfPieces):
    #         size_line = lines[3 + piece_idx].strip()
    #         piecesize = float(size_line) * 10.0
    #         self.totalpiecesize += piecesize
    #         self.pieces.append(Piece(piecesize, piece_idx))

    def loadInstance(self, instanceID: int) -> None:
        if instanceID < 2:
            ins = f"data/binpacking/falkenauer/falk1000-{instanceID+1}.txt"
        elif instanceID < 4:
            ins = f"data/binpacking/schoenfield/schoenfieldhard{instanceID-1}.txt"
        elif instanceID < 5:
            ins = "data/binpacking/2000/10-30/instance1.txt"
        elif instanceID < 6:
            ins = "data/binpacking/2000/10-30/instance2.txt"
        elif instanceID < 7:
            ins = "data/binpacking/trip1002/instance1.txt"
        elif instanceID < 8:
            ins = "data/binpacking/trip2004/instance1.txt"
        elif instanceID < 9:
            ins = "data/binpacking/testdual4/binpack0.txt"
        elif instanceID < 10:
            ins = "data/binpacking/testdual7/binpack0.txt"
        elif instanceID < 11:
            ins = "data/binpacking/2000/50-90/instance1.txt"
        elif instanceID < 12:
            ins = "data/binpacking/testdual10/binpack0.txt"
        else:
            raise ValueError(f"instance doesn't exist: {instanceID}")

        filename = Path(__file__).resolve().parents[2] / "resources" / ins
        self._loadInstance_from_filename(filename)
        self.solutionMemory = [None] * self.defaultmemorysize

    # --- initial solution ---
    def initialiseSolution(self, index: int) -> None:
        sol = BinPacking.Solution()
        sol.addBin(Bin())

        piecerandomiser = list(self.pieces)
        self.rng.shuffle(piecerandomiser)

        # Java fills pieces[] by removing last repeatedly
        self.pieces = []
        while piecerandomiser:
            self.pieces.append(piecerandomiser.pop())

        # First-fit packing; leaves one empty bin at end
        for currentPiece in self.pieces:
            numberOfBins = sol.size()
            for binNumber in range(numberOfBins):
                b = sol.get(binNumber)
                if currentPiece.getSize() <= (self.capacity - b.getFullness()):
                    b.addPiece(currentPiece)
                    sol.set(binNumber, b)
                    if binNumber == numberOfBins - 1:
                        sol.addBin(Bin())
                    break

        self.sortbins(sol.solution, self.HIGHEST_FIRST)
        self.solutionMemory[index] = sol

        val = self.getFunctionValue(index)
        if val < self.bestEverObjectiveFunction:
            self.bestEverObjectiveFunction = val

    # --- objective ---
    def evaluateObjectiveFunction(self, bins: List[Bin]) -> float:
        utilisation = 0.0
        binsused = 0.0
        for b in bins:
            if b.getFullness() != 0:
                utilisation += (b.getFullness() / self.capacity) ** 2
                binsused += 1.0
        if binsused == 0:
            return 1.0
        return 1.0 - (utilisation / binsused)

    # --- packing helpers ---
    def applyBestFit(self, array: List[Bin], v: List[Bin]) -> None:
        pieceVector: List[Piece] = []
        for b in array:
            while b.numberOfPiecesInThisBin() > 0:
                pieceVector.append(b.removePieceAt(0))

        pieceVector.sort()  # Piece.__lt__ sorts descending

        while pieceVector:
            currentPiece = pieceVector.pop(0)
            numberOfBins = len(v)
            bestgapsofar = float("inf")
            bestbinsofar = -1
            for binNumber in range(numberOfBins):
                b = v[binNumber]
                gap = self.capacity - currentPiece.getSize() - b.getFullness()
                if (gap < bestgapsofar) and (gap >= 0):
                    bestgapsofar = gap
                    bestbinsofar = binNumber

            if bestbinsofar < 0:
                # if nothing fits, use last bin (as Java often assumes a trailing empty bin exists)
                bestbinsofar = numberOfBins - 1

            b = v[bestbinsofar]
            b.addPiece(currentPiece)
            if bestbinsofar == numberOfBins - 1:
                v.append(Bin())

    # --- heuristics 0..7 ---
    def applyHeuristic0(self, temporary: List[Bin]) -> None:
        # swap random piece with another different-size piece; force feasibility by spilling into last bin if needed
        for _ in range(self.mrepeats):
            piece1 = self.pieces[self.rng.randrange(self.numberOfPieces)]
            piece2 = self.pieces[self.rng.randrange(self.numberOfPieces)]
            while piece1.getSize() == piece2.getSize():
                piece2 = self.pieces[self.rng.randrange(self.numberOfPieces)]

            bin1index = -1
            bin2index = -1
            for x, b in enumerate(temporary):
                if b.indexOfPiece(piece1) != -1:
                    bin1index = x
                if b.indexOfPiece(piece2) != -1:
                    bin2index = x
                if bin1index != -1 and bin2index != -1:
                    break

            bin1 = temporary[bin1index]
            bin2 = temporary[bin2index]

            bin2.removePiece(piece2)
            if (bin2.getFullness() + piece1.getSize()) <= self.capacity:
                bin2.addPiece(piece1)
            else:
                last = temporary[-1]
                last.addPiece(piece1)
                temporary.append(Bin())

            bin1.removePiece(piece1)
            if (bin1.getFullness() + piece2.getSize()) <= self.capacity:
                bin1.addPiece(piece2)
            else:
                last = temporary[-1]
                last.addPiece(piece2)
                temporary.append(Bin())

            self.sortbins(temporary, self.HIGHEST_FIRST)

    def ruinAndRecreate(self, numberOfBinsToRemove: int, temporary: List[Bin], highestOrLowest: bool) -> None:
        tempBinArray: List[Bin] = []
        if not highestOrLowest:
            self.sortbins(temporary, self.LOWEST_FIRST)
        for _ in range(numberOfBinsToRemove):
            tempBinArray.append(temporary.pop(0))
        if not highestOrLowest:
            self.sortbins(temporary, self.HIGHEST_FIRST)
        self.applyBestFit(tempBinArray, temporary)

    def applyHeuristic1(self, temporary: List[Bin]) -> None:
        iom = self.intensityOfMutation
        if iom <= 0.2:
            self.ruinAndRecreate(3, temporary, self.HIGHEST_FIRST)
        elif iom <= 0.4:
            self.ruinAndRecreate(6, temporary, self.HIGHEST_FIRST)
        elif iom <= 0.6:
            self.ruinAndRecreate(9, temporary, self.HIGHEST_FIRST)
        elif iom <= 0.8:
            self.ruinAndRecreate(12, temporary, self.HIGHEST_FIRST)
        else:
            self.ruinAndRecreate(15, temporary, self.HIGHEST_FIRST)

    def applyHeuristic2(self, temporary: List[Bin]) -> None:
        iom = self.intensityOfMutation
        if iom <= 0.2:
            self.ruinAndRecreate(3, temporary, self.LOWEST_FIRST)
        elif iom <= 0.4:
            self.ruinAndRecreate(6, temporary, self.LOWEST_FIRST)
        elif iom <= 0.6:
            self.ruinAndRecreate(9, temporary, self.LOWEST_FIRST)
        elif iom <= 0.8:
            self.ruinAndRecreate(12, temporary, self.LOWEST_FIRST)
        else:
            self.ruinAndRecreate(15, temporary, self.LOWEST_FIRST)

    def applyHeuristic3(self, temporary: List[Bin]) -> None:
        for _ in range(self.mrepeats):
            self.ruinAndRecreate(1, temporary, self.LOWEST_FIRST)
            self.sortbins(temporary, self.HIGHEST_FIRST)

    def applyHeuristic4(self, temporary: List[Bin]) -> None:
        # LS: take largest piece from lowest bin, try swap with smaller from random bin; else swap with two smaller; else revert
        for _ in range(self.lrepeats):
            self.sortbins(temporary, self.LOWEST_FIRST)
            lowestbin = temporary[0]

            largestpieceinthisbin = 0.0
            largestpieceindex = -1
            for x in range(lowestbin.numberOfPiecesInThisBin()):
                if lowestbin.getPieceSize(x) > largestpieceinthisbin:
                    largestpieceinthisbin = lowestbin.getPieceSize(x)
                    largestpieceindex = x

            if largestpieceindex == -1:
                continue

            p1 = lowestbin.removePieceAt(largestpieceindex)

            # pick a random bin != 0
            while True:
                bin2 = self.rng.randrange(len(temporary))
                if bin2 != 0:
                    break
            randomBin = temporary[bin2]

            largestsmallerpiece = 0.0
            largestsmallerpieceindex = -1
            for x in range(randomBin.numberOfPiecesInThisBin()):
                piecesize = randomBin.getPieceSize(x)
                if piecesize < largestpieceinthisbin:
                    if (randomBin.getFullness() - piecesize + largestpieceinthisbin) <= self.capacity:
                        if piecesize > largestsmallerpiece:
                            largestsmallerpiece = piecesize
                            largestsmallerpieceindex = x

            if largestsmallerpieceindex != -1:
                p2 = randomBin.removePieceAt(largestsmallerpieceindex)
                lowestbin.addPiece(p2)
                randomBin.addPiece(p1)
            else:
                piece1index = -1
                piece2index = -1
                for x in range(randomBin.numberOfPiecesInThisBin()):
                    piece1size = randomBin.getPieceSize(x)
                    for y in range(randomBin.numberOfPiecesInThisBin()):
                        if y == x:
                            continue
                        piece2size = randomBin.getPieceSize(y)
                        if (piece1size + piece2size) < largestpieceinthisbin:
                            if (randomBin.getFullness() - piece1size - piece2size + largestpieceinthisbin) <= self.capacity:
                                piece1index = x
                                piece2index = y
                if piece1index != -1:
                    tworemoved = randomBin.removeTwoPieces(piece1index, piece2index)
                    randomBin.addPiece(p1)
                    lowestbin.addPiece(tworemoved[0])
                    lowestbin.addPiece(tworemoved[1])
                else:
                    lowestbin.addPiece(p1)

    def applyHeuristic5(self, temporary: List[Bin]) -> None:
        # pick a bin with >= average #pieces (roulette by emptiness), split half its pieces into new bin
        for _ in range(self.mrepeats):
            if len(temporary) <= 2:
                continue
            aver = 0.0
            for x in range(len(temporary) - 1):  # skip trailing empty bin
                aver += temporary[x].numberOfPiecesInThisBin()
            aver /= (len(temporary) - 1)

            if aver == 1:
                continue

            candidate_idxs: List[int] = []
            for x in range(len(temporary) - 1):
                if temporary[x].numberOfPiecesInThisBin() >= aver:
                    candidate_idxs.append(x)

            if not candidate_idxs:
                continue

            emptinesses: List[float] = []
            totalemptiness = 0.0
            for idx in candidate_idxs:
                totalemptiness += self.capacity - temporary[idx].getFullness()
                emptinesses.append(totalemptiness)

            roulettenumber = self.rng.random() * totalemptiness
            for i, cum in enumerate(emptinesses):
                if roulettenumber <= cum:
                    binToHalf = temporary[candidate_idxs[i]]
                    n_pieces = binToHalf.numberOfPiecesInThisBin()
                    takeout = int(math.floor(n_pieces / 2.0))
                    newbin = Bin()
                    for y in range(takeout):
                        newbin.addPiece(binToHalf.removePieceAt(self.rng.randrange(n_pieces - y)))
                    temporary.append(newbin)
                    break

            self.sortbins(temporary, self.HIGHEST_FIRST)

    def applyHeuristic6(self, temporary: List[Bin]) -> None:
        # LS: swap two random different-size pieces only if feasible + "beneficial"
        for _ in range(self.lrepeats):
            piece1 = self.pieces[self.rng.randrange(self.numberOfPieces)]
            piece2 = self.pieces[self.rng.randrange(self.numberOfPieces)]
            while piece1.getSize() == piece2.getSize():
                piece2 = self.pieces[self.rng.randrange(self.numberOfPieces)]

            bin1index = -1
            bin2index = -1
            for x, b in enumerate(temporary):
                if b.indexOfPiece(piece1) != -1:
                    bin1index = x
                if b.indexOfPiece(piece2) != -1:
                    bin2index = x
                if bin1index != -1 and bin2index != -1:
                    break

            bin1 = temporary[bin1index]
            bin2 = temporary[bin2index]

            possible = True
            beneficial = False
            if bin1.getFullness() - piece1.getSize() + piece2.getSize() > self.capacity:
                possible = False
            if bin2.getFullness() - piece2.getSize() + piece1.getSize() > self.capacity:
                possible = False

            if possible:
                if bin1.getFullness() > bin2.getFullness():
                    if piece1.getSize() < piece2.getSize():
                        beneficial = True
                elif bin2.getFullness() > bin1.getFullness():
                    if piece2.getSize() < piece1.getSize():
                        beneficial = True
                else:
                    beneficial = True

            if beneficial:
                bin2.removePiece(piece2)
                bin1.removePiece(piece1)
                bin2.addPiece(piece1)
                bin1.addPiece(piece2)

            self.sortbins(temporary, self.HIGHEST_FIRST)

    def applyHeuristic7(self, child: List[Bin], other: List[Bin]) -> None:
        # crossover from "exon shuffling" style
        binlist: List[Bin] = []
        binlist.extend(child)
        binlist.extend(other)
        binlist.sort()  # descending fullness
        binlist.pop()   # remove the last (empty bin)

        child.clear()

        # phase 1: add mutually exclusive bins
        numberspacked: List[int] = []
        i = 0
        while i < len(binlist):
            b = binlist[i]
            newbin = True
            for p in numberspacked:
                if b.containsNumber(p):
                    newbin = False
                    break
            if newbin:
                child.append(b)
                b.copyPieceNumbers(numberspacked)
                binlist.pop(i)
            else:
                i += 1

        # phase 2: remove already-included pieces from remaining bins
        j = 0
        numbers_set = set(numberspacked)
        while j < len(binlist):
            b = binlist[j]
            b.pieces_in_this_bin = [p for p in b.pieces_in_this_bin if p.getNumber() not in numbers_set]
            # update packed set with anything left (matches Java intent)
            for p in b.pieces_in_this_bin:
                numbers_set.add(p.getNumber())
            if b.numberOfPiecesInThisBin() == 0:
                binlist.pop(j)
            else:
                j += 1

        self.applyBestFit(binlist, child)
        self.sanitycheck(child)

    # --- copying / sorting / sanity ---
    def deepCopyBins(self, vectorToCopy: List[Bin]) -> List[Bin]:
        return [b.clone() for b in vectorToCopy]

    def sortbins(self, bins: List[Bin], highestOrLowest: bool) -> None:
        bins.sort()  # uses Bin.__lt__ descending fullness

        if len(bins) < 2:
            return

        endbin = bins[-1]
        endbin2 = bins[-2]
        if endbin.getFullness() != 0.0:
            raise RuntimeError("The last bin is not empty, so there are no empty bins")
        if endbin2.getFullness() == 0.0:
            # Java prints debug and exits
            raise RuntimeError("There is more than one empty bin")

        if not highestOrLowest:
            # reverse everything except the last empty bin
            core = bins[:-1]
            core.reverse()
            bins[:] = core + [bins[-1]]

    def sanitycheck(self, v: List[Bin]) -> bool:
        totalnumberofpieces = 0
        totalfullness = 0.0
        allnumbers = [False] * self.numberOfPieces

        for x, b in enumerate(v):
            totalnumberofpieces += b.numberOfPiecesInThisBin()
            totalfullness += b.getFullness()
            for p in b.pieces_in_this_bin:
                allnumbers[int(p.getNumber())] = True
            if b.getFullness() > self.capacity:
                raise RuntimeError(f"bin {x} is overfilled")

        for x in range(self.numberOfPieces):
            if not allnumbers[x]:
                raise RuntimeError(f"piece number {x} is not present in the solution")

        if totalnumberofpieces != self.numberOfPieces:
            raise RuntimeError("there are not the correct number of pieces")

        if totalfullness != self.totalpiecesize:
            raise RuntimeError("the pieces do not add up to the correct size")

        return True

    # --- API expected by framework ---
    def getNumberOfHeuristics(self) -> int:
        return 8

    def applyHeuristicUnary(self, heuristicID: int, source: int, destination: int) -> float:
        import time
        start = time.time()

        if self.solutionMemory[source] is None:
            raise RuntimeError("Source solution not initialised")

        temporary = self.deepCopyBins(self.solutionMemory[source].solution)

        isCrossover = heuristicID in self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        if isCrossover:
            if self.solutionMemory[destination] is None:
                self.solutionMemory[destination] = BinPacking.Solution()
            self.solutionMemory[destination].solution = self.deepCopyBins(self.solutionMemory[source].solution)
        else:
            if heuristicID == 0:
                self.applyHeuristic0(temporary)
            elif heuristicID == 1:
                self.applyHeuristic1(temporary)
            elif heuristicID == 2:
                self.applyHeuristic2(temporary)
            elif heuristicID == 3:
                self.applyHeuristic3(temporary)
            elif heuristicID == 4:
                self.applyHeuristic4(temporary)
            elif heuristicID == 5:
                self.applyHeuristic5(temporary)
            elif heuristicID == 6:
                self.applyHeuristic6(temporary)
            else:
                raise ValueError(f"Heuristic {heuristicID} does not exist")

            self.heuristicCallRecord[heuristicID] += 1
            self.heuristicCallTimeRecord[heuristicID] += int((time.time() - start) * 1000)

        # normalise solution
        self.sortbins(temporary, self.HIGHEST_FIRST)
        self.sanitycheck(temporary)

        newobj = self.evaluateObjectiveFunction(temporary)
        if newobj < self.bestEverObjectiveFunction:
            self.bestEverObjectiveFunction = newobj
            self.bestEverNumberOfBins = len(temporary) - 1
            self.bestEverSolution = self.deepCopyBins(temporary)

        if self.solutionMemory[destination] is None:
            self.solutionMemory[destination] = BinPacking.Solution()
        self.solutionMemory[destination].solution = self.deepCopyBins(temporary)

        return newobj

    def applyHeuristicCrossover(self, heuristicID: int, source1: int, source2: int, destination: int) -> float:
        import time
        start = time.time()

        if self.solutionMemory[source1] is None or self.solutionMemory[source2] is None:
            raise RuntimeError("Source solution not initialised")

        temp1 = self.deepCopyBins(self.solutionMemory[source1].solution)
        temp2 = self.deepCopyBins(self.solutionMemory[source2].solution)

        isCrossover = heuristicID in self.getHeuristicsOfType(HeuristicType.CROSSOVER)
        if isCrossover:
            if heuristicID == 7:
                self.applyHeuristic7(temp1, temp2)
            else:
                raise ValueError(f"Heuristic {heuristicID} is not a crossover operator")
        else:
            # same as 1-parent call but using temp1
            if heuristicID == 0:
                self.applyHeuristic0(temp1)
            elif heuristicID == 1:
                self.applyHeuristic1(temp1)
            elif heuristicID == 2:
                self.applyHeuristic2(temp1)
            elif heuristicID == 3:
                self.applyHeuristic3(temp1)
            elif heuristicID == 4:
                self.applyHeuristic4(temp1)
            elif heuristicID == 5:
                self.applyHeuristic5(temp1)
            elif heuristicID == 6:
                self.applyHeuristic6(temp1)
            else:
                raise ValueError(f"Heuristic {heuristicID} does not exist")

        self.heuristicCallRecord[heuristicID] += 1
        self.heuristicCallTimeRecord[heuristicID] += int((time.time() - start) * 1000)

        self.sortbins(temp1, self.HIGHEST_FIRST)
        self.sanitycheck(temp1)

        newobj = self.evaluateObjectiveFunction(temp1)
        if newobj < self.bestEverObjectiveFunction:
            self.bestEverObjectiveFunction = newobj
            self.bestEverNumberOfBins = len(temp1) - 1
            self.bestEverSolution = self.deepCopyBins(temp1)

        if self.solutionMemory[destination] is None:
            self.solutionMemory[destination] = BinPacking.Solution()
        self.solutionMemory[destination].solution = self.deepCopyBins(temp1)

        return newobj
    
    def applyHeuristic(
        self,
        heuristicID: int,
        solutionSourceIndex1: int,
        solutionSourceIndex2: int = None,
        solutionDestinationIndex: int = None
    ) -> float:
        """
        HyFlex-compatible unified applyHeuristic.

        Supports:
        - Unary:     applyHeuristic(h, src, dst)
                    applyHeuristic(h, src, None, dst)
        - Crossover: applyHeuristic(h, src1, src2, dst)
        """

        # Called as applyHeuristic(h, src, dst)
        if solutionDestinationIndex is None:
            solutionDestinationIndex = solutionSourceIndex2
            solutionSourceIndex2 = None

        if solutionDestinationIndex is None:
            raise TypeError("applyHeuristic requires a destination index")

        is_crossover = heuristicID in set(self.getHeuristicsOfType(HeuristicType.CROSSOVER) or [])

        if is_crossover:
            if solutionSourceIndex2 is None:
                # This is the key: don’t silently copy for crossover.
                # Force the HH (or caller) to provide parent2 correctly.
                raise TypeError(
                    f"Heuristic {heuristicID} is CROSSOVER but was called without a second parent"
                )
            return self.applyHeuristicCrossover(
                heuristicID,
                solutionSourceIndex1,
                solutionSourceIndex2,
                solutionDestinationIndex,
            )

        # Not crossover -> unary path
        return self.applyHeuristicUnary(heuristicID, solutionSourceIndex1, solutionDestinationIndex)

    def copySolution(self, source: int, destination: int) -> None:
        if self.solutionMemory[source] is None:
            raise RuntimeError("Source solution not initialised")
        if self.solutionMemory[destination] is None:
            self.solutionMemory[destination] = BinPacking.Solution()
        self.solutionMemory[destination].solution = self.deepCopyBins(self.solutionMemory[source].solution)

    def solutionToString(self, index: int) -> str:
        if self.solutionMemory[index] is None:
            return ""
        s = ""
        for binNumber in range(self.solutionMemory[index].size()):
            b = self.solutionMemory[index].get(binNumber)
            s += f"{binNumber} "
            s = b.addToString(s)
        return s

    def bestSolutionToString(self) -> str:
        if self.bestEverSolution is None:
            return "Best Solution Found:\nObjective Function Value: inf\n"
        s = "Best Solution Found:\n"
        for b in self.bestEverSolution:
            s = b.addToString(s)
        s += f"Objective Function Value: {self.bestEverNumberOfBins}\n"
        return s

    def getFunctionValue(self, index: int) -> float:
        if self.solutionMemory[index] is None:
            return float("inf")
        return self.evaluateObjectiveFunction(self.solutionMemory[index].solution)

    def getBestSolutionValue(self) -> float:
        return self.bestEverObjectiveFunction

    def setMemorySize(self, size: int) -> None:
        newMem: List[Optional[BinPacking.Solution]] = [None] * size
        if self.solutionMemory:
            for x, s in enumerate(self.solutionMemory):
                if x < size:
                    newMem[x] = s
        self.solutionMemory = newMem

    def getNumberOfInstances(self) -> int:
        return 12

    def __str__(self) -> str:
        return "BinPacking"

    def compareSolutions(self, solutionIndex1: int, solutionIndex2: int) -> bool:
        s1 = self.solutionMemory[solutionIndex1]
        s2 = self.solutionMemory[solutionIndex2]
        if s1 is None or s2 is None:
            return False
        if s1.size() != s2.size():
            return False
        for i in range(s1.size()):
            b1 = s1.get(i)
            b2 = s2.get(i)
            if b1.compareTo(b2) != 0 or b1.numberOfPiecesInThisBin() != b2.numberOfPiecesInThisBin():
                return False
            for x in range(b1.numberOfPiecesInThisBin()):
                if b1.getPieceSize(x) != b2.getPieceSize(x):
                    return False
        return True

    def getHeuristicsOfType(self, hType) -> Optional[List[int]]:
        if hType == HeuristicType.LOCAL_SEARCH:
            return self.localSearches
        if hType == HeuristicType.RUIN_RECREATE:
            return self.ruinRecreates
        if hType == HeuristicType.MUTATION:
            return self.mutations
        if hType == HeuristicType.CROSSOVER:
            return self.crossovers
        return None

    # --- nested classes ---
    class returnSolution:
        def __init__(self, s: List[Piece], v: float):
            self.solution = s
            self.value = v

    class Solution:
        def __init__(self):
            self.solution: List[Bin] = []

        def addBin(self, b: Bin) -> None:
            self.solution.append(b)

        def size(self) -> int:
            return len(self.solution)

        def get(self, index: int) -> Bin:
            return self.solution[index]

        def set(self, index: int, b: Bin) -> None:
            self.solution[index] = b