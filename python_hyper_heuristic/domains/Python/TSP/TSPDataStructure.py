
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


class TspDataStructure(ABC):
    @staticmethod
    def create(tour: List[int]) -> "TspDataStructure":
        # arrayRepresentation is assumed to exist elsewhere (like in Java).
        if len(tour) <= 5000:
            return arrayRepresentation(tour)  # type: ignore[name-defined]
        return TwoLayList(tour)

    @abstractmethod
    def next(self, v: int) -> int:
        """Return the city index that is next to v."""
        raise NotImplementedError

    @abstractmethod
    def prev(self, v: int) -> int:
        """Return the city index that is previous to v."""
        raise NotImplementedError

    @abstractmethod
    def sequence(self, a: int, b: int, c: int) -> bool:
        """
        True if b is between a and c, false otherwise.
        False if b == a or b == c.
        """
        raise NotImplementedError

    @abstractmethod
    def flip(self, b: int, d: int) -> None:
        """Reverse the path between cities b and d."""
        raise NotImplementedError

    @abstractmethod
    def toString(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def toStringFrom(self, start_city: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def returnTour(self, start_city: int) -> List[int]:
        """Return permutation representing a tour starting at start_city."""
        raise NotImplementedError

    @abstractmethod
    def returnTourInto(self, start_city: int, tour_out: List[int]) -> None:
        """Save tour starting at start_city into tour_out (must be sized)."""
        raise NotImplementedError


# -----------------------------
# Helpers that TwoLayList needs
# -----------------------------

@dataclass
class Parent:
    id: int
    previous: Optional["Parent"] = None
    next: Optional["Parent"] = None
    reverse: bool = False
    startSegment: Optional["Node"] = None
    endSegment: Optional["Node"] = None

    # Java: returnStart/returnEnd
    def returnStart(self) -> "Node":
        assert self.startSegment is not None
        return self.startSegment

    def returnEnd(self) -> "Node":
        assert self.endSegment is not None
        return self.endSegment

    # Java: setStart/setEnd
    def setStart(self, n: "Node") -> None:
        self.startSegment = n

    def setEnd(self, n: "Node") -> None:
        self.endSegment = n


@dataclass
class Node:
    city: int
    id: int = 0
    next: Optional["Node"] = None
    previous: Optional["Node"] = None
    parent: Optional[Parent] = None

    # Java used setNext/setPrevious helpers that respect parent.reverse
    def setNext(self, n: "Node") -> None:
        assert self.parent is not None
        if not self.parent.reverse:
            self.next = n
        else:
            self.previous = n

    def setPrevious(self, n: "Node") -> None:
        assert self.parent is not None
        if not self.parent.reverse:
            self.previous = n
        else:
            self.next = n


class TwoLayList(TspDataStructure):
    """
    Two-level linked list structure.
    This is converted from the provided Java portion (first half).
    """

    def __init__(self, permutation: List[int]):
        self.list: List[Node] = []
        self.parents: List[Parent] = []
        self.numbCities: int = len(permutation)
        self.maxSegmentSize: int = int(0.5 * math.sqrt(self.numbCities)) + 1
        self.numbParents: int = int(math.sqrt(self.numbCities))
        numbCitiesPerSegment: int = int(self.numbCities / self.numbParents) + 1

        self.sumNodes: int = 0
        self.sumFlips: int = 0

        # Step 1
        self._initialise_list(permutation)
        self._initialise_parents(numbCitiesPerSegment)

        # Step 2: link nodes to parents
        count = 0
        for i in range(self.numbParents - 1):
            tempP = self.parents[i]
            self.list[permutation[count]].parent = tempP
            tempP.startSegment = self.list[permutation[count]]
            for _j in range(1, numbCitiesPerSegment):
                self.list[permutation[count]].parent = tempP
                count += 1
            tempP.endSegment = self.list[permutation[count - 1]]

        # Step 2 continuation: last segment
        tempP = self.parents[self.numbParents - 1]
        tempP.startSegment = self.list[permutation[count]]
        for count in range(count, self.numbCities):
            self.list[permutation[count]].parent = tempP
        tempP.endSegment = self.list[permutation[self.numbCities - 1]]

    # ---------- Node-level next/prev (respect segment reverse bit) ----------

    def _next_node(self, v: Node) -> Node:
        assert v.parent is not None
        if not v.parent.reverse:
            assert v.next is not None
            return v.next
        assert v.previous is not None
        return v.previous

    def _prev_node(self, v: Node) -> Node:
        assert v.parent is not None
        if not v.parent.reverse:
            assert v.previous is not None
            return v.previous
        assert v.next is not None
        return v.next

    # ---------- Public next/prev on city indices ----------

    def next(self, v: int) -> int:
        node = self.list[v]
        assert node.parent is not None
        if not node.parent.reverse:
            assert node.next is not None
            return node.next.city
        assert node.previous is not None
        return node.previous.city

    def prev(self, v: int) -> int:
        node = self.list[v]
        assert node.parent is not None
        if not node.parent.reverse:
            assert node.previous is not None
            return node.previous.city
        assert node.next is not None
        return node.next.city

    # ---------- sequence(a,b,c) ----------

    def sequence(self, a: int, b: int, c: int) -> bool:
        pa = self.list[a].parent
        pb = self.list[b].parent
        pc = self.list[c].parent
        assert pa is not None and pb is not None and pc is not None

        pa_id, pb_id, pc_id = pa.id, pb.id, pc.id

        # Case 1: all different parents
        if pa is not pb and pa is not pc and pb is not pc:
            if ((pa_id < pb_id < pc_id) or (pa_id > pc_id and (pc_id > pb_id or pb_id > pa_id))):
                return True
            return False

        a_id = self.list[a].id
        b_id = self.list[b].id
        c_id = self.list[c].id

        # Case 2: all same parent
        if pa is pb and pb is pc:
            if not pa.reverse:
                if ((a_id < b_id < c_id) or (a_id > c_id and (c_id > b_id or b_id > a_id))):
                    return True
                return False
            else:
                if ((a_id > b_id > c_id) or (a_id < c_id and (c_id < b_id or b_id < a_id))):
                    return True
                return False

        # Case 3: parents a and b same
        if pa is pb:
            if not pa.reverse:
                return a_id < b_id
            return a_id > b_id

        # Case 4: parents a and c same
        if pa is pc:
            if not pa.reverse:
                return a_id > c_id
            return a_id < c_id

        # Case 5: parents b and c same
        if not pb.reverse:
            return b_id < c_id
        return b_id > c_id

    # ---------- flip(ib,id) ----------

    def flip(self, ib: int, id_: int) -> None:
        if ib == id_:
            return

        b = self.list[ib]
        d = self.list[id_]

        # if reversing full tour
        if self._next_node(d) is b or self._prev_node(b) is self._next_node(d):
            self.reverseTour()
            return

        a = self._prev_node(b)
        c = self._next_node(d)

        assert b.parent is not None and d.parent is not None and a.parent is not None and c.parent is not None

        if b.parent is d.parent:
            # b and d in same segment
            if c.parent is a.parent:
                # a and c also same segment
                if ((b.id < d.id and not b.parent.reverse) or (b.id > d.id and b.parent.reverse)):
                    # case 1: b precedes d
                    if abs(b.id - d.id) > self.maxSegmentSize:
                        self.splitAFirst(b)
                        self.splitALast(d)  # defined later (not in this half)
                    self._flip_sub_segment(b, d)
                else:
                    # case 2: d precedes b
                    if abs(c.id - a.id) > self.maxSegmentSize:
                        self.splitAFirst(c)
                        self.splitALast(a)  # defined later
                    self._flip_sub_segment(c, a)
                    self.reverseTour()
            else:
                # b and d same segment but not a and c
                if abs(b.id - d.id) > self.maxSegmentSize:
                    self.splitAFirst(b)
                    self.splitALast(d)  # defined later
                self._flip_sub_segment(b, d)
        else:
            # b and d not in same segment
            if c.parent is a.parent:
                # c and a same segment
                if abs(c.id - a.id) > self.maxSegmentSize:
                    self.splitAFirst(c)
                    self.splitALast(a)  # defined later
                self._flip_sub_segment(c, a)
                self.reverseTour()
            else:
                # neither (b,d) nor (a,c) same segment
                # Case 1: path b-d shorter than a-c
                if (
                    abs(b.parent.id - d.parent.id) <= abs(a.parent.id - c.parent.id)
                    and d.parent.next is not b.parent
                ):
                    if b.parent.returnStart() is not b:
                        self.splitAFirst(b)
                    if b.parent is not d.parent and d.parent.returnEnd() is not d:
                        self.splitALast(d)  # defined later

                    if b.parent is d.parent:
                        if abs(b.id - d.id) > self.maxSegmentSize:
                            self.splitAFirst(b)
                            self.splitALast(d)  # defined later
                        self._flip_sub_segment(b, d)
                    else:
                        if b.parent.id > d.parent.id:
                            self._flip_contiguous_segments2(b.parent, d.parent)
                        else:
                            self._flip_contiguous_segments(b.parent, d.parent)
                else:
                    # Case 2: path a-c shorter than b-d
                    if c.parent.returnStart() is not c:
                        self.splitAFirst(c)
                    if c.parent is not a.parent and a.parent.returnEnd() is not a:
                        self.splitALast(a)  # defined later

                    if c.parent is a.parent:
                        if abs(c.id - a.id) > self.maxSegmentSize:
                            self.splitAFirst(c)
                            self.splitALast(a)  # defined later
                        self._flip_sub_segment(c, a)
                        self.reverseTour()
                    else:
                        if c.parent.id > a.parent.id:
                            self._flip_contiguous_segments2(c.parent, a.parent)
                        else:
                            self._flip_contiguous_segments(c.parent, a.parent)
                        self.reverseTour()

    # ---------- utility / tour output ----------

    def toStringFrom(self, start_city: int) -> str:
        pi = self.returnTour(start_city)
        return " ".join(str(x) for x in pi) + " "

    def toString(self) -> str:
        return self.toStringFrom(0)

    def returnTour(self, start_city: int) -> List[int]:
        pi = [0] * self.numbCities
        self.returnTourInto(start_city, pi)
        return pi

    def returnTourInto(self, start_city: int, tour_out: List[int]) -> None:
        count = 0
        start = self.list[start_city]
        aux = self._next_node(start)
        tour_out[0] = start.city
        count += 1
        while aux is not start:
            tour_out[count] = aux.city
            aux = self._next_node(aux)
            count += 1

    # ------------------------
    # PRIVATE METHODS (HALF 1)
    # ------------------------

    def _initialise_list(self, permutation: List[int]) -> None:
        self.list = [Node(i) for i in range(self.numbCities)]

        # initialise id values
        for i in range(self.numbCities):
            self.list[permutation[i]].id = self.numbCities + i

        # previous links
        self.list[permutation[0]].previous = self.list[permutation[self.numbCities - 1]]
        for i in range(1, self.numbCities):
            self.list[permutation[i]].previous = self.list[permutation[i - 1]]

        # next links
        self.list[permutation[self.numbCities - 1]].next = self.list[permutation[0]]
        for i in range(0, self.numbCities - 1):
            self.list[permutation[i]].next = self.list[permutation[i + 1]]

    def _initialise_parents(self, size: int) -> None:
        small_size = self.numbCities - (self.numbParents - 1) * size
        if small_size < self.numbParents / 2:
            size -= 1
            small_size += self.numbParents - 1

        self.parents = [Parent(i) for i in range(self.numbParents)]

        self.parents[0].previous = self.parents[self.numbParents - 1]
        for i in range(1, self.numbParents):
            self.parents[i].previous = self.parents[i - 1]

        self.parents[self.numbParents - 1].next = self.parents[0]
        for i in range(0, self.numbParents - 1):
            self.parents[i].next = self.parents[i + 1]

    def _flip_sub_segment(self, b: Node, d: Node) -> None:
        self.sumFlips += 1
        self.sumNodes += abs(b.id - d.id)

        p = b.parent
        assert p is not None

        # if b and d are the start and end of the segment
        if p.returnStart() is b and p.returnEnd() is d:
            self._flip_segment(b, d, self._prev_node(b), self._next_node(d))
            return

        # make sure b.id < d.id
        if b.id > d.id:
            b, d = d, b

        a = b.previous
        c = d.next
        assert a is not None and c is not None
        assert c.parent is not None and a.parent is not None

        # Step 1: reconnect c and a
        if c.parent.reverse != p.reverse:
            c.next = b
            a.next = d
        else:
            c.previous = b
            if a.parent.reverse != p.reverse:
                a.previous = d
            else:
                a.next = d

        # Step 2: reverse path b-d
        if b.next is d:
            # adjacent
            d.previous = a
            d.next = b
            b.previous = d
            b.next = c
        else:
            d.next = d.previous
            aux = d.next
            assert aux is not None
            while aux is not b:
                aux.next = aux.previous
                aux = aux.previous
                assert aux is not None

            b.next = c
            d.previous = a

            aux2 = d
            while aux2 is not b:
                assert aux2.next is not None
                aux2.next.previous = aux2
                aux2 = aux2.next

        # Step 3: update start/end segment refs
        if p.startSegment is b:
            p.startSegment = d
        else:
            if p.endSegment is d:
                p.endSegment = b

        # Step 4: update id values
        d.id = b.id
        aux3 = d.next
        assert aux3 is not None
        while aux3 is not b:
            assert aux3.previous is not None
            aux3.id = aux3.previous.id + 1
            aux3 = aux3.next
            assert aux3 is not None
        assert b.previous is not None
        b.id = b.previous.id + 1

        self.updateAllRanks(b.parent)

    def _flip_segment(self, b: Node, d: Node, a: Node, c: Node) -> None:
        assert b.parent is not None
        # 1. complement reverse bit in parent
        b.parent.reverse = not b.parent.reverse

        # 2. swap segment external links
        b.setNext(c)
        c.setPrevious(b)
        d.setPrevious(a)
        a.setNext(d)

    def _flip_contiguous_segments(self, pb: Parent, pd: Parent) -> None:
        # pb must precede pd: pb.id < pd.id
        b = pb.returnStart()
        c = pd.returnEnd()
        a = self._prev_node(b)
        d = self._next_node(c)

        a.setNext(c)
        b.setPrevious(d)
        c.setNext(a)
        d.setPrevious(b)

        pa = pb.previous
        pc = pd.next
        assert pa is not None and pc is not None
        pa.next = pd
        pc.previous = pb

        if pb.next is pd:
            pb.next = pc
            pb.previous = pd
            pd.previous = pa
            pd.next = pb
        else:
            pd.next = pd.previous
            ptemp = pd.next
            assert ptemp is not None
            while ptemp is not pb:
                ptemp.next = ptemp.previous
                ptemp = ptemp.previous
                assert ptemp is not None
            pb.next = pc
            pd.previous = pa
            ptemp2 = pd
            while ptemp2 is not pb:
                assert ptemp2.next is not None
                ptemp2.next.previous = ptemp2
                ptemp2 = ptemp2.next

        # update ranks + complement reverse bits
        pd.id = pb.id
        pd.reverse = not pd.reverse
        ptemp3 = pd.next
        assert ptemp3 is not None
        while ptemp3 is not pb:
            ptemp3.reverse = not ptemp3.reverse
            assert ptemp3.previous is not None
            ptemp3.id = ptemp3.previous.id + 1
            ptemp3 = ptemp3.next
            assert ptemp3 is not None

        assert pb.previous is not None
        pb.id = pb.previous.id + 1
        pb.reverse = not pb.reverse

    def _flip_contiguous_segments2(self, pb: Parent, pd: Parent) -> None:
        # pb must succeed pd: pb.id > pd.id
        b = pb.returnStart()
        c = pd.returnEnd()
        a = self._prev_node(b)
        d = self._next_node(c)

        a.setNext(c)
        b.setPrevious(d)
        c.setNext(a)
        d.setPrevious(b)

        pa = pb.previous
        pc = pd.next
        assert pa is not None and pc is not None
        pa.next = pd
        pc.previous = pb

        if pb.next is pd:
            pb.next = pc
            pb.previous = pd
            pd.previous = pa
            pd.next = pb
        else:
            pb.previous = pb.next
            ptemp = pb.next
            assert ptemp is not None
            while ptemp is not pd:
                ptemp.previous = ptemp.next
                ptemp = ptemp.previous
                assert ptemp is not None

            pd.previous = pa
            pb.next = pc

            ptemp2 = pb
            while ptemp2 is not pd:
                assert ptemp2.previous is not None
                ptemp2.previous.next = ptemp2
                ptemp2 = ptemp2.previous

        self._update_parent_ranks()

        # complement reverse bits walking backwards pb -> pd
        ptemp3: Parent = pb
        ptemp3.reverse = not ptemp3.reverse
        while ptemp3 is not pd:
            assert ptemp3.previous is not None
            ptemp3 = ptemp3.previous
            ptemp3.reverse = not ptemp3.reverse

    def _update_parent_ranks(self) -> None:
        start = self.list[0].parent
        assert start is not None
        count = 0
        start.id = count
        aux = start.next
        while aux is not start:
            count += 1
            aux.id = count
            aux = aux.next
            assert aux is not None

    def splitAFirst(self, a: Node) -> None:
        """
        Splits a's segment so that a becomes the first city of its segment.
        Mirrors the Java logic exactly (including the reverse-bit cases).
        """
        assert a.parent is not None

        # Case 0: Move cities before a to previous segment
        if abs(a.parent.returnStart().id - a.id) <= abs(a.parent.returnEnd().id - a.id) + 1:
            aux = a.parent.returnStart()
            ptemp = aux.parent.previous  # type: ignore[union-attr]
            assert ptemp is not None
            aux2 = ptemp.returnEnd()

            if aux.parent.reverse:  # type: ignore[union-attr]
                if aux.parent.reverse == ptemp.reverse:  # type: ignore[union-attr]
                    while aux is not a:
                        aux.parent = ptemp
                        aux = aux.previous  # type: ignore[assignment]
                        assert aux is not None
                else:
                    while aux is not a:
                        aux.parent = ptemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.next  # type: ignore[assignment]
                        assert aux is not None
            else:
                if aux.parent.reverse == ptemp.reverse:  # type: ignore[union-attr]
                    while aux is not a:
                        aux.parent = ptemp
                        aux = aux.next  # type: ignore[assignment]
                        assert aux is not None
                else:
                    while aux is not a:
                        aux.parent = ptemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.previous  # type: ignore[assignment]
                        assert aux is not None

            # ptemp.setEnd(prev(aux))
            ptemp.setEnd(self._prev_node(aux))
            a.parent.setStart(a)

            self.updateRanksForw(aux2)

        else:
            # Case 2: Move a and following cities to next segment
            aux = a.parent.returnEnd()
            ptemp = aux.parent.next  # type: ignore[union-attr]
            assert ptemp is not None
            aux2 = ptemp.returnStart()

            if aux.parent.reverse:  # type: ignore[union-attr]
                if aux.parent.reverse == ptemp.reverse:  # type: ignore[union-attr]
                    while aux is not a:
                        aux.parent = ptemp
                        aux = aux.next  # type: ignore[assignment]
                        assert aux is not None
                else:
                    while aux is not a:
                        aux.parent = ptemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.previous  # type: ignore[assignment]
                        assert aux is not None
            else:
                if aux.parent.reverse == ptemp.reverse:  # type: ignore[union-attr]
                    while aux is not a:
                        aux.parent = ptemp
                        aux = aux.previous  # type: ignore[assignment]
                        assert aux is not None
                else:
                    while aux is not a:
                        aux.parent = ptemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.next  # type: ignore[assignment]
                        assert aux is not None

            a.parent.setEnd(self._prev_node(a))

            if a.parent.reverse != ptemp.reverse:
                aux_swap = a.next
                a.next = a.previous
                a.previous = aux_swap

            a.parent = ptemp
            a.parent.setStart(a)

            self.updateRanksBack(aux2)

    def splitALast(self, a):
        if abs(a.parent.returnStart().id - a.id) + 1 <= abs(a.parent.returnEnd().id - a.id):
            aux = a.parent.returnStart()
            pTemp = aux.parent.previous
            aux2 = pTemp.returnEnd()
            if aux.parent.reverse:
                if aux.parent.reverse == pTemp.reverse:
                    while aux is not a:
                        aux.parent = pTemp
                        aux = aux.previous
                else:
                    while aux is not a:
                        aux.parent = pTemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.next
            else:
                if aux.parent.reverse == pTemp.reverse:
                    while aux is not a:
                        aux.parent = pTemp
                        aux = aux.next
                else:
                    while aux is not a:
                        aux.parent = pTemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.previous

            a.parent.setStart(self._next_node(a))
            if a.parent.reverse != pTemp.reverse:
                aux = a.next
                a.next = a.previous
                a.previous = aux
            a.parent = pTemp
            pTemp.setEnd(a)
            self.updateRanksForw(aux2)
            # self.updateAllRanks(pTemp)
        else:
            aux = a.parent.returnEnd()
            pTemp = aux.parent.next
            aux2 = pTemp.returnStart()
            if aux.parent.reverse:
                if aux.parent.reverse == pTemp.reverse:
                    while aux is not a:
                        aux.parent = pTemp
                        aux = aux.next
                else:
                    while aux is not a:
                        aux.parent = pTemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.previous
            else:
                if aux.parent.reverse == pTemp.reverse:
                    while aux is not a:
                        aux.parent = pTemp
                        aux = aux.previous
                else:
                    while aux is not a:
                        aux.parent = pTemp
                        aux3 = aux.next
                        aux.next = aux.previous
                        aux.previous = aux3
                        aux = aux.next

            pTemp.setStart(self._next_node(a))
            a.parent.setEnd(a)
            # self.updateAllRanks(pTemp)
            self.updateRanksBack(aux2)


    def updateRanksForw(self, a):
        if not a.parent.reverse:
            aux = a
            end = a.parent.endSegment
            count = a.id
            while aux is not end:
                aux.id = count
                aux = aux.next
                count += 1
            end.id = count
        else:
            aux = a
            end = a.parent.startSegment
            count = a.id
            while aux is not end:
                aux.id = count
                aux = aux.previous
                count -= 1
            end.id = count


    def updateRanksBack(self, a):
        if not a.parent.reverse:
            aux = a
            end = a.parent.startSegment
            count = a.id
            while aux is not end:
                aux.id = count
                aux = aux.previous
                count -= 1
            end.id = count
        else:
            aux = a
            end = a.parent.endSegment
            count = a.id
            while aux is not end:
                aux.id = count
                aux = aux.next
                count += 1
            end.id = count


    def reverseTour(self):
        # Step 1: swap previous and next links
        for i in range(self.numbParents):
            temp = self.parents[i]
            temp2 = temp.previous
            temp.previous = temp.next
            temp.next = temp2
            temp.reverse = not temp.reverse

        # Step 2: update ranks
        count = 0
        self.parents[0].id = count
        temp = self.parents[0].next
        while temp is not self.parents[0]:
            count += 1
            temp.id = count
            temp = temp.next


    def updateAllRanks(self, parent):
        aux = parent.startSegment
        end = parent.endSegment
        count = self.numbCities
        while aux is not end:
            aux.id = count
            aux = aux.next
            count += 1
        end.id = count


    def getSegmentSizes(self):
        segSizes = [0] * self.numbParents
        for i in range(self.numbParents):
            segSizes[i] = abs(self.parents[i].startSegment.id - self.parents[i].endSegment.id)
        return segSizes


    def getNumbParents(self):
        return self.numbParents


    def verifyEnumeration(self):
        for i in range(self.numbParents):
            p = self.parents[i]
            count = p.startSegment.id
            count2 = count
            nxt = p.startSegment.next
            while (nxt is not p.endSegment) and (p.startSegment is not p.endSegment):
                count += 1
                count2 -= 1
                if not (nxt.id == count or nxt.id == count2):
                    return False
                nxt = nxt.next
        return True


# ---------------------------
# OUTSIDE TwoLayList:
# class arrayRepresentation extends TspDataStructure
# ---------------------------

class arrayRepresentation(TspDataStructure):
    def __init__(self, tour):
        self.tour = tour
        self.numbCities = len(tour)
        self.inverse = [0] * self.numbCities  # inverse[i] = position of city i in tour
        self.reversed = False
        for i in range(self.numbCities):
            self.inverse[tour[i]] = i

    def next(self, v):
        if not self.reversed:
            if (self.inverse[v] + 1) == self.numbCities:
                return self.tour[0]
            return self.tour[self.inverse[v] + 1]
        else:
            if self.inverse[v] == 0:
                return self.tour[self.numbCities - 1]
            return self.tour[self.inverse[v] - 1]

    def prev(self, v):
        if not self.reversed:
            if self.inverse[v] == 0:
                return self.tour[self.numbCities - 1]
            return self.tour[self.inverse[v] - 1]
        else:
            if (self.inverse[v] + 1) == self.numbCities:
                return self.tour[0]
            return self.tour[self.inverse[v] + 1]

    def sequence(self, a, b, c):
        ia = self.inverse[a]
        ib = self.inverse[b]
        ic = self.inverse[c]
        if not self.reversed:
            if ia < ic:
                if ib < ic and ib > ia:
                    return True
            else:
                if ib > ia or ib < ic:
                    return True
        else:
            if ic < ia:
                if ib < ia and ib > ic:
                    return True
            else:
                if ib < ia or ib > ic:
                    return True
        return False

    def flip(self, b, d):
        # Special cases
        if b == d:  # illegal movement
            return
        if self.next(d) == b or self.prev(b) == self.next(d):  # reversing whole tour
            self.reversed = not self.reversed
            return

        # General cases (4)
        if not self.reversed:
            if self.inverse[b] < self.inverse[d]:
                self.flip1(self.inverse[b], self.inverse[d])  # Case 1
            else:
                self.flip2(self.inverse[b], self.inverse[d])  # Case 2
        else:
            if self.inverse[b] < self.inverse[d]:
                self.flip3(self.inverse[b], self.inverse[d])  # Case 3
            else:
                self.flip4(self.inverse[b], self.inverse[d])  # Case 4

    def toString(self):
        return self.toStringFrom(0)

    def toStringFrom(self, startCity):
        tour = self.returnTour(startCity)
        return " ".join(str(x) for x in tour) + " "

    def returnTour(self, startCity):
        tour = [0] * self.numbCities
        self.returnTourInto(startCity, tour)
        return tour

    def returnTourInto(self, startCity, tour):
        tour[0] = startCity
        aux = self.next(startCity)
        for i in range(1, self.numbCities):
            tour[i] = aux
            aux = self.next(aux)

    # PRIVATE METHODS

    def flip1(self, ib, id_):
        if (id_ - ib + 1) < (self.numbCities / 2):
            self.flipIn(ib, id_)
        else:
            self.reversed = not self.reversed
            if ib == 0:
                self.flipIn(id_ + 1, self.numbCities - 1)
            else:
                if id_ == self.numbCities - 1:
                    self.flipIn(0, ib - 1)
                else:
                    self.flipOut(ib - 1, id_ + 1)

    def flip2(self, ib, id_):
        ia = ib - 1
        ic = id_ + 1
        if (ia - ic + 1) < (self.numbCities / 2):
            self.reversed = not self.reversed
            self.flipIn(ic, ia)
        else:
            self.flipOut(id_, ib)

    def flip3(self, ib, id_):
        ia = ib + 1
        ic = id_ - 1
        if (ic - ia + 1) < (self.numbCities / 2):
            self.reversed = not self.reversed
            self.flipIn(ia, ic)
        else:
            self.flipOut(ib, id_)

    def flip4(self, ib, id_):
        if (ib - id_ + 1) < (self.numbCities / 2):
            self.flipIn(id_, ib)
        else:
            self.reversed = not self.reversed
            if id_ == 0:
                self.flipIn(ib + 1, self.numbCities - 1)
            else:
                if ib == self.numbCities - 1:
                    self.flipIn(0, id_ - 1)
                else:
                    self.flipOut(id_ - 1, ib + 1)

    def flipIn(self, ib, id_):
        q = (id_ - ib + 1) // 2
        for count in range(q):
            temp = self.tour[ib + count]
            self.inverse[self.tour[id_ - count]] = ib + count
            self.tour[ib + count] = self.tour[id_ - count]
            self.inverse[temp] = id_ - count
            self.tour[id_ - count] = temp

    def flipOut(self, ib, id_):
        q = (ib + 1) if (ib + 1) < (self.numbCities - id_) else (self.numbCities - id_)

        # Step 1: swap (ib with id), (ib-1 with id+1) ...
        for count in range(q):
            temp = self.tour[id_ + count]
            self.inverse[self.tour[ib - count]] = id_ + count
            self.tour[id_ + count] = self.tour[ib - count]
            self.inverse[temp] = ib - count
            self.tour[ib - count] = temp

        # Step 2: reverse leftover unswapped segment, if needed
        if (self.numbCities - 1) - id_ - q > 0:
            self.flipIn(id_ + q, self.numbCities - 1)
        else:
            if ib - q > 0:
                self.flipIn(0, ib - q)