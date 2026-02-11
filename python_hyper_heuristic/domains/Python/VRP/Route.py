# package VRP

# Assumes Location and RouteItem exist (or will be created) with the same methods used here:
# - Location: copyLocation(), getDemand(), getId(), getXCoord(), getYCoord()
# - RouteItem: constructor RouteItem(location, prev, next, timeArrived)
#             getCurrLocation(), getPrev(), getNext(), setPrev(), setNext(),
#             getTimeArrived(), getWaitingTime(), setWaitingTime(),
#             compareRouteItem(other)

from __future__ import annotations
from typing import Optional

from python_hyper_heuristic.domains.Python.VRP.RouteItem import RouteItem


class Route:
    def __init__(self, l, iD: int, t: int):
        self.id: int = 0
        self.first: Optional["RouteItem"] = None
        self.last: Optional["RouteItem"] = None
        self.volume: int = 0

        self.setId(iD)

        depot = RouteItem(l, None, None, 0)
        self.first = depot

        depot2 = RouteItem(l, self.first, None, t)
        self.last = depot2

        self.first.setNext(self.last)

    def copyRoute(self) -> "Route":
        newR = Route(self.first.getCurrLocation().copyLocation(), self.id, 0)
        newR.getFirst().setWaitingTime(self.first.getWaitingTime())

        currRI = self.first
        currNewRI = newR.getFirst()

        # Mirrors:
        # while((currRI=currRI.getNext()).getNext()!=null)
        while True:
            currRI = currRI.getNext()
            if currRI.getNext() is None:
                break

            currNewRI.setNext(
                RouteItem(
                    currRI.getCurrLocation().copyLocation(),
                    currNewRI,
                    currNewRI.getNext(),
                    currRI.getTimeArrived(),
                )
            )
            currNewRI = currNewRI.getNext()
            currNewRI.getNext().setPrev(currNewRI)
            currNewRI.setWaitingTime(currRI.getWaitingTime())

        newR.setVolume(self.volume)
        return newR

    def compareRoute(self, r: "Route") -> bool:
        identical = True

        thisRI = self.first
        thatRI = r.getFirst()

        while thisRI is not None:
            if not thisRI.compareRouteItem(thatRI):
                return False
            thisRI = thisRI.getNext()
            thatRI = thatRI.getNext()

        if not (self.id == r.getId() and self.volume == r.getVolume()):
            return False

        return identical

    def addPenultimate(self, l, t: float) -> None:
        ri = RouteItem(l, self.last.getPrev(), self.last, t)
        self.last.getPrev().setNext(ri)
        self.last.setPrev(ri)
        self.volume += l.getDemand()

    def insertAfter(self, ri: "RouteItem", l, t: float) -> None:
        if ri.getNext() is None:
            print("Last location must be depot")
        else:
            r = RouteItem(l, ri, ri.getNext(), t)
            ri.getNext().setPrev(r)
            ri.setNext(r)
            self.volume += l.getDemand()

    def removeRouteItem(self, ri: "RouteItem") -> None:
        if (ri.getPrev() is None) or (ri.getNext() is None):
            print("Cannot delete depot")
        else:
            ri.getPrev().setNext(ri.getNext())
            ri.getNext().setPrev(ri.getPrev())
            self.volume -= ri.getCurrLocation().getDemand()

    def printRoute(self) -> None:
        currItem = self.first
        while currItem is not None:
            loc = currItem.getCurrLocation()
            print(
                f"Location {loc.getId()} at ({loc.getXCoord()},{loc.getYCoord()}) "
                f"has been visited at {currItem.getTimeArrived()}"
            )
            currItem = currItem.getNext()
        print(self.volume)

    def sizeOfRoute(self) -> int:
        size = 1
        curr = self.first
        while True:
            curr = curr.getNext()
            if curr is None:
                break
            size += 1
        return size

    def calcVolume(self) -> int:
        ri = self.first
        volume = 0
        while ri is not None:
            volume += ri.getCurrLocation().getDemand()
            ri = ri.getNext()
        return volume

    def getFirst(self):
        return self.first

    def setFirst(self, first) -> None:
        self.first = first

    def getLast(self):
        return self.last

    def setLast(self, last) -> None:
        self.last = last

    def getVolume(self) -> int:
        return self.volume

    def setVolume(self, volume: int) -> None:
        self.volume = volume

    def setId(self, id: int) -> None:
        self.id = id

    def getId(self) -> int:
        return self.id
