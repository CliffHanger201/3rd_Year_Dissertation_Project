# package VRP

# Assumes Location exists (or will be created) with:
# - compareLocation(other_location)

from __future__ import annotations
from typing import Optional


class RouteItem:
    def __init__(self, cl, p: Optional["RouteItem"], n: Optional["RouteItem"], ta: float):
        self.currLocation = None
        self.prev: Optional["RouteItem"] = None
        self.next: Optional["RouteItem"] = None
        self.timeArrived: float = 0.0
        self.waitingTime: float = 0.0

        self.setCurrLocation(cl)
        self.setPrev(p)
        self.setNext(n)
        self.setTimeArrived(ta)

    def compareRouteItem(self, ri: "RouteItem") -> bool:
        identical = True
        if not (
            self.currLocation.compareLocation(ri.getCurrLocation())
            and self.timeArrived == ri.getTimeArrived()
            and self.waitingTime == ri.getWaitingTime()
        ):
            identical = False
        return identical

    def setCurrLocation(self, currLocation) -> None:
        self.currLocation = currLocation

    def getCurrLocation(self):
        return self.currLocation

    def setPrev(self, prev: Optional["RouteItem"]) -> None:
        self.prev = prev

    def getPrev(self) -> Optional["RouteItem"]:
        return self.prev

    def setNext(self, next: Optional["RouteItem"]) -> None:
        self.next = next

    def getNext(self) -> Optional["RouteItem"]:
        return self.next

    def setTimeArrived(self, timeArrived: float) -> None:
        self.timeArrived = timeArrived

    def getTimeArrived(self) -> float:
        return self.timeArrived

    def setWaitingTime(self, waitingTime: float) -> None:
        self.waitingTime = waitingTime

    def getWaitingTime(self) -> float:
        return self.waitingTime
