# package VRP

class Location:
    def __init__(self, iD: int, xc: int, yc: int, d: int, rt: int, dd: int, st: int):
        self.id: int = 0
        self.xCoord: int = 0
        self.yCoord: int = 0
        self.demand: int = 0
        self.dueDate: int = 0
        self.readyTime: int = 0
        self.serviceTime: int = 0
        self.serviced: bool = False

        self.setId(iD)
        self.setXCoord(xc)
        self.setYCoord(yc)
        self.setDemand(d)
        self.setDueDate(dd)
        self.setServiceTime(st)
        self.setReadyTime(rt)

    def copyLocation(self) -> "Location":
        return Location(
            self.id,
            self.xCoord,
            self.yCoord,
            self.demand,
            self.readyTime,
            self.dueDate,
            self.serviceTime,
        )

    def compareLocation(self, loc: "Location") -> bool:
        return (
            self.id == loc.getId()
            and self.xCoord == loc.getXCoord()
            and self.yCoord == loc.getYCoord()
            and self.demand == loc.getDemand()
            and self.dueDate == loc.getDueDate()
            and self.readyTime == loc.getReadyTime()
            and self.serviceTime == loc.getServiceTime()
        )

    def setId(self, id: int) -> None:
        self.id = id

    def getId(self) -> int:
        return self.id

    def setXCoord(self, xCoord: int) -> None:
        self.xCoord = xCoord

    def getXCoord(self) -> int:
        return self.xCoord

    def setYCoord(self, yCoord: int) -> None:
        self.yCoord = yCoord

    def getYCoord(self) -> int:
        return self.yCoord

    def setDemand(self, demand: int) -> None:
        self.demand = demand

    def getDemand(self) -> int:
        return self.demand

    def setDueDate(self, dueDate: int) -> None:
        self.dueDate = dueDate

    def getDueDate(self) -> int:
        return self.dueDate

    def setServiceTime(self, serviceTime: int) -> None:
        self.serviceTime = serviceTime

    def getServiceTime(self) -> int:
        return self.serviceTime

    def setServiced(self, serviced: bool) -> None:
        self.serviced = serviced

    def isServiced(self) -> bool:
        return self.serviced

    def setReadyTime(self, readyTime: int) -> None:
        self.readyTime = readyTime

    def getReadyTime(self) -> int:
        return self.readyTime
