from typing import List
from pathlib import Path

# Assume Location exists elsewhere in your project.
# from VRP.Location import Location
# or: from location import Location
# Keep the import consistent with your existing structure.
from python_hyper_heuristic.domains.Python.VRP.Location import Location


class Instance:
    def __init__(self, id: int):
        self.demands: List[Location] = []
        self.instanceName: str = ""
        self.vehicleNumber: int = 0
        self.vehicleCapacity: int = 0
        self.depot: Location | None = None

        fileName = "data/vrp/"
        if id == 0:
            fileName += "Solomon_100_customer_instances/RC/RC207.txt"
        elif id == 1:
            fileName += "Solomon_100_customer_instances/R/R101.txt"
        elif id == 2:
            fileName += "Solomon_100_customer_instances/RC/RC103.txt"
        elif id == 3:
            fileName += "Solomon_100_customer_instances/R/R201.txt"
        elif id == 4:
            fileName += "Solomon_100_customer_instances/R/R106.txt"
        elif id == 5:
            fileName += "Homberger_1000_customer_instances/C/C1_10_1.TXT"
        elif id == 6:
            fileName += "Homberger_1000_customer_instances/RC/RC2_10_1.TXT"
        elif id == 7:
            fileName += "Homberger_1000_customer_instances/R/R1_10_1.TXT"
        elif id == 8:
            fileName += "Homberger_1000_customer_instances/C/C1_10_8.TXT"
        elif id == 9:
            fileName += "Homberger_1000_customer_instances/RC/RC1_10_5.TXT"

        # Java tries filesystem, then classloader resources.
        # In Python, simplest equivalent is: try open the file; otherwise error out.
        project_root = Path(__file__).resolve().parents[2] / "resources"  # adjust if needed
        file_path = project_root / fileName
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as reader:
                try:
                    self.instanceName = reader.readline().rstrip("\n")

                    # skip 3 lines
                    reader.readline()
                    reader.readline()
                    reader.readline()

                    # read vehicle number and capacity
                    info = reader.readline().split()
                    self.vehicleNumber = int(info[0])
                    self.vehicleCapacity = int(info[1])

                    # skip 4 lines
                    reader.readline()
                    reader.readline()
                    reader.readline()
                    reader.readline()

                    # read locations
                    for line in reader:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        loc = Location(
                            int(parts[0]),
                            int(parts[1]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                            int(parts[5]),
                            int(parts[6]),
                        )
                        self.demands.append(loc)

                    self.setDepot(self.demands[0])

                except Exception as e:
                    print(f"Exception found: {e}")
                    print("Could not load instance, or instance does not exist")
                    raise

        except FileNotFoundError:
            print(f"cannot find file {fileName}")
            raise

    def getDemands(self) -> List[Location]:
        return self.demands

    def setDemands(self, demands: List[Location]) -> None:
        self.demands = demands

    def getInstanceName(self) -> str:
        return self.instanceName

    def setInstanceName(self, instanceName: str) -> None:
        self.instanceName = instanceName

    def getVehicleNumber(self) -> int:
        return self.vehicleNumber

    def setVehicleNumber(self, vehicleNumber: int) -> None:
        self.vehicleNumber = vehicleNumber

    def getVehicleCapacity(self) -> int:
        return self.vehicleCapacity

    def setVehicleCapacity(self, vehicleCapacity: int) -> None:
        self.vehicleCapacity = vehicleCapacity

    def setDepot(self, depot: Location) -> None:
        self.depot = depot

    def getDepot(self) -> Location:
        return self.depot