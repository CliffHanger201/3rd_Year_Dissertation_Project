

class TspSolution:
    def __init__(self, permutation, cost):
        # permutation: list or numpy array of city indices
        self.permutation = list(permutation)
        self.cost = float(cost)

    def clone(self):
        # Deep copy of the solution (HyFlex-style safe copy)
        return TspSolution(self.permutation.copy(), self.cost)

    def __str__(self):
        perm_str = " ".join(str(x) for x in self.permutation)
        return f"Cost = {self.cost}\n {perm_str}"
