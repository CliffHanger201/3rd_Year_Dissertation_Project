"""
Visualisation for the pre-training stats of the hyperheuristic
"""

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential, layers, regularizers

    

# if __name__ == "__main__":
#     # ── Stub problem domain (replace with real HyFlex domains) ──────────────
#     class StubProblem:
#         def getNumberOfHeuristics(self): return 6
#         def setMemorySize(self, n): self._mem = [0.0] * n
#         def initialiseSolution(self, i): self._mem[i] = random.uniform(50, 100)
#         def copySolution(self, s, d): self._mem[d] = self._mem[s]
#         def applyHeuristic(self, h, s, d): self._mem[d] = max(0, self._mem[s] + random.gauss(-0.3, 2.0))
#         def getFunctionValue(self, i): return self._mem[i]
#         def getBestSolutionValue(self): return min(self._mem)
#         def getHeuristicsOfType(self, t): return []
 
#     H_COUNT  = 6
#     STATE_DIM = H_COUNT * 4 + 3   # matches HHState.as_vector() with H_COUNT heuristics
 
#     # ── Pre-train on 3 stub instances ────────────────────────────────────────
#     training_problems = [StubProblem() for _ in range(3)]
 
#     hh = pretrain_and_deploy(
#         training_problems=training_problems,
#         h_count=H_COUNT,
#         state_dim=STATE_DIM,
#         time_limit_ms=3_000,
#         use_surrogate=False,    # set True if TensorFlow installed
#     )
 
#     # ── Deploy on a new test instance ────────────────────────────────────────
#     test_problem = StubProblem()
#     hh.setTimeLimit(10_000)
#     hh.loadProblemDomain(test_problem)
#     hh.run()
 
#     print(f"\nBest solution found: {hh.getBestSolutionValue():.4f}")
#     trace = hh.getFitnessTrace()
#     if trace:
#         print(f"Trace (first 10): {[round(v,2) for v in trace[:10]]}")