import matplotlib.pyplot as plt
from customhys.hyperheuristic import Hyperheuristic

def sphere(x):
    return sum(v**2 for v in x)

problem = {
    "function": sphere,
    "is_constrained": False,
    "boundaries": ([-5.12] * 10, [5.12] * 10)
}

parameters = dict(
    cardinality=3,
    cardinality_min=1,
    num_iterations=100,
    num_agents=30,
    as_mh=False,
    num_replicas=50,
    num_steps=200,
    stagnation_percentage=0.37,
    max_temperature=1,
    min_temperature=1e-6,
    cooling_rate=1e-3,
    temperature_scheme='fast',
    acceptance_scheme='exponential',
    allow_weight_matrix=True,
    trial_overflow=False,
    learnt_dataset=None,
    repeat_operators=True,
    verbose=True,
    learning_portion=0.37,
    solver='static'
)

hh = Hyperheuristic(problem=problem, parameters=parameters)

best_solution, best_performance, hist_current, hist_best = hh.solve()

plt.plot(hist_best)
plt.title("Best Fitness Over Time")
plt.xlabel("Step")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()
