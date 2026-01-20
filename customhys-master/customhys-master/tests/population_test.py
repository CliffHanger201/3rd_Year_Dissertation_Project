# import numpy as np
# import pytest

# from customhys

# @pytest.fixture
# def boundaries():
#     return ([-5.0, -5.0], [5.0, 5.0])


# @pytest.fixture
# def population(boundaries):
#     pop = Population(boundaries=boundaries, num_agents=10)
#     pop.initialise_positions()
#     return pop


# @pytest.fixture
# def sphere_function():
#     def f(x):
#         return np.sum(x ** 2)
#     return f

# ------------------------------------------------------
# Initialization Testing
# ------------------------------------------------------

# def test_population_initialization(boundaries):
#     pop = Population(boundaries, num_agents=5)

#     assert pop.num_agents == 5
#     assert pop.num_dimensions == 2
#     assert pop.positions.shape == (5, 2)
#     assert np.all(np.isnan(pop.fitness))


# def test_invalid_boundaries():
#     with pytest.raises(PopulationError):
#         Population(boundaries=([0, 0], [1]))

# # ------------------------------------------------------
# # Position Testing
# # ------------------------------------------------------

# def test_initialise_positions_random(population):
#     assert np.all(population.positions <= population.upper_boundaries)
#     assert np.all(population.positions >= population.lower_boundaries)


# def test_initialise_positions_vertex(boundaries):
#     pop = Population(boundaries, num_agents=4)
#     pop.initialise_positions(scheme="vertex")

#     assert pop.positions.shape == (4, 2)
#     assert np.all(np.abs(pop._positions) <= 1.0)


# def test_rescale_back(population):
#     pos = np.array([0.0, 0.0])
#     rescaled = population.rescale_back(pos)

#     assert np.allclose(rescaled, [0.0, 0.0])


# def test_set_and_get_positions(population):
#     new_positions = np.array([[1.0, 1.0]] * population.num_agents)
#     scaled = population.set_positions(new_positions)
#     population.positions = scaled

#     retrieved = population.get_positions()
#     assert np.allclose(retrieved, new_positions)

# # -----------------------------------------------------------------
# # Constraint handling
# # -----------------------------------------------------------------

# def test_check_simple_constraints(population):
#     population._positions[:] = 2.0
#     population._check_simple_constraints()

#     assert np.all(population._positions <= 1.0)
#     assert np.all(population.velocities == 0.0)

# # ---------------------------------------------------------------------
# # Fitness evaluation
# # ---------------------------------------------------------------------

# def test_evaluate_fitness(population, sphere_function):
#     population.evaluate_fitness(sphere_function)

#     assert np.all(np.isfinite(population.fitness))
#     assert np.all(population.fitness >= 0.0)


# def test_evaluate_fitness_callable_required(population):
#     with pytest.raises(AssertionError):
#         population.evaluate_fitness(None)


# # ---------------------------------------------------------------------
# # Selection logic
# # ---------------------------------------------------------------------

# @pytest.mark.parametrize(
#     "selector,new,old,expected",
#     [
#         ("greedy", 1.0, 2.0, True),
#         ("greedy", 2.0, 1.0, False),
#         ("all", 10.0, 1.0, True),
#         ("none", 0.0, 1.0, False),
#     ]
# )
# def test_selection_basic(population, selector, new, old, expected):
#     assert population._selection(new, old, selector) == expected


# def test_selection_probabilistic(population):
#     population.probability_selection = 1.0
#     assert population._selection(10.0, 1.0, "probabilistic") is True


# def test_selection_metropolis_accepts_better(population):
#     assert population._selection(1.0, 2.0, "metropolis") is True


# def test_invalid_selector(population):
#     with pytest.raises(PopulationError):
#         population._selection(1.0, 2.0, "invalid_selector")