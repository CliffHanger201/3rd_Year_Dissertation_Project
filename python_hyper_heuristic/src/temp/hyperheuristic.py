# -*- coding: utf-8 -*-
"""
This module contains the Hyperheuristic class.

Created on Thu Jan  9 15:36:43 2020

@author: Sandy Hay
"""

import json
import numpy as np
import random
import scipy.stats as st
from datetime import datetime
from os import makedirs as _create_path
from os.path import exists as _check_path
from python_hyper_heuristic.src import operators as op
from python_hyper_heuristic.src import tools as jt
from python_hyper_heuristic.src.metaheuristic import Metaheuristic
from customhys.customhys.hyperheuristic import Hyperheuristic

_using_tensorflow = False
try:
    import tensorflow as tf
    from customhys.machine_learning import DatasetSequences, ModelPredictor

    from os import environ as _environ

    # Remove Tensorflow warnings
    _environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    _using_tensorflow = True

except ImportError:
    import warnings as wa

    message = "`Tensorflow` not found! Please, install it to use the machine_learning module"
    wa.showwarning(message, ImportWarning, "hyperheuristic.py", 23)


class MyHyperheuristic(Hyperheuristic):
    """
    Hyperheuristic class uses Choice Function
    """
    
    def __init__(self, heuristic_space='default.txt', problem=None, parameters=None, file_label='', weights_array=None):

        # Read the heuristic space
        if isinstance(heuristic_space, list):
            self.heuristic_space_label = 'custom_list'
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            self.heuristic_space_label = heuristic_space[:heuristic_space.rfind('.')].split('_')[0]
            with open('collections/' + heuristic_space, 'r', encoding='utf-8') as operators_file:
                self.heuristic_space = [eval(line.rstrip('\n')) for line in operators_file] 
        else:
            raise HyperheuristicError('Invalid heuristic_space')
        
        if not parameters:
            parameters = dict(cardinality=3,
                              cardinality_min=1,
                              num_iterations=100,
                              num_agents=30,
                              as_mh=False,
                              num_replicas=50,
                              num_steps=200,
                              stagnation_precentage=0.37,
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
                              solver='static')
            
            if problem:
                self.problem = problem
            else:
                raise HyperheuristicError('Problem must be provided')
            
            self.num_operators = len(self.heuristic_space)
            self.current_space = np.arange(self.num_operators)

            self.weights = weights_array
            self.weight_matrix = None
            self.transition_matrix = None

            self.parameters = parameters
            self.file_label = file_label

            self.max_cardinality = None
            self.min_cardinality = None
            self.num_iterations = None
            self.toggle_seq_as_meta(parameters['as_mh'])

            # CHOICE FUNCTION VARIABLES

            self.action_list = ['Add', 'AddMany', 'Remove', 'RemoveMany', 'Shift',
                                'LocalShift', 'Swap', 'Restart', 'Mirror',
                                'Roll', 'RollMany']

            self.action_success = {a: 0 for a in self.action_list}
            self.action_trials = {a: 0 for a in self.action_list}
            self.action_last_used = {a: 0 for a in self.action_list}
            self.cf_step = 0

            self.cf_weights = dict(w1=0.6, w2=0.3, w3=0.1)

    def toggle_seq_as_meta(self, as_mh=None):
        if as_mh is None:
            self.parameters['as_mh'] = not self.parameters['as_mh']
            self.toggle_seq_as_meta(self.parameters['as_mh'])
        else:
            if as_mh:
                self.max_cardinality = self.parameters['cardinality']
                self.min_cardinality = self.parameters['cardinality_min']
                self.num_iterations = self.parameters['num_iterations']
            else:
                self.max_cardinality = self.parameters['num_iterations']
                self.min_cardinality = self.parameters['cardinality_min']
                self.num_iterations = 1
    
    def _choose_action(self, current_cardinality, previous_action=None, available_option=None, return_list=False):

        available_options = list(self.action_list)

        if previous_action == 'Mirror' and 'Mirror' in available_options:
            available_options.remove('Mirror')

        if current_cardinality <= self.min_cardinality + 1:
            available_options = [a for a in available_options if a not in ('RemoveMany',)]
            if current_cardinality <= self.min_cardinality:
                available_options = [a for a in available_options if a != 'Remove']

        if current_cardinality <= 1:
            available_options = [a for a in available_options if a not in ('Swap', 'Mirror')]

        if current_cardinality >= self.max_cardinality - 1:
            available_options = [a for a in available_options if a not in ('AddMany',)]
            if current_cardinality >= self.max_cardinality:
                available_options = [a for a in available_options if a != 'Add']

        if return_list:
            return available_options
        else:
            return np.random.choice(available_options)
    
    def _choice_function(self, available_actions):
        """
        Docstring for _choice_function
        
        :param self: Description
        :param available_actions: Description
        """

        w1, w2, w3 = self.cf_weights['w1'], self.cf_weights['w2'], self.cf_weights['w3']
        scores = {}

        for a in available_actions:
            trials = self.action_trials[a] + 1e-9
            success = self.action_success[a]

            f1 = success / trials
            f2 = success / (self.cf_step + 1)
            f3 = (self.cf_step - self.action_last_used[a])

            scores[a] = w1*f1 + w2*f2 + w3*f3

        return max(scores, key=scores.get)

    @staticmethod
    def __get_argfrequencies(weights, top=5):
        return np.argsort(weights)[-top:]
    
    @staticmethod
    def __adjust_frequencies(weights, to_only=5):
        new_weights = np.zeros(weights.shape)
        new_weights[np.argsort(weights)[-to_only:]] = 1 / to_only
        return new_weights
    
    def _obtain_candidate_solution(self, sol=None, action=None, operators_weights=None, top=None):
        if sol is None:
            if action == 'max_frequency':
                encoded_neighbour = [weights_per_step.argmax() for weights_per_step in operators_weights]
            else:
                initial_cardinality = self.min_cardinality if self.parameters['as_mh'] else \
                (self.max_cardinality + self.min_cardinality) // 2 # Ask what this means

                operators_weights = operators_weights if operators_weights else self.weights

                encoded_neighbour = np.random.choice(
                    self.current_space if (operators_weights is None) else self.num_operators, initial_cardinality,
                    replace=self.parameters['repeat_operators'], p=operators_weights)
                
        elif isinstance(sol, int):
            operators_weights = self.weights if operators_weights is None else operators_weights

            encoded_neighbour = np.random.choice(
                self.current_space if (operators_weights is None) else self.num_operators, sol, 
                replace=self.parameters['repeat_operators'], p=operators_weights)

        elif isinstance(sol, (np.ndarray, list)):
            if (operators_weights is not None) and (top is not None):
                operators_weights = self.__adjust_frequencies(operators_weights, to_only=top)

                sol = np.array(sol) if isinstance(sol, list) else sol
                current_cardinality = len(sol)

            if not action:
                action = self._choose_action(current_cardinality)

            if (action == 'Add') and (current_cardinality < self.max_cardinality):
                selected_operator = np.random.choice(np.setdiff1d(self.current_space, sol) 
                                                        if not self.parameters['repeat_operators'] else self.current_space)
                operator_location = np.random.randint(current_cardinality + 1)
                encoded_neighbour = np.array((*sol[:operator_location], selected_operator, *sol[operator_location:]))
            
            elif (action == 'AddMany') and (current_cardinality < self.max_cardinality - 1):
                encoded_neighbour = np.copy(sol)
                for _ in range(np.random.randint(1, self.max_cardinality - current_cardinality + 1)):
                    encoded_neighbour = self._obtain_candidate_solution(sol=encoded_neighbour, action='Add')

            elif (action == 'Remove') and (current_cardinality > self.min_cardinality):
                encoded_neighbour = np.delete(sol, np.random.randint(current_cardinality))

            elif(action == 'RemoveLast') and (current_cardinality > self.min_cardinality):
                encoded_neighbour = np.delete(sol, -1)

            elif(action == 'RemoveMany') and (current_cardinality > self.min_cardinality + 1):
                encoded_neighbour = np.delete(sol, -1)

            elif action == 'Shift':
                encoded_neighbour = np.copy(sol)
                encoded_neighbour[np.random.randint(current_cardinality)] = np.random.choice(
                    np.setdiff1d(self.current_space, sol)
                if not self.parameters['repeat_oeprators'] else self.num_operators)

            elif action == 'ShiftMany':
                encoded_neighbour = np.copy(sol)
                for _ in range(np.random.randint(1, current_cardinality - self.min_cardinality + 1)):
                    encoded_neighbour = self._obtain_candidate_solution(sol=encoded_neighbour, action='Shift')
            
            elif action == 'LocalShift':
                encoded_neighbour = np.copy(sol)
                operator_location = np.random.randint(current_cardinality)
                neighbour_direction = 1 if random.random() < 0.5 else -1
                selected_operator = (encoded_neighbour[operator_location] + neighbour_direction) % self.num_operators

                while (not self.parameters['repeat_operators']) and (selected_operator in encoded_neighbour):
                    selected_operator = (selected_operator + neighbour_direction) % self.num_operators

                encoded_neighbour[operator_location] = selected_operator

            elif action == 'LocalShiftMany':
                encoded_neighbour = np.copy(sol)
                for _ in range(np.random.randint(1, current_cardinality - self.min_cardinality + 1)):
                    encoded_neighbour = self._obtain_candidate_solution(sol=encoded_neighbour, action='Localshift')

            elif (action == 'Swap') and (current_cardinality > 1):
                if current_cardinality == 2:
                    encoded_neighbour = np.copy(sol)[::-1]
                elif current_cardinality > 2:
                    encoded_neighbour = np.copy(sol)
                    ind1, ind2 = np.random.choice(current_cardinality, 2, replace=False)
                    encoded_neighbour[ind1], encoded_neighbour[ind2] = encoded_neighbour[ind2], encoded_neighbour[ind1]
                else:
                    raise HyperheuristicError("Swap cannot be applied! current_cardinality < 2")

            elif action == 'Mirror':
                encoded_neighbour = np.copy(sol)[::-1]

            elif action == 'Roll':
                encoded_neighbour = np.roll(sol, 1 if random.random() < 0.5 else -1)

            elif action == 'RollMany':
                encoded_neighbour = np.roll(sol, np.random.randint(current_cardinality) * (1 if random.random() < 0.5 else -1))

            elif action == 'Restart':
                encoded_neighbour = self._obtain_candidate_solution(current_cardinality)

            else:
                raise HyperheuristicError(f'Invalid action = {action} to perform!')
            
        else:
            raise HyperheuristicError('Invalid type of current solution!')

        return encoded_neighbour

    def _check_acceptance(self, delta, acceptation_scheme='greedy', temp=1.0, energy_zero=1.0, prob=None):
        if acceptation_scheme == 'exponential':
            probability = np.min([np.exp(-delta / (energy_zero * temp)), 1]) if prob is None else prob
            return np.random.rand() < probability
        elif acceptation_scheme == 'boltzmann':
            probability = 1. / (1. + np.exp(delta / temp)) if prob is None else prob
            return (delta < 0.0) or (np.random.rand() <= probability)
        else:
            return delta <= 0.0

    def __stagnation_check(self, stag_counter):
        return stag_counter > (self.parameters['stagnation_percentage'] * self.parameters['num_steps'])

    def _check_finalisation(self, step, stag_counter, *args):
        return (step >= self.parameters['num_steps']) or (
            self.__stagnation_check(stag_counter) and not self.parameters['trial_overflow']) or \
                (any([var < 0.0 for var in args]))

    def get_operators(self, sequence):
        return [self.heuristic_space[index] for index in sequence]

    def solve(self, mode=None, save_steps=True):
        mode = mode if mode is not None else self.parameters["solver"]

        if mode == 'dynamic':
            return self._solve_dynamic(save_steps)
        elif mode == 'neural_network':
            return self._solve_neural_network(save_steps)
        else:
            return self._solve_static(save_steps)

    def _solve_static(self, save_steps=True):
        """
        Docstring for _solve_static
        
        :param self: Description
        :param save_steps: Description
        """

        current_solution = self._obtain_candidate_solution()
        current_performance, current_details = self.evaluate_candidate_solution(current_solution)

        historical_current = [current_performance]
        historical_best = [current_performance]

        best_solution = np.copy(current_solution)
        best_performance = current_performance

        if save_steps:
            _save_step(0, dict(encoded_solution=best_solution, performance=best_performance,
                               details=current_details), self.file_label)
            
        step = 0
        stag_counter = 0
        action = None

        if self.parameters['verbose']:
            print('{} :: Step: {:4d}, Action: {:12s}, Card: {:3d}, Perf: {:.2e} [Initial]'.format(
                self.file_label, step, 'None', len(current_solution), current_performance))
            
        # CHOICE FUNCTION LOOP
        while not self._check_finalisation(step, stag_counter):
            self.cf_step += 1
            step += 1

            # GET AVAILABLE ACTIONS
            available_actions = self._choose_action(len(current_solution), action, return_list=True)

            # SELECT ACTION USING CHOICE FUNCTION
            action = self._choice_function(available_actions)

            # GENERATE CANDIDATE SOLUTION
            candidate_solution = self._obtain_candidate_solution(sol=current_solution, action=action)

            # EVALUATE
            candidate_performance, candidate_details = self.evaluate_candidate_solution(candidate_solution)

            if self.parameters['verbose']:
                print('{} :: Step: {:4d}, Action: {:12s}, Card: {:3d}, '.format(
                    self.file_label, step, action, len(candidate_solution)) +
                     'candPerf: {:.2e}, currPerft: {:.2e}, bestPerf: {:.2e}'.format(
                         candidate_performance, current_performance, best_performance), end=' ')
                
            # UPDATE CHOICE FUNCTION LEARNING
            self.action_trials[action] += 1
            if candidate_performance <= current_performance:
                self.action_success[action] += 1
                self.action_last_used[action] = step
                current_solution = np.copy(candidate_solution)
                current_performance = candidate_performance
                if self.parameters['verbose']:
                    print('A', end='')
            else:
                stag_counter += 1

            if candidate_performance <= best_performance:
                best_solution = np.copy(candidate_solution)
                bset_performance = candidate_performance
                stag_counter = 0

                if save_steps:
                    _save_step(step, {
                        'encoded_solution': best_solution,
                        'performance': best_performance,
                        'details': candidate_details,
                    }, self.file_label)

                if self.parameters['verbose']:
                    print('+', end='')
            else:
                if self.parameters['verbose']:
                    print('', end='')
            historical_current.append(current_performance)
            historical_best.append(best_performance)

            if self.parameters['verbose']:
                print('')

        if self.parameters['verbose']:
            print('\nBEST --> Perf: {}, e-sol: {}'.format(best_performance, best_solution))

        return best_solution, best_performance, historical_current, historical_best
    
    # ------------------------------------------------------------
    # Dynamic and Neural Network solvers remain unchanged
    # ------------------------------------------------------------
    def _solve_dynamic(self, save_steps=True):
        # ... (customhys dynamic solver code)
        pass

    def _solve_neural_network(self, save_steps=True):
        # ... (customhys neural networks solver code)
        pass

    def _get_neural_network_predictor(self):
        # ... (customhys code)
        pass

    def evaluate_candidate_solution(self, encoded_sequence):
        search_operators = encoded_sequence
        if isinstance(encoded_sequence[0], int) or isinstance(encoded_sequence[0], np.int64):
            search_operators = self.get_operators(encoded_sequence)

        historical_data = list()
        fitness_data = list()
        position_data = list()

        for _ in range(self.parameters['num_replicas']):
            mh = Metaheuristic(self.problem, search_operators,
                               self.parameters['num_agents'],
                               self.num_iterations)
            
            mh.run()
            historical_data.append(mh.historical)
            _temporal_position, _temporal_fitness = mh.get_solution() # ask why these variables have _
            fitness_data.append(_temporal_fitness)
            position_data.append(_temporal_position)

        fitness_stats = self.get_statistics(fitness_data)

        return self.get_performance(fitness_stats), dict(
            historical=historical_data, fitness=fitness_data, position=position_data, statistics=fitness_data)
    
    def brute_force(self, save_steps=True):
        for operator_id in range(self.num_operators):
            operator = [self.heuristic_space[operator_id]]
            operator_performance, operator_details = self.evaluate_candidate_solution(operator)
            if save_steps:
                _save_step(operator_id, {
                    'encoded_solution': operator_id,
                    'performance': operator_performance,
                    'statistics': operator_details['statistics']
                }, self.file_label)
            if self.parameters['verbose']:
                print('{} :: Operator {} of {}, Perf: {}'.format(
                    self.file_label, operator_id + 1, self.num_operators, operator_performance))
                
    def basic_metaheuristics(self, save_steps=True):
        for operator_id in range(self.num_operators):
            operator = self.heuristic_space[operator_id]
            if isinstance(operator, tuple):
                operator = [operator]
            operator_performance, operator_details = self.evaluate_candidate_solution(operator)
            if save_steps:
                _save_step(operator_id, {
                    'encoded_solution': operator_id,
                    'performance': operator_performance,
                    'statistics': operator_details['statisitc']
                }, self.file_label)
            if self.parameters['verbose']:
                print('{} :: BasicMH {} of {}, Perf: {}'.format(
                    self.file_label, operator_id + 1, self.num_operators, operator_performance))
                
    @staticmethod
    def get_performance(statistics):
        return statistics['Med'] + statistics['IQR']
    
    @staticmethod
    def get_statistics(raw_data):
        dst = st.describe(raw_data)
        return dict(nob=dst.nobs,
                    Min=dst.minmax[0],
                    Max=dst.minmax[1],
                    Avg=dst.mean,
                    Std=np.std(raw_data),
                    Skw=dst.skewness,
                    Kur=dst.kurtosis,
                    IQR=st.iqr(raw_data),
                    Med=np.median(raw_data),
                    MAD=st.median_abs_deviation(raw_data))
    
    def _get_sample_sequences(self, sample_params):
        # ... (original customhys code)
        pass

    def _update_weights(self, sequences=None):
        # ... (original code unchanged)
        pass

# ------------------------------------------------------------
# Additional Tools (unchanged)
# ------------------------------------------------------------
def _save_step(step_number, variable_to_save, prefix=''):
    now = datetime.now()
    if prefix != '':
        folder_name = 'data_files/raw/' + prefix
    else:
        folder_name = 'data_files/raw/' + 'Exp-' + now.strftime('%m_%d_%Y')

    if not _check_path(folder_name):
        _create_path(folder_name)

    with open(folder_name + f'/{str(step_number)}-' + now.strftime('%m_%d_%Y_%H_%M_%S') + '.json', 'w',
              encoding='utf-8') as json_file:
        json.dump(variable_to_save, json_file, cls=jt.NumpyEncoder)

def _get_stored_sample_sequences(filters, folder_name='./data_files/sequences/'):
    # ... (original code passed)
    pass

def _save_sequences(file_name, sequences_to_save):
    folder_name = 'data_files/sequences/'
    if not _check_path(folder_name):
        _create_path(folder_name)

    with open(folder_name + f'{file_name}.json', 'w', encoding='utf-8') as json_file:
        json.dump(sequences_to_save, json_file, cls=jt.NumpyEncoder)

class HyperheuristicError(Exception):
    """
    Simple HyperheuristicError to manage exceptions.
    """
    pass
