"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Scripts that train the Function Approximation RL networks.
"""

import _pickle
import logging
from control_and_ai.helpers import *
from control_and_ai.function_approximation_q_learning import *
from main_simulation import *

verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

simulation_settings = {'Side Engines': True,
            'Clouds': True,
            'Vectorized Nozzle': True,
            'Graph': False,
            'Render': False,
            'Starting Y-Pos Constant': 1,
            'Initial Force': 'random',
            'Rows': 1,
            'Columns': 2,
            #'Initial Coordinates': (0.8,0.5,0),
            'Test': False,
            'Episodes': 5000}

evo_strategy_parameters = {
    'population_size': 100,
    'action_size': 3,
    'noise_standard_deviation': 0.1,
    'number_of_generations': 1000,
    'learning_rate': 0.00025,
    'state_size': 8,
    'max_num_actions': 250
}

env = []
for i in range(evo_strategy_parameters['population_size']+1):
    env.append(RocketLander(simulation_settings))


def rocket_rl_function_approximation(env, settings : dict, logger, load_path=None, save_path=None, low_discretization=True):
    if settings['Test']:
        print("Testing rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path, save_path))
    else:
        print("Training rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path,
                                                                                                      save_path))
    env.reset()
    s = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False) # remove this line if normal state
    reinforcedControl = FunctionApproximation(s, load=load_path, low_discretization=low_discretization, epsilon=0.001, alpha=0.001)
    max_steps = 1000
    steps = 0
    def train():
        episode = 1
        done = False
        for episode in range(settings['Episodes']):
            s = env.reset()
            steps = 0
            while steps < max_steps:
                a = reinforcedControl.act()
                s, r, done, info = env.step(a)
                s = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
                reinforcedControl.learn(s, r)
                if episode % 50 == 0 or settings['Render']:
                    env.refresh(render=True)

                if done:
                    logger.info('Episode:\t{0}\tReward:\t{1}'.format(episode, reinforcedControl.total_reward))

                    reinforcedControl.reset()
                    break

                steps += 1


            if episode % 50 == 0:
                if save_path is not None:
                    logger.info('Saving Model at Episode:\t{0}'.format(episode))
                    _pickle.dump(reinforcedControl.log, open(save_path, "wb"))

    def test():
        episode = 0
        while episode < settings['Episodes']:
            a = reinforcedControl.act()
            s, r, done, info = env.step(a)
            if settings['Render']:
                env.refresh(render=True)

            logger.info('Episode:\t{0}\tReward:\t{1}'.format(episode, reinforcedControl.total_reward))

            if done:
                env.reset()
                reinforcedControl.reset()
                episode += 1

    if isinstance(settings.get('Test'), bool):
        if settings['Test']:
            test()
        else:
            train()
    else:
        train()

def train_low_discretization_rl():
    print("Training LOW Discretization RL-Function Approximator")
    load_path = None#'rl_linear_function_approximation_low_function_discretization_squared_states_2000_episodes.p'
    save_path = 'rl_linear_function_approximation_low_function_discretization_trained_at_once.p'
    rocket_rl_function_approximation(env[0], settings=simulation_settings, logger=logger, load_path=load_path,
                                     save_path=save_path, low_discretization=True)

def train_high_discretization_rl():
    print("Training HIGH Discretization RL-Function Approximator")
    load_path = 'rl_linear_function_approximation_high_function_discretization_5000_episodes_trained_at_once.p'
    save_path = 'rl_linear_function_approximation_high_function_discretization_5000_episodes_trained_at_once.p'
    rocket_rl_function_approximation(env[0], settings=simulation_settings, logger=logger, load_path=load_path,
                                     save_path=save_path, low_discretization=False)


#train_low_discretization_rl()

train_high_discretization_rl()
