import _pickle
from control_and_ai.helpers import *
from main_simulation import *

verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

simulation_settings = {'Side Engines': True,
            'Clouds': False,
            'Vectorized Nozzle': True,
            'Graph': False,
            'Render': True,
            'Starting Y-Pos Constant': 1,
            'Initial Force': 'random',
            'Rows': 1,
            'Columns': 2,
            'Initial Coordinates': (0.8,0.5,0)}

evo_strategy_parameters = {
    'population_size': 100,
    'action_size': 3,
    'noise_standard_deviation': 0.1,
    'number_of_generations': 100,
    'learning_rate': 0.00025,
    'state_size': 8,
    'max_num_actions': 250
}

env = []
for i in range(evo_strategy_parameters['population_size']+1):
    env.append(RocketLander(simulation_settings))

def evolutionary_network_fixed_psi_momentum():
    filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_fixed_psi_momentum_learning.npy'

    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': False,
                           'Graph': True,
                           'Render': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}

    evo_strategy_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.2,
        'number_of_generations': 100,
        'learning_rate': 0.00025,
        'state_size': 8,
        'max_num_actions': 250,
        'momentum': True
    }

    env = []
    for i in range(evo_strategy_parameters['population_size'] + 1):
        env.append(RocketLander(simulation_settings))

    evolutionary_network(env, evo_strategy_parameters, logger, None, filepath)

def evolutionary_network_variable_psi_momentum():
    # saved in weights_es.npy
    filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi_momentum_learning.npy'

    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': True,
                           'Render': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}

    evo_strategy_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.1,
        'number_of_generations': 100,
        'learning_rate': 0.00025,
        'state_size': 8,
        'max_num_actions': 250,
        'momentum': True
    }

    env = []
    for i in range(evo_strategy_parameters['population_size'] + 1):
        env.append(RocketLander(simulation_settings))

    evolutionary_network(env, evo_strategy_parameters, logger, None, filepath)

def evolutionary_network_variable_psi_normal():
    # saved in weights_es.npy
    filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi_normal_learning.npy'

    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': True,
                           'Render': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}

    evo_strategy_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.1,
        'number_of_generations': 100,
        'learning_rate': 0.00025,
        'state_size': 8,
        'max_num_actions': 250
    }

    env = []
    for i in range(evo_strategy_parameters['population_size'] + 1):
        env.append(RocketLander(simulation_settings))

    evolutionary_network(env, evo_strategy_parameters, logger, None, filepath)

def evolutionary_network_variable_psi_transformable_state():
    # saved in weights_es.npy
    filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi_transformable_state.npy'

    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': True,
                           'Render': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}

    evo_strategy_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.1,
        'number_of_generations': 200,
        'learning_rate': 0.00025,
        'state_size': 14,
        'max_num_actions': 250
    }

    env = []
    for i in range(evo_strategy_parameters['population_size'] + 1):
        env.append(RocketLander(simulation_settings))

    evolutionary_network_transformable_state(env, evo_strategy_parameters, logger, None, filepath)