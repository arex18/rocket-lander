import _pickle
from control_and_ai.evolutionary_strategy import *
from control_and_ai.function_approximation_q_learning import *

verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


def evolutionary_network_fixed_psi_momentum(simulation_settings, load_path=None, save_path=None):
    print("evolutionary_network_fixed_psi_momentum + loading = {0}, saving = {1}".format(load_path, save_path))

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

    evolutionary_network_transformable_state(env, evo_strategy_parameters, logger, load_path, save_path)

def evolutionary_network_variable_psi_normal(simulation_settings, load_path=None, save_path=None):
    print("evolutionary_network_variable_psi_normal + loading = {0}, saving = {1}".format(load_path, save_path))
    # saved in weights_es.npy
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

    evolutionary_network_transformable_state(env, evo_strategy_parameters, logger, load_path, save_path)

def rocket_rl_function_approximation(env, settings : dict, logger, load_path=None, save_path=None):
    if settings['Test']:
        print("Testing rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path, save_path))
    else:
        print("Training rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path,
                                                                                                      save_path))
    i = 0
    from plotting.realtime_plot import RealTime_Graph_Thread
    s = env.reset()

    reinforcedControl = FunctionApproximation(s, load=load_path)
    done = False

    def train():
        episode = 1
        if settings['Graph']:
            data = []
            handles = RealTime_Graph_Thread()
            handles.start()

        while (episode <= 2000):
            a = reinforcedControl.act()
            s, r, done, info = env.step(a)
            reinforcedControl.learn(s, r)
            if episode % 100 == 0 or settings['Render']:
                env.refresh(render=True)

            if done:
                env.reset()
                logger.info('Episode:\t{0}\tReward:\t{1}'.format(episode, reinforcedControl.total_reward))
                if settings['Graph']:
                    if handles.isAlive():
                            data.append(reinforcedControl.total_reward)
                            handles.data[0] = data
                reinforcedControl.reset()
                episode += 1

        if save_path is not None:
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
            'Episodes': 10}


load_path = 'rl_linear_function_approximation_increased_Fe_and_psi.p'
save_path = 'rl_linear_function_approximation_increased_Fe_and_psi.p'
rocket_rl_function_approximation(RocketLander(simulation_settings), settings=simulation_settings, logger=logger, load_path=load_path, save_path=save_path)
# ----------------------------------------------------------------------------
filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_fixed_psi_momentum_learning.npy'
evolutionary_network_fixed_psi_momentum(simulation_settings, load_path=filepath, save_path=filepath) # Training











