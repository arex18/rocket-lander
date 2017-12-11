import _pickle
from control_and_ai.evolutionary_strategy import *
from main_simulation import *
from control_and_ai.pid import *
from control_and_ai.function_approximation_q_learning import *

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
            'Episodes': 10}

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


# Working

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

    evolutionary_network(env, evo_strategy_parameters, logger, load_path, save_path)

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

    evolutionary_network(env, evo_strategy_parameters, logger, load_path, save_path)

def rocket_pid_controlled(env, settings):
    from plotting.realtime_plot import RealTime_Graph_Thread
    x, y, y2 = [], [], []
    i = 0

    #s = env.reset()
    env.adjust_dynamics(y_dot=0, x_dot=0, theta=0.255, theta_dot=0)
    s = env.state
    if settings['Graph']:
        data = []
        handles = RealTime_Graph_Thread(settings)
        handles.start()

    done = False
    pid = PID_Benchmark()
    total_reward = 0
    while(i < 10):
        # start = time.clock()

        a, pid_state = pid.compute(env, s)
        s, r, done, info = env.step(a)
        total_reward += r
        if settings['Render']:
            env.render()

        if settings['Graph']:
            if handles.isAlive():
                handles.data[0].append(s[3])
                handles.data[1].append(s[1])
        #print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(s[0],s[1],s[2],s[3],s[4],s[5],env.lander.position.x, env.lander.position.y ))
        if done:
            env.reset()
            env.adjust_dynamics(x=0, y=0, y_dot=0, x_dot=0, theta=0.055, theta_dot=0)
            print("Total Reward:\t{0}".format(total_reward))
            total_reward = 0
            i += 1
    print(env.lander.mass)
            # print(time.clock()-start)
    #np.save("yvel_ypos_data", handles.data)

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

# ----------------------------------------------------------------------------
# Normal
# load_path = 'rl_linear_function_approximation.p'
# save_path = 'rl_linear_function_approximation.p'
# Increased degrees of freedom
load_path = 'rl_linear_function_approximation_increased_Fe_and_psi.p'
save_path = 'rl_linear_function_approximation_increased_Fe_and_psi.p'
rocket_rl_function_approximation(env[0], settings=simulation_settings, logger=logger, load_path=load_path, save_path=save_path)
# ----------------------------------------------------------------------------
#rocket_pid_controlled(env[0], simulation_settings)
# ----------------------------------------------------------------------------
# Not really working - saturating - Ignore but comment on this
# filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi_normal_learning.npy'
# evolutionary_network_variable_psi_normal(simulation_settings, load_path=filepath, save_path=filepath) # Training
# ----------------------------------------------------------------------------
filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_fixed_psi_momentum_learning.npy'
evolutionary_network_fixed_psi_momentum(simulation_settings, load_path=filepath, save_path=filepath) # Training











