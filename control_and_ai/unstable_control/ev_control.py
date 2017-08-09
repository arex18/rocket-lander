from control_and_ai.helpers import *
from rocketlander_v2 import *

hyper_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.1,
        'number_of_generations': 200,
        'learning_rate': 0.00025,
        'state_size': 8,
        'max_num_actions': 250
    }

gen_num = hyper_parameters['number_of_generations']
population = hyper_parameters['population_size']
max_num_actions = hyper_parameters['max_num_actions']
learning_strategy = hyper_parameters.get('momentum')
EV = EvolutionNetwork(hyper_parameters, None)

verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

savepath = None

def run_episode(env, weights, show=False, *args):
    state = env.reset()
    done = False
    total_reward = 0
    ground_touched_flag = False
    num_actions = 0
    while not done:
        if show:
            env.render()
        if ground_touched_flag or num_actions <= max_num_actions:
            action = np.matmul(weights.T, state)
        else:
            action = [0, 0, 0]
        state, reward, done, info = env.step(action)
        if (state[6] == 1 or state[7] == 1):
            ground_touched_flag = True
        total_reward += reward
        num_actions += 1
    return total_reward

def run(EV, env):
    # ---------------------------------------------------------------------------------------
    logger.info("{0}\t{1}\t{2}".format('Episode', 'Reward', 'Mean Genes Reward'))
    running_reward = 0
    # ---------------------------------------------------------------------------------------
    for episode in range(gen_num):
        # Run the episode result
        if episode % 20 == 0:
            running_reward = run_episode(env[0], EV.weights, show=True)

        population_weights, noise = EV.generateMutations()

        rewards = np.zeros(population)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(population):
                future = executor.submit(run_episode, env[i + 1], population_weights[i], False)
                rewards[i] = future.result()

        if learning_strategy is None:
            EV.updateWeights(noise, rewards)
        else:
            EV.updateWeights_momentum(noise, rewards)

        if savepath is not None:
            EV.save_genes(savepath)
        logger.info("{0}\t{1}\t{2}".format(episode, running_reward, np.mean(rewards)))

simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': True,
                           'Render': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}


env = []
for i in range(hyper_parameters['population_size'] + 1):
    env.append(RocketLander(simulation_settings))

run(EV, env)
