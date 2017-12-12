"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Evolutionary Networks implementation with runnable scripts.
"""

from .helpers import *
from environments.rocketlander import RocketLander
import numpy as np
import concurrent.futures

""" Difference between the two classes is in their implementation. """
class EvolutionNetwork():
    def __init__(self, hyper_parameters: dict, loadfilename=None):
        # Unpack ES hyper_parameters for readability
        self.population_size = hyper_parameters['population_size']
        self.action_size = hyper_parameters['action_size']
        self.state_size = hyper_parameters['state_size']
        self.noise_std = hyper_parameters['noise_standard_deviation']
        self.learning_rate = hyper_parameters['learning_rate']
        self.max_num_actions = hyper_parameters['max_num_actions']
        self.v = 0

        if hyper_parameters.get('gamma') is None:
            self.gamma = 0.2
        else:
            self.gamma = hyper_parameters['gamma']
        if loadfilename is not None:
            self.weights = np.load(loadfilename)
        else:
            self.weights = np.random.rand(self.state_size, self.action_size)

    def generateMutations(self):
        # Append mutations
        noise = np.random.normal(0, self.noise_std, (self.population_size, self.state_size, self.action_size))
        population_weights = []
        for n in noise:
            population_weights.append(self.weights + self.noise_std * n)
        return np.array(population_weights), noise

    def normalizeWeightingMatrix(self, matrix):
        # Normalize Rewards
        return (matrix - np.mean(matrix)) / np.std(matrix)

    def updateWeights_momentum(self, noise, rewards):
        # Take the expected value over the population, weighted by the reward matrix
        weighted_noise = np.matmul(noise.T, rewards).T
        self.v = self.gamma * self.v + self.learning_rate / (self.population_size * self.noise_std) * weighted_noise
        self.weights = self.weights + self.v

    def updateWeights(self, noise, rewards):
        # Take the expected value over the population, weighted by the reward matrix
        weighted_noise = np.matmul(noise.T, rewards).T
        constant = self.learning_rate / (self.population_size * self.noise_std) * weighted_noise
        self.weights = self.weights + constant

    def decreaseGamma(self, constant):
        self.gamma = self.gamma - constant

    def save_genes(self, filename):
        np.save(filename, self.weights)

class EvolutionaryNetWork:
    def __init__(self, sigma=0.01, state_size=8,
                 action_size=4, population_size=100, loadfilepath=None):
        self.population_size = population_size
        self.action_size = action_size
        self.state_size = state_size
        self.sigma = sigma
        self.gamma = 0.1
        self.v = 0
        if loadfilepath is not None:
            self.weight = np.load(loadfilepath)
        else:
            self.weight = -np.random.rand(state_size, action_size)

    def generate_mutations(self):
        mutations = []
        noise = np.random.randn(self.population_size, self.state_size, self.action_size)

        for i in range(self.population_size):
            mutations.append(self.weight + self.sigma * noise[i])

        np_mutations = np.array(mutations)

        return np_mutations.reshape(self.population_size, self.state_size, self.action_size), noise

    def update_genes_momentum(self, total_rewards, noise, learning_rate):
        weighted_noise = np.matmul(noise.T, total_rewards).T
        self.v = self.gamma * self.v + learning_rate / (self.population_size * self.sigma) * weighted_noise
        self.weight = self.weight + self.v

    def save_genes(self, savefilepath):
        np.save(savefilepath, self.weight)

def evolutionary_network_transformable_state(env, hyper_parameters: dict, logger, loadpath=None, savepath=None):
    gen_num = hyper_parameters['number_of_generations']
    population = hyper_parameters['population_size']
    max_num_actions = hyper_parameters['max_num_actions']
    learning_strategy = hyper_parameters.get('momentum')
    EV = EvolutionNetwork(hyper_parameters, loadpath)
    state_builder = State_Builder(integral_number=2)

    def run_episode(env, weights, show=False, *args):
        state = env.reset()
        state = buildState(state_builder, state)
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
            state = buildState(state_builder, state)
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

    run(EV, env)

def evolutionary_network_variable_psi_transformable_state(simulation_settings, logger, save=False):
    print("evolutionary_network_variable_psi_transformable_state")
    # saved in weights_es.npy
    filepath = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi_transformable_state.npy'

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

    if save:
        save_file = filepath
        load_file = None
    else:
        save_file = None
        load_file = filepath

    evolutionary_network_transformable_state(env, evo_strategy_parameters, logger, load_file, save_file)
