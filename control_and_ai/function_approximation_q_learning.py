"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Function Approximation Q-Learning.
"""

import _pickle
import numpy as np
from constants import *


class FunctionApproximation():
    """ Q-Learning implementation. """

    def __init__(self, state, low_discretization, load=None, epsilon=0.1, alpha=0.001):
        self.low_discretization = low_discretization
        self.gamma = 0.9
        self.total_reward = 0
        self.update_state(state)

        self.epsilon = epsilon
        self.alpha = alpha
        self.numberoffeatures = len(self.get_all_features(state, [0, 0, 0]))
        # Intialise theta optimistically with small positive values
        if load is not None:
            data = _pickle.load(open(load, 'rb'))
            _, theta = data[-1]
            self.theta = theta
            # print(self.theta)
        else:
            self.theta = np.random.uniform(-10, 10, (self.numberoffeatures, 1))
        self.actionSet = self.buildActionSet()
        self.number_of_actions = len(self.actionSet)
        self.reward = 0
        self.total_reward = 0
        self.log = []

    def act(self):
        self.state = self.next_state
        # if randomNumber < epsilon, choose action based on the boltzmann exploration
        # Use softmax for action selection
        if np.random.uniform(0., 1.) < self.epsilon:
            # Qs = self.Q_all_states(self.state)
            # probs = np.exp(Qs) / np.sum(np.exp(Qs))
            randomaction = np.random.randint(low=0, high=self.number_of_actions - 1, size=1)
            # idx = np.random.choice(self.number_of_actions, p=probs)
            action = self.actionSet[int(randomaction)]
        else:
            action = self.argmaxQ(self.state)

        # if (self.state[4] < 0): # Exploit symmetry in the problem
        #     action[1] = -1.0*action[1]
        #     action[2] = -1.0*action[2]

        self.action = np.array(action)
        if (self.state[-1] == 1 or self.state[-2] == 1):
            self.action[0] = 0
        # print(self.action)
        return self.action

    def reset(self):
        self.log.append((self.total_reward, self.theta))
        self.total_reward = 0
        # self.alpha = self.alpha*0.9999
        self.epsilon = self.epsilon * 0.999
        # self.alpha = self.alpha*0.999

    def update_state(self, state):
        self._sense(state)

    def learn(self, state, reward):
        self.update_state(state)
        self.updateReward(reward)
        self._learn()

    def updateReward(self, reward):
        self.reward = reward
        self.total_reward += self.reward

    def _sense(self, state):
        self.next_state = state

    def _learn(self):
        features = self.get_all_features(self.state, self.action)

        Qsa = self.Q(features)
        maxQ = self.maxQ(self.next_state)
        error = self.reward + self.gamma * maxQ - Qsa
        self.error = error

        grad = features.reshape(self.numberoffeatures, 1)

        self.theta = self.theta + self.alpha * error * grad

    def Q(self, features):
        return np.asscalar(np.dot(features, self.theta))

    def Q_all_states(self, s):
        features = self.features(s)
        return np.asarray(list(
            map(lambda a: self.Q(np.concatenate((features, self.action_dependent_features(s, a)))), self.actionSet)))

    def argmaxQ(self, s):
        return self.actionSet[np.argmax(self.Q_all_states(s))]

    def maxQ(self, s):
        return np.max(self.Q_all_states(s))

    def square(self, list):
        return [i ** 2 for i in list]

    # state_labels = ['dx','dy','x_vel','y_vel','theta','psi','theta_dot','left_ground_contact','right_ground_contact']
    def features(self, simulationDynamics):
        # simulationDynamics[5] = 0 # psi - don't use
        state = simulationDynamics
        squared_state = [1]  # self.square(simulationDynamics)
        binary_state_limits = [1, 1, 0.09]
        binary_states = np.array([0 for _ in binary_state_limits])

        # 0.09 = approximately 5 degrees per second rotation, 0.25 = psi limit
        for i, lim in enumerate(binary_state_limits):
            if (simulationDynamics[i + 2] > lim):
                binary_states[i] = 1  # use .append sparingly
            elif (simulationDynamics[i + 2] < -lim):
                binary_states[i] = -1
            else:
                binary_states[i] = 0
        np.append(binary_states, 1)  # free parameter

        # ------------------------------------------------------------------------------------
        # This entire section is ~1.5 seconds faster than a single line:
        # final_state = np.concatenate((state, binary_states, self.action_dependent_features(simulationDynamics, action)))

        # action_dependent_states = self.action_dependent_features(simulationDynamics, action)

        len_state = len(state)
        len_squaredStates = len(squared_state)
        len_binaryStates = len(binary_states)
        # len_action_dep_states = len(action_dependent_states)

        final_state = np.zeros((len_state + len_squaredStates + len_binaryStates,))

        # This replaces concatenation --> Faster
        final_state[0:len_state] = state
        final_state[len_state:len_state + len_squaredStates] = squared_state
        final_state[len_state + len_squaredStates:] = binary_states
        # # ------------------------------------------------------------------------------------
        return final_state

    def action_dependent_features(self, state, action):
        Fe, Fs, psi = action
        theta = state[THETA]
        parameters = theta, Fe, Fs, psi
        ddx, ddy, ddtheta = self.rocket_kinematics(parameters)

        bools = [theta > 0 and Fs < 0,
                 theta < 0 and Fs > 0,
                 theta > 0 and psi < 0 and Fe > 0,
                 theta < 0 and psi > 0 and Fe > 0,
                 theta > 0 and ddtheta > 0,
                 theta < 0 and ddtheta < 0,
                 state[XX] > 0 and ddx < 0,
                 ddy < 2]

        additional_features = np.array([0 for _ in range(len(bools))])
        for i, exp in enumerate(bools):
            additional_features[i] = self.evalBool(exp, 1, -1)

        return additional_features

    def get_all_features(self, state, action):
        s_features = self.features(state)
        s_a_features = self.action_dependent_features(state, action)

        len_s_features = len(s_features)
        len_s_a_features = len(s_a_features)
        # This is faster than np.concatenate
        all_features = np.zeros((len_s_a_features + len_s_features,))
        all_features[0:len_s_features] = s_features
        all_features[len_s_features:] = s_a_features
        return all_features

    def evalBool(self, expression, conditionMet, conditionNotMet):
        if (expression):
            return conditionMet
        else:
            return conditionNotMet

    # Can use LQR linear delta_x_dot template to get better accuracy at different values of theta
    def rocket_kinematics(self, parameters):
        theta, Fe, Fs, psi = parameters
        # -----------------------------
        ddot_x = (Fe * theta + Fe * psi + Fs) / MASS
        ddot_y = (Fe - Fe * theta * psi - Fs * theta - MASS * GRAVITY) / MASS
        ddot_theta = (Fe * psi * (L1 + LN) - L2 * Fs) / INERTIA
        return ddot_x, ddot_y, ddot_theta

    def buildActionSet(self):
        # Fs = [i / 10 for i in range(-11, 11, 1)]
        # Fe = [i / 100 for i in range(0, 105, 5)]
        # psi = [i * DEGTORAD for i in range(-15, 16, 1)]

        if self.low_discretization:
            Fs = [-1, 0, 1]
            Fe = [0.6, 0.8]
            psi = [-5 * DEGTORAD, -1.5 * DEGTORAD, 0, 1.5 * DEGTORAD, 5 * DEGTORAD]
        else:
            # Fs = [0, 1]
            # Fe = [0.5, 0.6, 0.8, 0.9]
            # psi = [0, 1.5 * DEGTORAD, 3 * DEGTORAD, 6*DEGTORAD, 10*DEGTORAD]#[i * DEGTORAD for i in range(-9, 10, 3)]
            Fs = [-1, 0, 1]
            Fe = [0.6, 0.7, 0.75, 0.8, 0.9]
            psi = [i * DEGTORAD for i in range(-8, 8, 2)]

        # action = []
        # for f in Fe:
        #     action.extend([f, 0, p] for p in psi)
        # action.extend([0, fs, 0] for fs in Fs)
        action = []
        for f in Fe:
            for fs in Fs:
                action.extend([f, fs, p] for p in psi)
        # action.extend([0, fs, 0] for fs in Fs)
        return action
