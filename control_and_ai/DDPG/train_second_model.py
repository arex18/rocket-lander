"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""

import sys
import os
import shutil
import gym
import argparse
import sklearn.preprocessing
sys.path.append('C://Users//REUBS_LEN//PycharmProjects//RocketLanding')
from environments.rocketlander import get_state_sample
from constants import *
from .utils import Utils
import numpy as np


def train(env, agent, FLAGS):
    print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 1000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE")
    #print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 2000")
    obs_size = env.observation_space.shape[0]

    util = Utils()
    state_samples = get_state_sample(samples=5000, normal_state=True)
    util.create_normalizer(state_sample=state_samples)

    for episode in range(1, FLAGS.num_episodes + 1):
        old_state = None
        done = False
        total_reward = 0

        state = env.reset()
        state = util.normalize(state)
        max_steps = 500

        left_or_right_barge_movement = np.random.randint(0, 2)
        epsilon = 0.05

        while not done:#for t in range(max_steps): # env.spec.max_episode_steps
            if FLAGS.show or episode % 10 == 0:
                env.refresh(render=True)

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not FLAGS.test)

            # take it
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)
            total_reward += reward

            if state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if not FLAGS.test:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        agent.log_data(total_reward, episode)

        if episode % 50 == 0 and not FLAGS.test:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        print("Reward:\t{0}".format(total_reward))