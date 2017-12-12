"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network.
"""

import tensorflow as tf
from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.train import set_up
from control_and_ai.DDPG.train_third_model_unnormalized import train as train_third_model_unnormalized

from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from main_simulation import RocketLander

# with tf.device('/cpu:0'):
FLAGS = set_up()

action_bounds = [1, 1, 15*DEGTORAD]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 500}
env = RocketLander(simulation_settings)
#env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True, write_upon_reset=True)

FLAGS.retrain = False # Restore weights if False
FLAGS.test = False
FLAGS.num_episodes = 500
model_dir = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//DDPG//model_2_longer_unnormalized_state'
with tf.device('/cpu:0'):
    agent = DDPG(
        action_bounds,
        eps,
        env.observation_space.shape[0],
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        retrain=FLAGS.retrain,
        log_dir=FLAGS.log_dir,
        model_dir=model_dir)

    #train(env, agent, FLAGS)
    #test(env, agent, simulation_settings)
    train_third_model_unnormalized(env, agent, FLAGS)
#train_second_model(env, agent, FLAGS)

