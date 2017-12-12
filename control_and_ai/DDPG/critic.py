"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Critic network definition using Tensorflow.
"""

import tensorflow as tf
import numpy as np
from .utils import layer, batch_layer

class Critic():

    def __init__(self, sess, action_space_size, env_space_size, learning_rate=0.001, gamma=0.99, tau=0.001,
            optimizer=tf.train.AdamOptimizer):
        self.sess = sess
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.loss_val = 0
        self.state_ph = tf.placeholder(tf.float32, shape=(None, env_space_size), name='state_ph')
        self.action_ph = tf.placeholder(tf.float32, shape=(None, action_space_size), name='action_ph')

        self.features_ph = tf.concat([self.state_ph, self.action_ph], axis=1)

        self.phase = tf.placeholder(tf.bool, name='phase')  # Are we trianing or not?

        #self.infer = self.create_nn_with_batch(self.features_ph, self.phase)
        self.infer = self.create_nn(self.features_ph)
        self.weights = [v for v in tf.trainable_variables() if 'critic' in v.op.name]

        # Target network code "repurposed" from Patrick Emani :^)
        self.target = self.create_nn(self.features_ph, name='critic_target')
        #self.target = self.create_nn_with_batch(self.features_ph, self.phase, name='critic_target')
        self.target_weights = [v for v in tf.trainable_variables() if 'critic' in v.op.name][len(self.weights):]

        self.update_target_weights = \
        [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]

        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.infer))

        self.train = optimizer(learning_rate).minimize(self.loss)

        self.gradient = tf.gradients(self.infer, self.action_ph) # Why gradients here?. Called in update

    def update(self, old_states, old_actions, rewards, new_states, new_actions, is_terminals):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            wanted_qs = self.sess.run(self.target,
                    feed_dict={
                        self.state_ph: new_states,
                        self.action_ph: new_actions,
                        self.phase: True
                    })

            for i in range(len(wanted_qs)):
                if is_terminals[i]:
                    wanted_qs[i] = rewards[i]
                else:
                    wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            self.loss_val, _ = self.sess.run([self.loss, self.train],
                    feed_dict={
                        self.state_ph: old_states,
                        self.action_ph: old_actions,
                        self.wanted_qs: wanted_qs,
                        self.phase: True
                    })

            self.sess.run(self.update_target_weights)

    def get_gradients(self, state, action):
        grads = self.sess.run(self.gradient,
                feed_dict={
                    self.state_ph: state,
                    self.action_ph: action,
                    self.phase: True
                })

        return grads[0]

    def create_nn_with_batch(self, state, phase, name='critic'):
        input = int(state.shape[1])
        with tf.variable_scope(name + '_fc_1'):
            fc1 = batch_layer(state, input, 400, phase, name+'Layer-0', nonlinearity=tf.nn.relu)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = batch_layer(fc1, 400, 400, phase, name+'Layer-1', nonlinearity=tf.sigmoid)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = batch_layer(fc2, 400, 1, phase, name+'Layer-2', nonlinearity=None)
        return fc3

    def create_nn(self, features, name='critic'):
        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 400, activation=tf.nn.relu)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 300, activation=tf.sigmoid)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 1, is_output=True)

        return fc3
