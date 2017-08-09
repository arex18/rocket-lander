import tensorflow as tf
import numpy as np
from .utils import layer, batch_layer

class Actor():

    def __init__(self, 
            sess,
            action_space_bounds,
            exploration_policies,
            env_space_size,
            learning_rate=0.0001,
            tau=0.001,
            optimizer=tf.train.AdamOptimizer):

        self.sess = sess
        self.learning_rate = learning_rate
        self.action_space_bounds = action_space_bounds
        self.action_space_size = len(action_space_bounds)
        self.exploration_policies = exploration_policies
        self.tau = tau


        self.phase = tf.placeholder(tf.bool, name='phase')  # Are we trianing or not?

        self.state_ph = tf.placeholder(tf.float32, shape=(None, env_space_size))

        #self.infer = self.create_nn_with_batch(self.state_ph, self.phase)
        self.infer = self.create_nn(self.state_ph)
        self.weights = [v for v in tf.trainable_variables() if 'actor' in v.op.name]

        # Target network code "repurposed" from Patrick Emani :^)
        #self.target = self.create_nn_with_batch(self.state_ph, self.phase, name='actor_target')
        self.target = self.create_nn(self.state_ph, name='actor_target')
        self.target_weights = [v for v in tf.trainable_variables() if 'actor' in v.op.name][len(self.weights):]

        self.update_target_weights = \
        [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]

        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        self.policy_gradient = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.train = optimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))

    def get_action(self, state, explore=True):
        actions = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.phase: True
                })

        if explore:
            for action in actions:
                for i in range(len(action)):
                    # Ornstein Uhlenbeck process for exploration
                    action[i] = self.exploration_policies[i].get_noise(action[i])
                    # bound it to the action space
                    action[i] = max(min(action[i], self.action_space_bounds[i]), -self.action_space_bounds[i])

        return actions

    def get_target_action(self, state):
        actions = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.phase: True
                })

        return actions

    def update(self, state, action_derivs):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            weights, policy_grad, _ = self.sess.run([self.weights, self.policy_gradient, self.train],
                    feed_dict={
                        self.state_ph: state,
                        self.action_derivs: action_derivs,
                        self.phase: True
                    })

            self.sess.run(self.update_target_weights)

    def create_nn_with_batch(self, state, phase, name='actor'):
        input = int(state.shape[1])
        with tf.variable_scope(name + '_fc_1'):
            fc1 = batch_layer(state, input, 400, phase, name+'Layer-0', nonlinearity=tf.nn.relu)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = batch_layer(fc1, 400, 400, phase, name+'Layer-1', nonlinearity=tf.sigmoid)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = batch_layer(fc2, 400, self.action_space_size, phase, name+'Layer-2', nonlinearity=tf.tanh)
        return fc3 * self.action_space_bounds

    def create_nn(self, state, name='actor'):
        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(state, 400, activation=tf.nn.relu)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 300, activation=tf.sigmoid)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, self.action_space_size, is_output=True)

        return tf.tanh(fc3) * self.action_space_bounds
