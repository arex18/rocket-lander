"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: NN layer helper functions, normalization.
"""

import tensorflow as tf
import sklearn.preprocessing
import numpy as np


def layer(input_layer, num_next_neurons, is_output=False, activation=tf.nn.relu):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]

    weights = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [num_next_neurons], initializer=tf.contrib.layers.xavier_initializer())

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    act = activation(dot)
    return act


def batch_layer(inputs, input_dim, output_dim, phase, layernumber, nonlinearity=tf.nn.tanh):
    with tf.variable_scope(layernumber):
        output = tf.contrib.layers.fully_connected(inputs, output_dim,
                                                   activation_fn=nonlinearity,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer())

        _batchOutput = tf.layers.batch_normalization(output,
                                                    center=True, scale=True,
                                                    training=phase,
                                                    name=layernumber)  # Are we training or not?
    if nonlinearity is None:
        return _batchOutput
    else:
        return nonlinearity(_batchOutput)

class Utils():

    def create_normalizer(self, state_sample):
        self.normalizer = self.get_sample_statistics_and_normalizer(state_sample)
        self.state_len = len(self.normalizer.mean_)

    def normalize(self, state):
        return self.normalizer.transform(np.array(state).reshape(1,-1)).flatten()

    def get_sample_statistics_and_normalizer(self, state_sample):
        normalizer = sklearn.preprocessing.StandardScaler()
        # The input to the NN should be stationary. Normalize the difference in states
        return normalizer.fit(state_sample)  # contains the mean and variance

    def get_difference(self, s, s_next):
        return (s_next - s).flatten()

    def process_state(self, state):
        return self.normalize(state)