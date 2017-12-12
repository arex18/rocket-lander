"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: General plotting helpers for the Rocket Lander.
"""


from plotting.plotty import  *
import numpy as np


def plot_trajectory(x, y, *args):
    plt.plot(x, y)
    plt.title("X-Z Trajectory Profile. Barge X-Position = 16.5.")
    plt.xlabel("X-Position Difference/metres")
    plt.ylabel("Z-Altitude Difference/metres")

def convert_state_and_plot_trajectory(state_history):
    test_history = np.matrix(state_history)
    plot_trajectory(test_history[:, 0], test_history[:, 1])

def convert_state_and_plot_trajectory_2(res, fig, axis, state_history):
    test_history = np.matrix(state_history)
    res.plot_graph(test_history[:, 0], test_history[:, 1], axis)