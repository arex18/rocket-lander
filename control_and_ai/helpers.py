"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Builds custom states for AI training purposes.
"""
import numpy as np

class State_Builder():
    """ Builds custom states associated with the environment of Rocket Lander. General, but Rocket Lander specific. """

    def __init__(self, integral_number):
        self.integrals = [0 for i in range(integral_number)]

    def incrementIntegrals(self, variables):
        for i, v in enumerate(variables):
            self.integrals[i] = self.integrals[i] + v

    def evaluateTarget(self, state):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = state

        angle_target = dx * 0.5 + vel_x * 1.0  # angle should point towards center (dx is horizontal coordinate, vel_x hor speed)
        if angle_target > 0.4: angle_target = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_target < -0.4: angle_target = -0.4
        hover_target = 0.55 * np.abs(dx)  # target y should be proportional to horizontal offset

        self.targets = [hover_target, angle_target]

    def evaluateBinary(self, condition, value_if_true, value_if_false):
        if condition:
            return value_if_true
        else:
            return value_if_false

def buildState(state_builder, state):
    state_builder.evaluateTarget(state)

    dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = state

    x_error = state_builder.targets[1] - theta
    x_dterror = -omega

    # ------------------------------------------
    y_error = state_builder.targets[0] - dy
    y_dterror = -vel_y

    # ------------------------------------------
    theta_error = theta
    theta_dterror = -omega - vel_x

    binary_theta = state_builder.evaluateBinary(theta > 0, 0, 1)
    binary_y_vel = state_builder.evaluateBinary(vel_y < -1, 0, 1)

    state_builder.incrementIntegrals([x_error, y_error])

    return [dx, vel_x, x_dterror, x_dterror, state_builder.integrals[0], y_error,
            legContact_left, legContact_right, y_dterror, state_builder.integrals[1], theta_error, theta_dterror,
            binary_theta, binary_y_vel]
