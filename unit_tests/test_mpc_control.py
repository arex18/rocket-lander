import unittest

import matplotlib.pyplot as plt
import numpy as np

from control_and_ai.unstable_control import mpc_control
from rocketlander_v2 import RocketLander


class MPC_Control_Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MPC_Control_Tests, self).__init__(*args, **kwargs)
        simulation_settings = {'Side Engines': True,
                               'Clouds': False,
                               'Vectorized Nozzle': True,
                               'Graph': False,
                               'Render': False,
                               'Starting Y-Pos Constant': 1.2,
                               'Initial Force': (0,0),
                               'Rows': 1,
                               'Columns': 2}
        self.env = RocketLander(simulation_settings)

    def test_optimizer(self):
        state_len = 6
        time_horizon = 50
        action_len = 3
        alpha = 0.2  # random purposes # Makes a difference if the problem is Optimal Feasible or not
        beta = 5  # random purposes
        A = np.eye(state_len) + alpha * np.random.randn(state_len, state_len)
        B = np.random.randn(state_len, action_len)

        x_initial = beta * np.random.randn(state_len, 1)

        controller = mpc_control.MPC(self.env)
        Q, R = controller.create_Q_R_matrices(Q_matrix_weight=2, R_matrix_weight=0.2)
        x,u = controller.optimize(A,B,x_initial,Q,R,time_horizon,verbose=True)
        # u[0, :].value.A.flatten()
        print(x,u)

    def test_guidance(self):
        controller = mpc_control.MPC(self.env)
        state = self.env.untransformed_state
        y_target_profile = controller.create_altitude_profile()

        state_len = 6
        final_x = 16.5
        final_y = 6
        time_horizon = 50
        x = [final_x, state[0]]
        y = [final_y, state[1]]

        [m,c] = controller.regression_fit(x,y,deg=1)

        targets = controller.guidance_target(state=state, final_x=final_x, y_profile=y_target_profile,
                                           current_time_iteration=200, time_horizon=time_horizon, time_step=1 / 60,
                                           polyfit=[m, c])
        assert np.size(targets) == state_len
        plt.plot(targets[0],targets[2])
        plt.show()
        plt.plot(targets[1],targets[3])
        plt.show()
        plt.plot(targets[4], targets[5])
        plt.show()














