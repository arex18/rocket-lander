import unittest
from main_simulation import simulate_kinematics
import numpy as np
import main_simulation

class Env_Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Env_Tests, self).__init__(*args, **kwargs)

    def test_kinematics(self):
        action = [0, 0, 0]

        simulation_settings = {'Side Engines': True,
                               'Clouds': False,
                               'Vectorized Nozzle': True,
                               'Graph': False,
                               'Render': False,
                               'Starting Y-Pos Constant': 1.3,
                               'Initial Force': (0, 0),
                               'Initial Coordinates': (0.5,0.7,0,'normalized')}

        env = main_simulation.RocketLander(simulation_settings)
        state = env.untransformed_state

        eps = 1/60
        len_state = len(state)
        len_action = len(action)
        ss = np.tile(state, (len_state, 1))
        x1 = ss + np.eye(len_state) * eps
        x2 = ss - np.eye(len_state) * eps
        aa = np.tile(action, (len_state, 1))
        f1 = simulate_kinematics(x1, aa, simulation_settings)
        f2 = simulate_kinematics(x2, aa, simulation_settings)
        delta_A = (f1 - f2) / 2 / eps # Jacobian
        assert (len(delta_A) == len_state)

        x3 = np.tile(state, (len_action, 1))
        u1 = np.tile(action, (len_action, 1)) + np.eye(len_action) * eps
        u2 = np.tile(action, (len_action, 1)) - np.eye(len_action) * eps
        f1 = simulate_kinematics(x3, u1, simulation_settings)
        f2 = simulate_kinematics(x3, u2, simulation_settings)
        delta_B = (f1 - f2) / 2 / eps
        delta_B = delta_B.T
        assert (len(delta_B) == len_state)

        # Arrays returned in the format x_next = A.x + B.u
        # A = 6x6, B = 6x3, u = 3x1, x = 6x1
