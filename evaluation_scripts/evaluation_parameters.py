"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Different states for testing and benchmarking the different controllers.
"""

from constants import DEGTORAD

"""
A total of 24 tests are defined below, varying in initial conditions and disturbances.
"""
# 0 Translational velocity
INITIAL_STATES = [[0.5, 1, 0, -15, 0, 0],  # 1. Free fall with no initial acceleration
                 [0.5, 1, 0, -15, 5 * DEGTORAD, 0],
                 [0.3, 1, 0, -15, 0, 0],
                 [0.3, 1, 0, -15, 5 * DEGTORAD, 0],
                 [0.3, 1, 0, -15, -5 * DEGTORAD, 0],
                 # With x-translational Velocity
                 [0.5, 1, 3, -15, 0, 0],
                 [0.5, 1, -3, -15, 5 * DEGTORAD, 0],
                 [0.3, 1, 3, -15, 0, 0],
                 [0.3, 1, -3, -15, 5 * DEGTORAD, 0],
                 [0.3, 1, -3, -15, -5 * DEGTORAD, 0],
                 # With higher z-velocity
                 [0.5, 1, 3, -19, 0, 0],
                 [0.5, 1, -3, -19, 5 * DEGTORAD, 0],
                 [0.3, 1, 3, -19, 0, 0],
                 [0.3, 1, -3, -19, 5 * DEGTORAD, 0],
                 [0.3, 1, -3, -19, -5 * DEGTORAD, 0],
                 # With higher theta
                 [0.5, 1, 0, -15, 10 * DEGTORAD, 0],
                 [0.3, 1, 0, -15, 10 * DEGTORAD, 0],
                 [0.3, 1, 0, -15, -10 * DEGTORAD, 0],
                 # Added for disturbances
                 [0.5, 1, 0, -15, 0, 0],
                 [0.5, 1, 0, -15, 5 * DEGTORAD, 0],
                 [0.5, 1, 0, -15, -5 * DEGTORAD, 0],
                  # Added for impulses
                  [0.5, 1, 0, -15, 0, 0],
                  [0.5, 1, 0, -15, 5 * DEGTORAD, 0],
                  [0.5, 1, 0, -15, -5 * DEGTORAD, 0]]

# ---------------------------------------------------------------------------------------------------
DISTURBANCES = [None for _ in INITIAL_STATES]  # [None, (1, (1000, -5000))]
DISTURBANCES[18] = (0, (12, 0))  # time, force
DISTURBANCES[19] = (0, (12, 0))  # time, force
DISTURBANCES[20] = (0, (12, 0))  # time, force
# ---------------------------------------------------------------------------------------------------
IMPULSES = [None for _ in INITIAL_STATES]  # [None, (1, (1000, -5000))]
IMPULSES[21] = (1, (2000, 0))  # time, force
IMPULSES[22] = (1, (2000, 0))  # time, force
IMPULSES[23] = (1, (2000, 0))  # time, force
# ---------------------------------------------------------------------------------------------------
INITIAL_FORCES = [(0,0) for _ in INITIAL_STATES]