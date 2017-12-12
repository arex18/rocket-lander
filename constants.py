import sys, math
import matplotlib
matplotlib.use('TkAgg')
from enum import Enum


"""State array definitions"""
class State(Enum):
    x = 0
    y = 1
    x_dot = 2
    y_dot = 3
    theta = 4
    theta_dot = 5
    left_ground_contact = 6
    right_ground_contact = 7
# --------------------------------
"""Simulation Update"""
FPS = 60
UPDATE_TIME = 1/FPS

# --------------------------------
"""Simulation view, Scale and Math Conversions"""
# NOTE: Dimensions do not change linearly with Scale
SCALE = 30  # Adjusts Pixels to Units conversion, Forces, and Leg positioning. Keep at 30.

VIEWPORT_W = 1000
VIEWPORT_H = 800

SEA_CHUNKS = 25

DEGTORAD = math.pi/180

W = int(VIEWPORT_W / SCALE)
H = int(VIEWPORT_H / SCALE)

# --------------------------------
"""Rocket Relative Dimensions"""
INITIAL_RANDOM = 20000.0  # Initial random force (if enabled through simulation settings)

LANDER_CONSTANT = 1  # Constant controlling the dimensions
LANDER_LENGTH = 227 / LANDER_CONSTANT
LANDER_RADIUS = 10 / LANDER_CONSTANT
LANDER_POLY = [
    (-LANDER_RADIUS, 0), (+LANDER_RADIUS, 0),
    (+LANDER_RADIUS, +LANDER_LENGTH), (-LANDER_RADIUS, +LANDER_LENGTH)
]

NOZZLE_POLY = [
    (-LANDER_RADIUS+LANDER_RADIUS/2, 0), (+LANDER_RADIUS-LANDER_RADIUS/2, 0),
    (-LANDER_RADIUS + LANDER_RADIUS/2, +LANDER_LENGTH/8), (+LANDER_RADIUS-LANDER_RADIUS/2, +LANDER_LENGTH/8)
]

LEG_AWAY = 30 / LANDER_CONSTANT
LEG_DOWN = 0.3/LANDER_CONSTANT
LEG_W, LEG_H = 3 / LANDER_CONSTANT, LANDER_LENGTH / 8 / LANDER_CONSTANT

SIDE_ENGINE_VERTICAL_OFFSET = 5  # y-distance away from the top of the rocket
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = 10.0

# --------------------------------
"""Forces, Costs, Torque, Friction"""
MAIN_ENGINE_POWER = FPS*LANDER_LENGTH / (LANDER_CONSTANT * 2.1)  # Multiply by FPS since we're using Forces not Impulses
SIDE_ENGINE_POWER = MAIN_ENGINE_POWER / 50  # Multiply by FPS since we're using Forces not Impulses

INITIAL_FUEL_MASS_PERCENTAGE = 0.2  # Allocate a % of the total initial weight of the rocket to fuel
MAIN_ENGINE_FUEL_COST = MAIN_ENGINE_POWER/SIDE_ENGINE_POWER
SIDE_ENGINE_FUEL_COST = 1

LEG_SPRING_TORQUE = LANDER_LENGTH/2
NOZZLE_TORQUE = 500 / LANDER_CONSTANT
NOZZLE_ANGLE_LIMIT = 15*DEGTORAD

BARGE_FRICTION = 2

# --------------------------------
"""Landing Calibration"""
LANDING_VERTICAL_CALIBRATION = 0.03
TERRAIN_CHUNKS = 16 # 0-20 calm seas, 20+ rough seas
BARGE_LENGTH_X1_RATIO = 0.35# 0.35#0.27 # 0 -1
BARGE_LENGTH_X2_RATIO = 0.65#0.65 #0.73 # 0 -1
# --------------------------------
"Kinematic Constants"
# NOTE: Recalculate if the dimensions of the rocket change in any way
MASS = 25.222
L1 = 3.8677
L2 = 3.7
LN = 0.1892
INERTIA = 482.2956
GRAVITY = 9.81

# ---------------------------------
"""State Reset Limits"""
THETA_LIMIT = 35*DEGTORAD

# ---------------------------------
"""State Definition"""
# Added for accessing state array in a readable manner
XX = State.x.value
YY = State.y.value
X_DOT = State.x_dot.value
Y_DOT = State.y_dot.value
THETA = State.theta.value
THETA_DOT = State.theta_dot.value
LEFT_GROUND_CONTACT = State.left_ground_contact.value
RIGHT_GROUND_CONTACT = State.right_ground_contact.value
