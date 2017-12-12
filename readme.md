# Rocket Lander OpenAI Environment

This is a vertical rocket landing simulator modelled from SpaceX's Falcon 9 first stage rocket. The simulation 
was developed in Python 3.5 and written using [OpenAI's gym environment](https://gym.openai.com/docs/). 
Box2D was the physics engine of choice and the environment is similar to the [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/). [This](https://www.youtube.com/watch?v=4_igzo4qNmQ) is a video of the simulator in action.

![environment 3](https://user-images.githubusercontent.com/16338481/33860598-870fd702-ded1-11e7-8bdb-86fa01e2db47.JPG)


Code used for:
* Proportional Integral Control (PID)
* Deep Deterministic Policy Gradients (DDPG)
* Modern Predictive Control (MPC)

is also available, but not generalized. Other sample code available:

* Evolutionary Strategy (ES)
* Function Approximation Q-Learning (FA Q-Learning)
* Linear Quadratic Regulator (LQR)

Main contributions of the project is the environment, with other scripts for controllers included for context and general reference.

Code for the simulation exists under ```environments```.
## Getting Started

Download the repo. The rocket lander might be forked and included as a separate package which can eventually be installed using pip.

### Prerequisites

List of libraries needed to run the project: (some, e.g. cvxpy require other pre-requisites). Windows users head to the 
life-saving list of [Windows Python Extension Libraries](Unofficial Windows Binaries for Python Extension Packages)
to install cvxpy and any other failing pip installs.

```
tensorflow
matplotlib
gym
numpy
Box2D
logging
pyglet
cvxpy
abc
concurrent

python pip install PATH_TO_YOUR_DOWNLOADED_LIBRARY (ending in whl)
```

### Checking Functionality

Run the main_simulation.py and check that the simulation starts. A window showing the rocket should pop up.
If running from terminal, simply:

```
python main_simulation.py
```
## Problem Definition
### Introduction

![rocket model](https://user-images.githubusercontent.com/16338481/33860716-021480d8-ded2-11e7-85b4-6fffbcea0258.PNG)

The point of this small project was to compare and contrast classical control methods with AI algorithms as applied to a
**continuous control problem.** This is unlike the lunar lander, where the action space is discretized.
Example of discretized action space:
```
lunar_lander_horizontal_thrusters = {-1, -0.5, 0, 0.5, 1}
lunar_lander_vertical_thruster = {0, 0.5, 1}
```
Example of continuous action space:
```
lunar_lander_left_thruster = [0, 1] (negated in code)
lunar_lander_right_thruster = [0, 1]
lunar_lander_vertical_thruster = [0, 1]
```
However, most real life problems exist in both continuous state and continuous action space. Both state and action
domains can be discretized, but this leads to various practical limitations.

The simulator was therefore built for continuous action purposes. PID, MPC, ES and DDPG algorithms were compared,
with the DDPG showing impressive results.  DDPG solves the discrete action space limitation of Q-
Learning controllers as well as the state approximation by using two separate NNs 
to implement the actor-critic architecture. NNs are capable of emulating a state on a 
continuous  level.  Although  somewhat  complex,  DDPG  managed  to  obtain  the 
highest efficiency and best overall control

### Simulation states and actions

In code, the state is defined as:
```
State = [x_pos, y_pos, x_vel, y_vel, lateral_angle, angular_velocity]
Actions = Fe, Fs, $psi$
```
* Fe = Main Engine (vertical thruster) ```[0, 1]```
* Fs = Side Nitrogen Thrusters ```[-1, 1]```
* Psi = Nozzle angle ```[-NOZZLE_LIMIT, NOZZLE_LIMIT]```

All simulation settings, limits, polygons, clouds, sea, etc. are defined as constants
in the file ```constants.py```.

### Controllers

Code for controllers exists under ```control_and_ai```, with the DDPG having a separate package. Several _unstructured_ 
scripts were written during prototyping and training, so apologies for messy untested code. Trained models are also
provided in different directories.

Script evaluations were left here for reference.


## Version

1.1.0

## Authors

* **Reuben Ferrante**

References: https://gym.openai.com/envs/LunarLander-v2/

## License

This project is licensed under the MIT License.
