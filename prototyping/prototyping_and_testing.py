from rocketlander_v2 import *
import copy
import cProfile
from control_and_ai.helpers import *
import timeit
import time
# Plot results.
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import matplotlib.pylab as pylab


def es():
    MAX_EPISODES = 200
    MAX_STEPS = 250
    DO_NOTHING_ACTION = 0
    POPULATION = 100
    LEARNING_RATE_EXPLORING = 0.00025  # Learning rate
    LEARNING_RATE_MATURE = 0.0002  # Learning rate
    SIGMA = 0.1

    mutation_environments = []

    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': True,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random'}  # (-5000, -INITIAL_RANDOM)


    main_env = RocketLander(simulation_settings)
    for i in range(POPULATION):
        mutation_environments.append(RocketLander(simulation_settings))  # gym.make('LunarLander-v2'))

    savefilepath='C://Users//REUBS_LEN//PycharmProjects//RocketLanding//weights_es_variable_psi.npy'

    #ES_2(main_env, mutation_environments)
    for i in range(POPULATION):
        mutation_environments[i].close()

def state_sample_test(samples):
    return get_state_sample(samples)

def test_rocket_initialisation():
    simulation_settings = {'Side Engines': True,
                'Clouds': False,
                'Vectorized Nozzle': True,
                'Graph': False,
                'Render': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': (0,0),
                'Rows': 1,
                'Columns': 2,
                'Initial Coordinates': (W/2, H/1.2, 0)}

    env = RocketLander(simulation_settings)
    print(env.state)
    print(env.lander.position.x)
    env.step([0,0,0])
    print(env.state)
    print(env.lander.position.x)
    env.adjust_dynamics(x=0.8,y=0.5,y_dot=1,x_dot=1,theta=5,theta_dot=1)
    env.step([0,0,0])
    print(env.state)
    print(env.lander.position.x)
    env.step([0.2,0.6,0])
    print(env.state)
    print(env.lander.position.x)
    env.step([0.2,0.6,0])
    print(env.state)
    print(env.lander.position.x)

def timed(f, show_output=True):
  start = time.time()
  ret = f()
  elapsed = time.time() - start
  if show_output:
      print(elapsed)
  return ret, elapsed

def test_time_list_flattening():
    from collections import Iterable
    from itertools import chain

    test_array = [(1,4),(12,23),(43,23.1)]

    def flatten(items):
        """Yield items from any nested iterable; see REF."""
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    total_time = 0
    for i in range(1000000):
        start = time.time()
        list(flatten(test_array))
        total_time += time.time()-start
    print(total_time/1000000)

    # Much faster
    total_time = 0
    for i in range(1000000):
        start = time.time()
        list(chain.from_iterable(test_array))
        total_time += time.time() - start
    print(total_time / 1000000)


test_time_list_flattening()


