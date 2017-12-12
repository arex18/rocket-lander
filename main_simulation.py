"""
Author: Reuben Ferrante
Date:   16/08/2017
Description: This is the  main running point of the simulation. Set settings, algorithm, episodes,...
"""

from environments.rocketlander import RocketLander
from constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
import numpy as np

if __name__ == "__main__":
    # Settings holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    s = env.reset()

    from control_and_ai.pid import PID_Benchmark

    # Initialize the PID algorithm
    pid = PID_Benchmark()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05
    total_reward = 0
    episode_number = 5

    for episode in range(episode_number):
        while (1):
            a = pid.pid_algorithm(s) # pass the state to the algorithm, get the actions
            # Step through the simulation (1 step). Refer to Simulation Update in constants.py
            s, r, done, info = env.step(a)
            total_reward += r   # Accumulate reward
            # -------------------------------------
            # Optional render
            env.render()
            # Draw the target
            env.draw_marker(env.landing_coordinates[0], env.landing_coordinates[1])
            # Refresh render
            env.refresh(render=False)

            # When should the barge move? Water movement, dynamics etc can be simulated here.
            if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                # Random Force on rocket to simulate wind.
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            # Touch down or pass abs(THETA_LIMIT)
            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                total_reward = 0
                env.reset()
                break
