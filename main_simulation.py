from environments.rocketlander import *

if __name__ == "__main__":

    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random',  # (6000, -10000)
                'Rows': 1,
                'Columns': 2}

    env = RocketLander(settings)
    s = env.reset()

    from control_and_ai.pid import PID_Benchmark

    pid = PID_Benchmark()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05
    total_reward = 0
    while (1):
        a = pid.pid_algorithm(s)
        s, r, done, info = env.step(a)
        total_reward += r
        # -------------------------------------
        env.render()
        env.drawMarker(env.landing_coordinates[0], env.landing_coordinates[1])
        env.refresh(render=False)

        if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
            env.move_barge_randomly(epsilon, left_or_right_barge_movement)
            env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
            env.apply_random_y_disturbance(epsilon=0.005)

        if done:
            print('Total Reward:\t{0}'.format(total_reward))
            total_reward = 0
            env.reset()
