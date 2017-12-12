import matplotlib.pyplot as plt
from environments.rocketlander import RocketLander, compute_derivatives
from constants import *
import numpy as np
from control_and_ai.pid import PID_psi
from control_and_ai.mpc import MPC

def plot_trajectory(controller, y_target_profile, initial_untransformed_state):
    x, y, x_dot, y_dot, theta, theta_dot = controller.guidance_target(state=initial_untransformed_state, final_x=16.5, y_profile=y_target_profile,
                                         time_horizon=10, time_step=1/60)

    plt.plot(x, y)
    plt.title("X-Z Trajectory Profile. Barge X-Position = 16.5.")
    plt.xlabel("X-Position/metres")
    plt.ylabel("Altitude/metres")
    plt.show()

def MPC_controller_run(env, simulation_settings, controller):
    s = env.state
    psi_PID = PID_psi()

    time_horizon = 20
    time_step = 1/10

    x_time, y_target_profile = controller.create_altitude_profile(max_altitude=26, ground_altitude=5, exponential_constant=3 / time_horizon,
                                                          timestep=time_step, timespan=100)

    final_x = 16.5
    finite_differences_step_size = 100

    total_reward = 0
    episodes = 0
    max_episodes = 10
    a = [0, 0, 0]
    original_action = [0, 0, 0]
    prev_state = np.zeros(6)

    done = False
    optimization_type = 1
    if optimization_type == 1:
        Q, R = controller.create_Q_R_matrices(5, 1e-2)
        R[2, 2] = 5
        # R[0, 0] = 0.5
        # Q[2, 2] = Q[4, 4] = Q[5, 5] = 5
    elif optimization_type == 2:
        Q, R = controller.create_Q_R_matrices(0.1, 1e-10)
        R[2, 2] = 5
        R[0, 0] = 0.5
        Q[2, 2] = Q[4, 4] = Q[5, 5] = 5
    else:
        Q, R = controller.create_Q_R_matrices(0.5, 1e-2, 4, 2)
        R[1, 1] = 1e-3

    for i in range(max_episodes):
        iteration = 0
        while (not done):
            if env.state[LEFT_GROUND_CONTACT] == 0 and env.state[LEFT_GROUND_CONTACT] == 0:
                print(iteration)
                # if iteration == 100:
                #     env.apply_disturbance((-5000,-10000))
                ss = env.untransformed_state
                targets = controller.guidance_target(state=ss, final_x=final_x, y_profile=y_target_profile,
                                                     time_horizon=time_horizon + 1, time_step=time_step)
                # ------------ DIFFERENT MODELS ------------
                psi = psi_PID.pid_algorithm(env.state, targets=None)
                A, B, x_0 = compute_derivatives(ss, a, finite_differences_step_size)

                if optimization_type == 1:
                    x, u = controller.optimize_linearized_model(A, B, ss, Q, R, np.array(targets), time_horizon=time_horizon, verbose=False)
                elif optimization_type == 2:
                    x, u = controller.optimize_analytical_model(ss, prev_state, Q, R, np.array(targets),
                                                               time_horizon=time_horizon, verbose=False, env=env)
                elif optimization_type == 3:
                    A = A[:-2, :-2]
                    B = B[:-2, :-1]
                    # A, B = compute_matrices_A_B_linearised_PID(env.untransformed_state, env.lander.mass, env.nozzle.angle)
                    x, u = controller.optimize_with_PID(A, B, ss[:-2], Q, R, np.array(targets)[:-2, :],
                                                    time_horizon=time_horizon, verbose=False)
                # -------------------------------------------
                #controller.plot_results(x,u)
                prev_state = ss
                if (u.value is not None):
                    action = np.array(u.value).squeeze()
                    for i in range(1):
                        original_action = action.T[i]
                        if optimization_type == 1 or optimization_type == 2:
                            a = original_action / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, 1]
                            a[2] = psi
                        else:
                            a = original_action / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER]
                            a = np.append(a, psi)
                        s, r, done, info = env.step(a)
                        total_reward += r
                        if iteration % 1 == 0:
                            if simulation_settings['Render']:
                                env.render()
                                env.draw_line(x=targets[XX], y=targets[YY], color=(0, 0, 0))
                                env.draw_line(x=np.array(x[0, :].value).squeeze(), y=np.array(x[1, :].value).squeeze(),
                                              color=(1, 0, 0))
                                env.refresh()

                print("X:\t{0}\t{1}".format(env.untransformed_state[0], env.untransformed_state[2]))
            else:
                s, r, done, info = env.step([0, 0, 0])
                if env.state[LEFT_GROUND_CONTACT] == 1 and env.state[LEFT_GROUND_CONTACT] == 1:
                    done = True

            if done:
                env.reset()
                print("Total Reward:\t{0}".format(total_reward))
                total_reward = 0
                episodes += 1
            iteration += 1

def MPC_controller_run_custom(env, simulation_settings, controller, Q, R, time_horizon, control_horizon,
                              y_target_profile, x_target, optimization_type=1, time_step=1/20, overwrite_psi=False):
    s = env.state
    psi_PID = PID_psi()

    finite_differences_step_size = 50

    total_reward = 0
    max_episodes = simulation_settings['Episodes']
    a = [0, 0, 0]
    original_action = [0, 0, 0]
    prev_state = np.zeros(6)

    done = False

    for i in range(max_episodes):
        iteration = 0
        while (not done):
            if env.state[LEFT_GROUND_CONTACT] == 0 or env.state[LEFT_GROUND_CONTACT] == 0:
                #ss = env.state[:-2]
                #targets = None
                ss = env.untransformed_state
                targets = controller.guidance_target(state=ss, final_x=x_target, y_profile=y_target_profile,
                                                     time_horizon=time_horizon + 1, time_step=time_step)
                #targets = np.diff(targets)
                # ------------ DIFFERENT MODELS ------------
                psi = psi_PID.pid_algorithm(env.state, targets=None)
                A, B, x_0 = compute_derivatives(ss, a, finite_differences_step_size)

                if optimization_type == 1:
                    x, u = controller.optimize_linearized_model(A, B, ss, Q, R, np.array(targets), time_horizon=time_horizon, verbose=False)
                elif optimization_type == 2:
                    x, u = controller.optimize_analytical_model(ss, prev_state, Q, R, np.array(targets),
                                                               time_horizon=time_horizon, verbose=False, env=env)
                elif optimization_type == 3:
                    A = A[:-2, :-2]
                    B = B[:-2, :-1]
                    # A, B = compute_matrices_A_B_linearised_PID(env.untransformed_state, env.lander.mass, env.nozzle.angle)
                    x, u = controller.optimize_with_PID(A, B, ss[:-2], Q, R, np.array(targets)[:-2, :],
                                                    time_horizon=time_horizon, verbose=False)
                # -------------------------------------------
                #controller.plot_results(x,u)
                prev_state = ss
                if (u.value is not None):
                    action = np.array(u.value).squeeze()
                    for i in range(control_horizon):
                        original_action = action.T[i]
                        if optimization_type == 1 or optimization_type == 2:
                            a = original_action / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, 1]
                        else:
                            a = original_action / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER]

                        if overwrite_psi:
                            a[2] = psi

                        s, r, done, info = env.step(a)
                        total_reward += r

                        if simulation_settings['Render'] or iteration % 5 == 0:
                            env.render()
                            env.draw_line(x=targets[XX], y=targets[YY], color=(0, 0, 0))
                            env.draw_line(x=np.array(x[0, :].value).squeeze(), y=np.array(x[1, :].value).squeeze(),
                                          color=(1, 0, 0))
                            env.refresh()
                else:
                    print("Action is None!")

                #print("X:\t{0}\t{1}".format(env.untransformed_state[0], env.untransformed_state[2]))
            else:
                s, r, done, info = env.step([0, 0, 0])
                if env.state[LEFT_GROUND_CONTACT] == 1 and env.state[LEFT_GROUND_CONTACT] == 1:
                    done = True

            if done:
                env.reset()
                print("Total Reward:\t{0}".format(total_reward))
                total_reward = 0
            iteration += 1

def call_MPC(env, simulation_settings, controller, mpc_settings : dict):

    time_horizon = mpc_settings['time_horizon']
    time_step = mpc_settings['time_step']
    time_span = mpc_settings['flight_time_span']
    ground_altitude = mpc_settings['ground_altitude']
    control_horizon = mpc_settings['control_horizon']
    x_target = mpc_settings['x_target']

    x_time, y_target_profile = controller.create_altitude_profile(max_altitude=26, ground_altitude=ground_altitude,
                                                                  exponential_constant=3 / time_horizon,
                                                                  timestep=time_step, timespan=time_span)

    optimization_type = 1
    if optimization_type == 1:
        k, p = 100, 1

        Q_weights = np.array([0.25, 4, 0.25, 0.25, 365, 131])
        Q = np.eye(6) * Q_weights * k

        R_weights = np.array([0.01, 0.1, 0.001])
        R = np.eye(3) * R_weights * p
        # R[0, 0] = 0.5
        # Q[2, 2] = Q[4, 4] = Q[5, 5] = 5
    elif optimization_type == 2:
        Q, R = controller.create_Q_R_matrices(0.1, 1e-10)
        R[2, 2] = 5
        R[0, 0] = 0.5
        Q[2, 2] = Q[4, 4] = Q[5, 5] = 5
    else:
        Q, R = controller.create_Q_R_matrices(0.5, 1e-2, 4, 2)
        R[1, 1] = 1e-3

    MPC_controller_run_custom(env, simulation_settings, controller, Q, R, time_horizon, control_horizon,
                              y_target_profile, x_target,
                              optimization_type, time_step, overwrite_psi=True)

simulation_settings = {'Side Engines': True,
            'Clouds': False,
            'Vectorized Nozzle': True,
            'Graph': False,
            'Render': True,
            'Starting Y-Pos Constant': 1,
            'Initial Coordinates': (0.5, 1, 0, 1),
            'Initial Force': (0,0),
            'Initial State': [0.5, 1, 0, -15, 0, 0],
            'Rows': 1,
            'Columns': 2,
            'Episodes': 1}

mpc_settings = {'time_horizon': 10,
                'time_step': 1/10,
                'flight_time_span': 100,
                'ground_altitude': 5,
                'control_horizon':2,
                'x_target': 16.5}

env = RocketLander(simulation_settings)
mpc_controller = MPC(env)
call_MPC(env, simulation_settings, mpc_controller, mpc_settings)




