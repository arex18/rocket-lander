"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: MPC definition.
"""

from constants import *
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as opt


class MPC:
    """ This class encapsulates the optimization related to MPC. Specific for Rocket Lander, not general. """

    def __init__(self, env):
        # last 2 state elements are not used since the environment includes left leg contact and right leg contact
        self.state_len = len(env.state[:-2])
        self.action_len = len(env.action_space)
        self.angle_limit = env.lander_tilt_angle_limit

    def create_Q_R_matrices(self, Q_matrix_weight, R_matrix_weight, Q_size=None, R_size=None):
        """
        Create Q, R matrices
        :param Q_matrix_weight:
        :param R_matrix_weight:
        :param Q_size:
        :param R_size:
        :return:
        """
        # state = x,y,x_dot,y_dot,theta,theta_dot
        if Q_size is None:
            Q_size = self.state_len
            R_size = self.action_len

        Q = np.eye(Q_size) * Q_matrix_weight
        R = np.eye(R_size) * R_matrix_weight
        return Q, R

    def optimize_analytical_model(self, initial_state, previous_state, Q, R, target, time_horizon=10, time_step=1 / 60,
                                  verbose=False, env=None):
        """
        Defines the optimisation problem for the analytical model defined explicity.
        :param initial_state:
        :param previous_state:
        :param Q:
        :param R:
        :param target:
        :param time_horizon:
        :param time_step:
        :param verbose:
        :param env:
        :return:
        """
        assert len(initial_state) == self.state_len
        # Create variables
        m = env.lander.mass
        J = env.lander.inertia

        x = opt.Variable(self.state_len, time_horizon + 2, name='x')
        # mass = opt.Variable(time_horizon + 1, name='mass')
        u = opt.Variable(self.action_len, time_horizon, name='actions')

        # Loop through the entire time_horizon and append costs
        cost_function = []
        # @TODO: Change to Matrix Form
        for t in range(time_horizon):
            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 2], Q) + opt.quad_form(u[:, t], R)

            A = [(u[0, t] * (x[4, t] + u[2, t]) + u[1, t]) * (time_step ** 2) / m + 2 * x[0, t + 1] - x[0, t],
                 (u[0, t] - u[1, t] * x[4, t] - m * GRAVITY) * time_step ** 2 / m + 2 * x[1, t + 1] - x[1, t],
                 (x[0, t + 1] - x[0, t]) / time_step,
                 (x[1, t + 1] - x[1, t]) / time_step,
                 (-u[0, t] * u[2, t] * (L1 + LN) + L2 * u[1, t]) * time_step ** 2 / J + 2 * x[4, t + 1] - x[4, t],
                 (x[4, t + 1] - x[4, t]) / time_step]
            # Dynamics
            dynamic_constraints = [x[i, t + 2] == expr for i, expr in enumerate(A)]
            limit_constraints = [
                # mass[t + 1] == mass[t] - 0.007, #* opt.exp(
                # -u[0, t] * MAIN_ENGINE_FUEL_COST - u[1, t] * SIDE_ENGINE_FUEL_COST),
                # --------------------------------------------------------------------------------
                # Actions
                u[0, t] >= MAIN_ENGINE_POWER / 6, u[0, t] <= MAIN_ENGINE_POWER,
                u[1, t] >= -SIDE_ENGINE_POWER, u[1, t] <= SIDE_ENGINE_POWER,
                u[2, t] >= -self.angle_limit, u[2, t] <= self.angle_limit]

            _constraints = np.append(dynamic_constraints, limit_constraints)
            cost_function.append(opt.Problem(opt.Minimize(_cost), constraints=list(limit_constraints)))

        # Add final cost
        problem = sum(cost_function)
        problem.constraints += [opt.norm(target[:, time_horizon] - x[:, time_horizon], 1) <= 10.01,
                                x[:, 0] == previous_state,
                                x[:, 1] == initial_state]

        # Minimize Problem
        problem.solve(verbose=verbose, solver=opt.SCS)
        # u[0, :].value.A.flatten()
        return x[:, 2:], u

    def optimize_linearized_model(self, A, B, initial_state, Q, R, target, time_horizon=10, verbose=False):
        """
        Optimisation problem defined for the linearised model, where the linearised state space is computed with
        finite differences (matrices A, B)
        :param A: Computed with finite differences
        :param B: Computed with finite differences
        :param initial_state:
        :param Q:
        :param R:
        :param target:
        :param time_horizon:
        :param verbose:
        :return:
        """
        assert len(initial_state) == self.state_len
        # Create variables
        x = opt.Variable(self.state_len, time_horizon + 1, name='states')
        u = opt.Variable(self.action_len, time_horizon, name='actions')

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(time_horizon):
            # _cost = opt.norm(x[2, t + 1],2)*50 + opt.norm(x[3, t + 1],2)*50 + opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(u[:, t], R) + opt.quad_form(u[:, t]-u[:, t-1], R*0.1)
            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(u[:, t], R) + opt.quad_form(
                u[:, t] - u[:, t - 1], R * 0.1)
            _constraints = [x[:, t + 1] == A * x[:, t] + B * u[:, t],
                            u[0, t] >= MAIN_ENGINE_POWER/1.8, u[0, t] <= MAIN_ENGINE_POWER / 1.05,
                            u[1, t] >= -SIDE_ENGINE_POWER, u[1, t] <= SIDE_ENGINE_POWER,
                            u[2, t] >= -self.angle_limit, u[2, t] <= self.angle_limit,
                            opt.norm(target[:, t + 1] - x[:, t + 1], 1) <= 0.01]

            cost_function.append(opt.Problem(opt.Minimize(_cost), constraints=_constraints))

        # Add final cost
        problem = sum(cost_function)
        problem.constraints += [opt.norm(target[:, time_horizon] - x[:, time_horizon], 1) <= 0.001,
                                x[:, 0] == initial_state]
        # Minimize Problem
        problem.solve(verbose=verbose, solver=opt.SCS)
        # u[0, :].value.A.flatten()
        return x, u


    def optimize_with_PID(self, A, B, initial_state, Q, R, target, time_horizon=10, verbose=False):
        """
        Optimisation problem defined for the PID-MPC
        :param A:
        :param B:
        :param initial_state:
        :param Q:
        :param R:
        :param target:
        :param time_horizon:
        :param verbose:
        :return:
        """
        # Create variables
        x = opt.Variable(4, time_horizon + 1, name='states')
        u = opt.Variable(2, time_horizon, name='actions')

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(time_horizon):
            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(u[:, t], R) + opt.quad_form(
                u[:, t] - u[:, t - 1], R * 0.1)
            _constraints = [x[:, t + 1] == A * x[:, t] + B * u[:, t],
                            u[0, t] >= MAIN_ENGINE_POWER / 6, u[0, t] <= MAIN_ENGINE_POWER,
                            u[1, t] >= -SIDE_ENGINE_POWER, u[1, t] <= SIDE_ENGINE_POWER]

            cost_function.append(opt.Problem(opt.Minimize(_cost), constraints=_constraints))

        # Add final cost
        problem = sum(cost_function)
        problem.constraints += [opt.norm(target[:, time_horizon] - x[:, time_horizon], 1) <= 0.01,
                                x[:, 0] == initial_state]
        # Minimize Problem
        problem.solve(verbose=verbose, solver=opt.SCS)
        # u[0, :].value.A.flatten()
        return x, u

    def regression_fit(self, x, y, deg):
        """
        Fit a polynomial to the supplied data
        :param x:
        :param y:
        :param deg: polynomial degree
        :return: [ao,a1,...,a_deg] where a0 is the coefficient of the highest degree
        """
        return np.polyfit(x, y, deg=deg)

    def guidance_target(self, state, final_x, y_profile, time_horizon, time_step, polyfit=None):
        """
        Creates a trajectory for each state element, i.e. [x,y,x_dot,y_dot,theta,theta_dot]
        :param state:
        :param final_x:
        :param y_profile:
        :param time_horizon:
        :param time_step:
        :param polyfit:
        :return:
        """
        current_x, current_y, current_theta, current_y_dot = state[XX], state[YY], state[THETA], state[Y_DOT]
        y_index = np.where(y_profile == np.extract(y_profile <= current_y, y_profile)[0])[0][0]

        # m, c = polyfit
        beta = 3
        gamma = 0.5

        delta_t = [t * time_step for t in range(0, time_horizon)]
        y = y_profile[y_index : y_index+time_horizon]
        y = np.clip(y, y_profile[-1], 26)
        # x_adjust = final_x * (1 - np.exp(-0.05 * np.array(delta_t)))
        # x = current_x + x_adjust * (final_x - current_x)
        x = current_x + (final_x - current_x) / (1 + np.exp(-np.linspace(-4, 4, num=time_horizon)))
        if current_x > final_x:
            x = np.clip(x, final_x, 33)
        else:
            x = np.clip(x, 0, final_x)

        y_dot = current_y_dot * np.exp(-np.linspace(0, 2, time_horizon))  # -2 * np.exp(-1/abs(y))
        x_dot = (x - final_x)

        theta = current_theta * np.exp(-beta * np.array(delta_t)) + current_theta * 0.3
        if theta[0] > 0:
            theta = np.clip(theta, 0, 15 * DEGTORAD)
        else:
            theta = np.clip(theta, -15 * DEGTORAD, 0)
        theta_dot = gamma * theta

        targets = [x, y, x_dot, y_dot, theta, theta_dot]
        return targets

    def create_altitude_profile(self, max_altitude=32, exponential_constant=0.05, ground_altitude=6, timespan=30,
                                timestep=1 / 60):
        """
        Creates the y-trajectory profile
        :param max_altitude: Max height
        :param exponential_constant:  decay constant in exponential term
        :param ground_altitude: Altitude of ground target
        :param timespan: time to reach target
        :param timestep: time step taken to reach target
        :return:
        """
        x_time = np.arange(0, timespan, timestep)
        y_target_profile = max_altitude * np.exp(-exponential_constant * x_time) + ground_altitude
        return x_time, y_target_profile

    def compute_matrices_A_B(self, state, action, env):
        """
        This method is used to compute matrices A and B using the analytical solution.
        A and B are the Jacobians with respect to x and u respectively. In other words, this function computes
        the partial derivatives as A = df/dx and B = df/du
        :param state: 6x1 array [x,y,x_dot,y_dot,theta,theta_dot]
        :param action: 3x1 array [Fe, Fs, psi]
        :param env: The current gym environment handle
        :return: A, B
        """
        Fe, Fs, psi = action
        theta = state[THETA]
        m = env.lander.mass
        J = env.lander.inertia

        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        cos_t_cos_p = cos_theta * cos_psi
        sin_t_cos_p = sin_theta * cos_psi
        sin_t_sin_p = sin_theta * sin_psi
        sin_t_cos_t = sin_theta * cos_theta
        cos_t_sin_p = cos_theta * sin_psi

        a_25 = (Fe * (cos_t_cos_p - sin_psi * sin_theta) - Fs * sin_theta) / m
        a_45 = (Fe * (sin_t_cos_t - cos_t_sin_p) - Fs * cos_theta) / m

        A = [[0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, a_25, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, a_45, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0]]

        b_21 = (sin_t_cos_p + cos_t_cos_p) / m
        b_22 = cos_theta / m
        b_23 = -Fe * sin_t_sin_p / m

        b_41 = (cos_t_cos_p - sin_t_sin_p) / m
        b_42 = -sin_theta / m
        b_43 = Fe * (-cos_t_sin_p - sin_t_cos_p) / m

        b_61 = -sin_psi * L1 / J
        b_62 = L2 / J
        b_63 = -Fe * cos_psi * L1 / J

        B = [[0, 0, 0],
             [b_21, b_22, b_23],
             [0, 0, 0],
             [b_41, b_42, b_43],
             [0, 0, 0],
             [b_61, b_62, b_63]]

        return np.array(A), np.array(B)

    def plot_results(self, x, u):
        from plotting import plotty

        y_labels = [r"$u_t$", r"$x_t, y_t$", r"$theta_t$"]
        x_labels = ['Time Horizon', 'Time Horizon', 'Time Horizon']

        res = plotty.Graphing()

        fig, [ax1, ax2, ax3] = res.create_figure_and_subplots(new_figure=True, y_labels=y_labels, x_labels=None,
                                                              row_number=3, column_number=1)
        # plt.title('Inputs and States vs. Time Horizon')

        res.plot_graph(None, u[0, :].value.A.flatten() / MAIN_ENGINE_POWER, ax1, labeltext='Main Engine', marker='o')
        res.plot_graph(None, u[1, :].value.A.flatten() / SIDE_ENGINE_POWER, ax1, labeltext='Side Thrusters', marker='o')
        res.plot_graph(None, u[2, :].value.A.flatten() / NOZZLE_ANGLE_LIMIT, ax1, labeltext='Nozzle Angle', marker='o')

        res.plot_graph(None, x[0, :].value.A.flatten(), ax2, labeltext='x-position', marker='o')
        res.plot_graph(None, x[1, :].value.A.flatten(), ax2, labeltext='y-position', marker='o')
        res.plot_graph(None, x[4, :].value.A.flatten(), ax3, labeltext='theta', marker='o')

        res.show_legend([ax1, ax2, ax3])
        res.show_plot()
        plt.show()

def compute_matrices_A_B_linearised_PID(state, mass, nozzle_angle):
    """
    This function is for the minimized state: [x,y,x_dot,y_dot] used in the linearised MPC with psi controlled
    with a PID
    :param state: State vector np.array: [x,y,x_dot,y_dot]
    :param mass:  obtained from env.lander.mass
    :param nozzle_angle: obtained from env.nozzle.angle
    :return: A = 4x4, B = 4x2
    """
    theta = state[THETA]

    a = math.sin(theta + nozzle_angle) / mass
    b = math.cos(theta) / mass
    c = math.cos(theta + nozzle_angle) / mass
    d = -math.sin(theta) / mass

    A = [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    B = [[0, 0],
         [0, 0],
         [a, b],
         [c, d]]

    return np.array(A), np.array(B)

