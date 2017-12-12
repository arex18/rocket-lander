"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Quick scripts written for evaluation.
"""

from threading import Timer
from main_simulation import RocketLander
from evaluation_scripts.evaluation_parameters import *
from control_and_ai.controllers import *
from constants import *


class Evaluation_Framework():
    """
    This Framework is intended for the evaluation_scripts of controllers. Once instantiated, execute_evaluation should be called
    , with the required parameters.
    """
    def __init__(self, simulation_settings : dict, controller_settings : dict = None):
        # Dictionaries holding various simulation and controller specific settings
        self.simulation_settings = simulation_settings
        self.controller_settings = controller_settings
        self.reset()

    def reset_and_adjust_initial_conditions(self, env, initial_state, initial_force):
        # Adjust Initial Coordinates
        x, y, x_dot, y_dot, theta, theta_dot = initial_state
        env.settings['Initial Coordinates'] = (x, y, 0, True)  # x, y, randomness, normalized values
        env.settings['Initial Force'] = initial_force
        env.reset()
        env.adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        return env

    def reset(self):
        self.IMPULSE_FLAG = False
        self.DISTURBANCE_FLAG = False

    def landing_check(self, state):
        """
        Checks if the rocket landed or not.
        :param state: n x 1 state
        :return: 1 = Landed, 0 = One Leg Touched Down, -1 = Unsuccessful Landing
        """
        landed = 0  # -1 = not landed, 0 = partial landing (e.g. half leg out), 1 = landed successfully
        if state[LEFT_GROUND_CONTACT] == 1 and state[RIGHT_GROUND_CONTACT] == 1:
            landed = 1
        elif state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
            landed = -1
        else:
            landed = 0
        return landed

    def start_disturbance_timer(self, disturbance, function):
        disturbance_time, disturbance_force = disturbance
        t = Timer(disturbance_time, function)
        t.start()
        return t, disturbance_force

    def set_impulse_flag(self):
        self.IMPULSE_FLAG = True

    def set_disturbance_flag(self):
        self.DISTURBANCE_FLAG = True

    def get_final_cost_and_fuel(self, env, state):
        total_fuel_consumed = env.get_consumed_fuel()
        final_state_cost = env.compute_cost(np.append(state, total_fuel_consumed))
        return final_state_cost, total_fuel_consumed

        # Cost, Fuel, Reward, Trajectory

    def render(self, env, controller):
        env.render()
        controller.draw(env)
        env.refresh(render=False)

    def execute_evaluation(self, env, controller, initial_states: list, initial_forces: list, disturbances: list, impulses: list):
        assert len(initial_states) == len(disturbances)

        number_of_tests = len(initial_states)
        len_actions = len(env.action_space)


        state_history = [[] for _ in range(number_of_tests)]
        untransformed_state_history = [[] for _ in range(number_of_tests)]

        reward_results = [[] for _ in range(number_of_tests)]
        action_history = [[] for _ in range(number_of_tests)]

        done = False
        total_reward = 0
        disturbance_force = (0, 0)
        impulse = (0, 0)
        max_steps = 1000

        print("Test Number\tTotal Reward\tFinal Cost\tFuel Consumed\tAverage Theta\tLanded\tFe\tFl\tFr\t|psi| > 3")


        for i in range(number_of_tests):
            steps = 0
            done = False
            env = self.reset_and_adjust_initial_conditions(env, initial_states[i], initial_forces[i])
            s = env.state

            if disturbances[i] is not None:
                t1, disturbance_force = self.start_disturbance_timer(disturbances[i], self.set_disturbance_flag)

            if impulses[i] is not None:
                t2, impulse = self.start_disturbance_timer(impulses[i], self.set_impulse_flag)

            while not done:
                steps += 1
                if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                    a = controller.act(env, s)
                else:
                    a = [0, 0, 0] # override controls
                a = np.array(a)

                if a.shape == (len_actions,):
                    # a is (1 by len_actions)
                    s, r, done, info = env.step(a)
                else:
                    # a is an array != (1 by len_actions)
                    # E.g. used in Model Predictive Control (MPC)
                    if self.controller_settings.get('control_horizon'):
                        for j in range(self.controller_settings.get('control_horizon')):
                            s, r, done, info = env.step(a[j])
                            state_history[i].append(s) # state history not saved in this case
                            untransformed_state_history[i].append(env.untransformed_state)
                            # rendering not done either
                        del state_history[i][-1]
                    else:
                        s, r, done, info = env.step(a[0])


                total_reward += r

                # Force is continuous
                if self.DISTURBANCE_FLAG:
                    env.apply_disturbance(disturbance_force)

                # Impulse is only done once
                if self.IMPULSE_FLAG:
                    env.apply_disturbance(impulse)
                    self.IMPULSE_FLAG = False

                if self.simulation_settings['Render']:
                    self.render(env, controller)

                state_history[i].append(s)

                if steps > max_steps:
                    done = True # Override

                if done:
                    self.render(env, controller)
                    if disturbances[i] is not None:
                        if t1.is_alive(): t1.cancel()  # Cancel disturbance if the time surpasses the simulation

                    if impulses[i] is not None:
                        if t2.is_alive(): t2.cancel()  # Cancel disturbance if the time surpasses the simulation

                    self.reset()
                    steps = 0

                    final_state_cost, total_fuel_consumed = self.get_final_cost_and_fuel(env, s)
                    action_history[i] = env.get_action_history()

                    landed = self.landing_check(s)

                    # print only the last
                    reward_results[i].append([total_reward, final_state_cost, total_fuel_consumed, landed])
                    average_theta = np.average(np.matrix(state_history[i])[:,4]/DEGTORAD)

                    Fe, Fl, Fr, psi_percentage = calculate_stats_percentages(action_history[i])
                    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}".format(i+1,
                                                                                    total_reward,
                                                                                    final_state_cost,
                                                                                    total_fuel_consumed,
                                                                                    average_theta,
                                                                                    landed,
                                                                                    Fe, Fl, Fr, psi_percentage))
                    total_reward = 0



        return reward_results, state_history, action_history

''' Helper Functions '''
def load_numpy_files(file_path):
    file = np.load(file_path)
    return file

def print_action_percentages(action_history):
    """
    Helper function for saved actions.
    :param action_history: 2D list containing all actions taken for that eoisode
    :return:
    """
    # action_history = load_numpy_files(file_path)
    print("Fe\tFl\tFr\t|psi| > 3")
    for i in range(len(action_history)):
        Fe, Fl, Fr, psi_percentage = calculate_stats_percentages(action_history[i])
        print("{0}\t{1}\t{2}\t{3}".format(Fe, Fl, Fr, psi_percentage))

def calculate_stats_percentages(action_history, psi_degree_threshold=3):
    """
    Calculates the percentage of time that inputs were active throughout the episode.
    :param action_history: list of actions. This is converted into a numpy matrix.
    :param psi_degree_threshold: Angle threshold for the nozzle angle.
    :return: list of percentages in this order: [Fe, Fsleft, Fsright, psi]
    """
    action_history = np.matrix(action_history)
    number_of_iterations = len(action_history)
    Fe_percentage = action_history[np.where(action_history[:, 0] > 0)].size / number_of_iterations
    Fs_left_percentage = action_history[np.where(action_history[:, 1] > 0)].size / number_of_iterations
    Fs_right_percentage = action_history[np.where(action_history[:, 1] < 0)].size / number_of_iterations
    psi_percentage = action_history[np.where((action_history[:, 2] < -psi_degree_threshold * DEGTORAD) | (
        action_history[:, 2] > psi_degree_threshold * DEGTORAD))].size / number_of_iterations

    return [Fe_percentage, Fs_left_percentage, Fs_right_percentage, psi_percentage]

''' Evaluation Functions '''
# PID
def evaluate_pid():
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//pid'

    env = RocketLander(simulation_settings)
    pid_controller = PID_Controller()
    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, pid_controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

# Low Discretization Q-Learning
def evaluate_low_discretization_q_learning_function_approximation():
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//rl_q_learning//low_discretization'

    load_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//function_approximation_rl' \
                '//rl_linear_function_approximation_low_function_discretization_trained_at_once.p'

    env = RocketLander(simulation_settings)
    controller = Q_Learning_Controller(load_path=load_path, low_discretization=True,
                                       simulation_settings=simulation_settings)
    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

# High Discretization Q-Learning
def evaluate_high_discretization_q_learning_function_approximation():
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//rl_q_learning//high_discretization'

    load_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//function_approximation_rl' \
                '//rl_linear_function_approximation_high_function_discretization_trained_at_once.p'

    env = RocketLander(simulation_settings)
    controller = Q_Learning_Controller(load_path=load_path, low_discretization=False,
                                       simulation_settings=simulation_settings)
    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

#rl_linear_function_approximation_high_function_discretization_5000_episodes_trained_at_once
# High Discretization Q-Learning
def evaluate_high_discretization_q_learning_function_approximation_longer_state():
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//rl_q_learning//high_discretization_longer_state'

    load_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//function_approximation_rl' \
                '//rl_linear_function_approximation_high_function_discretization_5000_episodes_trained_at_once.p'

    env = RocketLander(simulation_settings)
    controller = Q_Learning_Controller_Longer_State(load_path=load_path, low_discretization=False,
                                       simulation_settings=simulation_settings)
    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

# Model 1 - Normalized state = [x,z,x_dot,z_dot,theta,theta_dot]
def evaluate_normalized_normal_state_ddpg():
    # Model 1
    # Fuel Cost = 0, Max Steps = 500, Episode Training = 2000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE
    load_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//DDPG//model_normal_state'
    
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//DDPG//model_normal_state'

    normal_state_FLAG = True
    untransformed_state_FLAG = False
    normalized_FLAG = True

    simulation_settings['Observation Space Size'] = 8
    env = RocketLander(simulation_settings)
    controller = Normalized_DDPG_Controller(env=env, load_path=load_path, normal_state_FLAG=normal_state_FLAG,
                                              untransformed_state_FLAG=untransformed_state_FLAG,
                                              normalized_FLAG=normalized_FLAG)#, simulation_settings=simulation_settings)
    
    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

# Model 2 - Unormalized state = [x,z,x_dot,z_dot,theta,theta_dot,fuel,mass,barge left edge coordinates,barge right edge coordinates, landing coordinates]
def evaluate_unnormalized_longer_state_ddpg():
    load_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//DDPG//model_2_longer_unnormalized_state'

    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//ddpg//model_2_unnormalized_longer_state'

    normal_state_FLAG = False
    untransformed_state_FLAG = False
    normalized_FLAG = False

    simulation_settings['Observation Space Size'] = 16
    env = RocketLander(simulation_settings)
    controller = Unnormalized_DDPG_Controller_Longer_State(env=env, load_path=load_path, normal_state_FLAG=normal_state_FLAG,
                                              untransformed_state_FLAG=untransformed_state_FLAG,
                                              normalized_FLAG=normalized_FLAG)  # , simulation_settings=simulation_settings)

    testing_framework = Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

# MPC
def evaluate_MPC():
    file_path = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//mpc'

    k, p = 100, 10

    Q_weights = np.array([0.25, 4, 0.25, 0.25, 365, 131])
    Q = np.eye(6)*Q_weights*k

    R_weights = np.array([0.1, 0.1, 10])
    R = np.eye(3)*R_weights*p

    mpc_settings = {'time_horizon':30,
                    'control_horizon': 5,
                    'time_step': 1 / 30,
                    'time_span': 30,
                    'flight_time_span': 50,
                    'ground_altitude': 5.2,
                    'max_altitude': 26,
                    'x_target': 16.5,
                    'finite_differences_step_size': 50,
                    'Optimisation Type': 1,
                    'Q': Q,
                    'R': R}

    simulation_settings['Observation Space Size'] = 6
    env = RocketLander(simulation_settings)
    controller = MPC_Controller(env, mpc_settings=mpc_settings)

    testing_framework = Evaluation_Framework(simulation_settings, mpc_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, controller,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

    return file_path, reward_results, final_state_history, action_history

if __name__ == "__main__":
    simulation_settings = {'Side Engines': True,
                           'Clouds': True,
                           'Vectorized Nozzle': True,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': (0, 0),
                           'Render': True,
                           'Evaluation Averaging Loops': 3,
                           'Gather Stats': True,
                           'Episodes': 50}

    file_path, reward_results, final_state_history, action_history = evaluate_pid()

    # np.savez(file_path +'//evaluation_results', reward_results, final_state_history, action_history)
    # np.save(file_path + '//reward_results', reward_results)
    # np.save(file_path + '//final_state_history', final_state_history)
    # np.save(file_path + '//action_history', action_history)