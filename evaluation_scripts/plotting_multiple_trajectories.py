"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: General scripts for graphing instead of notebook.
"""

from evaluation_scripts.plotting_trajectory import *

def plot_pid_and_ddpg_trajectories(res):
    # Low Disc
    # state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//rl_and_control//evaluation_scripts//'
    #                         'rl_q_learning//low_discretization//final_state_history.npy')
    # -------------------------------
    state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//'
                            'pid//final_state_history.npy')

    fig = res.create_figure()
    ax = res.add_subplot(fig, 111, "X-Position Displacement/metres", "Z-Altitude Displacement/metres", grid=False)

    # tests = [1, 7, 16, 22]
    tests = [1, 7, 16]
    for i in tests:
        convert_state_and_plot_trajectory_2(res, fig, ax, state_history[i])

    state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//'
                            'ddpg//model_2_unnormalized_longer_state//final_state_history.npy')

    for i in tests:
        convert_state_and_plot_trajectory_2(res, fig, ax, state_history[i])

    res.add_title('X-Z Trajectories for Multiple Tests - PID and DDPG')
    res.add_legend(np.append(['PID Test '+str(i+1) for i in tests], ['DDPG Test '+str(i+1) for i in tests]))
    res.show_plot()

def plot_ddpg1_and_ddpg2_trajectories(res):
    state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//'
                            'ddpg//model_1_normal_state//final_state_history.npy')

    fig = res.create_figure()
    ax = res.add_subplot(fig, 111, "X-Position Displacement/metres", "Z-Altitude Displacement/metres", grid=False)

    tests = [1, 7, 16, 22]
    for i in tests:
        convert_state_and_plot_trajectory_2(res, fig, ax, state_history[i])

    state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//'
                            'ddpg//model_2_unnormalized_longer_state//final_state_history.npy')

    for i in tests:
        convert_state_and_plot_trajectory_2(res, fig, ax, state_history[i])

    res.add_title('X-Z Trajectories for Multiple Tests - DDPG Models 1 and 2')
    res.add_legend(np.append(['M1 Test ' + str(i+1) for i in tests], ['M2 Test ' + str(i+1) for i in tests]))
    res.show_plot()

def plot_single_trajectories(res):
    # Low Disc
    state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//'
                            'rl_q_learning//low_discretization//final_state_history.npy')
    # -------------------------------
    # state_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//rl_and_control//evaluation_scripts//'
    #                         'pid//final_state_history.npy')

    fig = res.create_figure()
    ax = res.add_subplot(fig, 111, "X-Position Displacement/metres", "Z-Altitude Displacement/metres", grid=False)

    test_history = np.matrix(state_history)
    tests = [1, 2, 7, 8, 9, 12]
    for i in tests:
        convert_state_and_plot_trajectory_2(res, fig, ax, state_history[i])

    res.add_title('X-Z Trajectories for Multiple Tests - Low Action Discretization')
    res.add_legend(['Test ' + str(i+1) for i in tests])
    res.show_plot()

plot_colors = ['darkred', 'red', 'salmon', 'darkblue', 'blue', 'deepskyblue', 'cyan']
res = Graphing(plot_colors=plot_colors, fig_size=(8, 5))
plot_pid_and_ddpg_trajectories(res)