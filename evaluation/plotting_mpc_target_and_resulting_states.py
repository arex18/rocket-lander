from evaluation.plotting_trajectory import *


def plot_single_trajectories(res):
    # Low Disc
    x_planned = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//'
                            'mpc//x_planned.npy')
    y_planned = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//'
                        'mpc//y_planned.npy')

    x_target = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//'
                        'mpc//x_target.npy')
    y_target = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//'
                       'mpc//y_target.npy')

    resulting_states = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//'
                       'mpc//resulting_states.npy')


    fig = res.createFig()
    ax = res.addsubplot(fig, 111, "X-Position/metres", "Z-Altitude/metres", grid=False)

    res.plotGraph(x_target, y_target, ax)
    res.plotGraph(x_target, y_target, ax, plottype='scatter')

    res.plotGraph(x_planned, y_planned, ax)
    res.plotGraph(x_planned, y_planned, ax, plottype='scatter')

    res.plotGraph(resulting_states[:-1,0], resulting_states[:-1,1], ax)
    res.plotGraph(resulting_states[:-1, 0], resulting_states[:-1, 1], ax, plottype='scatter')

    res.addTitle('X-Z Target, Planned and Actual Trajectories for a Single Optimization Iteration')
    res.add_legend(['Target', 'Planned', 'Actual'])
    res.showPlot()

res = Graphing(plot_colors=None, fig_size=(8, 5))
plot_single_trajectories(res)