import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.finance
matplotlib.rcParams.update({'font.size':9})
from cycler import cycler
import itertools
import pandas as pd
import numpy as np


class RealTimeGraph:
    """
    Creates a figure that is able to handle low number of data points in "real time". It still uses Matplotlib,
    which makes it unscalable for real time plotting. See realtime_plot.py as a replacement.
    Known crashing error. Not meant to be deployed.
    """
    def __init__(self, subplots=111):
        self.ptr = 0
        self.lines = []
        self.ax = []

        self.ylimflag = False
        self.xlimflag = False

        self.fig = plt.figure()
        self.addsubplot(subplots)

        self.fig.canvas.draw()
        plt.ion()

    def addsubplot(self, subplotnumber):
        ax = self.fig.add_subplot(subplotnumber)

        line, = ax.plot([], linewidth=1.0)

        self.ax.append(ax)
        self.lines.append(line)
        self.ptr += 1

    def updateScatterPlot(self, x, y, subplot=0):
        self.ax[subplot].scatter(x, y, linewidths=0.5)

        # self.ax[subplot].set_ylim([np.min(y), np.max(y)])
        # self.ax[subplot].set_xlim([np.min(x), np.max(x)])

        plt.pause(1e-30)

    def setYLimits(self, y1, y2, subplot=0):
        self.ylimflag = True
        self.ax[subplot].set_ylim([y1, y2])

    def setXLimits(self, x1, x2, subplot=0):
        self.xlimflag = True
        self.ax[subplot].set_xlim([x1, x2])

    def updateLinePlot(self, x, y, subplot=0):
        self.lines[subplot].set_xdata(np.array(x))
        self.lines[subplot].set_ydata(np.array(y))

        if not self.ylimflag:
            self.ax[subplot].set_ylim([np.min(y), np.max(y)])
        if not self.xlimflag:
            self.ax[subplot].set_xlim([np.min(x), np.max(x)])

        self.fig.canvas.blit(self.ax[subplot].bbox)                             # fill in the self.axes rectangle
        plt.pause(1e-30)

class Graphing:
    """
    Graphing helper class developed on Matplotlib.

    Example Use:

    res = Graphing.Results(create_figure=True)
    ax = res.add_subplot(0, 211, "Epoch", "Error")
    ax2 = res.add_subplot(0, 212, "Epoch", "Accuracy")

    res.plot_graph(training_epochvalues, trainingerror, ax, "Train. Error"+labeltext)

    res.show_legend(ax)
    res.show_legend(ax2)
    plt.show()

    """
    figurenumber = 0
    figures = []

    def __init__(self, plot_colors=None, fig_size=(9, 9)):
        """
        Parameters are automatically set for consistency, but can be overwritten from pylab.
        :param plot_colors: What colors should the plots have in order?
        :param fig_size: Tuple of 2 integer values, e.g. (4, 6). If none, colors will be automatically allocated.
        """
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': fig_size,
                  'axes.labelsize': 'x-large',
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 'x-large',
                  'ytick.labelsize': 'x-large',
                  'font.family': 'Times New Roman',
                  'font.size': 10}

        if plot_colors is None:
            self.plot_colors = ['dodgerblue', 'orange', 'lime','b', 'g', 'r', 'c', 'm', 'k', 'firebrick', 'darksalmon', 'darkblue', 'dodgerblue',
                   'k', 'silver', 'darkorchid', 'plum']
        else:
            self.plot_colors = plot_colors

        pylab.rcParams.update(params)
        self.scattercolors = itertools.cycle(self.plot_colors)

    def create_figure(self, *args, **kwargs):
        """
        This should be the starting point after instantiation. It creates a new figure and appends it to a list
        of figures in the class.
        :param args:
        :param kwargs:
        :return: figure
        """
        newFig = plt.figure(self.figurenumber, *args, **kwargs)
        self.figures.append(newFig)
        self.figurenumber += 1
        return newFig

    def create_figure_and_subplots(self, new_figure=True, y_labels=None, x_labels=None, row_number=1, column_number=1, *args, **kwargs):
        """
        This is simply an extension of add_subplot and create_figure
        :param new_figure:
        :param y_labels:
        :param x_labels:
        :param row_number:
        :param column_number:
        :param args:
        :param kwargs:
        :return:
        """
        axis = []

        if x_labels is None:
            x_labels = ['' for _ in range(row_number*column_number)]
        if y_labels is None:
            y_labels = ['' for _ in range(row_number*column_number)]

        if isinstance(new_figure, bool):
            fig = self.create_figure(*args, **kwargs)
        else:
            fig = new_figure

        for i in range(row_number*column_number):
            subplot_number = int(str(row_number)+str(column_number)+str(i+1))
            ax = self.add_subplot(fig, subplot_number, x_labels[i], y_labels[i])
            axis.append(ax)

        return fig, axis

    def add_subplot(self, figure, subplotnumber=111, xtitle='', ytitle='', grid=True, *args, **kwargs):
        ax = figure.add_subplot(subplotnumber, *args, **kwargs)
        ax.set_xlabel(xtitle, **kwargs)
        ax.set_ylabel(ytitle, **kwargs)
        ax.set_prop_cycle(cycler('color', self.plot_colors) + cycler('lw', [2 for _ in self.plot_colors]))
        if grid:
            plt.grid()
        return ax

    def plot_3D_graph(self, x, y, z, subplot, labeltext='', marker='^', markersize=8, alpha=0.5):
        subplot.plot(x,y,z,marker,markersize=markersize,alpha=alpha,label=labeltext, color=next(self.scattercolors))

    def plot_graph(self, x, y, subplot, labeltext='', hold=True, plottype='plot', **kwargs):
        lines = None
        if plottype is 'plot':
            if x is None:
                lines = subplot.plot(y, label=labeltext, **kwargs)
            else:
                lines = subplot.plot(x, y, label=labeltext, **kwargs)
        elif plottype is 'hist':
            lines = self.plot_hist(x, subplot, y, labeltext, **kwargs)
        elif plottype is 'scatter':
            lines = subplot.scatter(x, y, label=labeltext, color=next(self.scattercolors), **kwargs)
        return lines


    @staticmethod
    def add_legend(name_of_data, *args, **kwargs):
        plt.legend(name_of_data, *args, **kwargs)

    @staticmethod
    def show_legend(subplot):
        if isinstance(subplot,list):
            for s in subplot:
                s.legend(loc=0)
        else:
            subplot.legend(loc=0)

    @staticmethod
    def add_title(titlename):
        plt.title(titlename)

    @staticmethod
    def show_plot():
        plt.show()

    @staticmethod
    def plot_hist(y, subplot, bins, labeltext, **kwargs):
        subplot.hist(y, bins, label=labeltext, **kwargs)

    @staticmethod
    def plot_error_bar(x, y, subplot, labeltext, hold=True):
        subplot.errorbar(x, y, label=labeltext)

    @staticmethod
    def plot_hist_and_density_plot(y):
        y = pd.DataFrame(y)
        y.hist()
        y.plot(kind='kde')

def create_realtime_graphs(initialSubplot=111, x1=0, x2=0, y1=0, y2=0):
    """
    Creates a real time graph with a specified number of subplots.
    :param initialSubplot: Matplotlib style subplot convention, e.g. 211
    :param x1: First x-axis limit
    :param x2: Second x-axis limit
    :param y1: First y-axis limit
    :param y2: Second y-axis limit
    :return:
    """
    graph = RealTimeGraph(initialSubplot)
    if x2 != 0:
        graph.setXLimits(x1, x2)
    if y2 != 0:
        graph.setYLimits(y1, y2)

    return graph