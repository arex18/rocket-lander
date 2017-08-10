import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from pyqtgraph import mkPen
import threading
import time

## Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class RealTime_Graph_Thread(threading.Thread):
    """
    This class uses another thread on which to plot a real time graph in a scalable manner. Note that it is not
    implemented in a fully scalable way: updating of the data is being done externally, with the update being called
    periodically inside the thread.
    """
    def __init__(self, settings=None):
        super(RealTime_Graph_Thread, self).__init__()

        if settings is None: self.settings = {'Rows':1, 'Columns':1}
        else: self.settings = settings

        self.cancelled = False

        number_of_objects = self.settings['Rows']*self.settings['Columns']
        self.plots = []
        self.plotHandles = []
        self.data = [[] for _ in range(number_of_objects)]
        self.previous_data = [[] for _ in range(number_of_objects)]
        self.previousLenData = np.zeros(number_of_objects)
        self.win = None

    def _configurePlots(self):
        for i in range(self.settings['Rows']):
            if i > 0:
                self.win.nextRow()
            for _ in range(self.settings['Columns']):
                subplotGraph = self.win.addPlot()
                subplotGraph.setClipToView(True)
                subplotGraph.setDownsampling(mode='peak')
                self.plots.append(subplotGraph)
                self.plotHandles.append(subplotGraph.plot(pen=mkPen(width=2, color='k')))

    # Parameters defined explicitly due to repeated x&y, readability and ease of use. Can be integrated in **kwargs
    def _setPlotLabels(self, subplot=0, title="", x_label="", y_label="", x_units="", y_units="", color='k', width=2, **kwargs):
        # kwargs e.g. labelStyle = {'color': '#FFF', 'font-size': '14pt'}
        assert subplot >= 0, "Subplot must be a non-negative integral number."
        if self.plots is not None:
            self.plots[subplot].setTitle(title)
            self.plots[subplot].setLabel('bottom', x_label, units=x_units, **kwargs)
            self.plots[subplot].setLabel('left', y_label, units=y_units, **kwargs)
            self.plots[subplot].showGrid(x=True,y=True, alpha=0.3)
            self.plotHandles[subplot] = self.plots[subplot].plot(pen=mkPen(width=width, color=color))

            font = QtGui.QFont()
            font.setPixelSize(18)
            self.plots[subplot].getAxis("bottom").tickFont = font
            self.plots[subplot].getAxis("left").tickFont = font

    # API Stylet
    def setPlotLabels(self, **kwargs):
        self._setPlotLabels(**kwargs)

    def run(self):
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('Scrolling Plots')

        self._configurePlots()
        self.setPlotLabels(labelStyle={'color': '#FFF', 'font-size': '18pt', 'showValues': False})

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.__updateGraphs)
        timer.start(50)

        QtGui.QApplication.instance().exec_()

    def __updateGraphs(self):
        for i,data in enumerate(self.data):
            if (len(data) != self.previousLenData[i]):
                self.manual_updateGraph(i, data)
                self.previousLenData[i] = len(data)

    def manual_updateGraph(self, subplot, data):
        self.plotHandles[subplot].setData(data)


def printing():
    while (1):
        print("Hello")
        time.sleep(1)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    # Does not work this way
    q = threading.Thread(target=printing)
    q.start()
    data = []
    handles = RealTime_Graph_Thread({'Rows':1, 'Columns':2})
    handles.start()

    time.sleep(1)
    handles.setPlotLabels()
    # handles.setPlotLabels(title="hello", labelStyle={'color': '#FFF', 'font-size': '18pt', 'showValues': False})
    for i in range(1000):
        time.sleep(1)
        data.append(np.random.rand())
        handles.data[0] = data
        handles.data[1] = data