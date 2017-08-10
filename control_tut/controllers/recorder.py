import numpy as np

class Recorder():
    """
    A class for recording a signal and writing out at command, 
    clearing the cache after writing out to a compressed file.
    """
    def __init__(self, name, task, controller):
        """
        name string: the name of the file to write out to
        """

        self.name = name
        self.task = task
        self.controller = controller

        self.count = 0
        self.signal = []

        self.start_recording = False
        self.write_out = False

    def record(self, t, x):
        """ 
        The function to hook up to a node.

        t float: the time signal (ignored)
        x float, np.array: the signal to record
        """

        if self.start_recording == True:
            self.signal.append(x.copy())

        if self.write_out == True:
            print('saving')
            np.savez_compressed(
                    'data/%s/%s/%s%.3i'%(self.task, \
                            self.controller, self.name, self.count),
                        array1=np.asarray(self.signal))
            self.count += 1
            self.signal = []
            self.start_recording = False
            self.write_out = False
