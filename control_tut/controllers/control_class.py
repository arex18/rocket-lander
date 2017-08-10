'''
Copyright (C) 2014 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np

class Control(object):
    """
    The base class for controllers.
    """
    def __init__(self, kp=10, kv=np.sqrt(10),
                    additions=[], task='', write_to_file=False):
        """
        additions list: list of Addition classes to append to
                        the outgoing control signal
        kp float: the position error term gain value
        kv float: the velocity error term gain value
        """

        self.u = np.zeros((2,1)) # control signal

        self.additions = additions
        self.kp = kp
        self.kv = kv
        self.task = task
        self.target = None

        self.write_to_file = write_to_file
        self.recorders = []

    def check_distance(self, arm):
        """Checks the distance to target"""
        return np.sum(abs(arm.x - self.target)) + np.sum(abs(arm.dq))

    def control(self):
        """Generates a control signal to apply to the arm"""
        raise NotImplementedError

