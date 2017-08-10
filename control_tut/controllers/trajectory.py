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

from . import gc
from . import osc
from . import shell

import numpy as np

class Shell(shell.Shell):
    """
    """

    def __init__(self, gain, tau, trajectory, threshold=.01, **kwargs):
        """
        control Control instance: the controller to use
        trajectory np.array: the time series of points to follow
                             [DOFs, time], with a column of None
                             wherever the pen should be lifted
        tau float: the time scaling term
        threshold float: how close the system must get to initial target
        """

        super(Shell, self).__init__(**kwargs)

        self.done = False
        self.gain = gain
        self.not_at_start = True
        self.num_seq = 0
        self.tau = tau
        self.threshold = threshold
        self.x = None

        self.gen_path(trajectory)
        self.set_target()

    def check_pen_up(self):
        """Check to see if the pen should be lifted.
        """
        raise NotImplementedError

    def control(self, arm):
        """Apply a given control signal in (x,y)
           space to the arm"""

        self.x = np.copy(arm.x)

        if self.controller.check_distance(arm) < self.threshold:
            self.not_at_start = False

        if self.not_at_start or self.done:
            self.u = self.controller.control(arm)

        else:
            self.set_target()

            # check to see if it's pen up time
            if self.check_pen_up():
                self.pen_down = False

                if self.num_seq >= self.num_seqs - 1:
                    # if we're finished the last DMP
                    self.done = True
                    import sys; sys.exit()
                else:
                    # else move on to the next DMP
                    self.not_at_start = True
                    self.num_seq += 1
                    self.set_next_seq()
                    self.set_target()
            else:
                self.pen_down = True

            if isinstance(self.controller, osc.Control):
                pos = arm.x
            elif isinstance(self.controller, gc.Control):
                pos = arm.q

            pos_des = self.gain * (self.controller.target - pos)
            self.u = self.controller.control(arm, pos_des)

        return self.u

    def gen_path(self, trajectory):
        """Generate the sequences to follow.
        """
        raise NotImplementedError

    def set_next_seq(self):
        """Get the next sequence in the list.
        """
        raise NotImplementedError

    def set_target(self):
        """Get the next target in the sequence.
        """
        raise NotImplementedError
