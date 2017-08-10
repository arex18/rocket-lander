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

from . import signal

import numpy as np

class Addition(signal.Signal):
    """This adds a forcefield as described in Shademehr 1994.
    """
    def __init__(self, scale=1.0):
        """
        """
        self.force_matrix = np.array([[-10.1, -11.1, -10.1],
                                      [-11.2, 11.1, 10.1],
                                      [-11.2, 11.1, -10.1]]) 
        self.force_matrix *= scale

    def generate(self, u, arm):
        """Generate the signal to add to the control signal.

        u np.array: the outgoing control signal
        arm Arm: the arm currently being controlled
        """
        # calculate force to add
        force = np.dot(self.force_matrix[:arm.DOF, :arm.DOF],
                       arm.dq)

        # translate to joint torques
        return force
