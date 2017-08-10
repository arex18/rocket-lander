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

class Signal(object):
    """
    The base class for classes generating signals to be 
    used with to the outgoing control signal.
    """
    def __init__(self):
        """
        """

    def generate(self, u, arm):
        """Generate the signal to add to the control signal.
    
        u np.array: the outgoing control signal
        arm Arm: the arm currently being controlled
        """
        raise NotImplementedError
