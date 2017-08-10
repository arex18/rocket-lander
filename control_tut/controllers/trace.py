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

from . import trajectory

import numpy as np

class Shell(trajectory.Shell):
    """
    A controller that uses a given trajectory to 
    control a robotic arm end-effector.
    """

    def __init__(self, **kwargs):
        """
        """
        self.time = 0.0
        super(Shell, self).__init__(**kwargs)

    def check_pen_up(self):
        """Check to see if the pen should be lifted.
        """
        if self.time >= 1. - self.tau: 
            self.time = 0.0
            return True
        else: 
            return False

    def gen_path(self, trajectory):
        """Generates the trajectories for the 
        position, velocity, and acceleration to follow
        during run time to reproduce the given trajectory.

        trajectory np.array: a list of points to follow
        """

        if trajectory.ndim == 1: 
            trajectory = trajectory.reshape(1,len(trajectory))
        dt = 1.0 / trajectory.shape[1]

        # break up the trajectory into its different words
        # NaN or None signals a new word / break in drawing
        breaks = np.where(trajectory != trajectory)
        # some vector manipulation to get what we want
        breaks = breaks[1][:len(breaks[1])/2]
        self.num_seqs = len(breaks) - 1
       
        import scipy.interpolate
        self.seqs_x = [] 
        self.seqs_y = [] 
        for ii in range(self.num_seqs):
            # get the ii'th sequence
            seq_x = trajectory[0, breaks[ii]+1:breaks[ii+1]]
            seq_y = trajectory[1, breaks[ii]+1:breaks[ii+1]]
            
            # generate function to interpolate the desired trajectory
            vals = np.linspace(0, 1, len(seq_x))
            self.seqs_x.append(scipy.interpolate.interp1d(vals, seq_x))
            self.seqs_y.append(scipy.interpolate.interp1d(vals, seq_y))

        self.trajectory = [self.seqs_x[0], self.seqs_y[0]]

    def set_next_seq(self):
        """Get the next sequence in the list.
        """
        self.trajectory = [self.seqs_x[self.num_seq], 
                              self.seqs_y[self.num_seq]]

    def set_target(self):
        """Get the next target in the sequence.
        """
        self.controller.target = np.array([self.trajectory[d](self.time) \
                                                for d in range(2)])
        if self.time < 1.:
            self.time += self.tau
