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

from pydmps import dmp_discrete as DMP_discrete
from pydmps import dmp_rhythmic as DMP_rhythmic
from . import trajectory

import numpy as np


class Shell(trajectory.Shell):
    """
    A shell that uses dynamic movement primitives to
    control a robotic arm end-effector.
    """

    def __init__(self, bfs, add_to_goals=None,
                 pattern='discrete', **kwargs):
        """
        bfs int: the number of basis functions per DMP
        add_to_goals np.array: floats to add to the DMP goals
                               used to scale the DMPs spatially
        pattern string: specifies either 'discrete' or 'rhythmic' DMPs
        """

        self.bfs = bfs
        self.add_to_goals = add_to_goals
        self.pattern = pattern

        super(Shell, self).__init__(**kwargs)

        if add_to_goals is not None:
            for ii, dmp in enumerate(self.dmp_sets):
                dmp.goal[0] += add_to_goals[ii*2]
                dmp.goal[1] += add_to_goals[ii*2+1]

    def check_pen_up(self):
        """Check to see if the pen should be lifted.
        """
        if (self.dmps.cs.x <
                np.exp(-self.dmps.cs.ax * self.dmps.cs.run_time)):
            return True
        else:
            return False

    def gen_path(self, trajectory):
        """Generate the DMPs necessary to follow the
        specified trajectory.

        trajectory np.array: the time series of points to follow
                             [DOFs, time], with a column of None
                             wherever the pen should be lifted
        """

        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, len(trajectory))

        num_DOF = trajectory.shape[0]
        # break up the trajectory into its different words
        # NaN or None signals a new word / break in drawing
        breaks = np.array(np.where(trajectory[0] != trajectory[0]))[0]
        self.num_seqs = len(breaks) - 1

        self.dmp_sets = []
        for ii in range(self.num_seqs):
            # get the ii'th sequence
            seq = trajectory[:, breaks[ii]+1:breaks[ii+1]]

            if self.pattern == 'discrete':
                dmps = DMP_discrete.DMPs_discrete(n_dmps=num_DOF, n_bfs=self.bfs)
            elif self.pattern == 'rhythmic':
                dmps = DMP_rhythmic.DMPs_rhythmic(n_dmps=num_DOF, n_bfs=self.bfs)
            else:
                raise Exception('Invalid pattern type specified. Valid choices \
                                 are discrete or rhythmic.')

            dmps.imitate_path(y_des=seq)
            self.dmp_sets.append(dmps)

        self.dmps = self.dmp_sets[0]

    def set_next_seq(self):
        """Get the next sequence in the list.
        """
        self.dmps = self.dmp_sets[self.num_seq]

    def set_target(self):
        """Get the next target in the sequence.
        """
        error = 0.0
        if self.controller.target is not None:
            error = np.sqrt(np.sum((self.x -
                                    self.controller.target)**2)) * 1000
        self.controller.target, _, _ = self.dmps.step(tau=self.tau,
                                                      error=error)
