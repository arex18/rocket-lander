'''
Copyright (C) 2015 Travis DeWolf

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

from .arm2base import Arm2Base
from . import py2LinkArm

import numpy as np


class Arm(Arm2Base):
    """A wrapper around a MapleSim generated C simulation
    of a two link arm."""

    def __init__(self, **kwargs):

        Arm2Base.__init__(self, **kwargs)

        # stores information returned from maplesim
        self.state = np.zeros(7)
        # maplesim arm simulation
        self.sim = py2LinkArm.pySim(dt=1e-5)
        self.sim.reset(self.state)
        self.reset()  # set to init_q and init_dq

    def apply_torque(self, u, dt=None):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly.

        u np.array: the control signal to apply
        dt float: the timestep
        """
        if dt is None:
            dt = self.dt
        u = np.array(u, dtype='float')

        for i in range(int(np.ceil(dt/1e-5))):
            self.sim.step(self.state, u)
        self.update_state()

    def reset(self, q=[], dq=[]):
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if q:
            assert len(q) == self.DOF
        if dq:
            assert len(dq) == self.DOF

        state = np.zeros(self.DOF*2)
        state[::2] = self.init_q if not q else np.copy(q)
        state[1::2] = self.init_dq if not dq else np.copy(dq)

        self.sim.reset(self.state, state)
        self.update_state()

    def update_state(self):
        """Separate out the state variable into time, angles,
        velocities, and accelerations."""

        self.t = self.state[0]
        self.q = self.state[1:3]
        self.dq = self.state[3:5]
        self.ddq = self.state[5:]
