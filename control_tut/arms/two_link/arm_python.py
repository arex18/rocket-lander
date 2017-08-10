'''
Copyright (C) 2015 Travis DeWolf & Terry Stewart

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
import numpy as np


class Arm(Arm2Base):
    """
    A class that holds the simulation and control dynamics for
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, **kwargs):

        Arm2Base.__init__(self, **kwargs)

        # compute non changing constants
        self.K1 = ((1/3. * self.m1 + self.m2) * self.l1**2. +
                   1/3. * self.m2 * self.l2**2.)
        self.K2 = self.m2 * self.l1 * self.l2
        self.K3 = 1/3. * self.m2 * self.l2**2.
        self.K4 = 1/2. * self.m2 * self.l1 * self.l2

        self.reset()  # set to init_q and init_dq

    def apply_torque(self, u, dt=None):
        """Takes in a torque and time step and updates the
        arm simulation accordingly.

        u np.array: the control signal to apply
        dt float: the time step
        """
        if dt is None:
            dt = self.dt

        # equations solved for angles
        C2 = np.cos(self.q[1])
        S2 = np.sin(self.q[1])
        M11 = (self.K1 + self.K2*C2)
        M12 = (self.K3 + self.K4*C2)
        M21 = M12
        M22 = self.K3
        H1 = (-self.K2*S2*self.dq[0]*self.dq[1] -
              1/2.0*self.K2*S2*self.dq[1]**2.0)
        H2 = 1./2.*self.K2*S2*self.dq[0]**2.0

        ddq1 = ((H2*M11 - H1*M21 - M11*u[1] + M21*u[0]) /
                (M12**2. - M11*M22))
        ddq0 = (-H2 + u[1] - M22*ddq1) / M21
        self.dq += np.array([ddq0, ddq1]) * dt
        self.q += self.dq * dt

        # transfer to next time step
        self.t += dt
