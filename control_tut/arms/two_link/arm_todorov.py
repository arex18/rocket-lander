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

from arm2base import Arm2Base
import numpy as np

class Arm(Arm2Base):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    This implementation is based off of the model described in 
    'Optimal control for nonlinear biological movement systems' 
    by Weiwei Li.
    """
    def __init__(self, **kwargs): 
        
        Arm2Base.__init__(self, **kwargs)

        self.reset() # set to init_q and init_dq

    def apply_torque(self, u, dt=None):
        if dt is None: 
            dt = self.dt

        tau = 0.04     # actuator time constant (sec)

        # arm model parameters
        m = np.array([1.4, 1.1]) # segment mass
        l = np.array([0.3, 0.33]) # segment length
        s = np.array([0.11, 0.16]) # segment center of mass
        i = np.array([0.025, 0.045]) # segment moment of inertia

        # inertia matrix
        a1 = i[0] + i[1] + m[1]*l[0]**2
        a2 = m[1]*l[0]*s[1]
        a3 = i[1]
        I = np.array([[a1 + 2*a2*np.cos(q[1]), a3 + a2*np.cos(q[1])],
                      [a3 + a2*np.cos(q[1]), a3]])

        # centripital and Coriolis effects
        C = np.array([[-dq[1] * (2 * dq[0] + dq[1])],
                      [dq[0]]]) * a2 * np.sin(q[1])

        # joint friction
        B = np.array([[.05, .025],
                      [.025, .05]])

        # calculate forward dynamics
        ddq = np.linalg.pinv(I) * (u - C - np.dot(B, dq))

        # transfer to next time step 
        self.q += dt * self.dq
        self.dq += dt * ddq

        self.t += dt
