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

from ..Arm import Arm
import numpy as np

class Arm1Link(Arm):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, dt=1e-2, l1=.31, **kwargs): 
        self.DOF = 1
        Arm.__init__(self, **kwargs)

        self.dt = dt # timestep 
        
        # length of arm links
        self.l1=l1
        self.L = np.array([self.l1])
        # mass of links
        self.m1=1.98
        # z axis inertia moment of links
        izz1=15
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*self.m1

        # initial arm joint and end-effector position
        self.q = np.array([0.0]) # matching the C simulation
        # initial arm joint and end-effector velocity
        self.dq = np.zeros(1)
        # initial arm joint and end-effector acceleration
        self.ddq = np.zeros(1)

        self.t = 0.0

    def apply_torque(self, u, dt=None):
        if dt is None: 
            dt = self.dt

        # equations solved for angles
        self.ddq = 1.0 / (self.m1 * self.l1**2) * u
        self.dq += self.ddq * self.dt
        self.q += self.dq * self.dt

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
        if q is None:
            q = self.q
        q0 = q[0]   

        JCOM1 = np.zeros((6,1))
        JCOM1[0,0] = self.l1 * -np.sin(q0) 
        JCOM1[1,0] = self.l1 * np.cos(q0) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from end-effector to
           the origin frame"""
        if q is None:
            q = self.q
        q0 = q[0]   

        JEE = np.zeros((2,1))
        # define column entries right to left
        JEE[0,0] = self.l1 * -np.sin(q0) 
        JEE[1,0] = self.l1 * np.cos(q0)
        
        return JEE

    def gen_Mq(self, q=None):
        """Generates the mass matrix for the arm in joint space"""
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(q=q)
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1)) 
        
        return Mq

    def position(self, q=None):
        """Compute x,y position of the hand"""
        if q is None: q0 = self.q[0]
        else: q0 = q[0]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0)])

        return np.array([x, y])
