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

from ..ArmBase import ArmBase
import numpy as np

class Arm2Base(ArmBase):
    """A wrapper around a MapleSim generated C simulation
    of a two link arm."""

    def __init__(self, init_q=[.75613, 1.8553], init_dq=[0.,0.],
                    l1=.31, l2=.27, **kwargs):

        self.DOF = 2
        ArmBase.__init__(self, init_q=init_q, init_dq=init_dq,
                         singularity_thresh=.00025, **kwargs)

        # length of arm links
        self.l1 = l1
        self.l2 = l2
        self.L = np.array([self.l1, self.l2])
        # mass of links
        self.m1=1.98
        self.m2=1.32
        # z axis inertia moment of links
        izz1=15.; izz2=8.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M2 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*self.m1
        self.M1[3:,3:] = np.eye(3)*izz1
        self.M2[0:3,0:3] = np.eye(3)*self.m2
        self.M2[3:,3:] = np.eye(3)*izz2

        self.rest_angles = np.array([np.pi/4.0, np.pi/4.0])

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q

        JCOM1 = np.zeros((6,2))
        JCOM1[0,0] = self.l1 / 2. * -np.sin(q[0])
        JCOM1[1,0] = self.l1 / 2. * np.cos(q[0])
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self, q=None):
        """Generates the Jacobian from the COM of the second
        link to the origin frame"""
        q = self.q if q is None else q

        JCOM2 = np.zeros((6,2))
        # define column entries right to left
        JCOM2[0,1] = self.l2 / 2. * -np.sin(q[0]+q[1])
        JCOM2[1,1] = self.l2 / 2. * np.cos(q[0]+q[1])
        JCOM2[5,1] = 1.0

        JCOM2[0,0] = self.l1 * -np.sin(q[0]) + JCOM2[0,1]
        JCOM2[1,0] = self.l1 * np.cos(q[0]) + JCOM2[1,1]
        JCOM2[5,0] = 1.0

        return JCOM2

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from end-effector to
        the origin frame"""
        q = self.q if q is None else q

        JEE = np.zeros((2,2))
        # define column entries right to left
        JEE[0,1] = self.l2 * -np.sin(q[0]+q[1])
        JEE[1,1] = self.l2 * np.cos(q[0]+q[1])

        JEE[0,0] = self.l1 * -np.sin(q[0]) + JEE[0,1]
        JEE[1,0] = self.l1 * np.cos(q[0]) + JEE[1,1]

        return JEE

    def gen_Mq(self, q=None):
        """Generates the mass matrix for the arm in joint space"""
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(self.M2, JCOM2))

        return Mq

    def inv_kinematics(self, xy):
        """Calculate the joint angles for a given (x,y)
        hand position"""
        import scipy.optimize
        # function to optimize
        def distance_to_target(q, xy, L):
            x = L[0] * np.cos(q[0]) + L[1] * np.cos(q[0] + q[1])
            y = L[0] * np.sin(q[0]) + L[1] * np.sin(q[0] + q[1])
            return np.sqrt((x - xy[0])**2 + (y - xy[1])**2)

        return scipy.optimize.minimize(fun=distance_to_target, x0=self.q,
                args=([xy[0], xy[1]], self.L))['x']

    def position(self, q=None):
        """Compute x,y position of the hand

        q np.array: a set of angles to return positions for
        rotate float: how much to rotate the first joint by
        """
        q = self.q if q is None else q

        x = np.cumsum([0,
                       self.l1 * np.cos(q[0]),
                       self.l2 * np.cos(q[0]+q[1])])
        y = np.cumsum([0,
                       self.l1 * np.sin(q[0]),
                       self.l2 * np.sin(q[0]+q[1])])
        return np.array([x, y])
