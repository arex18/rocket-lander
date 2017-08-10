'''
Copyright (C) 2013 Travis DeWolf

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
import importlib

from ..ArmBase import ArmBase

import numpy as np


class Arm(ArmBase):
    """A wrapper around a MapleSim generated C simulation
    of a three link arm."""

    def __init__(self, init_q=[np.pi/5.5, np.pi/1.7, np.pi/6.],
                 init_dq=[0., 0., 0.], **kwargs):

        self.DOF = 3
        ArmBase.__init__(self, init_q=init_q, init_dq=init_dq,
                         **kwargs)

        # build the arm you would like to use by editing
        # the setup file to import the desired model and running
        # python setup.py build_ext -i
        # name the resulting .so file to match and go
        arm_import_name = 'arms.three_link.py3LinkArm'
        arm_import_name = (arm_import_name if self.options is None
                           else '_' + self.options)
        print(arm_import_name)
        pyArm = importlib.import_module(name=arm_import_name)

        # length of arm links
        l1 = 2.0
        l2 = 1.2
        l3 = .7
        self.L = np.array([l1, l2, l3])
        # mass of links
        m1 = 10
        m2 = m1
        m3 = m1
        # inertia moment of links
        izz1 = 100
        izz2 = 100
        izz3 = 100
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6, 6))
        self.M2 = np.zeros((6, 6))
        self.M3 = np.zeros((6, 6))
        self.M1[0:3, 0:3] = np.eye(3)*m1
        self.M1[5, 5] = izz1
        self.M2[0:3, 0:3] = np.eye(3)*m2
        self.M2[5, 5] = izz2
        self.M3[0:3, 0:3] = np.eye(3)*m3
        self.M3[5, 5] = izz3
        if self.options == 'smallmass':
            self.M1 *= .001
            self.M2 *= .001
            self.M3 *= .001

        self.rest_angles = np.array([np.pi/4.0, np.pi/4.0, np.pi/4.0])

        # stores information returned from maplesim
        self.state = np.zeros(7)
        # maplesim arm simulation
        self.sim = pyArm.pySim(dt=1e-5)
        self.sim.reset(self.state)
        self.reset()
        self.update_state()

    def apply_torque(self, u, dt=None):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly.

        u np.array: the control signal to apply
        dt float: the timestep
        """
        if dt is None:
            dt = self.dt

        u = -1 * np.array(u, dtype='float')

        for ii in range(int(np.ceil(dt/1e-5))):
            self.sim.step(self.state, u)
        self.update_state()

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q

        JCOM1 = np.zeros((6, 3))
        JCOM1[0, 0] = self.L[0] / 2. * -np.sin(q[0])
        JCOM1[1, 0] = self.L[0] / 2. * np.cos(q[0])
        JCOM1[5, 0] = 1.0

        return JCOM1

    def gen_jacCOM2(self, q=None):
        """Generates the Jacobian from the COM of the second
        link to the origin frame"""
        q = self.q if q is None else q
        q0 = q[0]
        q01 = q[0] + q[1]

        JCOM2 = np.zeros((6, 3))
        # define column entries right to left
        JCOM2[0, 1] = self.L[1] / 2. * -np.sin(q01)
        JCOM2[1, 1] = self.L[1] / 2. * np.cos(q01)
        JCOM2[5, 1] = 1.0

        JCOM2[0, 0] = self.L[0] * -np.sin(q0) + JCOM2[0, 1]
        JCOM2[1, 0] = self.L[0] * np.cos(q0) + JCOM2[1, 1]
        JCOM2[5, 0] = 1.0

        return JCOM2

    def gen_jacCOM3(self, q=None):
        """Generates the Jacobian from the COM of the third
        link to the origin frame"""
        q = self.q if q is None else q

        q0 = q[0]
        q01 = q[0] + q[1]
        q012 = q[0] + q[1] + q[2]

        JCOM3 = np.zeros((6, 3))
        # define column entries right to left
        JCOM3[0, 2] = self.L[2] / 2. * -np.sin(q012)
        JCOM3[1, 2] = self.L[2] / 2. * np.cos(q012)
        JCOM3[5, 2] = 1.0

        JCOM3[0, 1] = self.L[1] * -np.sin(q01) + JCOM3[0, 2]
        JCOM3[1, 1] = self.L[1] * np.cos(q01) + JCOM3[1, 2]
        JCOM3[5, 1] = 1.0

        JCOM3[0, 0] = self.L[0] * -np.sin(q0) + JCOM3[0, 1]
        JCOM3[1, 0] = self.L[0] * np.cos(q0) + JCOM3[1, 1]
        JCOM3[5, 0] = 1.0

        return JCOM3

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from end-effector to
        the origin frame"""
        q = self.q if q is None else q

        q0 = q[0]
        q01 = q[0] + q[1]
        q012 = q[0] + q[1] + q[2]

        JEE = np.zeros((2, 3))

        l3 = self.L[2]
        l2 = self.L[1]
        l1 = self.L[0]

        # define column entries right to left
        JEE[0, 2] = l3 * -np.sin(q012)
        JEE[1, 2] = l3 * np.cos(q012)

        JEE[0, 1] = l2 * -np.sin(q01) + JEE[0, 2]
        JEE[1, 1] = l2 * np.cos(q01) + JEE[1, 2]

        JEE[0, 0] = l1 * -np.sin(q0) + JEE[0, 1]
        JEE[1, 0] = l1 * np.cos(q0) + JEE[1, 1]

        return JEE

    def gen_Mq(self, q=None):
        """Generates the mass matrix of the arm in joint space"""

        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        JCOM3 = self.gen_jacCOM3(q=q)

        M1 = self.M1
        M2 = self.M2
        M3 = self.M3
        # generate the mass matrix in joint space
        Mq = (np.dot(JCOM1.T, np.dot(M1, JCOM1)) +
              np.dot(JCOM2.T, np.dot(M2, JCOM2)) +
              np.dot(JCOM3.T, np.dot(M3, JCOM3)))

        return Mq

    def inv_kinematics(self, xy):
        """Calculate the joint angles for a given (x,y)
        hand position, using minimal distance to resting
        joint angles to solve kinematic redundancies."""
        import scipy.optimize

        # function to optimize
        def distance_to_default(q, *args):
            # weights found with trial and error,
            # get some wrist bend, but not much
            weight = [1, 1, 1.3]
            return np.sqrt(np.sum([(qi - q0i)**2 * wi
                           for qi, q0i, wi in zip(q,
                                                  self.rest_angles,
                                                  weight)]))

        # constraint functions
        def x_constraint(q, xy):
            x = (self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) +
                 self.L[2]*np.cos(np.sum(q))) - xy[0]
            return x

        def y_constraint(q, xy):
            y = (self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) +
                 self.L[2]*np.sin(np.sum(q))) - xy[1]
            return y

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.rest_angles, eqcons=[x_constraint, y_constraint],
            args=((xy[0], xy[1]),), iprint=0)

    def position(self, q=None):
        """Compute x,y position of the hand

        q np.array: a set of angles to return positions for
        """
        if q is None:
            q0 = self.q[0]
            q1 = self.q[1]
            q2 = self.q[2]
        else:
            q0 = q[0]
            q1 = q[1]
            q2 = q[2]

        x = np.cumsum([0,
                       self.L[0] * np.cos(q0),
                       self.L[1] * np.cos(q0+q1),
                       self.L[2] * np.cos(q0+q1+q2)])
        y = np.cumsum([0,
                       self.L[0] * np.sin(q0),
                       self.L[1] * np.sin(q0+q1),
                       self.L[2] * np.sin(q0+q1+q2)])
        return np.array([x, y])

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
        """Update the local variables"""
        self.t = self.state[0]
        self.q = self.state[1:4]
        self.dq = self.state[4:]
