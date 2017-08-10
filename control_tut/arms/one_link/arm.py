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

from ..ArmBase import ArmBase
import numpy as np
import py1LinkArm

class Arm(ArmBase):
    """A wrapper around a MapleSim generated C simulation 
    of a one link arm"""

    def __init__(self, JEE='x', **kwargs):
        """
        JEE string: specifies whether the Jacobian for OSC control of 
                    the end-effector should relay the x or y position.
        """
        self.DOF = 1
        ArmBase.__init__(self, singularity_thresh=1e-10, **kwargs)

        # length of arm links
        self.l1 = .8
        self.L = np.array([self.l1])
        # mass of links
        m1=1.0
        # z axis inertia moment of links
        izz1=15.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*m1
        self.M1[3:,3:] = np.eye(3)*izz1

        self.resting_position = np.array([np.pi/4.0])
        
        # stores information returned from maplesim
        self.state = np.zeros(3) 
        # maplesim arm simulation
        self.sim = py1LinkArm.pySim(dt=1e-5)
        self.sim.reset(self.state)
        self.reset()
        self.update_state()

    def apply_torque(self, u, dt):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly. 

        u np.array: the control signal to apply
        dt float: the timestep
        """
        u = -1 * np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/1e-5))):
            self.sim.step(self.state, u)
        self.update_state()

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
        q = self.q if q is None else q
    
        JCOM1 = np.zeros((6,1))
        JCOM1[0,0] = self.l1 / 2. * -np.sin(q[0]) 
        JCOM1[1,0] = self.l1 / 2. * np.cos(q[0]) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from end-effector to
           the origin frame"""
        q = self.q if q is None else q

        JEE = np.zeros((2,1))
        JEE[0,0] = self.l1 * -np.sin(q[0])
        JEE[1,0] = self.l1 * np.cos(q[0])
        
        return JEE

    def gen_Mq(self, q=None):
        """Generates the mass matrix for the arm in joint space"""
        q = self.q if q is None else q
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1()
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1))
        
        return Mq
        
    def position(self, q=None):
        """Compute x,y position of the hand

        q np.array: a set of angles to return positions for
        ee_only boolean: only return the (x,y) of the end-effector
        """
        q = self.q if q is None else q

        x = np.cumsum([0,
                       self.l1 * np.cos(q[0])])
        y = np.cumsum([0,
                       self.l1 * np.sin(q[0])])
        return np.array([x, y])

    def reset(self, q=[], dq=[]):
        if isinstance(q, np.ndarray): q = q.tolist()
        if isinstance(dq, np.ndarray): dq = dq.tolist()

        if q: assert len(q) == self.DOF
        if dq: assert len(dq) == self.DOF

        state = np.zeros(self.DOF*2)
        state[::2] = self.init_q if not q else np.copy(q)
        state[1::2] = self.init_dq if not dq else np.copy(dq)

        self.sim.reset(self.state, state)
        self.update_state()

    def update_state(self):
        """Separate out the state variable into time, angles, 
        velocities, and accelerations."""

        self.t = self.state[0]
        self.q = self.state[1:2]
        self.dq = self.state[2:] 
