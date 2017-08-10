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

from . import control_class

import numpy as np

class Control(control_class.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, null_control=True, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Control, self).__init__(**kwargs)

        self.DOF = 2 # task space dimensionality
        self.null_control = null_control

        if self.write_to_file is True:
            from recorder import Recorder
            # set up recorders
            self.u_recorder = Recorder('control signal', self.task, 'osc')
            self.xy_recorder = Recorder('end-effector position', self.task, 'osc')
            self.dist_recorder = Recorder('distance from target', self.task, 'osc')
            self.recorders = [self.u_recorder,
                            self.xy_recorder,
                            self.dist_recorder]

    def control(self, arm, x_des=None):
        """Generates a control signal to move the
        arm to the specified target.

        arm Arm: the arm model being controlled
        des list: the desired system position
        x_des np.array: desired task-space force,
                        system goes to self.target if None
        """

        # calculate desired end-effector acceleration
        if x_des is None:
            self.x = arm.x
            x_des = self.kp * (self.target - self.x)

        # generate the mass matrix in end-effector space
        Mq = arm.gen_Mq()
        Mx = arm.gen_Mx()

        # calculate force
        Fx = np.dot(Mx, x_des)

        # calculate the Jacobian
        JEE = arm.gen_jacEE()
        # tau = J^T * Fx + tau_grav, but gravity = 0
        # add in velocity compensation in GC space for stability
        self.u = (np.dot(JEE.T, Fx).reshape(-1,) -
                  np.dot(Mq, self.kv * arm.dq))

        # if null_control is selected and the task space has
        # fewer DOFs than the arm, add a control signal in the
        # null space to try to move the arm to its resting state
        if self.null_control and self.DOF < len(arm.L):

            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            prop_val = ((arm.rest_angles - arm.q) + np.pi) % (np.pi*2) - np.pi
            q_des = (self.kp * prop_val + \
                     self.kv * -arm.dq).reshape(-1,)

            Mq = arm.gen_Mq()
            u_null = np.dot(Mq, q_des)

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
            null_filter = np.eye(len(arm.L)) - np.dot(JEE.T, Jdyn_inv)

            null_signal = np.dot(null_filter, u_null).reshape(-1,)

            self.u += null_signal

        if self.write_to_file is True:
            # feed recorders their signals
            self.u_recorder.record(0.0, self.u)
            self.xy_recorder.record(0.0, self.x)
            self.dist_recorder.record(0.0, self.target - self.x)

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, arm)

        return self.u

    def gen_target(self, arm):
        """Generate a random target"""
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0

        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()
