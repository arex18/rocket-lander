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
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, **kwargs): 

        super(Control, self).__init__(**kwargs)

        # generalized coordinates
        self.target_gain = 2*np.pi
        self.target_bias = -np.pi

        if self.write_to_file is True:
            from controllers.recorder import Recorder
            # set up recorders
            self.u_recorder = Recorder('control signal', self.task, 'gc')
            self.xy_recorder = Recorder('end-effector position', self.task, 'gc')
            self.dist_recorder = Recorder('distance from target', self.task, 'gc')
            self.recorders = [self.u_recorder, 
                            self.xy_recorder, 
                            self.dist_recorder]

    def check_distance(self, arm):
        """Checks the distance to target"""
        return np.sum(abs(arm.q - self.target))

    def control(self, arm, q_des=None):
        """Generate a control signal to move the arm through
           joint space to the desired joint angle position"""
        
        # calculated desired joint angle acceleration
        if q_des is None:
            prop_val = ((self.target.reshape(1,-1) - arm.q) + np.pi) % \
                                                        (np.pi*2) - np.pi
        else: 
            # if a desired location is specified on input
            prop_val = q_des - arm.q

        # add in velocity compensation
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq).reshape(-1,)

        Mq = arm.gen_Mq()

        # tau = Mq * q_des + tau_grav, but gravity = 0
        self.u = np.dot(Mq, q_des).reshape(-1,)

        if self.write_to_file is True:
            # feed recorders their signals
            self.u_recorder.record(0.0, self.U)
            self.xy_recorder.record(0.0, self.arm.x)
            self.dist_recorder.record(0.0, self.target - self.arm.x)

        return self.u

    def gen_target(self, arm):
        """Generate a random target"""
        self.target = np.random.random(size=arm.DOF,) * \
            self.target_gain + self.target_bias

        return self.target
