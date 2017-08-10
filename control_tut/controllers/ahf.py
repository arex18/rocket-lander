'''
Copyright (C) 2016 Travis DeWolf

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

import control

import numpy as np

from hessianfree.rnnet import RNNet
from hessianfree.nonlinearities import (Tanh, Linear)

class Control(control.Control):
    """
    A controller that loads in a neural network trained using the
    hessianfree (https://github.com/drasmuss/hessianfree) library 
    to control a simulated arm.
    """
    def __init__(self, **kwargs): 

        super(Control, self).__init__(**kwargs)

        self.old_target = [None, None]

        # load up our network
        import glob

        # this code goes into the weights folder, finds the most 
        # recent trial, and loads up the weights
        files = sorted(glob.glob('controllers/weights/rnn*'))
        print 'loading weights from %s'%files[-1]
        W = np.load(files[-1])['arr_0']
        num_states = 4
        self.rnn = RNNet(shape=[num_states * 2, 32, 32, num_states, num_states], 
                     layers=[Linear(), Tanh(), Tanh(), Linear(), Linear()],
                     rec_layers=[1,2],
                     conns={0:[1, 2], 1:[2], 2:[3], 3:[4]},
                     load_weights=W,
                     use_GPU=False)

        offset, W_end, b_end = self.rnn.offsets[(3,4)]
        self.rnn.mask = np.zeros(self.rnn.W.shape, dtype=bool)
        self.rnn.mask[offset:b_end] = True
        self.rnn.W[offset:W_end] = np.eye(4).flatten()

        self.joint_targets = None
        self.act = None

        # set up recorders
        if self.write_to_file is True:
            from recorder import Recorder
            self.u_recorder = Recorder('control signal', self.task, 'hf')
            self.xy_recorder = Recorder('end-effector position', self.task, 'hf')
            self.dist_recorder = Recorder('distance from target', self.task, 'hf')
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

        self.x = arm.x

        # if the target has changed, convert into joint angles again
        if np.any(self.old_target != self.target):
            self.joint_targets = arm.inv_kinematics(xy=self.target)
            self.old_target = self.target
            
        inputs = np.concatenate([self.joint_targets, np.zeros(2), arm.q, arm.dq])[None,None,:]
        self.act = [a[:,-1,:] for a in self.rnn.forward(inputs, init_activations=self.act)]
        u = self.act[-1][0]
        # NOTE: Make sure this is set up the same way as in training
        # use all the network output is the control signal
        self.u = np.array([np.sum(u[ii::arm.DOF]) for ii in range(arm.DOF)])

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
        pass


