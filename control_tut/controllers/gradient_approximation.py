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
import scipy.linalg as spla

from arms.two_link.arm_python import Arm as Arm2_python

class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, **kwargs): 

        super(Control, self).__init__(**kwargs)

        self.DOF = 2 # task space dimensionality 
        self.u = None

        # specify which gradient approximation method to use
        self.gradient_approximation = self.spsa

        if self.write_to_file is True:
            from recorder import Recorder
            # set up recorders
            self.xy_recorder = Recorder('end-effector position', self.task, 'gradient_spsa')
            self.recorders = [self.xy_recorder]

    def fdsa(self, x, u, ck):
        """ Estimate the gradient of a self.cost using 
        Finite Differences Stochastic Approximation (FDSA). 

        x np.array: state of the function
        u np.array: control signal
        ck: magnitude of perturbation
        """
        gk = np.zeros(u.shape)
        for ii in range(gk.shape[0]):
            # Generate perturbations one parameter at a time
            inc_u = np.copy(u)
            inc_u[ii] += ck
            dec_u = np.copy(u) 
            dec_u[ii] -= ck

            # Function evaluation
            cost_inc = self.cost(np.copy(x), inc_u)
            cost_dec = self.cost(np.copy(x), dec_u)

            # Gradient approximation
            gk[ii] = (cost_inc - cost_dec) / (2.0 * ck)
        return gk

    def spsa(self, x, u, ck):
        """ Estimate the gradient of a self.cost using 
        Simultaneous Perturbation Stochastic Approximation (SPSA). 
        Implemented base on (Spall, 1998).

        x np.array: state of the function
        u np.array: control signal
        ck: magnitude of perturbation
        """
        # Generate simultaneous perturbation vector.
        # Choose each component from a bernoulli +-1 distribution 
        # with probability of .5 for each +-1 outcome.
        delta_k = np.random.choice([-1,1], size=self.arm.DOF, 
                                   p=[.5, .5])

        # Function evaluations
        inc_u = np.copy(u) + ck * delta_k
        cost_inc = self.cost(np.copy(x), inc_u)
        dec_u = np.copy(u) - ck * delta_k
        cost_dec = self.cost(np.copy(x), dec_u)

        # Gradient approximation
        gk = np.dot((cost_inc - cost_dec) / (2.0*ck), delta_k)
        return gk

    def cost(self, x, u): 
        """ Calculate the cost of applying u in state x. """
        dt = .1 if self.arm.DOF == 3 else .01
        next_x = self.plant_dynamics(x, u, dt=dt)
        vel_gain = 100 if self.arm.DOF == 3 else 10
        return (np.sqrt(np.sum((self.arm.x - self.target)**2)) * 1000 \
                + np.sum((next_x[self.arm.DOF:])**2) * vel_gain)

    def control(self, arm, x_des=None):
        """ Use gradient approximation to calculate a 
        control signal that minimizes self.cost()

        arm Arm: the arm model being controlled
        x_des np.array: desired task-space force, 
                        system goes to self.target if None
        """  

        self.x = arm.x
        self.arm, state = self.copy_arm(arm)

        # Step 1: Initialization and coefficient selection
        max_iters = 10
        converge_thresh = 1e-5

        alpha = 0.602 # from (Spall, 1998)
        gamma = 0.101
        a = .101 # found empirically using HyperOpt
        A = .193
        c = .0277

        delta_K = None
        delta_J = None
        u = np.copy(self.u) if self.u is not None \
                else np.zeros(self.arm.DOF)

        for k in range(max_iters):
            ak = a / (A + k + 1)**alpha
            ck = c / (k + 1)**gamma

            # Estimate gradient 
            gk = self.gradient_approximation(state, u, ck)

            # Update u estimate
            old_u = np.copy(u)
            u -= ak * gk

            # Check for convergence
            if np.sum(abs(u - old_u)) < converge_thresh:
                break

        self.u = np.copy(u)

        if self.write_to_file is True:
            # feed recorders their signals
            self.xy_recorder.record(0.0, self.x)

        # add in any additional signals 
        for addition in self.additions:
            self.u += addition.generate(self.u, arm)

        return self.u
 
    def copy_arm(self, real_arm):
        """ Make a copy of the arm for local simulation. """
        arm = real_arm.__class__()
        arm.dt = real_arm.dt

        # reset arm position to x_0
        arm.reset(q = real_arm.q, dq = real_arm.dq)

        return arm, np.hstack([real_arm.q, real_arm.dq])

    def plant_dynamics(self, x, u, dt=None):
        """ Simulate the arm dynamics locally. """
        dt = self.arm.dt if dt is None else dt

        if x.ndim == 1:
            x = x[:,None]
            u = u[None,:]

        xnext = np.zeros((x.shape))
        for ii in range(x.shape[1]):
            # set the arm position to x
            self.arm.reset(q=x[:self.arm.DOF, ii], 
                          dq=x[self.arm.DOF:self.arm.DOF*2, ii])

            # apply the control signal
            self.arm.apply_torque(u[ii], dt)
            # get the system state from the arm
            xnext[:,ii] = np.hstack([np.copy(self.arm.q), 
                                   np.copy(self.arm.dq)])

        return xnext

    def gen_target(self, arm):
        pass

