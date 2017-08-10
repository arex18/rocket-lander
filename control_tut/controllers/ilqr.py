'''
Copyright (C) 2016 Travis DeWolf 

Implemented from 'Control-limited differential dynamic programming'
by Yuval Tassa, Nicolas Mansard, and Emo Todorov (2014).

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

import lqr as lqr

import numpy as np
from copy import copy

class Control(lqr.Control):
    """
    A controller that implements iterative Linear Quadratic Gaussian control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, n=50, max_iter=100, **kwargs): 
        '''
        n int: length of the control sequence
        max_iter int: limit on number of optimization iterations
        '''

        super(Control, self).__init__(**kwargs)

        self.old_target = [None, None]

        self.tN = n # number of timesteps

        self.max_iter = max_iter
        self.lamb_factor = 10
        self.lamb_max = 1000
        self.eps_converge = 0.001 # exit if relative improvement below threshold

        if self.write_to_file is True:
            from controllers.recorder import Recorder
            # set up recorders
            self.u_recorder = Recorder('control signal', self.task, 'ilqr')
            self.xy_recorder = Recorder('end-effector position', self.task, 'ilqr')
            self.dist_recorder = Recorder('distance from target', self.task, 'ilqr')
            self.recorders = [self.u_recorder, 
                            self.xy_recorder, 
                            self.dist_recorder]

    def control(self, arm, x_des=None):
        """Generates a control signal to move the 
        arm to the specified target.
            
        arm Arm: the arm model being controlled
        des list: the desired system position
        x_des np.array: desired task-space force, 
                        irrelevant here.
        """

        # if the target has changed, reset things and re-optimize 
        # for this movement
        if self.old_target[0] != self.target[0] or \
            self.old_target[1] != self.target[1]:
                self.reset(arm, x_des)


        # Reset k if at the end of the sequence
        if self.t >= self.tN-1:
            self.t = 0
        # Compute the optimization
        if self.t % 1 == 0:
            x0 = np.zeros(arm.DOF*2)
            self.arm, x0[:arm.DOF*2] = self.copy_arm(arm)
            U = np.copy(self.U[self.t:])
            self.X, self.U[self.t:], cost = \
                    self.ilqr(x0, U)
        self.u = self.U[self.t]

        # move us a step forward in our control sequence
        self.t += 1

        if self.write_to_file is True:
            # feed recorders their signals
            self.u_recorder.record(0.0, self.U)
            self.xy_recorder.record(0.0, self.arm.x)
            self.dist_recorder.record(0.0, self.target - self.arm.x)

        # add in any additional signals (noise, external forces)
        for addition in self.additions:
            self.u += addition.generate(self.u, arm)

        return self.u

    def copy_arm(self, real_arm):
        """ make a copy of the arm model, to make sure that the
        actual arm model isn't affected during the iLQR process

        real_arm Arm: the arm model being controlled
        """ 

        # need to make a copy of the arm for simulation
        arm = real_arm.__class__()
        arm.dt = real_arm.dt

        # reset arm position to x_0
        arm.reset(q = real_arm.q, dq = real_arm.dq)

        return arm, np.hstack([real_arm.q, real_arm.dq])

    def cost(self, x, u):
        """ the immediate state cost function """
        # compute cost
        dof = u.shape[0]
        num_states = x.shape[0]

        l = np.sum(u**2)

        # compute derivatives of cost
        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))
        l_u = 2 * u
        l_uu = 2 * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        # returned in an array for easy multiplication by time step 
        return l, l_x, l_xx, l_u, l_uu, l_ux

    def cost_final(self, x):
        """ the final state cost function """
        num_states = x.shape[0]
        l_x = np.zeros((num_states))
        l_xx = np.zeros((num_states, num_states))

        wp = 1e4 # terminal position cost weight
        wv = 1e4 # terminal velocity cost weight

        xy = self.arm.x
        xy_err = np.array([xy[0] - self.target[0], xy[1] - self.target[1]])
        l = (wp * np.sum(xy_err**2) +
                wv * np.sum(x[self.arm.DOF:self.arm.DOF*2]**2))

        l_x[0:self.arm.DOF] = wp * self.dif_end(x[0:self.arm.DOF])
        l_x[self.arm.DOF:self.arm.DOF*2] = (2 * 
                wv * x[self.arm.DOF:self.arm.DOF*2])

        eps = 1e-4 # finite difference epsilon
        # calculate second derivative with finite differences
        for k in range(self.arm.DOF): 
            veps = np.zeros(self.arm.DOF)
            veps[k] = eps
            d1 = wp * self.dif_end(x[0:self.arm.DOF] + veps)
            d2 = wp * self.dif_end(x[0:self.arm.DOF] - veps)
            l_xx[0:self.arm.DOF, k] = ((d1-d2) / 2.0 / eps).flatten()

        l_xx[self.arm.DOF:self.arm.DOF*2, self.arm.DOF:self.arm.DOF*2] = 2 * wv * np.eye(self.arm.DOF)

        # Final cost only requires these three values
        return l, l_x, l_xx

    # Compute derivative of endpoint error 
    def dif_end(self, x):

        xe = -self.target.copy()
        for ii in range(self.arm.DOF):
            xe[0] += self.arm.L[ii] * np.cos(np.sum(x[:ii+1]))
            xe[1] += self.arm.L[ii] * np.sin(np.sum(x[:ii+1]))

        edot = np.zeros((self.arm.DOF,1))
        for ii in range(self.arm.DOF):
            edot[ii,0] += (2 * self.arm.L[ii] * 
                    (xe[0] * -np.sin(np.sum(x[:ii+1])) + 
                     xe[1] * np.cos(np.sum(x[:ii+1]))))
        edot = np.cumsum(edot[::-1])[::-1][:]

        return edot

    def finite_differences(self, x, u): 
        """ calculate gradient of plant dynamics using finite differences

        x np.array: the state of the system
        u np.array: the control signal 
        """  
        dof = u.shape[0]
        num_states = x.shape[0]

        A = np.zeros((num_states, num_states))
        B = np.zeros((num_states, dof))

        eps = 1e-4 # finite differences epsilon
        for ii in range(num_states):
            # calculate partial differential w.r.t. x
            inc_x = x.copy()
            inc_x[ii] += eps
            state_inc,_ = self.plant_dynamics(inc_x, u.copy())
            dec_x = x.copy()
            dec_x[ii] -= eps
            state_dec,_ = self.plant_dynamics(dec_x, u.copy())
            A[:, ii] = (state_inc - state_dec) / (2 * eps)

        for ii in range(dof):
            # calculate partial differential w.r.t. u
            inc_u = u.copy()
            inc_u[ii] += eps
            state_inc,_ = self.plant_dynamics(x.copy(), inc_u)
            dec_u = u.copy()
            dec_u[ii] -= eps
            state_dec,_ = self.plant_dynamics(x.copy(), dec_u)
            B[:, ii] = (state_inc - state_dec) / (2 * eps)

        return A, B

    def gen_target(self, arm):
        """Generate a random target"""
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0
        
        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()

    def ilqr(self, x0, U=None): 
        """ use iterative linear quadratic regulation to find a control 
        sequence that minimizes the cost function 

        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        U = self.U if U is None else U

        tN = U.shape[0] # number of time steps
        dof = self.arm.DOF # number of degrees of freedom of plant 
        num_states = dof * 2 # number of states (position and velocity)
        dt = self.arm.dt # time step

        lamb = 1.0 # regularization parameter
        sim_new_trajectory = True

        for ii in range(self.max_iter):

            if sim_new_trajectory == True: 
                # simulate forward using the current control trajectory
                X, cost = self.simulate(x0, U)
                oldcost = np.copy(cost) # copy for exit condition check

                # now we linearly approximate the dynamics, and quadratically 
                # approximate the cost function so we can use LQR methods 

                # for storing linearized dynamics
                # x(t+1) = f(x(t), u(t))
                f_x = np.zeros((tN, num_states, num_states)) # df / dx
                f_u = np.zeros((tN, num_states, dof)) # df / du
                # for storing quadratized cost function 
                l = np.zeros((tN,1)) # immediate state cost 
                l_x = np.zeros((tN, num_states)) # dl / dx
                l_xx = np.zeros((tN, num_states, num_states)) # d^2 l / dx^2
                l_u = np.zeros((tN, dof)) # dl / du
                l_uu = np.zeros((tN, dof, dof)) # d^2 l / du^2
                l_ux = np.zeros((tN, dof, num_states)) # d^2 l / du / dx
                # for everything except final state
                for t in range(tN-1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    A, B = self.finite_differences(X[t], U[t])
                    f_x[t] = np.eye(num_states) + A * dt
                    f_u[t] = B * dt
                
                    (l[t], l_x[t], l_xx[t], l_u[t], 
                        l_uu[t], l_ux[t]) = self.cost(X[t], U[t])
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt
                # aaaand for final state
                l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

                sim_new_trajectory = False

            # optimize things! 
            # initialize Vs with final state cost and set up k, K 
            V = l[-1].copy() # value function
            V_x = l_x[-1].copy() # dV / dx
            V_xx = l_xx[-1].copy() # d^2 V / dx^2
            k = np.zeros((tN, dof)) # feedforward modification
            K = np.zeros((tN, dof, num_states)) # feedback gain

            # NOTE: they use V' to denote the value at the next timestep, 
            # they have this redundant in their notation making it a 
            # function of f(x + dx, u + du) and using the ', but it makes for 
            # convenient shorthand when you drop function dependencies

            # work backwards to solve for V, Q, k, and K
            for t in range(tN-2, -1, -1):

                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                
                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

                # Calculate Q_uu^-1 with regularization term set by 
                # Levenberg-Marquardt heuristic (at end of this loop)
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                Q_uu_inv = np.dot(Q_uu_evecs, 
                        np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

                # 5b) k = -np.dot(Q_uu^-1, Q_u)
                k[t] = -np.dot(Q_uu_inv, Q_u)
                # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

            Unew = np.zeros((tN, dof))
            # calculate the optimal change to the control trajectory
            xnew = x0.copy() # 7a)
            for t in range(tN - 1): 
                # use feedforward (k) and feedback (K) gain matrices 
                # calculated from our value function approximation
                # to take a stab at the optimal control signal
                Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t]) # 7b)
                # given this u, find our next state
                _,xnew = self.plant_dynamics(xnew, Unew[t]) # 7c)

            # evaluate the new trajectory 
            Xnew, costnew = self.simulate(x0, Unew)

            # Levenberg-Marquardt heuristic
            if costnew < cost: 
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X = np.copy(Xnew) # update trajectory 
                U = np.copy(Unew) # update control signal
                oldcost = np.copy(cost)
                cost = np.copy(costnew)

                sim_new_trajectory = True # do another rollout

                # print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldcost-cost)/cost) < self.eps_converge):
                    print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                            " logLambda = %.1f"%np.log(lamb))
                    break

            else: 
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lamb_max: 
                    print("lambda > max_lambda at iteration = %d;"%ii + 
                        " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                          np.log(lamb)))
                    break

        return X, U, cost

    def plant_dynamics(self, x, u):
        """ simulate a single time step of the plant, from 
        initial state x and applying control signal u

        x np.array: the state of the system
        u np.array: the control signal
        """ 

        # set the arm position to x
        self.arm.reset(q=x[:self.arm.DOF], 
                       dq=x[self.arm.DOF:self.arm.DOF*2])

        # apply the control signal
        self.arm.apply_torque(u, self.arm.dt)
        # get the system state from the arm
        xnext = np.hstack([np.copy(self.arm.q), 
                           np.copy(self.arm.dq)])
        # calculate the change in state
        xdot = ((xnext - x) / self.arm.dt).squeeze()

        return xdot, xnext

    def reset(self, arm, q_des):
        """ reset the state of the system """

        # Index along current control sequence
        self.t = 0
        self.U = np.zeros((self.tN, arm.DOF))

        self.old_target = self.target.copy()

    def simulate(self, x0, U):
        """ do a rollout of the system, starting at x0 and 
        applying the control sequence U

        x0 np.array: the initial state of the system
        U np.array: the control sequence to apply
        """ 
        tN = U.shape[0]
        num_states = x0.shape[0]
        dt = self.arm.dt

        X = np.zeros((tN, num_states))
        X[0] = x0
        cost = 0

        # Run simulation with substeps
        for t in range(tN-1):
            _,X[t+1] = self.plant_dynamics(X[t], U[t])
            l,_,_,_,_,_ = self.cost(X[t], U[t])
            cost = cost + dt * l

        # Adjust for final cost, subsample trajectory
        l_f,_,_ = self.cost_final(X[-1])
        cost = cost + l_f

        return X, cost
