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

import controllers.target_list as target_list
import controllers.signal as signal

import numpy as np

class Addition(signal.Signal):
    def __init__(self, index=0):
        forces = np.array([[-0.90937641, -0.64614133, -0.89626731],
                           [ 0.56228211,  0.80956427, -0.47827398],
                           [ 0.77974938, -0.5750045 ,  0.17895108],
                           [-0.25071541,  0.54737452, -0.92822244],
                           [ 0.19228047,  0.034985  , -0.37703873],
                           [-0.7975092 , -0.56974833,  0.31316189],
                           [-0.52069927,  0.44450567, -0.55619732],
                           [-0.65013087, -0.88908786,  0.07065782]])
        self.scale = 400
        self.force_vector = forces[index] * self.scale

    def generate(self, u, arm):
        return self.force_vector[:arm.DOF]


def Task(arm, controller_type, x_bias=0., y_bias=2., dist=.4, 
         force_index=7, write_to_file=False, **kwargs):
    """
    This task sets up the arm to move to 8 target points
    around the unit circle, and then attempt to hold these
    positions while an unexpected force is applied.
    """

    # set arm specific parameters ------------
    repeat = 0 
    if arm.DOF == 2:
        dist = .075
        y_bias = .35
        threshold = .0075 
    if arm.DOF == 3:
        threshold = .015

    # generate the path to follow -------------
    # set up the reaching trajectories, 8 points around unit circle
    targets_x = [dist * np.cos(theta) + x_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    targets_x += targets_x * repeat
    targets_y = [dist * np.sin(theta) + y_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    targets_y += targets_y * repeat
    trajectory = np.ones((2*len(targets_x)+3, 2))*np.nan

    for ii in range(len(targets_x)): 
        trajectory[ii*2] = [targets_x[ii], targets_y[ii]]
    trajectory[-2] = [0, y_bias]


    # generate control shell -----------------
    additions = []
    force_index = np.random.randint(8) if force_index is None else force_index
    print 'applying force %i...'%force_index
    additions.append(Addition(index=force_index))
    task = 'arm%i/postural%i'%(arm.DOF, force_index)

    control_pars = {'additions':additions,
                    'task':task,
                    'write_to_file':write_to_file}
    if controller_type.__class__ == 'osc':
        kp = 100 # position error gain on the PD controller
        control_pars['kp'] = kp
        control_pars['kv'] = np.sqrt(kp)
    controller = controller_type.Control(**control_pars)

    control_pars = {'target_list':trajectory,
                    'threshold':threshold, # how close to get to each target
                    'timer_time':2000,# how many ns to stay at each target 
                    'postural':True}

    runner_pars = {'infinite_trail':True, 
                   'title':'Task: Reaching',
                   'trajectory':trajectory}

    control_shell = target_list.Shell(controller=controller, **control_pars)

    return (control_shell, runner_pars)
