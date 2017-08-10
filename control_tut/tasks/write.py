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

import controllers.osc as osc
import controllers.forcefield as forcefield

import tasks.write_data.read_path as rp

import numpy as np


def Task(arm, controller_class, sequence=None, scale=None,
         force=None, write_to_file=False, **kwargs):
    """
    This task sets up the arm to write numbers inside
    a specified area (-x_bias, x_bias, -y_bias, y_bias).
    """

    # check controller type ------------------
    controller_name = controller_class.__name__.split('.')[1]
    if controller_name not in ('dmp', 'trace'):
        raise Exception('Cannot perform reaching task with this controller.')

    # set arm specific parameters ------------
    if arm.DOF == 2:
        kp = 20  # position error gain on the PD controller
        threshold = .01
        writebox = np.array([-.1, .1, .2, .25])
    elif arm.DOF == 3:
        kp = 100  # position error gain on the PD controller
        threshold = .05
        writebox = np.array([-.25, .25, 1.65, 2.])

    # generate the path to follow -------------
    sequence = 'hello' if sequence is None else sequence
    sequence = [c for c in sequence]

    if scale is None:
        scale = [1.0] * len(sequence)
    else:
        scale = [float(c) for c in scale]

    trajectory = rp.get_sequence(sequence, writebox, spaces=True)

    # generate control shell -----------------
    additions = []
    if force is not None:
        print('applying joint velocity based forcefield...')
        additions.append(forcefield.Addition(scale=force))

    control_pars = {'additions': additions,
                    'gain': 1000,  # pd gain for trajectory following
                    'pen_down': False,
                    'threshold': threshold,
                    'trajectory': trajectory.T}

    controller_name = controller_class.__name__.split('.')[1]
    if controller_name == 'dmp':
        # number of goals is the number of (NANs - 1) * number of DMPs
        num_goals = (np.sum(trajectory[:, 0] != trajectory[:, 0]) - 1) * 2
        # respecify goals for spatial scaling by changing add_to_goals
        control_pars['add_to_goals'] = [0]*num_goals
        control_pars['bfs'] = 1000  # number of basis function per DMP
        control_pars['tau'] = .1  # how fast the trajectory rolls out
    elif controller_name == 'trace':
        control_pars['tau'] = .005  # how fast the trajectory rolls out

    print('Using operational space controller...')
    controller = osc.Control(kp=kp, kv=np.sqrt(kp),
                             write_to_file=write_to_file)
    control_shell = controller_class.Shell(controller=controller,
                                           **control_pars)

    # generate runner parameters -----------
    runner_pars = {'infinite_trail': True,
                   'title': 'Task: Writing numbers',
                   'trajectory': trajectory}

    return (control_shell, runner_pars)
