'''
Copyright (C) 2014 Travis DeWolf

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

import controllers.shell as shell
import controllers.forcefield as forcefield

import numpy as np

def Task(arm, controller_class, 
        force=None, write_to_file=False, **kwargs):
    """
    This task sets up the arm to move to random 
    target positions ever t_target seconds. 
    """

    # check controller type ------------------
    controller_name = controller_class.__name__.split('.')[1]
    if controller_name not in ('gc', 'lqr', 'osc'):
        raise Exception('Cannot perform reaching task with this controller.')

    # set arm specific parameters ------------
    if arm.DOF == 1:
        kp = 5
    elif arm.DOF == 2:
        kp = 10
    elif arm.DOF == 3: 
        kp = 50

    # generate control shell -----------------
    additions = []
    if force is not None:
        print 'applying joint velocity based forcefield...'
        additions.append(forcefield.Addition(scale=force))
        task = 'arm%i/forcefield'%arm.DOF

    controller = controller_class.Control(
                                        additions=additions,
                                        kp=kp, 
                                        kv=np.sqrt(kp),
                                        task='arm%i/random'%arm.DOF,
                                        write_to_file=write_to_file)
    control_shell = shell.Shell(controller=controller)

    # generate runner parameters -----------
    runner_pars = {'control_type':'random',
                'title':'Task: Random movements'}

    return (control_shell, runner_pars)
