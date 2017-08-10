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

Usage:
    run ARM CONTROLLER TASK [options]

Arguments:
    ARM         the arm to control
    CONTROLLER  the controller to use
    TASK        the task to perform

Options:
    --dt=DT                     specify float step size for simulations
    --end_time=ENDTIME          specify float time to end (seconds)
                                                  default is None
    --force=FORCEPARS           specify float strength of force to add
                                                  default is None
    --sequence=PHRASEPARS       specify sequence details for a given TASK
    --scale=SCALEPARS           specify the scale of the DMP if TASK=write
    --write_to_file=WRITEPARS   specify boolean for writing to file,
                                                  default is False
'''

from control_tut.sim_and_plot import Runner
from docopt import docopt
import importlib

#args = docopt(__doc__)
args = {'CONTROLLER': 'lqr', 'TASK': 'follow_mouse'}


dt = 1e-2 if args['CONTROLLER'] == 'ilqr' else 1e-3
dt = dt if args.get('--dt', None) is None else float(args['--dt'])

# get and initialize the arm
# if args['ARM'][:4] == 'arm1':
#     subfolder = 'one_link'
# if args['ARM'][:4] == 'arm2':
#     subfolder = 'two_link'
# elif args['ARM'][:4] == 'arm3':
#     subfolder = 'three_link'
subfolder = 'three_link'
arm_name = 'arms.%s.%s' % (subfolder, 'arm')#args['ARM'][4:])
arm_module = importlib.import_module(name=arm_name)
arm = arm_module.Arm(dt=dt)

# get the chosen controller class
controller_name = 'controllers.%s' % args['CONTROLLER'].lower()
controller_class = importlib.import_module(name=controller_name)

# get the chosen task class
task_name = 'tasks.%s' % args['TASK']
task_module = importlib.import_module(name=task_name)
print('task: ', task_module)
task = task_module.Task

# instantiate the controller for the chosen task
# and get the sim_and_plot parameters
control_shell, runner_pars = task(
    arm, controller_class,
    sequence=args['--sequence'], scale=args['--scale'],
    force=float(args['--force']) if args['--force'] is not None else None,
    write_to_file=bool(args['--write_to_file']))

# set up simulate and plot system
runner = Runner(dt=dt, **runner_pars)
runner.run(arm=arm, control_shell=control_shell,
           end_time=(float(args['--end_time'])
                     if args['--end_time'] is not None else None))
runner.show()
