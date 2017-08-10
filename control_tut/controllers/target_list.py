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

import numpy as np

class Shell(object):
    """
    """

    def __init__(self, controller, target_list, 
                    threshold=.01, pen_down=False, 
                    timer_time=1, timer2_time=50,
                    postural=False):
        """
        controller Control instance: the controller to use 
        pen_down boolean: True if the end-effector is drawing
        """

        self.controller = controller
        self.pen_down = pen_down 
        self.target_list = target_list
        self.threshold = threshold
   
        self.postural = postural
        self.not_at_start = True 
        self.target_index = 0
        self.set_target()

        # starts when a movement is complete before starting the next one
        self.run_timer = False 
        self.timer = 0 
        self.hold_time = timer_time # ns

        # starts when target is presented, while running arm can't move
        self.run_timer2 = False
        self.timer2 = 0 
        self.hold_time2 = timer2_time # ns

        if self.postural == True:
            self.additions = self.controller.additions[0]
            self.controller.additions = []
            print('additional force removed...')

    def control(self, arm): 
        """Move to a series of targets.
        """

        if self.controller.check_distance(arm) < self.threshold and self.run_timer == False:
            # start the timer, holding for self.hold_time ms before beginning next movement
            self.run_timer = True

            if self.postural == True and \
                    self.target_index != len(self.target_list) - 1:
                self.pen_down = not self.pen_down
                self.controller.additions = [self.additions]
                print('additional force added...')

                print('start recording')
                for recorder in self.controller.recorders:
                    recorder.start_recording = True

            elif (self.target_index % 3) == 1:
                print('start recording')
                for recorder in self.controller.recorders:
                    recorder.start_recording = True

        if self.run_timer == True:

            self.timer += 1

            if self.timer == self.hold_time:
                # then move the target list forward now
            
                if self.target_index < len(self.target_list)-1:
                    self.target_index += 1
                self.set_target()

                self.not_at_start = not self.not_at_start
                self.pen_down = not self.pen_down

                self.timer = 0
                self.run_timer = False
                self.run_timer2 = True
                self.controller.block_output = True

                print('target shown...')

                if self.postural == True:
                    self.controller.additions = []
                    print('additional force removed...')

                    print('write to file')
                    for recorder in self.controller.recorders:
                        recorder.write_out = True


        if self.run_timer2 == True:

            self.timer2 += 1

            if self.timer2 == self.hold_time2:
                # stop blocking output, let control signal be applied

                self.timer2 = 0
                self.run_timer2 = False
                self.controller.block_output = False

                print('start movement...')

        self.u = self.controller.control(arm)

        return self.u

    def set_target(self):
        """
        Set the current target for the controller.
        """

        # write to file if it's time
        if self.postural is not True and \
                self.target_index % 3 == 0 and \
                    self.target_index > 0: # 24 == 0
            print('write to file')
            for recorder in self.controller.recorders:
                recorder.write_out = True

        # and then exit or go to the next target
        if self.target_index == len(self.target_list)-1:
            exit()
        else:
            target = self.target_list[self.target_index]

        # if it's NANs then skip to next target
        if target[0] != target[0]: 
            self.target_index += 1
            self.set_target()
        else:
            self.controller.target = target
