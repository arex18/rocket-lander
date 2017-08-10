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

import numpy as np
import pygame
import pygame.locals
import sys

class ArmPart:
    """
    A class for storing relevant arm segment information.
    """
    def __init__(self, pic, scale=1.0):
        self.base = pygame.image.load(pic)
        self.offset = self.base.get_rect()[2] / 2. * scale

    def rotate(self, rotation):
        """
        Rotates and re-centers the arm segment.
        """
        self.rotation = rotation

        # rotate our image
        image = pygame.transform.rotozoom(self.base, np.degrees(rotation), 1)
        # reset the center
        rect = image.get_rect()
        rect.center = np.zeros(2)

        return image, rect

def transform(rect, base, arm_part):
    rect.center += np.asarray(base)
    rect.center += np.array([np.cos(arm_part.rotation) * arm_part.offset,
                            -np.sin(arm_part.rotation) * arm_part.offset])
def transform_lines(rect, base, arm_part):
    transform(rect, base, arm_part)
    rect.center += np.array([-rect.width / 2.0, -rect.height / 2.0])

class Runner:
    """
    A class for drawing the arm simulation using PyGame
    """
    def __init__(self, title='', dt=1e-4, control_steps=10,
                       display_steps=100, t_target=1.0,
                       box=[-1,1,-1,1], rotate=0.0,
                       control_type='', trajectory=None,
                       infinite_trail=False, mouse_control=False):
        self.dt = dt
        self.control_steps = control_steps
        self.display_steps = display_steps
        self.target_steps = int(t_target/float(dt*display_steps))
        self.trajectory = trajectory

        self.box = box
        self.control_type = control_type
        self.infinite_trail = infinite_trail
        self.mouse_control = mouse_control
        self.rotate = rotate
        self.title = title

        self.sim_step = 0
        self.trail_index = -1
        self.pen_lifted = False

        self.width = 642
        self.height = 600
        self.base_offset = np.array([self.width / 2.0, self.height*.9])

    def run(self, arm, control_shell, video=None, video_time=None):

        self.arm = arm
        self.shell = control_shell

        # load arm images
        arm1 = ArmPart('img/three_link/svgupperarm2.png',
                        scale = .7)
        arm2 = ArmPart('img/three_link/svgforearm2.png',
                        scale = .8)
        arm3 = ArmPart('img/three_link/svghand2.png',
                        scale= 1)

        scaling_term = np.ones(2) * 105
        upperarm_length = self.arm.L[0] * scaling_term[0]
        forearm_length = self.arm.L[1] * scaling_term[0]
        hand_length = self.arm.L[2] * scaling_term[0]
        line_width = .15 * scaling_term[0]

        # create transparent arm lines
        line_upperarm_base = pygame.Surface((upperarm_length, line_width),
                pygame.SRCALPHA, 32)
        line_forearm_base = pygame.Surface((forearm_length, line_width),
                pygame.SRCALPHA, 32)
        line_hand_base = pygame.Surface((hand_length, line_width),
                pygame.SRCALPHA, 32)

        white = (255, 255, 255)
        red = (255, 0, 0)
        black = (0, 0, 0)
        arm_color = (75, 75, 75)
        line_color = (50, 50, 50, 200) # fourth value is transparency

        # color in transparent arm lines
        line_upperarm_base.fill(line_color)
        line_forearm_base.fill(line_color)
        line_hand_base.fill(line_color)

        fps = 20 # frames per second
        fpsClock = pygame.time.Clock()

        # constants for magnify plotting
        magnify_scale = 1.75
        magnify_window_size = np.array([200, 200])
        first_target = np.array([321, 330])
        magnify_offset = first_target * magnify_scale - magnify_window_size / 2

        # setup pen trail and appending functions
        self.trail_data = []
        def pen_down1():
            self.pen_lifted = False
            x,y = self.arm.position()
            x = int( x[-1] * scaling_term[0] + self.base_offset[0])
            y = int(-y[-1] * scaling_term[1] + self.base_offset[1])
            self.trail_data.append([[x,y],[x,y]])

            self.trail_data[self.trail_index].append(points[3])
            self.pen_down = pen_down2
        def pen_down2():
            self.trail_data[self.trail_index].append(points[3])

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

        background = pygame.image.load('img/whiteboard.jpg')

        # enter simulation / plotting loop
        while True:

            self.display.fill(white)

            self.target = self.shell.controller.target * np.array([1, -1]) * \
                            scaling_term + self.base_offset

            # before drawing
            for j in range(self.display_steps):
                # update control signal
                if self.sim_step % self.control_steps == 0 or \
                    'tau' not in locals():
                        tau = self.shell.control(self.arm)
                # apply control signal and simulate
                self.arm.apply_torque(u=tau, dt=self.dt)

                self.sim_step +=1

            # get (x,y) positions of the joints
            x,y = self.arm.position()
            points = [(int(a * scaling_term[0] + self.base_offset[0]),
                       int(-b * scaling_term[1] + self.base_offset[1]))
                       for a,b in zip(x,y)]

            arm1_image, arm1_rect = arm1.rotate(self.arm.q[0])
            arm2_image, arm2_rect = arm2.rotate(self.arm.q[1] + arm1.rotation)
            arm3_image, arm3_rect = arm3.rotate(self.arm.q[2] + arm2.rotation)

            # recenter the image locations appropriately
            transform(arm1_rect, points[0], arm1)
            transform(arm2_rect, points[1], arm2)
            transform(arm3_rect, points[2], arm3)
            arm3_rect.center += np.array([np.cos(arm3.rotation),
                                          -np.sin(arm3.rotation)]) * -10

            # transparent upperarm line
            line_upperarm = pygame.transform.rotozoom(line_upperarm_base, np.degrees(arm1.rotation), 1)
            rect_upperarm = line_upperarm.get_rect()
            transform_lines(rect_upperarm, points[0], arm1)

            # transparent forearm line
            line_forearm = pygame.transform.rotozoom(line_forearm_base, np.degrees(arm2.rotation), 1)
            rect_forearm = line_forearm.get_rect()
            transform_lines(rect_forearm, points[1], arm2)

            # transparent hand line
            line_hand = pygame.transform.rotozoom(line_hand_base, np.degrees(arm3.rotation), 1)
            rect_hand = line_hand.get_rect()
            transform_lines(rect_hand, points[2], arm3)

            # update trail
            if self.shell.pen_down is True:
                self.pen_down()
            elif self.shell.pen_down is False and self.pen_lifted is False:
                self.pen_down = pen_down1
                self.pen_lifted = True
                self.trail_index += 1

            # draw things!
            self.display.blit(background, (0,0)) # draw on the background

            for trail in self.trail_data:
                pygame.draw.aalines(self.display, black, False, trail, True)

            # draw arm images
            self.display.blit(arm1_image, arm1_rect)
            self.display.blit(arm2_image, arm2_rect)
            self.display.blit(arm3_image, arm3_rect)

            # draw original arm lines
            # pygame.draw.lines(self.display, arm_color, False, points, 18)

            # draw transparent arm lines
            self.display.blit(line_upperarm, rect_upperarm)
            self.display.blit(line_forearm, rect_forearm)
            self.display.blit(line_hand, rect_hand)

            # draw circles at shoulder
            pygame.draw.circle(self.display, black, points[0], 30)
            pygame.draw.circle(self.display, arm_color, points[0], 12)

            # draw circles at elbow
            pygame.draw.circle(self.display, black, points[1], 20)
            pygame.draw.circle(self.display, arm_color, points[1], 7)

            # draw circles at wrist
            pygame.draw.circle(self.display, black, points[2], 15)
            pygame.draw.circle(self.display, arm_color, points[2], 5)

            # draw target
            pygame.draw.circle(self.display, red, [int(val) for val in self.target], 10)

            # now display magnification of drawing area
            magnify = pygame.Surface(magnify_window_size)
            magnify.blit(background, (-200,-200)) # draw on the background
            # magnify.fill(white)
            # put a border on it
            pygame.draw.rect(magnify, black, (2.5, 2.5, 195, 195), 1)
            # now we need to rescale the trajectory and targets
            # using the first target position, which I know to be the
            # desired center of the magnify area
            for trail in self.trail_data:
                pygame.draw.aalines(magnify, black, False,
                        np.asarray(trail) * magnify_scale - magnify_offset, True)
            pygame.draw.circle(magnify, red,
                    np.array(self.target * magnify_scale - magnify_offset,
                        dtype=int), 5)

            # now draw the target and hand line
            self.display.blit(magnify, (32, 45))

            # check for quit
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()
            fpsClock.tick(fps)

    def show(self):
        try:
            plt.show()
        except AttributeError:
            pass
