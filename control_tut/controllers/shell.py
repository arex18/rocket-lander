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


class Shell(object):

    def __init__(self, controller, pen_down=False, **kwargs):
        """
        control Control instance: the controller to use
        pen_down boolean: True if the end-effector is drawing
        """

        self.controller = controller
        self.pen_down = pen_down
        self.kwargs = kwargs

    def control(self, arm):
        """Call the controllers control function.
        """
        self.u = self.controller.control(arm, **self.kwargs)
        return self.u
