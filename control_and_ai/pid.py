"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: PID algorithm definitions. Slight deviation from PEP convention: action variables are defined in capitals.
"""

from constants import *
import abc


class PID():
    """ Called from the children of PID_Framework"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0

    def increment_intregral_error(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if (self.accumulated_error > pi_limit):
            self.accumulated_error = pi_limit
        elif (self.accumulated_error < pi_limit):
            self.accumulated_error = -pi_limit

    def compute_output(self, error, dt_error):
        self.increment_intregral_error(error)
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error

class PID_Framework():
    """ Sets the skeleton code for the actual pid algorithms (children) to inherit. """

    @abc.abstractmethod
    def pid_algorithm(self, s, x_target, y_target):
        pass


class PID_Benchmark(PID_Framework):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self):
        super(PID_Benchmark, self).__init__()
        self.Fe_PID = PID(10, 0, 10)
        self.psi_PID = PID(0.085, 0.001, 10.55)
        self.Fs_theta_PID = PID(5, 0, 6)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        if x_target is not None:
            dx = dx - x_target
        if y_target is not None:
            dy = dy - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x

        Fe = self.Fe_PID.compute_output(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.compute_output(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.compute_output(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi


class PID_Heuristic_Benchmark(PID_Framework):
    """ Heuristic PID Benchmark """

    def __init__(self):
        super(PID_Heuristic_Benchmark, self).__init__()
        self.Fe = PID(10, 0, 10)
        self.psi = PID(0.01, 0, 0.01)
        self.Fs = PID(10, 0, 30)

    def pid_algorithm(self, s, x_target, y_target):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        # ------------------------------------------
        x_error = x_target - theta
        x_dterror = -omega
        Fs = -self.Fs.compute_output(x_error, x_dterror)
        # ------------------------------------------
        y_error = y_target - dy
        y_dterror = -vel_y
        Fe = self.Fe.compute_output(y_error, y_dterror) - 1
        # ------------------------------------------
        theta_error = theta
        theta_dterror = -omega - vel_x
        psi = self.psi.compute_output(theta_error, theta_dterror)
        # ------------------------------------------
        if legContact_left and legContact_right:  # legs have contact
            Fe = 0

        return Fe, Fs, psi


class PID_psi(PID_Framework):
    """ PID for controlling just the angle of the rocket nozzle. """

    def __init__(self):
        super(PID_psi, self).__init__()
        self.psi = PID(0.1, 0, 0.01)

    def pid_algorithm(self, s, x_target=None, y_target=None):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        theta_error = theta
        theta_dterror = -omega - vel_x
        psi = self.psi.compute_output(theta_error, theta_dterror)
        return psi