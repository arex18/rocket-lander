import logging
import _pickle
from constants import *

""" PID """


class PID():
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0

    def incrementIntegralError(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if (self.accumulated_error > pi_limit):
            self.accumulated_error = pi_limit
        elif (self.accumulated_error < pi_limit):
            self.accumulated_error = -pi_limit

    def computeOutput(self, error, dt_error):
        self.incrementIntegralError(error)
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error


class PID_Framework():
    def compute(self, env, s):
        # Unpack for clarity
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s

        angle_target = dx * 0.5 + vel_x * 1.0  # angle should point towards center (dx is horizontal coordinate, vel_x hor speed)
        if angle_target > 0.4: angle_target = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_target < -0.4: angle_target = -0.4
        hover_target = 0.55 * np.abs(dx)  # target y should be proportional to horizontal offset

        Fe, Fs, psi = self.pid_algorithm(s, [hover_target, angle_target])

        pid_state = [Fe, Fs, psi]

        if env.continuous:
            a = np.array([Fe, Fs, psi])
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if Fe > np.abs(Fs) and Fe > 0.05:
                a = 2
            elif Fs < -0.05:
                a = 3
            elif Fs > +0.05:
                a = 1
            elif psi > 15 * DEGTORAD:
                a = 15 * DEGTORAD
            elif psi < 15 * DEGTORAD:
                a = 15 * DEGTORAD

        return a, pid_state

    @abc.abstractmethod
    def pid_algorithm(self, s, targets):
        NotImplementedError()


class PID_Benchmark(PID_Framework):
    def __init__(self):
        super(PID_Benchmark, self).__init__()
        # self.Fe_PID = PID(10, 0, 10)
        # self.psi_PID = PID(0.05, 0, 0.05)
        # self.Fs_x_PID = PID(5, 0, 6)
        # self.Fs_theta_PID = PID(10, 0.01, 20)

        self.Fe_PID = PID(10, 0, 10)
        self.psi_PID = PID(0.085, 0.001, 10.55)
        # self.Fs_theta_PID = PID(10, 0.01, 20)
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

        Fe = self.Fe_PID.computeOutput(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.computeOutput(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.computeOutput(theta_error, theta_dterror)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi

        # def pid_algorithm(self, s, targets):
        #     dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        #     # ------------------------------------------
        #     y_ref = -0.1  # Adjust speed
        #     y_error = y_ref - dy + 0.1 * dx
        #     y_dterror = -vel_y + 0.1 * vel_x
        #
        #     Fe = self.Fe_PID.computeOutput(y_error, y_dterror)
        #     # ------------------------------------------
        #     x_ref = 0
        #     x_error = x_ref - dx
        #     x_dterror = -vel_x
        #     Fs_x = self.Fs_x_PID.computeOutput(x_error, x_dterror)
        #     # ------------------------------------------
        #     theta_ref = 0
        #     theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        #     theta_dterror = -omega + 0.2 * vel_x
        #     Fs_theta = self.Fs_x_PID.computeOutput(theta_error, theta_dterror)
        #     Fs = -Fs_theta  # + Fs_x
        #     # ------------------------------------------
        #     theta_ref = 0
        #     theta_error = theta_ref - theta - 0.05 * dx
        #     theta_dterror = -omega - 0.05 * vel_x
        #
        #     psi = self.psi_PID.computeOutput(theta_error, theta_dterror)
        #
        #     if legContact_left and legContact_right:  # legs have contact
        #         Fe = 0
        #         Fs = 0
        #
        #     return Fe, Fs, psi


class PID_Heuristic_Benchmark(PID_Framework):
    def __init__(self):
        super(PID_Heuristic_Benchmark, self).__init__()
        self.Fe = PID(10, 0, 10)
        self.psi = PID(0.01, 0, 0.01)
        self.Fs = PID(10, 0, 30)

    def pid_algorithm(self, s, targets):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        # ------------------------------------------
        x_error = targets[1] - theta
        x_dterror = -omega
        Fs = -self.Fs.computeOutput(x_error, x_dterror)
        # ------------------------------------------
        y_error = targets[0] - dy
        y_dterror = -vel_y
        Fe = self.Fe.computeOutput(y_error, y_dterror) - 1
        # ------------------------------------------
        theta_error = theta
        theta_dterror = -omega - vel_x
        psi = self.psi.computeOutput(theta_error, theta_dterror)
        # ------------------------------------------
        if legContact_left and legContact_right:  # legs have contact
            Fe = 0

        return Fe, Fs, psi


class PID_psi(PID_Framework):
    def __init__(self):
        super(PID_psi, self).__init__()
        self.psi = PID(0.1, 0, 0.01)

    def pid_algorithm(self, s, targets):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s
        theta_error = theta
        theta_dterror = -omega - vel_x
        psi = self.psi.computeOutput(theta_error, theta_dterror)
        return psi