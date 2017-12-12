import scipy.linalg
from constants import *
from environments.rocketlander import simulate_kinematics, RocketLander


class LQR_Control():
    def __init__(self, env, continuous_solver=False):
        self.state_len = len(env.state[:-2]) # last 2 states are not used
        self.action_len = len(env.action_space)
        self._create_Q_R_matrices(100, 1000)
        self.continuous_solver = continuous_solver

    def solve_continuous_lqr(self, A, B, Q, R):
        """Solve the continuous time lqr controller.
        dx/dt = A x + B u
        cost = integral x.T*Q*x + u.T*R*u
        reference: http://www.mwm.im/lqr-controllers-with-python/
        @NOTE: Control library can also be used
        """

        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

        eigVals, eigVecs = scipy.linalg.eig(A - B * K)
        #K,X,eigVals = control.lqr(A,B,Q,R)
        return np.array(K), np.array(X), eigVals

    def solve_discrete_lqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        reference: http://www.mwm.im/lqr-controllers-with-python/
        """

        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))
        eigVals, eigVecs = scipy.linalg.eig(A - B * K)
        return np.array(K), np.array(X), eigVals

    def _create_Q_R_matrices(self, Q_matrix_weight, R_matrix_weight):
        # state = x,y,x_dot,y_dot,theta,theta_dot
        self.Q = np.zeros((self.state_len, self.state_len))
        self.Q[:self.state_len, :self.state_len] = np.eye(self.state_len)
        self.Q[4,4] = self.Q[5,5] = Q_matrix_weight
        self.R = np.eye(self.action_len)
        self.R[2,2] = R_matrix_weight

    def evaluate_control(self, env, state, action, target_state=np.zeros(6).T):
        assert len(state) == self.state_len
        x_error = state - target_state
        A, B = self.compute_derivatives(state, action)
        if self.continuous_solver:
            K, X, _ = self.solve_continuous_lqr(A, B, self.Q, self.R)
        else:
            K, X, _ = self.solve_discrete_lqr(A, B, self.Q, self.R)
        u = -np.array(np.dot(K, env.untransformed_state)).squeeze()
        # if u[2] > 0:
        #     u[2] = u[2] % (15*DEGTORAD)
        # else:
        #     u[2] = -(u[2] % (15 * DEGTORAD))
        u[2] = u[2]/100
        u[0] = u[0]+env.lander.mass*GRAVITY
        return u

    def compute_derivatives(self, state, action):
        simulation_settings = {'Side Engines': True,
                               'Clouds': False,
                               'Vectorized Nozzle': True,
                               'Graph': False,
                               'Render': False,
                               'Starting Y-Pos Constant': 1,
                               'Initial Force': (0, 0)}

        eps = 1 / FPS
        len_state = self.state_len
        len_action = self.action_len
        ss = np.tile(state, (len_state, 1))
        x1 = ss + np.eye(len_state) * eps
        x2 = ss - np.eye(len_state) * eps
        aa = np.tile(action, (len_state, 1))
        f1 = simulate_kinematics(x1, aa, simulation_settings)
        f2 = simulate_kinematics(x2, aa, simulation_settings)
        delta_A = (f1 - f2) / 2 / eps  # Jacobian

        x3 = np.tile(state, (len_action, 1))
        u1 = np.tile(action, (len_action, 1)) + np.eye(len_action) * eps
        u2 = np.tile(action, (len_action, 1)) - np.eye(len_action) * eps
        f1 = simulate_kinematics(x3, u1, simulation_settings)
        f2 = simulate_kinematics(x3, u2, simulation_settings)
        delta_B = (f1 - f2) / 2 / eps
        delta_B = delta_B.T

        return delta_A, delta_B

def run(env, simulation_settings, controller):
    s = env.state
    if simulation_settings['Graph']:
        data = []
        handles = RealTime_Graph_Thread(simulation_settings)
        handles.start()

    total_reward = 0
    episodes = 0
    max_episodes = 10
    a = [0,0,0]
    while(episodes < max_episodes):
        a = controller.evaluate_control(env, env.untransformed_state, a)
        s, r, done, info = env.step(a)
        total_reward += r

        if simulation_settings['Render']:
            env.render()

        if simulation_settings['Graph']:
            if handles.isAlive():
                handles.data[0].append(s[3])
                handles.data[1].append(s[1])
        # print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(s[0],s[1],s[2],s[3],s[4],s[5],env.lander.position.x, env.lander.position.y ))
        if done:
            env.reset()
            print("Total Reward:\t{0}".format(total_reward))
            total_reward = 0
            episodes += 1
    print(env.lander.mass)

simulation_settings = {'Side Engines': True,
            'Clouds': False,
            'Vectorized Nozzle': True,
            'Graph': False,
            'Render': True,
            'Starting Y-Pos Constant': 1,
            'Initial Force': 'random',
            'Rows': 1,
            'Columns': 2}
env = RocketLander(simulation_settings)
lqr_controller = LQR_Control(env, continuous_solver=False)
run(env, simulation_settings, lqr_controller)
