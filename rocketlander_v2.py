from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from constants import *
# from control_and_ai.mpc_control import MPC
from itertools import chain
from threading import Timer


# This contact detector is equivalent the one implemented in Lunar Lander
class ContactDetector(contactListener):
    """
    Creates a contact listener to check when the rocket touches down.
    """
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    """
    Continuous VTOL of a rocket.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, settings):
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, -GRAVITY))
        self.mainBase = None
        self.bargeBase = None
        self.CONTACT_FLAG = False

        self.lander = None
        self.particles = []
        self.state = []
        self.prev_shaping = None

        # self.action_space = spaces.Box(-1, +1, (3,))
        if settings.get('Observation Space Size'):
            self.observation_space = spaces.Box(-np.inf, +np.inf, (settings.get('Observation Space Size'),))
        else:
            self.observation_space = spaces.Box(-np.inf, +np.inf, (8,))
        self.lander_tilt_angle_limit = THETA_LIMIT

        self.game_over = False

        self.settings = settings
        self.dynamicLabels = {}
        self.staticLabels = {}

        self.impulsePos = (0, 0)

        self.action_space = [0, 0, 0]  # Main Engine, Nozzle Angle, Left/Right Engine
        self.untransformed_state = [0] * 6 # Non-normalized state

        self.reset()

    """ INHERITED """

    def _seed(self, seed=None):
        self.np_random, returned_seed = seeding.np_random(seed)
        return returned_seed

    def _reset(self):
        self._destroy()
        self.game_over = False
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        smoothedTerrainEdges, terrainDividerCoordinates_x = self._createTerrain(TERRAIN_CHUNKS)

        self.initialMass = 0
        self.remainingFuel = 0
        self.prev_shaping = 0
        self.CONTACT_FLAG = False

        # Engine Stats
        self.action_history = []

        # gradient of 0.009
        # Reference y-trajectory
        self.y_pos_ref = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
        self.y_pos_speed = [-1.9, -1.8, -1.64, -1.5, -1.5, -1.3, -1.0, -0.9]
        self.y_pos_flags = [False for _ in self.y_pos_ref]

        # Create the simulation objects
        self._createClouds()
        self._createBarge()
        self._createBaseStaticEdges(TERRAIN_CHUNKS, smoothedTerrainEdges, terrainDividerCoordinates_x)

        # Adjust the initial coordinates of the rocket
        initial_coordinates = self.settings.get('Initial Coordinates')
        if initial_coordinates is not None:
            xx, yy, randomness_degree, normalized = initial_coordinates
            x = xx * W + np.random.uniform(-randomness_degree, randomness_degree)
            y = yy * H + np.random.uniform(-randomness_degree, randomness_degree)
            if not normalized:
                x = x / W
                y = y / H
        else:
            x, y = W / 2 + np.random.uniform(-0.1, 0.1), H / self.settings['Starting Y-Pos Constant']
        self.initial_coordinates = (x, y)

        self._createRocket(self.initial_coordinates)

        # @TODO: Replace Initial Coordinates by Initial State
        if self.settings.get('Initial State'):
            x, y, x_dot, y_dot, theta, theta_dot = self.settings.get('Initial State')
            self.adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        # Step through one action = [0, 0, 0] and return the state, reward etc.
        return self._step(np.array([0, 0, 0]))[0]

    def _destroy(self):
        if not self.mainBase: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.mainBase)
        self.mainBase = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _step(self, action):
        assert len(action) == 3  # Fe, Fs, psi

        # Check for contact with the ground
        if (self.legs[0].ground_contact or self.legs[1].ground_contact) and self.CONTACT_FLAG == False:
            self.CONTACT_FLAG = True

        # Shutdown all Engines upon contact with the ground
        if self.CONTACT_FLAG:
            action = [0, 0, 0]

        if self.settings.get('Vectorized Nozzle'):
            part = self.nozzle
            part.angle = self.lander.angle + float(action[2])  # This works better than motorSpeed
            if part.angle > NOZZLE_ANGLE_LIMIT:
                part.angle = NOZZLE_ANGLE_LIMIT
            elif part.angle < -NOZZLE_ANGLE_LIMIT:
                part.angle = -NOZZLE_ANGLE_LIMIT
            # part.joint.motorSpeed = float(action[2]) # action[2] is in radians
            # That means having a value of 2*pi will rotate at 360 degrees / second
            # A transformation can be done on the action, such as clipping the value
        else:
            part = self.lander

        # "part" is used to decide where the main engine force is applied (whether it is applied to the bottom of the
        # nozzle or the bottom of the first stage rocket

        # Main Force Calculations
        if self.remainingFuel == 0:
            logging.info("Strictly speaking, you're out of fuel, but act anyway.")
        m_power = self.__mainEngineForceComputation(action, rocketPart=part)
        s_power, engine_dir = self.__sideEnginesForceComputation(action)

        if self.settings.get('Gather Stats'):
            self.action_history.append([m_power, s_power * engine_dir, part.angle])

        # Decrease the rocket ass
        self._decreaseMass(m_power, s_power)

        # State Vector
        self.previous_state = self.state  # Keep a record of the previous state
        state, self.untransformed_state = self.__generateState()  # Generate state
        self.state = state  # Keep a record of the new state

        # Rewards for reinforcement learning
        reward = self.__computeRewards(state, m_power, s_power,
                                       part.angle)  # part angle can be used as part of the reward

        # Check if the game is done, adjust reward based on the final state of the body
        state_reset_conditions = [
            self.game_over,  # Evaluated depending on body contact
            abs(state[XX]) >= 1.0,  # Rocket moves out of x-space
            state[YY] < 0 or state[YY] > 1.3,  # Rocket moves out of y-space or below barge
            abs(state[THETA]) > THETA_LIMIT]  # Rocket tilts greater than the "controllable" limit
        done = False
        if any(state_reset_conditions):
            done = True
            reward = -10
        if not self.lander.awake:
            done = True
            reward = +10

        self._updateParticles()

        return np.array(state), reward, done, {}  # {} = info (required by parent class)

    """ PROBLEM SPECIFIC - PHYSICS, STATES, REWARDS"""

    def __mainEngineForceComputation(self, action, rocketPart, *args):
        # ----------------------------------------------------------------------------
        # Nozzle Angle Adjustment

        # For readability
        sin = math.sin(rocketPart.angle)
        cos = math.cos(rocketPart.angle)

        # Random dispersion for the particles
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Main engine
        m_power = 0
        try:
            if (action[0] > 0.0):
                # Limits
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.3  # 0.5..1.0
                assert m_power >= 0.3 and m_power <= 1.0
                # ------------------------------------------------------------------------
                ox = sin * (4 / SCALE + 2 * dispersion[0]) - cos * dispersion[
                    1]  # 4 is move a bit downwards, +-2 for randomness
                oy = -cos * (4 / SCALE + 2 * dispersion[0]) - sin * dispersion[1]
                impulse_pos = (rocketPart.position[0] + ox, rocketPart.position[1] + oy)

                # rocketParticles are just a decoration, 3.5 is here to make rocketParticle speed adequate
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power,
                                          radius=7)

                rocketParticleImpulse = (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power)
                bodyImpulse = (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power)
                point = impulse_pos
                wake = True

                # Force instead of impulse. This enables proper scaling and values in Newtons
                p.ApplyForce(rocketParticleImpulse, point, wake)
                rocketPart.ApplyForce(bodyImpulse, point, wake)
        except:
            print("Error in main engine power.")

        return m_power

    def __sideEnginesForceComputation(self, action):
        # ----------------------------------------------------------------------------
        # Side engines
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        sin = math.sin(self.lander.angle)  # for readability
        cos = math.cos(self.lander.angle)
        s_power = 0.0
        y_dir = 1 # Positioning for the side Thrusters
        engine_dir = 0
        if (self.settings['Side Engines']):  # Check if side gas thrusters are enabled
            if (np.abs(action[1]) > 0.5): # Have to be > 0.5
                # Orientation engines
                engine_dir = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0

                # if (self.lander.worldCenter.y > self.lander.position[1]):
                #     y_dir = 1
                # else:
                #     y_dir = -1

                # Positioning
                constant = (LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET) / SCALE
                dx_part1 = - sin * constant  # Used as reference for dy
                dx_part2 = - cos * engine_dir * SIDE_ENGINE_AWAY / SCALE
                dx = dx_part1 + dx_part2

                dy = np.sqrt(
                    np.square(constant) - np.square(dx_part1)) * y_dir - sin * engine_dir * SIDE_ENGINE_AWAY / SCALE

                # Force magnitude
                oy = -cos * dispersion[0] - sin * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)
                ox = sin * dispersion[0] - cos * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)

                # Impulse Position
                impulse_pos = (self.lander.position[0] + dx,
                               self.lander.position[1] + dy)

                # Plotting purposes only
                self.impulsePos = (self.lander.position[0] + dx, self.lander.position[1] + dy)

                try:
                    p = self._create_particle(1, impulse_pos[0], impulse_pos[1], s_power, radius=3)
                    p.ApplyForce((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
                    self.lander.ApplyForce((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)
                except:
                    logging.error("Error due to Nan in calculating y during sqrt(l^2 - x^2). "
                                  "x^2 > l^2 due to approximations on the order of approximately 1e-15.")

        return s_power, engine_dir

    def __generateState(self):
        # ----------------------------------------------------------------------------
        # Update
        self.world.Step(1.0 / FPS, 6 * 30, 6 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        target = (self.initialBargeCoordinates[1][0] - self.initialBargeCoordinates[0][0]) / 2 + \
                 self.initialBargeCoordinates[0][0]
        state = [
            (pos.x - target) / (W / 2),
            (pos.y - (self.maximumBargeHeight + (LEG_DOWN / SCALE))) / (W / 2) - LANDING_VERTICAL_CALIBRATION,
            # affects controller
            # self.bargeHeight includes height of helipad
            vel.x * (W / 2) / FPS,
            vel.y * (H / 2) / FPS,
            self.lander.angle,
            # self.nozzle.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]

        untransformed_state = [pos.x, pos.y, vel.x, vel.y, self.lander.angle, self.lander.angularVelocity]

        return state, untransformed_state

    # ['dx','dy','x_vel','y_vel','theta','theta_dot','left_ground_contact','right_ground_contact']
    def __computeRewards(self, state, main_engine_power, side_engine_power, part_angle):
        reward = 0
        shaping = -200 * np.sqrt(np.square(state[0]) + np.square(state[1])) \
                  - 100 * np.sqrt(np.square(state[2]) + np.square(state[3])) \
                  - 1000 * abs(state[4]) - 30 * abs(state[5]) \
                  + 20 * state[6] + 20 * state[7]

        # Introduce the concept of options by making reference markers wrt altitude and speed
        # if (state[4] < 0.052 and state[4] > -0.052):
        #     for i, (pos, speed, flag) in enumerate(zip(self.y_pos_ref, self.y_pos_speed, self.y_pos_flags)):
        #         if state[1] < pos and state[3] > speed and flag is False:
        #             shaping = shaping + 20
        #             self.y_pos_flags[i] = True
        #
        #         elif state[1] < pos and state[3] < speed and flag is False:
        #             shaping = shaping - 20
        #             self.y_pos_flags[i] = True

        # penalize increase in altitude
        if (state[3] > 0):
            shaping = shaping - 1

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # penalize the use of engines
        reward += -main_engine_power * 0.3
        if self.settings['Side Engines']:
            reward += -side_engine_power * 0.3
        # if self.settings['Vectorized Nozzle']:
        #     reward += -100*np.abs(nozzle_angle) # Psi

        return reward / 10

    """ PROBLEM SPECIFIC - RENDERING and OBJECT CREATION"""

    # Problem specific - LINKED
    def _createTerrain(self, CHUNKS):
        # Terrain Coordinates
        # self.helipad_x1 = W / 5
        # self.helipad_x2 = self.helipad_x1 + W / 5
        divisor_constant = 8  # Control the height of the sea
        self.helipad_y = H / divisor_constant

        # Terrain
        # height = self.np_random.uniform(0, H / 6, size=(CHUNKS + 1,))
        height = np.random.normal(H / divisor_constant, 0.5, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        # self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        # self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y

        return [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)], chunk_x  # smoothed Y

    # Problem specific - LINKED
    def _createRocket(self, initial_coordinates=(W / 2, H / 1.2)):
        bodyColor = (1, 1, 1)
        # ----------------------------------------------------------------------------------------
        # LANDER

        initial_x, initial_y = initial_coordinates
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = bodyColor
        self.lander.color2 = (0, 0, 0)

        if isinstance(self.settings['Initial Force'], str):
            self.lander.ApplyForceToCenter((
                self.np_random.uniform(-INITIAL_RANDOM * 0.3, INITIAL_RANDOM * 0.3),
                self.np_random.uniform(-1.3 * INITIAL_RANDOM, -INITIAL_RANDOM)
            ), True)
        else:
            self.lander.ApplyForceToCenter(self.settings['Initial Force'], True)

        # COG is set in the middle of the polygon by default. x = 0 = middle.
        # self.lander.mass = 25
        # self.lander.localCenter = (0, 3) # COG
        # ----------------------------------------------------------------------------------------
        # LEGS
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=5.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x005)
            )
            leg.ground_contact = False
            leg.color1 = bodyColor
            leg.color2 = (0, 0, 0)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(-i * 0.3 / LANDER_CONSTANT, 0),
                localAnchorB=(i * 0.5 / LANDER_CONSTANT, LEG_DOWN),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = 40 * DEGTORAD
                rjd.upperAngle = 45 * DEGTORAD
            else:
                rjd.lowerAngle = -45 * DEGTORAD
                rjd.upperAngle = -40 * DEGTORAD
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        # ----------------------------------------------------------------------------------------
        # NOZZLE
        self.nozzle = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in NOZZLE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x003,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.nozzle.color1 = (0, 0, 0)
        self.nozzle.color2 = (0, 0, 0)
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0.2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=NOZZLE_TORQUE,
            motorSpeed=0,
            referenceAngle=0,
            lowerAngle=-13 * DEGTORAD,  # +- 15 degrees limit applied in practice
            upperAngle=13 * DEGTORAD
        )
        # The default behaviour of a revolute joint is to rotate without resistance.
        self.nozzle.joint = self.world.CreateJoint(rjd)
        # ----------------------------------------------------------------------------------------
        # self.drawlist = [self.nozzle] + [self.lander] + self.legs
        self.drawlist = self.legs + [self.nozzle] + [self.lander]
        self.initialMass = self.lander.mass
        self.remainingFuel = INITIAL_FUEL_MASS_PERCENTAGE * self.initialMass
        return

    # Problem specific - LINKED
    def _createBarge(self):
        # Landing Barge
        # The Barge can be modified in shape and angle
        self.bargeHeight = self.helipad_y * (1 + 0.6)

        # self.landingBargeCoordinates = [(self.helipad_x1, 0.1), (self.helipad_x2, 0.1),
        #                                 (self.helipad_x2, self.bargeHeight), (self.helipad_x1, self.bargeHeight)]
        assert BARGE_LENGTH_X1_RATIO < BARGE_LENGTH_X2_RATIO, 'Barge Length X1 must be 0-1 and smaller than X2'

        x1 = BARGE_LENGTH_X1_RATIO*W
        x2 = BARGE_LENGTH_X2_RATIO*W
        self.landingBargeCoordinates = [(x1, 0.1), (x2, 0.1),
                                        (x2, self.bargeHeight), (x1, self.bargeHeight)]

        self.initialBargeCoordinates = self.landingBargeCoordinates
        self.minimumBargeHeight = min(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        self.maximumBargeHeight = max(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        # Used for the actual area inside the Barge
        # bargeLength = self.helipad_x2 - self.helipad_x1
        # padRatio = 0.2
        # self.landingPadCoordinates = [self.helipad_x1 + bargeLength * padRatio,
        #                               self.helipad_x2 - bargeLength * padRatio]
        bargeLength = x2 - x1
        padRatio = 0.2
        self.landingPadCoordinates = [x1 + bargeLength * padRatio,
                                      x2 - bargeLength * padRatio]

        self.landing_coordinates = self.get_landing_coordinates()

    # Problem specific - LINKED
    def _createBaseStaticEdges(self, CHUNKS, smooth_y, chunk_x):
        # Sky
        self.sky_polys = []
        # Ground
        self.ground_polys = []
        self.sea_polys = [[] for i in range(SEA_CHUNKS)]

        # Main Base
        self.mainBase = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self._createStaticEdges(self.mainBase, [p1, p2], 0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

            self.ground_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])

            for i in range(SEA_CHUNKS - 1):
                k = 1 - (i + 1) / SEA_CHUNKS
                self.sea_polys[i].append([(p1[0], p1[1] * k), (p2[0], p2[1] * k), (p2[0], 0), (p1[0], 0)])

        self._updateBargeStaticEdges()

    def _updateBargeStaticEdges(self):
        if self.bargeBase is not None:
            self.world.DestroyBody(self.bargeBase)
        self.bargeBase = None
        bargeEdgeCoordinates = [self.landingBargeCoordinates[2], self.landingBargeCoordinates[3]]
        self.bargeBase = self.world.CreateStaticBody(shapes=edgeShape(vertices=bargeEdgeCoordinates))
        self._createStaticEdges(self.bargeBase, bargeEdgeCoordinates, friction=BARGE_FRICTION)

    def _createStaticEdges(self, base, vertices, friction):
        base.CreateEdgeFixture(
            vertices=vertices,
            density=0,
            friction=friction)
        return

    def _create_particle(self, mass, x, y, ttl, radius=3):
        """
        Used for both the Main Engine and Side Engines
        :param mass: Different mass to represent different forces
        :param x: x position
        :param y:  y position
        :param ttl:
        :param radius:
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _createCloud(self, x_range, y_range, y_variance=0.1):
        self.cloud_poly = []
        numberofdiscretepoints = 3

        initial_y = (VIEWPORT_H * np.random.uniform(y_range[0], y_range[1], 1)) / SCALE
        initial_x = (VIEWPORT_W * np.random.uniform(x_range[0], x_range[1], 1)) / SCALE

        Y_coordinates = np.random.normal(0, y_variance, numberofdiscretepoints)
        x_step = np.linspace(initial_x, initial_x + np.random.uniform(1, 6), numberofdiscretepoints + 1)

        for i in range(0, numberofdiscretepoints):
            self.cloud_poly.append((x_step[i], initial_y + math.sin(3.14 * 2 * i / 50) * Y_coordinates[i]))

        return self.cloud_poly

    def _createClouds(self):
        self.clouds = []
        for i in range(10):
            self.clouds.append(self._createCloud([0.2, 0.4], [0.65, 0.7], 1))
            self.clouds.append(self._createCloud([0.7, 0.8], [0.75, 0.8], 1))

    def _decreaseMass(self, main_engine_power, side_engine_power):
        x = np.array([float(main_engine_power), float(side_engine_power)])
        consumedFuel = 0.009 * np.sum(x * (MAIN_ENGINE_FUEL_COST, SIDE_ENGINE_FUEL_COST)) / SCALE
        self.lander.mass = self.lander.mass - consumedFuel
        self.remainingFuel -= consumedFuel
        if self.remainingFuel < 0:
            self.remainingFuel = 0

    def _createLabels(self, labels):
        labels_dict = {}
        y_spacing = 0
        for text in labels:
            labels_dict[text] = pyglet.text.Label(text, font_size=15, x=W / 2, y=H / 2,  # - y_spacing*H/10,
                                                  anchor_x='right', anchor_y='center', color=(0, 255, 0, 255))
            y_spacing += 1
        return labels_dict

    """ RENDERING """

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # This can be removed since the code is being update to utilize env.refresh() instead
        # Kept here for backwards compatibility purposes
        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        self._renderEnvironment()
        self._renderLander()
        self.drawMarker(x=self.lander.worldCenter.x, y=self.lander.worldCenter.y)  # Center of Gravity
        # self.drawMarker(x=self.impulsePos[0], y=self.impulsePos[1])              # Side Engine Forces Positions
        # self.drawMarker(x=self.lander.position[0], y=self.lander.position[1])    # (0,0) position

        # Commented out to be able to draw from outside the class using self.refresh
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def refresh(self, mode='human', render=False):
        """
        Used instead of _render in order to draw user defined drawings from controllers, e.g. trajectories
        for the MPC or a a marking e.g. Center of Gravity
        :param mode:
        :param render:
        :return: Viewer
        """
        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        if render:
            self.render()
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _renderLander(self):
        # --------------------------------------------------------------------------------------------------------------
        # Rocket Lander
        # --------------------------------------------------------------------------------------------------------------
        # Lander and Particles
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False,
                                            linewidth=2).add_attr(t)
                else:
                    # Lander
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    def _renderClouds(self):
        for x in self.clouds:
            self.viewer.draw_polygon(x, color=(1.0, 1.0, 1.0))

    def _updateParticles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

    def _renderEnvironment(self):
        # --------------------------------------------------------------------------------------------------------------
        # ENVIRONMENT
        # --------------------------------------------------------------------------------------------------------------
        # Sky Boundaries

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0.83, 0.917, 1.0))

        # Landing Barge
        self.viewer.draw_polygon(self.landingBargeCoordinates, color=(0.1, 0.1, 0.1))

        for g in self.ground_polys:
            self.viewer.draw_polygon(g, color=(0, 0.5, 1.0))

        for i, s in enumerate(self.sea_polys):
            k = 1 - (i + 1) / SEA_CHUNKS
            for poly in s:
                self.viewer.draw_polygon(poly, color=(0, 0.5 * k, 1.0 * k + 0.5))

        if self.settings["Clouds"]:
            self._renderClouds()

        # Landing Flags
        for x in self.landingPadCoordinates:
            flagy1 = self.landingBargeCoordinates[3][1]
            flagy2 = self.landingBargeCoordinates[2][1] + 25 / SCALE

            polygonCoordinates = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
            self.viewer.draw_polygon(polygonCoordinates, color=(1, 0, 0))
            self.viewer.draw_polyline(polygonCoordinates, color=(0, 0, 0))
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0.5, 0.5, 0.5))

        # --------------------------------------------------------------------------------------------------------------

    """ CALLABLE DURING RUNTIME """

    def drawMarker(self, x, y):
        """
        Draws a black '+' sign at the x and y coordinates.
        :param x: normalized x position (0-1)
        :param y: normalized y position (0-1)
        :return:
        """
        offset = 0.2
        self.viewer.draw_polyline([(x, y - offset), (x, y + offset)], linewidth=2)
        self.viewer.draw_polyline([(x - offset, y), (x + offset, y)], linewidth=2)

    def drawPolygon(self, color=(0.2, 0.2, 0.2), **kwargs):
        # path expected as (x,y)
        if self.viewer is not None:
            path = kwargs.get('path')
            if path is not None:
                self.viewer.draw_polygon(path, color=color)
            else:
                x = kwargs.get('x')
                y = kwargs.get('y')
                self.viewer.draw_polygon([(xx, yy) for xx, yy in zip(x, y)], color=color)

    def drawLine(self, x, y, color=(0.2, 0.2, 0.2)):
        self.viewer.draw_polyline([(xx, yy) for xx, yy in zip(x, y)], linewidth=2, color=color)

    def move_barge(self, x_movement, left_height, right_height):
        self.landingBargeCoordinates[0] = (
        self.landingBargeCoordinates[0][0] + x_movement, self.landingBargeCoordinates[0][1])
        self.landingBargeCoordinates[1] = (
        self.landingBargeCoordinates[1][0] + x_movement, self.landingBargeCoordinates[1][1])
        self.landingBargeCoordinates[2] = (
        self.landingBargeCoordinates[2][0] + x_movement, self.landingBargeCoordinates[2][1] + right_height)
        self.landingBargeCoordinates[3] = (
        self.landingBargeCoordinates[3][0] + x_movement, self.landingBargeCoordinates[3][1] + left_height)
        self.minimumBargeHeight = min(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        self.maximumBargeHeight = max(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        self._updateBargeStaticEdges()
        self.update_landing_coordinate(x_movement, x_movement)
        self.landing_coordinates = self.get_landing_coordinates()
        return self.landing_coordinates

    def get_consumed_fuel(self):
        if self.lander is not None:
            return self.initialMass - self.lander.mass

    def get_landing_coordinates(self):
        x = (self.landingBargeCoordinates[1][0] - self.landingBargeCoordinates[0][0]) / 2 + \
            self.landingBargeCoordinates[0][0]
        y = abs(self.landingBargeCoordinates[2][1] - self.landingBargeCoordinates[3][1]) / 2 + \
            min(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        return [x, y]

    def get_barge_top_edge_points(self):
        return flatten_array(self.landingBargeCoordinates[2:])

    def get_state_with_barge_and_landing_coordinates(self, untransformed_state=False):
        if untransformed_state:
            state = self.untransformed_state
        else:
            state = self.state
        return flatten_array([state, [self.remainingFuel, self.lander.mass], self.get_barge_top_edge_points(),
                              self.get_landing_coordinates()])

    def get_barge_to_ground_distance(self):
        """
        Calculates the max barge height offset from the start to end
        :return:
        """
        initialBargeCoordinates = np.array(self.initialBargeCoordinates)
        currentBargeCoordinates = np.array(self.landingBargeCoordinates)

        bargeHeightOffset = initialBargeCoordinates[:, 1] - currentBargeCoordinates[:, 1]
        return np.max(bargeHeightOffset)

    def update_landing_coordinate(self, leftLanding_x, rightLanding_x):
        self.landingPadCoordinates[0] += leftLanding_x
        self.landingPadCoordinates[1] += rightLanding_x

        x_lim_1 = self.landingBargeCoordinates[0][0]
        x_lim_2 = self.landingBargeCoordinates[1][0]

        if (self.landingPadCoordinates[0] <= x_lim_1):
            self.landingPadCoordinates[0] = x_lim_1

        if (self.landingPadCoordinates[1] >= x_lim_2):
            self.landingPadCoordinates[1] = x_lim_2

    def get_action_history(self):
        return self.action_history

    def clear_forces(self):
        self.world.ClearForces()

    def get_nozzle_and_lander_angles(self):
        assert self.nozzle is not None, "Method called prematurely before initialization"
        return np.array([self.nozzle.angle, self.lander.angle, self.nozzle.joint.angle])

    def evaluate_kinematics(self, actions):
        Fe, Fs, psi = actions
        theta = self.untransformed_state[THETA]
        # -----------------------------
        ddot_x = (Fe * theta + Fe * psi + Fs) / MASS
        ddot_y = (Fe - Fe * theta * psi - Fs * theta - MASS * GRAVITY) / MASS
        ddot_theta = (Fe * psi * (L1 + LN) - L2 * Fs) / INERTIA
        return ddot_x, ddot_y, ddot_theta

    def apply_random_x_disturbance(self, epsilon, left_or_right, x_force=2000):
        if np.random.rand() < epsilon:
            if (left_or_right):
                self.apply_disturbance('random', x_force, 0)
            else:
                self.apply_disturbance('random', -x_force, 0)

    def apply_random_y_disturbance(self, epsilon, y_force=2000):
        if np.random.rand() < epsilon:
            self.apply_disturbance('random', 0, -y_force)

    def move_barge_randomly(self, epsilon, left_or_right, x_movement=0.05):
        if np.random.rand() < epsilon:
            if (left_or_right):
                self.move_barge(x_movement=x_movement, left_height=0, right_height=0)
            else:
                self.move_barge(x_movement=-x_movement, left_height=0, right_height=0)

    def adjust_dynamics(self, **kwargs):
        if kwargs.get('mass'):
            self.lander.mass = kwargs['mass']

        if kwargs.get('x_dot'):
            self.lander.linearVelocity.x = kwargs['x_dot']

        if kwargs.get('y_dot'):
            self.lander.linearVelocity.y = kwargs['y_dot']

        if kwargs.get('theta'):
            self.lander.angle = kwargs['theta']

        if kwargs.get('theta_dot'):
            self.lander.angularVelocity = kwargs['theta_dot']

        self.state, self.untransformed_state = self.__generateState()

    def apply_disturbance(self, force, *args):
        if force is not None:
            if isinstance(force, str):
                x, y = args
                self.lander.ApplyForceToCenter((
                    self.np_random.uniform(x),
                    self.np_random.uniform(y)
                ), True)
            elif isinstance(force, tuple):
                self.lander.ApplyForceToCenter(force, True)

    def compute_cost(self, state, untransformed_state=False, *args):
        len_state = len(state)
        cost_matrix = np.ones(len_state)
        cost_matrix[XX] = 10
        cost_matrix[X_DOT] = 5
        cost_matrix[Y_DOT] = 10
        cost_matrix[THETA] = 4
        cost_matrix[THETA_DOT] = 10

        state_target = np.zeros(len_state)
        if untransformed_state is True:
            state_target[XX] = args[XX]
            state_target[YY] = args[YY]

        ss = (state_target - abs(np.array(state)))
        return np.dot(ss, cost_matrix)


def get_state_sample(samples, normal_state=True, untransformed_state=True):
    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}
    env = RocketLander(simulation_settings)
    env.reset()
    state_samples = []
    while (len(state_samples) < samples):
        F_main = np.random.uniform(0, 1)
        F_side = np.random.uniform(-1, 1)
        psi = np.random.uniform(-90 * DEGTORAD, 90 * DEGTORAD)
        action = [F_main, F_side, psi]
        s, r, done, info = env.step(action)
        if normal_state:
            state_samples.append(s)
        else:
            state_samples.append(
                env.get_state_with_barge_and_landing_coordinates(untransformed_state=untransformed_state))
        if done:
            env.reset()
    env.close()
    return state_samples


def flatten_array(the_list):
    return list(chain.from_iterable(the_list))


def compute_derivatives(state, action, sample_time=1 / FPS):
    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': (0, 0)}

    eps = sample_time
    len_state = len(state)
    len_action = len(action)
    ss = np.tile(state, (len_state, 1))
    x1 = ss + np.eye(len_state) * eps
    x2 = ss - np.eye(len_state) * eps
    aa = np.tile(action, (len_state, 1))
    f1 = simulate_kinematics(x1, aa, simulation_settings)
    f2 = simulate_kinematics(x2, aa, simulation_settings)
    delta_x = f1 - f2
    delta_A = delta_x / 2 / eps  # Jacobian

    x3 = np.tile(state, (len_action, 1))
    u1 = np.tile(action, (len_action, 1)) + np.eye(len_action) * eps
    u2 = np.tile(action, (len_action, 1)) - np.eye(len_action) * eps
    f1 = simulate_kinematics(x3, u1, simulation_settings)
    f2 = simulate_kinematics(x3, u2, simulation_settings)
    delta_B = (f1 - f2) / 2 / eps
    delta_B = delta_B.T

    return delta_A, delta_B, delta_x


def simulate_kinematics(state, action, simulation_settings, render=False):
    next_state = np.zeros(state.shape)
    envs = [None for _ in range(len(state))]  # separate environment for memory management
    for i, (s, a) in enumerate(zip(state, action)):
        x, y, x_dot, y_dot, theta, theta_dot = s
        simulation_settings['Initial Coordinates'] = (x, y, 0, False)

        envs[i] = RocketLander(simulation_settings)
        if render:
            envs[i].render()
        envs[i].adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        ss, r, done, info = envs[i].step(a)
        if render:
            envs[i].render()
        next_state[i, :] = envs[i].untransformed_state
        envs[i].close()

    return next_state


def swap_array_values(array, indices_to_swap):
    for i, j in indices_to_swap:
        array[i], array[j] = array[j], array[i]
    return array


if __name__ == "__main__":

    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random',  # (6000, -10000)
                'Rows': 1,
                'Columns': 2}

    env = RocketLander(settings)
    s = env.reset()

    from control_and_ai.pid import PID_Benchmark

    pid = PID_Benchmark()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05
    total_reward = 0
    while (1):
        a = pid.pid_algorithm(s)
        s, r, done, info = env.step(a)
        total_reward += r
        # -------------------------------------
        env.render()
        env.drawMarker(env.landing_coordinates[0], env.landing_coordinates[1])
        env.refresh(render=False)

        if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
            env.move_barge_randomly(epsilon, left_or_right_barge_movement)
            env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
            env.apply_random_y_disturbance(epsilon=0.005)

        if done:
            print('Total Reward:\t{0}'.format(total_reward))
            total_reward = 0
            env.reset()
