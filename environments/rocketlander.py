import sys, math
import numpy as np

import Box2D
# Ignore the warning "Unresolved Reference" in this case.
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from gym.envs.classic_control import rendering
import gym
from gym import spaces
from gym.utils import seeding


FPS = 60

INITIAL_RANDOM = 9000.0

LANDER_CONSTANT = 1
LANDER_LENGTH = 120/LANDER_CONSTANT
LANDER_RADIUS = 10/LANDER_CONSTANT
LANDER_POLY =[
    (-LANDER_RADIUS,0), (+LANDER_RADIUS,0),
    (+LANDER_RADIUS,+LANDER_LENGTH),(-LANDER_RADIUS,+LANDER_LENGTH)
    ]



LEG_AWAY = 12/LANDER_CONSTANT
LEG_DOWN = 7/LANDER_CONSTANT
LEG_W, LEG_H = 3/LANDER_CONSTANT, 9/LANDER_CONSTANT
LEG_SPRING_TORQUE = LANDER_LENGTH/1.7

MAIN_ENGINE_POWER  = LANDER_LENGTH/(LANDER_CONSTANT*4)
SIDE_ENGINE_POWER  =  1.6/LANDER_CONSTANT

SIDE_ENGINE_HEIGHT = LANDER_LENGTH-5
SIDE_ENGINE_AWAY   = 10.0

VIEWPORT_W = 800
VIEWPORT_H = 600

BARGE_FRICTION = 0.2

SCALE = 30

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class RocketLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0,-14.81))
        self.mainBase = None
        self.bargeBase = None
        
        self.lander = None
        self.particles = []
        self.prev_shaping = None

        self.continuous = True
        self.game_over = False

        self.action_space = [0, 0, 0] # Main Engine, Nozzle Angle, Left/Right Engine
        self.reset()

    """ INHERITED """
    def _seed(self, seed=None):
        self.np_random, returned_seed = seeding.np_random(seed)
        return returned_seed

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

        #assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power, radius=5)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)


        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - ((self.helipad_x2-self.helipad_x1)/2 + self.helipad_x1)) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.minimumBargeHeight + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2), # self.bargeHeight includes height of helipad
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        assert len(state)==8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0 or state[1] < 0:
            done   = True
            reward = -100
        if not self.lander.awake:
            done   = True
            reward = +100
        return np.array(state), reward, done, {}

    def _reset(self):
        self._destroy()
        self.game_over = False
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        numberofTerrainDividers_x = 11
        smoothedTerrainEdges, terrainDividerCoordinates_x = self._createTerrain(numberofTerrainDividers_x)

        self._createBarge()

        self._createBaseStaticEdges(numberofTerrainDividers_x, smoothedTerrainEdges, terrainDividerCoordinates_x)

        self._createRocket()

        return self._step(np.array([0,0]) if self.continuous else 0)[0]

    """ PROBLEM SPECIFIC """
    # Problem specific - LINKED
    def _createTerrain(self, CHUNKS):
        # Terrain Coordinates
        self.helipad_x1 = W / 5
        self.helipad_x2 = self.helipad_x1 + W / 5
        self.helipad_y = H / 6

        # Terrain
        #height = self.np_random.uniform(0, H / 6, size=(CHUNKS + 1,))
        height = np.random.normal(H/6, 0.5, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y

        return [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)], chunk_x # smoothed Y

    # Problem specific - LINKED
    def _createRocket(self):
        bodyColor = (1, 1, 1)

        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
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
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = bodyColor
            leg.color2 = (0, 0, 0)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        return

    # Problem specific - LINKED
    def _createBarge(self):
        # Landing Barge
        # The Barge can be modified in shape and angle
        self.bargeHeight = self.helipad_y * (1 + 0.6)

        self.landingBargeCoordinates = [(self.helipad_x1, self.helipad_y), (self.helipad_x2, self.helipad_y),
                                        (self.helipad_x2, self.bargeHeight), (self.helipad_x1, self.bargeHeight)]

        self.initialBargeCoordinates = self.landingBargeCoordinates
        self.minimumBargeHeight = min(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        # Used for the actual area inside the Barge
        bargeLength = self.helipad_x2 - self.helipad_x1
        padRatio = 0.2
        self.landingPadCoordinates = [self.helipad_x1 + bargeLength * padRatio,
                                      self.helipad_x2 - bargeLength * padRatio]

    # Problem specific - LINKED
    def _createBaseStaticEdges(self, CHUNKS, smooth_y, chunk_x):
        # Sky
        self.sky_polys = []
        # Ground
        self.ground_polys = []

        # Main Base
        self.mainBase = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self._createStaticEdges(self.mainBase, [p1,p2], 0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
            self.ground_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])

        self.updateBargeStaticEdges()

    def updateBargeStaticEdges(self):
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

    def _create_particle(self, mass, x, y, ttl, radius=2):
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
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    """ RENDERING """
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        self._renderParticles()
        self._renderEnvironment()
        self._renderLander()

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

    def _renderParticles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

    def _renderEnvironment(self):
        # --------------------------------------------------------------------------------------------------------------
        # ENVIRONMENT
        # --------------------------------------------------------------------------------------------------------------
        # Sky Boundaries
        for p, g in zip(self.sky_polys, self.ground_polys):
            self.viewer.draw_polygon(p, color=(0.45, 0.7, 1.0))
            self.viewer.draw_polygon(g, color=(0, 0.25, 0.7))

        # Landing Flags
        for x in self.landingPadCoordinates:
            flagy1 = self.landingBargeCoordinates[3][1]
            flagy2 = self.landingBargeCoordinates[2][1] + 25 / SCALE

            polygonCoordinates = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
            self.viewer.draw_polygon(polygonCoordinates, color=(1, 0, 0))
            self.viewer.draw_polyline(polygonCoordinates, color=(0, 0, 0))
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))

        # Landing Barge
        self.viewer.draw_polygon(self.landingBargeCoordinates, color=(0.1, 0.1, 0.1))
        # --------------------------------------------------------------------------------------------------------------

    """ UPDATABLE DURING RUNTIME """
    def updateBargeHeight(self, leftHeight, rightHeight):
        self.landingBargeCoordinates[2] = (
        self.landingBargeCoordinates[2][0], self.landingBargeCoordinates[2][1] + rightHeight)
        self.landingBargeCoordinates[3] = (
        self.landingBargeCoordinates[3][0], self.landingBargeCoordinates[3][1] + leftHeight)
        self.minimumBargeHeight = min(self.landingBargeCoordinates[2][1], self.landingBargeCoordinates[3][1])
        self.updateBargeStaticEdges()
        return self.get_BargetoGroundDistance() # Max Vertical Offset

    def get_BargetoGroundDistance(self):
        initialBargeCoordinates = np.array(self.initialBargeCoordinates)
        currentBargeCoordinates = np.array(self.landingBargeCoordinates)

        bargeHeightOffset = initialBargeCoordinates[:, 1] - currentBargeCoordinates[:, 1]
        return np.max(bargeHeightOffset)

    def updateLandingCoordinate(self, leftLanding_x, rightLanding_x):
        self.landingPadCoordinates[0] += leftLanding_x
        self.landingPadCoordinates[1] += rightLanding_x
        flag = 0

        if (self.landingPadCoordinates[0] <= self.helipad_x1):
            self.landingPadCoordinates[0] = self.helipad_x1
            flag = 1

        if (self.landingPadCoordinates[1] >= self.helipad_x2):
            self.landingPadCoordinates[1] = self.helipad_x2
            flag = 1

        return flag

    def clearForces(self):
        self.world.ClearForces()

def PID(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])  # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

if __name__ == "__main__":
    import gym

    env = RocketLander()
    s = env.reset()

    flag = 0
    i = 0
    while(1):
        i += 1
        a = PID(env, s)
        s, r, done, info = env.step(a)
        env.render()
        #env.clear_forces()
        if i > 100 and i < 200:
            flag = env.updateLandingCoordinate(0.001, 0.001)
            env.updateBargeHeight(-0.008, -0.006)
        if done:
            #input("Press Enter to Restart")
            env.reset()
            i = 0
