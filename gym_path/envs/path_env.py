"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym_path.bot import Bot
from gym_path.path import Path


class PathEnv(gym.Env):
    """
    Loosely adapted from the standard cartpole environment.
    #TODO
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, maximum_error=.2):
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.max_speed = 1.
        self.max_rotational_vel = 1.

        self.maximum_error = maximum_error
        self.x_threshold = 2.4
        self.env_size = 4

        action_limits = np.array([self.max_speed, self.max_rotational_vel])
        self.action_space = spaces.Box(-action_limits, action_limits)  # rotational and forward velocity
        border_offsets = np.array([self.env_size / 2, self.env_size / 2])
        #TODO think about what the observations should be. The path points? The next point to go to? Probably the first,
        # since the second it too easy, then it's just going to a point instead of path following
        self.observation_space = spaces.Box(-border_offsets, border_offsets, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None # state will be of the form: (x,x_dot,theta,theta_dot). Should maybe just be the bot object


        self.path: Path = None # TODO
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        self.bot.apply_action(action[0], action[1], self.tau)

        error_from_path = self.path.distance(self.bot.pose.location)
        error_too_large = error_from_path > self.maximum_error
        goal_reached = self.path.goal_reached(self.bot.pose.location)
        self.done = error_too_large or goal_reached

        #TODO think about this true reward function
        if error_too_large:
            reward = -100.
        elif goal_reached:
            reward = 100.
        else:
            reward = 0. # Give reward for staying on path? Maybe define a certain distance under which it gets a reward

        return np.array(self.state), reward, self.done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        raise NotImplementedError()
        self.path = Path()
        self.bot = Bot(None, self.kinematics_integrator) # TODO spawn at beginning of track (with correct yaw?)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
