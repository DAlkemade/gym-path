"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from abc import abstractmethod

import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding

from gym_path.bot import Bot
from gym_path.coordination import Point
from gym_path.path import Path


class PathEnvShared(gym.Env):
    """Provides the based shared functionality for all environments in this package."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, clean_viewer, maximum_error=.25, goal_reached_threshold=.2):
        self.clean_viewer = clean_viewer
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.max_speed = 1.
        self.max_rotational_vel = 10.  # TODO handle this better
        self.goal_reached_threshold = goal_reached_threshold

        self.maximum_error = maximum_error
        self.x_threshold = 3.
        self.path_window_size = 30
        self.seed()
        self.viewer = None

        self.path: Path = None  # TODO
        self.done = False
        self.cumulative_run_error = None
        self.bot: Bot = None

        max_point_distances = np.array([self.x_threshold * 2, self.x_threshold * 2])
        self.observation_space = spaces.Box(low=np.array((list(-max_point_distances) + [0]) * self.path_window_size),
                                            high=np.array((list(max_point_distances) + [1]) * self.path_window_size),
                                            dtype=np.float32)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human', extra_objects: list = None):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        length = 50.0
        width = 30.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            right, front, left, right = -length / 2, length / 2, width / 2, -width / 2
            cart = rendering.FilledPolygon([(right, right), (right, left), (front, 0)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            screen_path_points = [[point.x * scale + screen_width / 2.0, point.y * scale + screen_height / 2.0] for
                                  point in self.path.points]
            path = rendering.make_polyline(screen_path_points)
            path.set_linewidth(4)
            self.viewer.add_geom(path)

            if extra_objects is not None:
                for object in extra_objects:
                    self.viewer.add_geom(object)

        if self.bot is None: return None

        x = self.bot.pose.location.x
        y = self.bot.pose.location.y
        yaw = self.bot.pose.yaw
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = y * scale + screen_height / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(yaw)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calc_reward(self):
        error_from_path = self.path.distance(self.bot.pose.location)
        self.cumulative_run_error += error_from_path
        error_too_large = error_from_path > self.maximum_error
        goal_reached = self.path.goal_reached(self.bot.pose.location)
        self.done = error_too_large or goal_reached
        # TODO think about this true reward function
        if error_too_large:
            reward = -100.
            print(f'Error too large, breaking off. Reward: {reward}')
        elif goal_reached:
            reward = 100000000 / self.cumulative_run_error
            # higher reward for lower cumulative error? This incentives faster driving and staying on the path,
            # while still encouraging goint till the end
            print(f'Reached goal! Reward is {reward}')
        else:
            reward = 0.
        return reward

    def reset(self):
        self.path = self.create_path()

        self.done = False
        self.cumulative_run_error = 0.
        observations = self.bot.get_future_path_in_local_coordinates(self.path)
        if self.clean_viewer:
            self.close()
        return observations

    @abstractmethod
    def create_path(self):
        raise NotImplementedError()


class PathEnvAbstract(PathEnvShared):
    """
    Loosely adapted from the standard cartpole environment.
    #TODO
    """

    def __init__(self, clean_viewer=False):
        super().__init__(clean_viewer)
        action_limits = np.array([self.max_speed, self.max_rotational_vel])
        self.action_space = spaces.Box(-action_limits, action_limits,
                                       dtype=np.float32)  # rotational and forward velocity

    def step(self, action: np.array):
        """Advance the environment with 1 step.

        @param action: [forward velocity, rotational velocity]
        @return: the new observation, reward, whether the run is done and extra info
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        u = action[0]
        w = action[1]
        self.bot.apply_action(u, w)

        reward = self.calc_reward()

        observation = self.bot.get_future_path_in_local_coordinates(self.path)
        assert self.observation_space.contains(np.array(observation)), "%r (%s) invalid" % (
            observation, type(observation))

        return observation, reward, self.done, {}

    def reset(self):
        self.bot = Bot(0., 0., np.pi / 2, self.kinematics_integrator,
                       self.path_window_size, self.tau)
        return super().reset()


class PathEnv(PathEnvAbstract):

    def create_path(self):
        return create_constant_path(self.goal_reached_threshold)


def create_constant_path(goal_reached_threshold: float):
    xs = np.linspace(0., 2., 100)
    points = [Point(x, np.sin(3 * x) * .45 + .1 * x) for x in xs]
    return Path(points, goal_reached_threshold)


def main():
    env = PathEnv()
    for i_episode in range(20):
        env.reset()
        for t in range(100):
            env.render()
            action_to_take = env.action_space.sample()
            action_to_take[0] = abs(action_to_take[0])  # Only positive velocity
            observation, reward, done, info = env.step(action_to_take)
            print(observation)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()
