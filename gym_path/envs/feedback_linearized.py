from abc import ABCMeta, abstractmethod

import numpy as np
from gym import spaces, logger

from gym_path.bot import FeedBackLinearizationBot
from gym_path.envs.path_env import PathEnvShared
from gym_path.envs.random_paths import create_random_path


class PathFeedbackLinearizedAbstract(PathEnvShared, metaclass=ABCMeta):
    def __init__(self):
        super().__init__(clean_viewer=True)
        # TODO
        self.action_space = spaces.Box(np.array([0.000001, 0.]), np.array([1., 1.]), dtype=np.float32)  # length of epsilon/pole
        self.latest_epsilon = None

    @abstractmethod
    def create_path(self):
        pass

    def render(self, mode='human', extra_objects: list = None):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        length = self.latest_epsilon * scale
        width = 5.
        r, f, l, r = 0, length, width / 2, -width / 2
        if self.viewer is None:
            self.pole = rendering.FilledPolygon([(r, r), (r, l), (f, l), (f, r)])
            self.poletrans = rendering.Transform()
            self.pole.add_attr(self.poletrans)

        self.pole.v = [(r, r), (r, l), (f, l), (f, r)]

        x = self.bot.pose.location.x
        y = self.bot.pose.location.y
        yaw = self.bot.pose.yaw
        polex = x * scale + screen_width / 2.0  # MIDDLE OF CART
        poley = y * scale + screen_height / 2.0  # MIDDLE OF CART
        # polex = (x + .5* np.sin(self.latest_epsilon)) * scale + screen_width / 2.0  # MIDDLE OF CART
        # poley = (y + .5 * np.cos(self.latest_epsilon)) * scale + screen_height / 2.0  # MIDDLE OF CART
        self.poletrans.set_translation(polex, poley)  # TODO
        self.poletrans.set_rotation(yaw)  # TODO
        super().render(extra_objects=[self.pole])

    def step(self, action: np.array):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        # TODO use the feedback lin functions
        epsilon_length: float = action[0]
        kp: float = action[1]
        self.latest_epsilon = epsilon_length
        num_states = len(self.observation_space.sample())
        self.bot.move_feedback_linearized(epsilon_length, self.path, num_states, kp, len(self.action_space.sample()))
        observation = self.bot.get_future_path_in_local_coordinates(self.path)
        assert self.observation_space.contains(np.array(observation)), "%r (%s) invalid" % (
            observation, type(observation))
        reward = self.calc_reward()
        return observation, reward, self.done, {}

    def reset(self):
        self.bot = FeedBackLinearizationBot(0., 0., np.pi / 2, self.kinematics_integrator,
                                            self.path_window_size, self.tau)
        self.latest_epsilon = 0.
        # TODO add lineariazation stuff to inherited bot class
        return super().reset()


class PathFeedbackLinearized(PathFeedbackLinearizedAbstract):

    def create_path(self):
        return create_random_path(self.goal_reached_threshold)


def main():
    env = PathFeedbackLinearized()
    for i_episode in range(20):
        env.reset()
        for t in range(200):
            env.render()
            action_to_take = env.action_space.sample()
            observation, reward, done, info = env.step(action_to_take)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()
