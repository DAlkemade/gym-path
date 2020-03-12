import numpy as np
from gym import spaces, logger

from gym_path.bot import FeedBackLinearizationBot
from gym_path.envs.path_env import PathEnvShared
from gym_path.envs.random_paths import create_random_path


class PathFeedbackLinearized(PathEnvShared):

    def __init__(self):
        super().__init__(clean_viewer=True)
        # TODO
        self.action_space = spaces.Box(np.array([0.]), np.array([1.]), dtype=np.float32)  # length of epsilon/pole

    def create_path(self):
        return create_random_path(self.goal_reached_threshold)

    def render(self, mode='human', extra_objects: list = None):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            width = 5.
            height = 20.
            l, r, t, b = -width / 2, width / 2, height / 2, -height / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.poletrans = rendering.Transform()
            pole.add_attr(self.poletrans)

        self.poletrans.set_translation(50., 100.)  # TODO
        self.poletrans.set_rotation(1.)  # TODO
        super().render()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        # TODO use the feedback lin functions
        epsilon_length: float = action
        num_states = len(env.observation_space.sample())
        self.bot.move_feedback_linearized(epsilon_length, self.path, num_states)
        observation = self.bot.get_future_path_in_local_coordinates(self.path)
        assert self.observation_space.contains(np.array(observation)), "%r (%s) invalid" % (
            observation, type(observation))
        reward = self.calc_reward()
        return observation, reward, self.done, {}

    def reset(self):
        self.bot = FeedBackLinearizationBot(0., 0., np.pi / 2, self.kinematics_integrator,
                                            self.path_window_size, self.tau)
        # TODO add lineariazation stuff to inherited bot class
        return super().reset()


# class PathFeedbackLinearized(PathEnvDifferentPaths):
#
#     def __init__(self):
#         super().__init__()
#         self.pole = None
#
#     def render(self, mode='human', extra_objects: list = None):
#         from gym.envs.classic_control import rendering
#         if self.viewer is None:
#             width = 5.
#             height = 20.
#             l, r, t, b = -width / 2, width / 2, height / 2, -height / 2
#             self.pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
#             self.poletrans = rendering.Transform()
#             self.pole.add_attr(self.poletrans)
#
#         self.poletrans.set_translation(50., 100.)  # TODO
#         self.poletrans.set_rotation(1.)  # TODO
#         super().render(extra_objects=[self.pole])


if __name__ == "__main__":
    env = PathFeedbackLinearized()
    for i_episode in range(20):
        env.reset()
        for t in range(200):
            env.render()
            action_to_take = env.action_space.sample()
            action_to_take[0] = abs(action_to_take[0])  # Only positive velocity
            observation, reward, done, info = env.step(action_to_take)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
