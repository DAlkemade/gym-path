from gym_path.envs import PathEnvDifferentPaths


# class PathFeedbackLinearized(PathEnvShared):
#
#     def __init__(self):
#         super().__init__(clean_viewer=True)
#
#     def render(self, mode='human', extra_objects: list = None):
#         from gym.envs.classic_control import rendering
#         if self.viewer is None:
#             width = 5.
#             height = 20.
#             l, r, t, b = -width / 2, width / 2, height / 2, -height / 2
#             pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
#             self.poletrans = rendering.Transform()
#             pole.add_attr(self.poletrans)
#
#         self.carttrans.set_translation(50., 100.) #TODO
#         self.carttrans.set_rotation(1.) #TODO
#         super().render()

class PathFeedbackLinearized(PathEnvDifferentPaths):

    def __init__(self):
        super().__init__()
        self.pole = None

    def render(self, mode='human', extra_objects: list = None):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            width = 5.
            height = 20.
            l, r, t, b = -width / 2, width / 2, height / 2, -height / 2
            self.pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.poletrans = rendering.Transform()
            self.pole.add_attr(self.poletrans)

        self.poletrans.set_translation(50., 100.)  # TODO
        self.poletrans.set_rotation(1.)  # TODO
        super().render(extra_objects=[self.pole])


if __name__ == "__main__":
    env = PathFeedbackLinearized()
    for i_episode in range(20):
        env.reset()
        for t in range(100):
            env.render()
            action_to_take = env.action_space.sample()
            action_to_take[0] = abs(action_to_take[0])  # Only positive velocity
            observation, reward, done, info = env.step(action_to_take)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
