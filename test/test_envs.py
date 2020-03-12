import unittest

from gym_path.envs import PathEnv, PathEnvDifferentPaths
from gym_path.envs.feedback_linearized import PathFeedbackLinearized


def run_env(env):
    for i_episode in range(5):
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


class TestEnvs(unittest.TestCase):

    def test_integration_basic_env(self):
        """Test whether there are no errors while running the environment. User should visually inspect viewer."""
        env = PathEnv()
        run_env(env)

    def test_integration_random_paths(self):
        """Test whether there are no errors while running the random path generation environment.

        User should visually inspect viewer.
        """
        env = PathEnvDifferentPaths()
        run_env(env)

    def test_integration_feedback_linearized_env(self):
        """Test whether there are no errors while running the random path generation environment.

        User should visually inspect viewer.
        """
        env = PathFeedbackLinearized()
        run_env(env)


if __name__ == '__main__':
    unittest.main()
