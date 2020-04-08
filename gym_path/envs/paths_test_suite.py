import random

from gym_path.envs.feedback_linearized import PathFeedbackLinearizedAbstract
from gym_path.envs.path_env import PathEnvAbstract
from gym_path.envs.random_paths import generate_x_values, create_hooked_path, create_sin_path, create_random_path
from gym_path.path import Path


def generate_paths_test(goal_reached_threshold: float):
    #TODO instead of manually defining some, just make 30 straight lines, 30 sins and 30 hooked
    paths = list()
    random.seed(30)
    nr_per_category = 100
    for i in range(nr_per_category):
        paths.append(create_random_path(goal_reached_threshold, hooked=True))
    for i in range(nr_per_category):
        paths.append(create_random_path(goal_reached_threshold, hooked=False))
    return paths


class PathEnvTestSuite(PathEnvAbstract):

    def __init__(self):
        super().__init__(clean_viewer=True)
        self.paths = generate_paths_test(self.goal_reached_threshold)
        self.counter = 0

    def create_path(self):
        if self.counter > len(self.paths):
            raise RuntimeWarning('All test paths have been used')
        path = self.paths[self.counter]
        self.counter += 1
        return path


class PathFeedbackLinearizedTestSuite(PathFeedbackLinearizedAbstract):

    def __init__(self):
        super().__init__()
        self.paths = generate_paths_test(self.goal_reached_threshold)
        self.counter = 0

    def create_path(self):
        if self.counter > len(self.paths):
            raise RuntimeWarning('All test paths have been used')
        path = self.paths[self.counter]
        self.counter += 1
        return path

def main():
    env = PathFeedbackLinearizedTestSuite()
    for i_episode in range(len(generate_paths_test(0.1))):
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


if __name__ == "__main__":
    main()