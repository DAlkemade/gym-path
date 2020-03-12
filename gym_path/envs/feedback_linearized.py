import random

import numpy as np

from gym_path.coordination import Point
from gym_path.envs import PathEnvDifferentPaths
from gym_path.envs.path_env import PathEnvAbstract
from gym_path.path import Path


class PathFeedbackLinearized(PathEnvDifferentPaths):

    def __init__(self):
        super().__init__()




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
