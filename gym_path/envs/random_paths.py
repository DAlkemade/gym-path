import random

import numpy as np
from gym_path.coordination import Point
from gym_path.envs.path_env import PathEnvAbstract
from gym_path.path import Path


class PathEnvDifferentPaths(PathEnvAbstract):

    def __init__(self):
        super().__init__(clean_viewer=True)

    def create_path(self):
        return create_random_path(self.goal_reached_threshold)


def create_random_path(goal_threshold: float):
    nr_points = 100
    xs = np.linspace(0., 2., nr_points)
    if random.random() < .5:
        a = .1 * (random.random() - .5)
        b = .1 * (random.random() - .5)
        c = .1 * (random.random() - .5)
        points = list()
        last = 0
        for i, x in enumerate(xs):
            if i < nr_points / 3:
                delta = a
            elif i < 2 * nr_points / 3:
                delta = b
            else:
                delta = c

            last = last + delta
            points.append(Point(x, last))
    else:
        a = random.random() * 3
        b = random.random()
        c = random.random()
        points = [Point(x, np.sin(a * x) * b + c * x) for x in xs]
    return Path(points, goal_threshold)


def main():
    env = PathEnvDifferentPaths()
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


if __name__ == "__main__":
    main()
