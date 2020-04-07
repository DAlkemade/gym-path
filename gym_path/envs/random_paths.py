import random

import numpy as np
from gym_path.coordination import Point
from gym_path.envs.path_env import PathEnvAbstract
from gym_path.path import Path

NR_POINTS = 100

class PathEnvDifferentPaths(PathEnvAbstract):

    def __init__(self):
        super().__init__(clean_viewer=True)

    def create_path(self):
        return create_random_path(self.goal_reached_threshold)


def create_hooked_path(a: float, b: float, c: float, xs: np.array):
    nr_points = len(xs)
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
        point = Point(x, last)
        points.append(point)
    return points


def generate_x_values():
    return np.linspace(0., 2., NR_POINTS)

def create_random_path(goal_threshold: float):
    xs = generate_x_values()
    if random.random() < .5:
        a = .1 * (random.random() - .5)
        b = .1 * (random.random() - .5)
        c = .1 * (random.random() - .5)
        points = create_hooked_path(a, b, c, xs)
    else:
        a = random.random() * 3
        b = random.random()
        c = random.random()
        points = create_sin_path(a, b, c, xs)
    return Path(points, goal_threshold)


def create_sin_path(a: float, b: float, c: float, xs: np.array):
    points = [Point(x, np.sin(a * x) * b + c * x) for x in xs]
    return points


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
