import numpy as np


class Point():
    def __init__(self, x, y):
        self.array = np.array([x, y])

    @property
    def x(self):
        return self.array[0]

    @x.setter
    def x(self, x):
        self.array[0] = x

    @property
    def y(self):
        return self.array[1]

    @y.setter
    def y(self, y):
        self.array[1] = y


class Pose(object):
    def __init__(self, x=None, y=None, yaw=None):
        self.location = Point(x, y)
        self.yaw = yaw
