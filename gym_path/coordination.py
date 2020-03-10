class Point():
    def __init__(self,x, y):
        self.x = x
        self.y = y


class Pose(object):
    def __init__(self, x=None, y=None, yaw=None):
        self.location = Point(x, y)
        self.yaw = yaw