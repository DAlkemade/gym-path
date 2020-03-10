import numpy as np

class Point():
    def __init__(self,x, y):
        self.x = x
        self.y = y

class Pose(object):
    def __init__(self, x=None, y=None, yaw=None):
        self.location = Point(x, y)
        self.yaw = yaw


class Bot(object):
    def __init__(self, pose: Pose, kinematics):
        self.kinematics = kinematics
        self.pose = pose

    def _update_pose_euler(self, u, w, dt):
        next_pose = Pose()
        next_pose.x = self.pose.location.x + dt * u * np.cos(self.pose.yaw)
        next_pose.y = self.pose.location.y + dt * u * np.sin(self.pose.yaw)
        next_pose.yaw = self.pose.yaw + dt * w
        self.pose = next_pose

    def apply_action(self, u, w, tau):
        if self.kinematics == 'euler':
            self._update_pose_euler(u, w, tau)
