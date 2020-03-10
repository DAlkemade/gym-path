import numpy as np
from gym_path.coordination import Pose


class Bot(object):
    def __init__(self, x:float, y: float, yaw: float, kinematics):
        self.kinematics = kinematics
        self.pose = Pose(x, y, yaw)

    def _update_pose_euler(self, u, w, dt):
        next_pose = Pose()
        next_pose.location.x = self.pose.location.x + dt * u * np.cos(self.pose.yaw)
        next_pose.location.y = self.pose.location.y + dt * u * np.sin(self.pose.yaw)
        next_pose.yaw = self.pose.yaw + dt * w
        self.pose = next_pose

    def apply_action(self, u, w, tau):
        if self.kinematics == 'euler':
            self._update_pose_euler(u, w, tau)
