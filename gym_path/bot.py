import numpy as np
from gym_path.coordination import Pose, Point
from gym_path.path import Path


class Bot(object):
    def __init__(self, x: float, y: float, yaw: float, kinematics: str, path_window_size: int):
        self.kinematics = kinematics
        self.pose = Pose(x, y, yaw)
        self.path_window_size = path_window_size

    def _update_pose_euler(self, u, w, dt):
        next_pose = Pose()
        next_pose.location.x = self.pose.location.x + dt * u * np.cos(self.pose.yaw)
        next_pose.location.y = self.pose.location.y + dt * u * np.sin(self.pose.yaw)
        next_pose.yaw = self.pose.yaw + dt * w
        self.pose = next_pose

    def apply_action(self, u, w, tau):
        if self.kinematics == 'euler':
            self._update_pose_euler(u, w, tau)

    def get_future_path_in_local_coordinates(self, path: Path):
        future_points = path.find_future_points(self.pose.location)
        path_relative = []
        for point in future_points:
            path_relative.append(self.get_relative_position(point))

        res = []
        for i in range(self.path_window_size):
            try:
                res.append(list(path_relative[i]))
            except IndexError:
                res.append([0., 0.])
        return list(np.array(res).flatten())

    def get_relative_position(self, absolute_position: Point):
        delta = absolute_position.array - self.pose.location.array
        theta = -1 * self.pose.yaw
        x = delta[0]
        y = delta[1]
        relative_position = [x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)]

        return np.array(relative_position)
