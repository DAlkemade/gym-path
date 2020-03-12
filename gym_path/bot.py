import numpy as np
from gym_path.coordination import Pose, Point
from gym_path.path import Path

X = 0
Y = 1
YAW = 2
SPEED = 1.

class Bot(object):
    def __init__(self, x: float, y: float, yaw: float, kinematics: str, path_window_size: int, tau: float):
        self.tau = tau
        self.kinematics = kinematics
        self.pose = Pose(x, y, yaw)
        self.path_window_size = path_window_size

    def _update_pose_euler(self, u, w, dt):
        next_pose = Pose()
        next_pose.location.x = self.pose.location.x + dt * u * np.cos(self.pose.yaw)
        next_pose.location.y = self.pose.location.y + dt * u * np.sin(self.pose.yaw)
        next_pose.yaw = self.pose.yaw + dt * w
        self.pose = next_pose

    def apply_action(self, u, w):
        if self.kinematics == 'euler':
            self._update_pose_euler(u, w, self.tau)

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


def feedback_linearized(pose, velocity, epsilon):
    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))
    return u, w


def get_velocity(position, path_points):
    v = np.zeros_like(position)
    if len(path_points) == 0:
        print("Reached goal 1")
        return v

    # strip zeroes from path points
    tmp = []
    for point in path_points:
        if not np.linalg.norm(point) <= 1E-10:
            tmp.append(point)
    path_points = tmp
    closest = float('inf')
    closest_point_index = None
    for i, point in enumerate(path_points):
        d = np.linalg.norm(position - point)
        if d < closest:
            closest = d
            closest_point_index = i
    if closest_point_index >= len(path_points) - 1:
        # Reached goal
        return v

    delta = path_points[closest_point_index + 1] - position
    v = SPEED * delta / np.linalg.norm(delta)
    return v


class FeedBackLinearizationBot(Bot):
    def move_feedback_linearized(self, epsilon: float, path, num_states):
        observation_old = self.get_future_path_in_local_coordinates(path)
        path_points = np.reshape(observation_old, (int(num_states / 2), 2))
        # TODO think about the coordinates below
        velocity = get_velocity([epsilon, 0.], path_points)
        u, w = feedback_linearized([0., 0., 0.], velocity, epsilon)
        self.apply_action(u, w)