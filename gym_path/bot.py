import numpy as np

from gym_path.coordination import Pose, Point
from gym_path.path import Path

X = 0
Y = 1
YAW = 2


class Bot(object):
    def __init__(self, x: float, y: float, yaw: float, kinematics: str, path_window_size: int, tau: float):
        self.tau = tau
        self.kinematics = kinematics
        self.pose = Pose(x, y, yaw)
        self.path_window_size = path_window_size

    def _update_pose_euler(self, u, w, dt):
        next_x = self.pose.location.x + dt * u * np.cos(self.pose.yaw)
        next_y = self.pose.location.y + dt * u * np.sin(self.pose.yaw)
        next_yaw = self.pose.yaw + dt * w
        assert type(next_x) is np.float64
        assert type(next_y) is np.float64
        assert type(next_yaw) is np.float64
        next_pose = Pose(next_x, next_y, next_yaw)
        self.pose = next_pose

    def apply_action(self, u, w):
        if self.kinematics == 'euler':
            self._update_pose_euler(u, w, self.tau)

    def get_future_path_in_local_coordinates(self, path: Path):
        path_relative = self.get_local_path(path)

        res = []
        for i in range(self.path_window_size):
            try:
                point = list(path_relative[i])
                point_and_valid = point + [1.]
                res.append(point_and_valid)
            except IndexError:
                res.append(np.array([0., 0., 0.], dtype=np.float32))
        result = list(np.array(res).flatten())
        return result

    def get_local_path(self, path):
        future_points = path.find_future_points(self.pose.location)
        path_relative = []
        for point in future_points:
            path_relative.append(self.get_relative_position(point))
        return path_relative

    def get_relative_position(self, absolute_position: Point):
        delta = absolute_position.array - self.pose.location.array
        theta = -1 * self.pose.yaw
        x = delta[0]
        y = delta[1]
        assert type(x) is np.float64
        assert type(y) is np.float64
        relative_position = [x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)]

        return np.array(relative_position)


def feedback_linearized(pose, velocity, epsilon):
    """Determine the forward and rotational velocity for differential drive robot using feedback linearization.

    @param pose: position of holonomic point
    @param velocity: velocity of holonomic point
    @param epsilon: distance that the holonomic point is in front of the differential drive robot.
    @return: forward velocity u and rotational velocity w
    """
    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))
    return u, w


def get_velocity(position, path_points, Kp: float):
    """Get velocity that a holonomic point at a position should have to track the path.

    @param path_points: points of path in front of holonomic robot
    @param position: position of holonomic point
    @return: velocity v of holonomic point to track path
    """
    # TODO clean this method up
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
    v = Kp * delta / np.linalg.norm(delta)
    return v


class FeedBackLinearizationBot(Bot):
    """Bot that has a manual feedback linearized path-tracking algorithm built in."""

    def move_feedback_linearized(self, epsilon: float, path, num_states, Kp: float, state_dim: int):
        """Move bot using feedbacklinearization from a holonomic point at a distance epsilon in front of it."""
        #TODO reshape is incorrect due to addition of the valid points
        path_points = self.get_local_path(path)
        # TODO think about the coordinates below
        velocity = get_velocity([epsilon, 0.], path_points, Kp)
        u, w = feedback_linearized([0., 0., 0.], velocity, epsilon)
        self.apply_action(u, w)
