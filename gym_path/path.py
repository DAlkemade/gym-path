from typing import List

import numpy as np
from gym import logger
from gym_path.coordination import Point


class Path(object):
    def __init__(self, points: List[Point] = None):
        if points is None:
            points = []
        self.points = points

    def _distances(self, point: Point):
        return [np.linalg.norm(p.array - point.array) for p in self.points]

    def distance(self, point: Point) -> (float, int):
        """Return minimum distance from point to path."""
        logger.warn("Distance function assumes very fine-grained path points")
        distances = self._distances(point)
        return np.min(distances)

    def _find_index_closest_point(self, point: Point) -> int:
        distances = self._distances(point)
        return np.argmin(distances)

    def find_future_points(self, point: Point):
        closest_index = self._find_index_closest_point(point)
        return self.points[closest_index:]

    def goal_reached(self, point: Point) -> bool:
        """Return whether the point is within a certain distance of the final point on the path."""
        logger.warn("PATH NOT IMPLEMENTED CORRECTLY")
        return False
