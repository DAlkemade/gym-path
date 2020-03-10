from gym_path.bot import Point
from gym import logger


class Path(object):
    def __init__(self):
        logger.warn("PATH NOT IMPLEMENTED CORRECTLY")

    def distance(self, point: Point) -> float:
        """Return minimum distance from point to path."""
        logger.warn("PATH NOT IMPLEMENTED CORRECTLY")
        return 0.

    def goal_reached(self, point: Point) -> bool:
        """Return whether the point is within a certain distance of the final point on the path."""
        logger.warn("PATH NOT IMPLEMENTED CORRECTLY")
        return False