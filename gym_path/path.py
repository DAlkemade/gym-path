from gym_path.bot import Point


class Path(object):
    def __init__(self):
        raise NotImplementedError()

    def distance(self, point: Point) -> float:
        """Return minimum distance from point to path."""
        raise NotImplementedError()

    def goal_reached(self, point: Point) -> bool:
        """Return whether the point is within a certain distance of the final point on the path."""
        raise NotImplementedError()