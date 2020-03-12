import unittest

import numpy as np

from gym_path.bot import Bot
from gym_path.envs.path_env import create_constant_path


class TestBot(unittest.TestCase):
    def test_local_coordinates(self):
        num_states = 30
        path = create_constant_path(.01)
        late_point = path.points[-5]
        bot = Bot(late_point.x, late_point.y, np.pi / 2, 'euler', num_states, .02)
        local_points = bot.get_future_path_in_local_coordinates(path)
        self.assertEquals(type(local_points), list)
        self.assertEquals(len(local_points), 2 * num_states)


if __name__ == '__main__':
    unittest.main()
