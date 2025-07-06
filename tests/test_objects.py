import unittest

from util.objects import Line
import numpy as np


class TestLine(unittest.TestCase):

    def setUp(self):
        # line through (0,0,0) in direction (1,0,0)
        self.line = Line(np.array([0, 0, 0]), np.array([1, 0, 0]))

    def test_point_on_line(self):
        p1 = np.array([2, 0, 0])  # on line
        p2 = np.array([-3, 0, 0])  # on line
        self.assertTrue(self.line.is_on_line(p1))
        self.assertTrue(self.line.is_on_line(p2))

    def test_point_off_line(self):
        p3 = np.array([0, 1, 0])  # off line
        p4 = np.array([1, 1, 0])  # off line
        self.assertFalse(self.line.is_on_line(p3))
        self.assertFalse(self.line.is_on_line(p4))

    def test_distance_to_point_on_line(self):
        p1 = np.array([5, 0, 0])
        distance = self.line.distance_to(p1)
        self.assertAlmostEqual(distance, 0)

    def test_distance_to_point_off_line(self):
        p2 = np.array([0, 2, 0])
        distance = self.line.distance_to(p2)
        self.assertAlmostEqual(distance, 2)

    def test_dimension_mismatch_init(self):
        with self.assertRaises(ValueError):
            Line(np.array([0, 0]), np.array([1, 0, 0]))

    def test_dimension_mismatch_is_on_line(self):
        point_wrong_dim = np.array([0, 0])
        self.assertFalse(self.line.is_on_line(point_wrong_dim))


class TestLineEquality(unittest.TestCase):
    def setUp(self):
        # Line along x-axis through origin
        self.line1 = Line(np.array([0, 0, 0]), np.array([1, 0, 0]))

    def test_equal_same_point_same_direction(self):
        line2 = Line(np.array([0, 0, 0]), np.array([1, 0, 0]))
        self.assertEqual(self.line1, line2)

    def test_equal_point_on_line_same_direction(self):
        # Same line but point at x=5
        line2 = Line(np.array([5, 0, 0]), np.array([1, 0, 0]))
        self.assertEqual(self.line1, line2)

    def test_equal_scaled_direction(self):
        # Same line but direction vector scaled
        line2 = Line(np.array([0, 0, 0]), np.array([10, 0, 0]))
        self.assertEqual(self.line1, line2)

    def test_not_equal_different_direction(self):
        # Different direction (y-axis)
        line2 = Line(np.array([0, 0, 0]), np.array([0, 1, 0]))
        self.assertNotEqual(self.line1, line2)

    def test_not_equal_parallel_different_line(self):
        # Parallel line offset in y-direction
        line2 = Line(np.array([0, 1, 0]), np.array([1, 0, 0]))
        self.assertNotEqual(self.line1, line2)

    def test_not_equal_wrong_type(self):
        self.assertNotEqual(self.line1, "not a line")


class TestPlane(unittest.TestCase):

    def setUp(self):
        # line through (0,0,0) in direction (1,0,0)
        self.line = Line(np.array([0, 0, 0]), np.array([1, 0, 0]))

    def test_point_on_line(self):
        p1 = np.array([2, 0, 0])  # on line
        p2 = np.array([-3, 0, 0])  # on line
        self.assertTrue(self.line.is_on_line(p1))
        self.assertTrue(self.line.is_on_line(p2))

    def test_point_off_line(self):
        p3 = np.array([0, 1, 0])  # off line
        p4 = np.array([1, 1, 0])  # off line
        self.assertFalse(self.line.is_on_line(p3))
        self.assertFalse(self.line.is_on_line(p4))

    def test_distance_to_point_on_line(self):
        p1 = np.array([5, 0, 0])
        distance = self.line.distance_to(p1)
        self.assertAlmostEqual(distance, 0)

    def test_distance_to_point_off_line(self):
        p2 = np.array([0, 2, 0])
        distance = self.line.distance_to(p2)
        self.assertAlmostEqual(distance, 2)

    def test_dimension_mismatch_init(self):
        with self.assertRaises(ValueError):
            Line(np.array([0, 0]), np.array([1, 0, 0]))

    def test_dimension_mismatch_is_on_line(self):
        point_wrong_dim = np.array([0, 0])
        self.assertFalse(self.line.is_on_line(point_wrong_dim))


if __name__ == "__main__":
    unittest.main()
