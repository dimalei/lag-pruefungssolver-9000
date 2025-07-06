import unittest

from util.objects3d import Point3D


class TestObjects3D(unittest.TestCase):

    def test_point_eq(self):
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(1, 2, 3)
        self.assertTrue(p1 == p2)


if __name__ == "__main__":
    unittest.main()
