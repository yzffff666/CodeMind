import unittest

from math_utils import clamp


class ClampTests(unittest.TestCase):
    def test_keeps_value_inside_range(self):
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_uses_lower_bound(self):
        self.assertEqual(clamp(-3, 0, 10), 0)

    def test_uses_upper_bound(self):
        self.assertEqual(clamp(12, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()
