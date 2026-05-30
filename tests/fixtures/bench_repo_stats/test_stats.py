import unittest

from stats import average


class AverageTests(unittest.TestCase):
    def test_average_multiple_values(self):
        self.assertEqual(average([2, 4, 6, 8]), 5)

    def test_average_single_value(self):
        self.assertEqual(average([7]), 7)

    def test_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            average([])


if __name__ == "__main__":
    unittest.main()

