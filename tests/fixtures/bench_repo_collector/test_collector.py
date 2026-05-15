import unittest

from collector import append_item


class AppendItemTests(unittest.TestCase):
    def test_keeps_single_call_result(self):
        self.assertEqual(append_item("alpha", None), ["alpha"])

    def test_does_not_share_default_bucket_between_calls(self):
        append_item("alpha")
        self.assertEqual(append_item("beta"), ["beta"])


if __name__ == "__main__":
    unittest.main()
