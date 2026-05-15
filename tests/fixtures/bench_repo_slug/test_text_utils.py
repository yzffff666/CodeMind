import unittest

from text_utils import slugify


class SlugifyTests(unittest.TestCase):
    def test_lowercases_and_joins_words(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_collapses_extra_whitespace(self):
        self.assertEqual(slugify("  Hello   World  "), "hello-world")


if __name__ == "__main__":
    unittest.main()
