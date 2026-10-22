# -*- coding: utf-8 -*-
import unittest

from tevforge.model.backends import get_backend, list_backends


class TestBackends(unittest.TestCase):
    def test_all_backends_registered(self):
        names = list_backends()
        for n in ["mock", "cpu", "openai", "vllm", "transformers"]:
            self.assertIn(n, names)

    def test_get_backend(self):
        b = get_backend("mock", None)
        self.assertIsNotNone(b.make_policy())

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            get_backend("nope", None)


if __name__ == "__main__":
    unittest.main()
