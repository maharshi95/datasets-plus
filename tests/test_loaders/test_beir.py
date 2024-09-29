import unittest
from datasets_plus.loaders.beir import BeIRLoader


class TestBeIRLoader(unittest.TestCase):
    def setUp(self):
        self.loader = BeIRLoader()

    def test_load(self):
        with self.assertRaises(NotImplementedError):
            self.loader.load("nfcorpus", None, "train")


if __name__ == "__main__":
    unittest.main()
