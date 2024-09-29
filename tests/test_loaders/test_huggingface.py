import unittest
from datasets_plus.loaders.huggingface import HuggingFaceLoader


class TestHuggingFaceLoader(unittest.TestCase):
    def setUp(self):
        self.loader = HuggingFaceLoader()

    def test_load(self):
        dataset = self.loader.load("squad", "v1.1", "train")
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) > 0)


if __name__ == "__main__":
    unittest.main()
