import unittest
from datasets_plus.core import load_dataset


class TestCore(unittest):
    def test_load_dataset(self):
        # Test loading a HuggingFace dataset
        dataset = load_dataset("squad:v1.1:train")
        self.assertIsNotNone(dataset)

        # Test loading a BeIR dataset (this will fail until BeIR loader is implemented)
        with self.assertRaises(NotImplementedError):
            dataset = load_dataset("beir/nfcorpus:train")


if __name__ == "__main__":
    unittest.main()
