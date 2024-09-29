from datasets_plus.loaders import BeIRLoader, HuggingFaceLoader
from datasets_plus.utils.parsing import parse_dataset_string


def load_dataset(dataset_string: str, **kwargs):
    """
    Load a dataset based on a complex string like "dataset_name:config:split".

    Args:
        dataset_string (str): A string in the format "dataset_name:config:split"
        **kwargs: Additional arguments to pass to the dataset loader

    Returns:
        A loaded dataset object
    """
    dataset_name, config, split = parse_dataset_string(dataset_string)

    if dataset_name.startswith("beir/"):
        loader = BeIRLoader()
    else:
        loader = HuggingFaceLoader()

    return loader.load(dataset_name, config, split, **kwargs)
