import tempfile

import pytest

from datasets_plus import load_dataset, load_hf_dataset, process_dataset_name


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        ("dataset", ("dataset", None, None)),
        ("dataset:split", ("dataset", None, "split")),
        ("dataset::split", ("dataset", None, "split")),
        ("dataset:config:split", ("dataset", "config", "split")),
    ],
)
def test_process_dataset_name(input_name, expected_output):
    assert process_dataset_name(input_name) == expected_output


def test_process_dataset_name_invalid():
    with pytest.raises(ValueError, match="Invalid dataset name format"):
        process_dataset_name("dataset:config:split:extra")


def test_load_as_hf_dataset(mocker):
    mock_load_dataset = mocker.patch("datasets.load_dataset")
    mock_load_from_disk = mocker.patch("datasets.load_from_disk")

    # Test JSON file
    load_dataset("test.json")
    mock_load_dataset.assert_called_with("json", data_files="test.json")

    # Test JSONL file
    load_dataset("test.jsonl")
    mock_load_dataset.assert_called_with("json", data_files="test.jsonl")

    # Test dataset name
    load_dataset("test_dataset")
    mock_load_dataset.assert_called_with("test_dataset", name=None, split=None)

    # Test dataset with config
    load_dataset("test_dataset:fold")
    mock_load_dataset.assert_called_with("test_dataset", name=None, split="fold")

    # Test dataset with split
    load_dataset("test_dataset::fold")
    mock_load_dataset.assert_called_with("test_dataset", name=None, split="fold")

    # Test dataset with config and split
    load_dataset("test_dataset:config:fold")
    mock_load_dataset.assert_called_with("test_dataset", name="config", split="fold")

    # Test directory
    temp_dir = tempfile.TemporaryDirectory()
    load_dataset(temp_dir.name)
    mock_load_from_disk.assert_called_with(temp_dir.name, split=None)
    temp_dir.cleanup()

    # Test directory with split
    temp_dir = tempfile.TemporaryDirectory()
    load_dataset(f"{temp_dir.name}::train")
    mock_load_from_disk.assert_called_with(temp_dir.name, split="train")
    temp_dir.cleanup()
