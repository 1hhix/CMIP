import os
import pickle


def check_extension(filename: str) -> str:
    """
    Ensure the filename has a .pkl extension.
    Args:
        filename: Input filename
    Returns:
        Filename with .pkl extension
    """
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename: str) -> None:
    """
    Save a dataset to a pickle file, creating directories if needed.
    Args:
        dataset: The dataset object to save
        filename: Target filename
    """
    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename: str):
    """
    Load a dataset from a pickle file.
    Args:
        filename: Source filename
    Returns:
        Loaded dataset object
    """
    with open(check_extension(filename), "rb") as f:
        return pickle.load(f)
