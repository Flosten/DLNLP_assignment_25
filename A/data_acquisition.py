import os

import pandas as pd


def load_data(folder_path, file_name):
    """
    Load data from a CSV file.

    Args:
        folder_path (str): The path to the folder containing the CSV file.
        file_name (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8", errors="replace") as file:
            data = pd.read_csv(file)
        return data
    else:
        raise FileNotFoundError(
            f"The file {file_name} does not exist in the folder {folder_path}."
        )


def save_data(data, folder_path, file_name):
    """
    Save data to a CSV file.

    Args:
        data (pd.DataFrame): The data to save.
        folder_path (str): The path to the folder where the CSV file will be saved.
        file_name (str): The name of the CSV file to save.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    data.to_csv(file_path, index=False)
