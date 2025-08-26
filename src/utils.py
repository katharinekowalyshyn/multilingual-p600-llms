"""
utils.py

Utility functions for the multilingual garden path LLMs project.
Add new functions here as the project progresses.
"""

import os
import pandas as pd

def save_dataframe_to_csv(df: pd.DataFrame, output_path: str):
    """
    Save a pandas DataFrame to a CSV file, creating the output directory if it doesn't exist.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path to the output CSV file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False) 