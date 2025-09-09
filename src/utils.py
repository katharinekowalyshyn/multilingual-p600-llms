"""
utils.py

Utility functions for the multilingual garden path LLMs project.
Add new functions here as the project progresses.
"""

import os
import requests
import numpy as np
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


def get_neuronpedia_api_key(env_var: str = "NEURONPEDIA_API_KEY") -> str:
    """
    Fetch Neuronpedia API key from environment. Raises a clear error if missing.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(f"{env_var} environment variable not set")
    return key


def neuronpedia_fetch_sae_features(api_key: str, prompt: str, model_id: str,
                                   source: str, index: int,
                                   base_url: str = "https://www.neuronpedia.org"):
    """
    Minimal helper to call Neuronpedia's SAE activation endpoint and return a feature vector.

    Returns a numpy array if successful, otherwise None.
    """
    endpoint = f"{base_url}/api/activation/new"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "feature": {"modelId": model_id, "source": source, "index": str(index)},
        "customText": prompt,
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict):
            for key in ("values", "sae_features", "features", "activations"):
                if key in data:
                    return np.array(data[key])
    except Exception:
        return None
    return None