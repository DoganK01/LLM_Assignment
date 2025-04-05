import os
import json
import logging
import pandas as pd
from typing import Dict
import ast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads a CSV dataset containing text data."""
    if not os.path.exists(file_path):
        msg = f"File {file_path} does not exist."
        logging.error(msg)
        raise FileNotFoundError(msg)
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset from {file_path} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Dataset saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")
        raise

def update_metadata(metadata: Dict, metadata_path: str) -> None:
    """Writes a dictionary to a JSON file as metadata."""
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata updated in {metadata_path}")
    except Exception as e:
        logging.error(f"Error updating metadata: {e}")
        raise


def parse_embedding(embedding_str):
    """Parses the embedding string into a list of floats."""
    try:
        embedding_str = embedding_str.strip()

        embedding_list = ast.literal_eval(embedding_str)
        
        if isinstance(embedding_list, list) and all(isinstance(i, (int, float)) for i in embedding_list):
            return embedding_list
        else:
            raise ValueError(f"Embedding format is incorrect for: {embedding_str}")
    except Exception as e:
        print(f"Error parsing embedding: {e}")
        return None