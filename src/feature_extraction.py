import logging
import concurrent.futures
import pandas as pd
from typing import List, Dict, Union
import json

from pydantic import BaseModel

from src.utils import load_dataset, save_dataset, update_metadata
from src.config import PipelineConfig
from src.openai_client import init_openai_client

class Features(BaseModel):
    entities: List[str]
    sentiment: str
    theme: str

def extract_features_from_text(client, text: str) -> Dict[str, Union[List[str], str]]:
    """
    Uses GPT-4o to extract features such as entities, sentiment, and themes from text.
    WARNING: In production, avoid using eval; instead, parse a valid JSON response.
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Extract entities, sentiment, and theme from by going through step by step: '{text}'"}],
            response_format=Features,
        )
        extracted_data = json.loads(response.choices[0].message.content)
        if not isinstance(extracted_data, dict) or not all(k in extracted_data for k in ["entities", "sentiment", "theme"]):
            raise ValueError("Invalid response format from OpenAI")
        return extracted_data
    except Exception as e:
        logging.error(f"Error extracting features for text '{text}': {e}")
        return {"entities": [], "sentiment": "unknown", "theme": "unknown"}

def extract_features_parallel(client, texts: List[str], num_workers: int = 5) -> List[Dict[str, Union[List[str], str]]]:
    """
    Runs feature extraction in parallel for efficiency.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda t: extract_features_from_text(client, t), texts))

    return pd.DataFrame(results)

def vectorize_texts(client, df: pd.DataFrame, method: str = "tfidf") -> pd.DataFrame:
    """
    Vectorizes text using OpenAI embeddings or TF-IDF.
    Adds an 'embedding' column to the DataFrame.
    """
    if method == "openai":
        try:
            df["embedding"] = [
                client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
                for text in df["text"]
            ]
            logging.info("Text vectorized using OpenAI embeddings.")
        except Exception as e:
            logging.error(f"Error in OpenAI embedding vectorization: {e}")
            raise
    elif method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df["text"])
        df["embedding"] = tfidf_matrix.toarray().tolist()
        logging.info("Text vectorized using TF-IDF.")
    else:
        raise ValueError("Invalid vectorization method. Choose 'openai' or 'tfidf'.")
    return df

def main_pipeline_feature_extraction(client_text, client_embedding, config: PipelineConfig, vectorization_method: str = "tfidf") -> None:
    """
    Pipeline step 2: Feature Extraction & Vectorization.
      - Loads data.
      - Extracts features in parallel.
      - Merges extracted features into the DataFrame.
      - Saves intermediate feature extraction output.
      - Vectorizes the text data.
      - Saves vectorized data and updates metadata.
    """
    df = load_dataset(config.prepared_data_file)
    features = extract_features_parallel(client_text, df["text"].tolist())
    df_features = pd.DataFrame(features)
    df = pd.concat([df, df_features], axis=1)
    save_dataset(df, config.feature_extraction_file)
    df_vectorized = vectorize_texts(client_embedding, df, method=vectorization_method)
    save_dataset(df_vectorized, config.vectorized_output_file)
    metadata = {
        "feature_rows": len(df),
        "vectorization_method": vectorization_method
    }
    update_metadata(metadata, config.metadata_feature_extraction)


if __name__ == "__main__":
    config = PipelineConfig()
    client_text, client_embedding = init_openai_client(config)
    main_pipeline_feature_extraction(client_text, client_embedding, config)