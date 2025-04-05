import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union
import uuid

from pydantic import BaseModel

from src.utils import load_dataset, save_dataset, update_metadata
from src.config import PipelineConfig
from src.openai_client import init_openai_client


class AugmentedText(BaseModel):
    id: int
    text: str
    timestamp: str

def augment_texts_with_gpt4o(client, texts: List[str]) -> List[Dict[str, Union[int, str]]]:
    """
    Uses OpenAI GPT-4o to generate augmented versions of input texts.
    In case of errors, returns the original text.
    """
    synthetic_texts = []
    for idx in range(10):
        for text in texts:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Paraphrase this as much different and unique as you can. Use your knowledge and imagination: '{text}'"}],
                    max_tokens=100,
                    temperature=1.5,
                )
                paraphrased_text = response.choices[0].message.content.strip()
                synthetic_texts.append({
                    "id": uuid.uuid4(),
                    "text": paraphrased_text,
                    "timestamp": datetime.now().isoformat()
                })
                logging.info(f"Augmented text generated: {paraphrased_text}")
            except Exception as e:
                logging.error(f"Error generating augmented text for '{text}': {e}")
                synthetic_texts.append({
                    "id": uuid.uuid4(),
                    "text": text,
                    "timestamp": datetime.now().isoformat()
                })
    return synthetic_texts

def main_pipeline_data_preparation(client, config: PipelineConfig) -> None:
    """
    Pipeline step 1: Data Preparation.
      - Loads the dataset.
      - Augments text data using GPT-4o.
      - Saves the augmented dataset.
      - Updates metadata (original rows, augmented rows, augmentation ratio).
    """
    df = load_dataset(config.data_file)
    augmented_data = augment_texts_with_gpt4o(client, df["text"].tolist())
    df_augmented = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
    save_dataset(df_augmented, config.prepared_data_file)
    metadata = {
        "original_rows": len(df),
        "augmented_rows": len(df_augmented),
        "augmentation_ratio": round(len(df_augmented) / len(df), 2)
    }
    update_metadata(metadata, config.metadata_data_preparation)

if __name__ == "__main__":
    config = PipelineConfig()
    client_text, _ = init_openai_client(config)
    main_pipeline_data_preparation(client_text, config)