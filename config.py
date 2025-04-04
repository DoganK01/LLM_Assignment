from enum import Enum
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class VectorizationMethod(str, Enum):
    openai = "openai"
    tfidf = "tfidf"

class PipelineConfig(BaseSettings):
    # File paths
    data_file: str = Field(..., description="Input CSV file containing raw text data")
    prepared_data_file: str = Field(..., description="Output CSV file for prepared (augmented) data")
    feature_extraction_file: str = Field(..., description="Output CSV file after feature extraction")
    vectorized_output_file: str = Field(..., description="Output CSV file after text vectorization")
    anomaly_detection_file: str = Field(..., description="Output CSV file after anomaly detection")

    # Metadata file paths
    metadata_data_preparation: str = Field(..., description="JSON file for data preparation metadata")
    metadata_feature_extraction: str = Field(..., description="JSON file for feature extraction metadata")
    metadata_anomaly_detection: str = Field(..., description="JSON file for anomaly detection metadata")
    metadata_evaluation: str = Field(..., description="JSON file for evaluation metadata")

    # Experiment configuration
    contamination_values: List[float] = Field(default=[0.05, 0.1, 0.15],
                                              description="List of contamination values for experiments")
    true_anomalies: int = Field(default=0, description="List of indices representing true anomalies for evaluation")

    # Pipeline options
    vectorization_method: VectorizationMethod = Field(default=VectorizationMethod.openai, description="Vectorization method")

    # OpenAI API configuration
    openai_api_key: str = Field(..., description="Your OpenAI API key")
    azure_endpoint: str = Field(..., description="Your Azure OpenAI endpoint")
    api_version: str = Field(..., description="API version for Azure OpenAI")
    
    openai_embedding_api_key: str = Field(..., description="Your OpenAI embedding API key")
    azure_embedding_endpoint: str = Field(..., description="Your Azure OpenAI embedding endpoint")
    embedding_api_version: str = Field(..., description="API version for Azure OpenAI embedding")

    images_file: str = Field(..., description="Output file for images")
    class Config:
        env_file = ".env"
