from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from src.config import VectorizationMethod

class PipelineRunRequest(BaseModel):
    data_file: Optional[str] = None
    prepared_data_file: Optional[str] = None
    feature_extraction_file: Optional[str] = None
    vectorized_output_file: Optional[str] = None
    anomaly_detection_file: Optional[str] = None
    metadata_data_preparation: Optional[str] = None
    metadata_feature_extraction: Optional[str] = None
    metadata_anomaly_detection: Optional[str] = None
    metadata_evaluation: Optional[str] = None
    contamination_values: Optional[List[float]] = None
    true_anomalies: Optional[List[int]] = None
    vectorization_method: Optional[VectorizationMethod] = None
    openai_api_key: Optional[str] = None

class PipelineStatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    timestamp: datetime
