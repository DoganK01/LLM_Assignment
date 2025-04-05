import uvicorn
import logging
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.config import PipelineConfig
from src.models import PipelineRunRequest, PipelineStatusResponse
from src.openai_client import init_openai_client

from src.data_preparation import main_pipeline_data_preparation
from src.feature_extraction import main_pipeline_feature_extraction
from src.anomaly_detection import main_pipeline_anomaly_detection
from src.evaluation import main_pipeline_evaluation

pipeline_status = {"status": "idle", "timestamp": datetime.utcnow()}

app = FastAPI(title="Advanced Pipeline REST API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

def get_pipeline_config() -> PipelineConfig:
    try:
        return PipelineConfig()
    except Exception as e:
        logging.error("Configuration error: " + str(e))
        raise HTTPException(status_code=500, detail="Configuration error")

def run_complete_pipeline(config: PipelineConfig) -> None:
    global pipeline_status
    logging.info("Starting complete pipeline...")
    
    client_text, client_embedding = init_openai_client(config)
    
    logging.info("=== Data Preparation ===")
    main_pipeline_data_preparation(
        client_text,
        config,
    )
    
    logging.info("=== Feature Extraction & Vectorization ===")
    main_pipeline_feature_extraction(
        client_text,
        client_embedding,
        config,
    )
    
    logging.info("=== Anomaly Detection ===")
    main_pipeline_anomaly_detection(
        client_text,
        config,
    )
    
    logging.info("=== Evaluation ===")
    main_pipeline_evaluation(
        config
    )
    
    pipeline_status = {"status": "completed", "timestamp": datetime.utcnow()}
    logging.info("Pipeline execution completed at " + str(pipeline_status["timestamp"]))

@app.get("/pipeline/status", response_model=PipelineStatusResponse)
def get_status() -> PipelineStatusResponse:
    """Returns the current status of the pipeline."""
    return PipelineStatusResponse(
        status=pipeline_status["status"],
        timestamp=pipeline_status["timestamp"],
        message=f"Pipeline is {pipeline_status['status']}"
    )

@app.post("/pipeline/run", response_model=PipelineStatusResponse)
def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    config: PipelineConfig = Depends(get_pipeline_config)
):
    """
    Triggers the complete pipeline run in the background.
    Overrides configuration via request payload if provided.
    """
    if request.data_file:
        config.data_file = request.data_file
    if request.prepared_data_file:
        config.prepared_data_file = request.prepared_data_file
    if request.feature_extraction_file:
        config.feature_extraction_file = request.feature_extraction_file
    if request.vectorized_output_file:
        config.vectorized_output_file = request.vectorized_output_file
    if request.anomaly_detection_file:
        config.anomaly_detection_file = request.anomaly_detection_file
    if request.metadata_data_preparation:
        config.metadata_data_preparation = request.metadata_data_preparation
    if request.metadata_feature_extraction:
        config.metadata_feature_extraction = request.metadata_feature_extraction
    if request.metadata_anomaly_detection:
        config.metadata_anomaly_detection = request.metadata_anomaly_detection
    if request.metadata_evaluation:
        config.metadata_evaluation = request.metadata_evaluation
    if request.contamination_values:
        config.contamination_values = request.contamination_values
    if request.true_anomalies:
        config.true_anomalies = request.true_anomalies
    if request.vectorization_method:
        config.vectorization_method = request.vectorization_method
    if request.openai_api_key:
        config.openai_api_key = request.openai_api_key

    background_tasks.add_task(run_complete_pipeline, config)
    return PipelineStatusResponse(
        status="started",
        message="Pipeline execution started",
        timestamp=datetime.utcnow()
    )

if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
