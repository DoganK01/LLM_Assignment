import requests
import json
import time
import uvicorn
import threading
from time import sleep
import sys
import os
# Optional

# from app import app

# Optional
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_URL = "http://localhost:8000"

def check_pipeline_status():
    """
    Check the current status of the pipeline.
    
    Returns:
        dict: The pipeline status response
    """
    response = requests.get(f"{BASE_URL}/pipeline/status")
    return response.json()

def run_pipeline_with_defaults():
    """
    Run the pipeline with default configuration from .env file.
    
    Returns:
        dict: The pipeline start response
    """
    response = requests.post(f"{BASE_URL}/pipeline/run", json={})
    return response.json()

def run_pipeline_with_custom_config():
    """
    Run the pipeline with custom configuration parameters.
    
    Returns:
        dict: The pipeline start response
    """

    custom_config = {
        "data_file": "data/custom_data.csv",
        "prepared_data_file": "data/custom_prepared_data.csv",
        "feature_extraction_file": "data/custom_feature_extraction.csv",
        "vectorized_output_file": "data/custom_vectorized_output.csv",
        "anomaly_detection_file": "data/custom_anomaly_detection.csv",
        "metadata_data_preparation": "data/custom_metadata_data_preparation.json",
        "metadata_feature_extraction": "data/custom_metadata_feature_extraction.json",
        "metadata_anomaly_detection": "data/custom_metadata_anomaly_detection.json",
        "metadata_evaluation": "data/custom_metadata_evaluation.json",
        "contamination_values": [0.03, 0.07, 0.12],
        "true_anomalies": [5, 15, 25], 
        "vectorization_method": "openai",
        "openai_api_key": "your_api_key"
    }
    
    response = requests.post(f"{BASE_URL}/pipeline/run", json=custom_config)
    return response.json()

def poll_until_pipeline_completion(timeout_seconds=5000, interval_seconds=10):
    """
    Poll the status endpoint until the pipeline completes or times out.
    
    Args:
        timeout_seconds (int): Maximum time to wait in seconds
        interval_seconds (int): Time between status checks in seconds
        
    Returns:
        dict: The final status response or None if timed out
    """
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        status_response = check_pipeline_status()
        
        if status_response["status"] == "completed":
            print(f"Pipeline completed at: {status_response['timestamp']}")
            return status_response
        
        print(f"Pipeline status: {status_response['status']} - {status_response.get('message', '')}")
        time.sleep(interval_seconds)
    
    print(f"Timed out after {timeout_seconds} seconds")
    return None

def full_pipeline_execution_example():
    """
    Complete example of running a pipeline and waiting for results.
    """
    start_response = run_pipeline_with_defaults()
    print("Pipeline started:")
    print(json.dumps(start_response, indent=2))
    
    final_status = poll_until_pipeline_completion()
    
    if final_status and final_status["status"] == "completed":
        print("Pipeline execution successful!")
    else:
        print("Pipeline execution failed or timed out.")

if __name__ == "__main__":
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
        ),
        daemon=True
    )
    server_thread.start()
    print("Waiting for server startup...")
    sleep(10)

    try:
        print("\n===== Simple Status Check =====")
        status = check_pipeline_status()
        print(json.dumps(status, indent=2))
        
        print("\n===== Running Full Pipeline Example =====")
        full_pipeline_execution_example()
    
    except requests.exceptions.ConnectionError:
        print("\n⚠️ Connection failed! Ensure:")
        print("1. Server is running")
        print("2. No port conflicts (other services using port 8000)")
        print("3. Firewall allows localhost connections")