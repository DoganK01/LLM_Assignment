import logging
import concurrent.futures
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc

from utils import load_dataset, update_metadata, parse_embedding
from anomaly_detection import train_isolation_forest
from config import PipelineConfig
from openai_client import init_openai_client

def experiment_with_contamination_parallel(df: pd.DataFrame, contamination_values: List[float], num_workers: int = 3) -> Dict[float, int]:
    """
    Runs experiments in parallel over different contamination values.
    Returns a dictionary mapping each contamination value to the number of detected anomalies.
    """
    X = np.array(df["embedding"].tolist())
    
    def run_experiment(contam: float) -> Tuple[float, int]:
        _, _, anomaly_predictions = train_isolation_forest(X, contamination=contam)
        anomaly_count = int(sum(anomaly_predictions == -1))
        return contam, anomaly_count

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_contam = {executor.submit(run_experiment, c): c for c in contamination_values}
        for future in concurrent.futures.as_completed(future_to_contam):
            contam, anomaly_count = future.result()
            results[contam] = anomaly_count

    logging.info(f"Parallel experiment results: {results}")
    return results

def main_pipeline_evaluation(config: PipelineConfig, contamination_values: List[float]=[0.05, 0.15], 
                             true_anomalies=["ff7c18a8-9b81-4612-a89a-10f63471c854"]) -> None:
    """
    Pipeline step 4: Evaluation.
      - Loads the anomaly detection results.
      - Calculates performance metrics (precision, recall) by comparing detected anomalies with known true anomalies.
      - Runs parallel experiments for different contamination values.
      - Updates evaluation metadata.
    """
    df = load_dataset(config.anomaly_detection_file)
    df['embedding'] = df['embedding'].apply(parse_embedding)
    df = df[df['embedding'].notnull()]
    try:
        y_true = [1 if idx in true_anomalies else 0 for idx in range(len(df))]
        y_pred = [0 if is_anomaly else 1 for is_anomaly in df.get("is_anomaly", [False]*len(df))]
        y_scores = -df["anomaly_score"].values

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_curve, precision_curve)
        avg_precision = average_precision_score(y_true, y_scores)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    except Exception as e:
        logging.error(f"Error during performance evaluation: {e}")
        precision, recall, f1, roc_auc, pr_auc, avg_precision, tn, fp, fn, tp = [None] * 10

    logging.info(f"Evaluation - Precision: {precision}, Recall: {recall}, F1: {f1}")
    logging.info(f"ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}, Average Precision: {avg_precision}")
    logging.info(f"Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    experiment_results = experiment_with_contamination_parallel(df, contamination_values)
    metadata = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": avg_precision,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "experiments": experiment_results
    }
    update_metadata(metadata, config.metadata_evaluation) #ff7c18a8-9b81-4612-a89a-10f63471c854

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = PipelineConfig()
    client_text, _ = init_openai_client(config)
    main_pipeline_evaluation(config)