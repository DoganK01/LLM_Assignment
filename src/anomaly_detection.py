import logging
import concurrent.futures
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import graphviz
from io import StringIO

from src.utils import load_dataset, save_dataset, update_metadata, parse_embedding
from src.config import PipelineConfig
from src.openai_client import init_openai_client


def train_isolation_forest(embeddings: np.ndarray, contamination: float = 0.1) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    Trains an Isolation Forest model on the embedding data.
    Returns the trained model, anomaly scores, and predictions.
    """
    try:
        model = IsolationForest(contamination=contamination, n_estimators=200, max_samples=50, max_features=100, random_state=42, n_jobs=-1)
        model.fit(embeddings)
        anomaly_scores = model.decision_function(embeddings)
        anomaly_predictions = model.predict(embeddings)
        return model, anomaly_scores, anomaly_predictions
    except Exception as e:
        logging.error(f"Error training Isolation Forest: {e}")
        raise

def detect_anomalies(df: pd.DataFrame, model: IsolationForest, anomaly_scores: np.ndarray, anomaly_predictions: np.ndarray) -> pd.DataFrame:
    """
    Attaches anomaly scores and flags rows as anomalies.
    """
    try:
        df["anomaly_score"] = anomaly_scores
        df["is_anomaly"] = anomaly_predictions == -1
        logging.info(f"Anomalies detected: {df['is_anomaly'].sum()} out of {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")
        raise

def summarize_anomalies(client, text: str) -> str:
    """
    Uses GPT-4o to summarize why a text might be an anomaly.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize why this text might be an anomaly: '{text}'"}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing anomaly: {e}")
        return "Summary unavailable"

def summarize_anomalies_parallel(client, df: pd.DataFrame, num_workers: int = 5) -> List[str]:
    """
    Runs anomaly summarization in parallel for texts flagged as anomalies.
    """
    anomaly_texts = df[df["is_anomaly"]]["text"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        summaries = list(executor.map(lambda t: summarize_anomalies(client, t), anomaly_texts))
    return summaries

def visualize_anomaly_results(df: pd.DataFrame, embeddings: np.ndarray, model: IsolationForest, output_dir: str):
    """
    Visualizes the results of the anomaly detection.
    Generates scatter plots and histograms of anomaly scores.
    """
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    df["pca_x"] = embeddings_2d[:, 0]
    df["pca_y"] = embeddings_2d[:, 1]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df["pca_x"], df["pca_y"], c=df["anomaly_score"], cmap="RdYlGn", alpha=0.6, s=50)
    plt.colorbar(scatter, label="Anomaly Score")
    anomalies = df[df["is_anomaly"]]
    plt.scatter(anomalies["pca_x"], anomalies["pca_y"], c="red", label="Anomalies", edgecolor="black", s=100)
    plt.title("2D Embedding Space with Anomaly Scores")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(f"{output_dir}/anomaly_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["anomaly_score"], bins=30, kde=True, color="blue")
    plt.axvline(x=0, color="red", linestyle="--", label="Decision Boundary (0)")
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{output_dir}/anomaly_score_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    def export_tree_to_dot(tree, feature_names):
        dot_data = StringIO()
        dot_data.write("digraph Tree {\nnode [shape=box] ;\n")
        def recurse(node_id, depth, parent_id=None):
            if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
                dot_data.write(f'{node_id} [label="Leaf"] ;\n')
            else:
                feature = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                dot_data.write(f'{node_id} [label="{feature} <= {threshold:.3f}"] ;\n')
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
                if parent_id is not None:
                    dot_data.write(f'{parent_id} -> {node_id} ;\n')
                recurse(left_child, depth + 1, node_id)
                recurse(right_child, depth + 1, node_id)
        recurse(0, 0)
        dot_data.write("}\n")
        return dot_data.getvalue()

    first_tree = model.estimators_[0]
    feature_names = [f"dim_{i}" for i in range(embeddings.shape[1])]
    dot_data = export_tree_to_dot(first_tree.tree_, feature_names)
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/isolation_tree_1", format="png", cleanup=True)

    n_features = embeddings.shape[1]  # 100
    feature_usage = np.zeros(n_features)
    for tree in model.estimators_:
        features_used = tree.tree_.feature[tree.tree_.feature >= 0]
        for feat in features_used:
            feature_usage[feat] += 1
    feature_importance = feature_usage / feature_usage.sum()
    top_features = np.argsort(feature_importance)[-10:]

    plt.figure(figsize=(12, 8))
    sns.heatmap(embeddings[df["is_anomaly"]][:, top_features], cmap="YlOrRd", annot=False)
    plt.title("Heatmap of Top 10 Features for Anomalous Samples")
    plt.xlabel("Top Features (Indices)")
    plt.ylabel("Anomalous Samples")
    plt.xticks(ticks=np.arange(10), labels=top_features)
    plt.savefig(f"{output_dir}/feature_importance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

def main_pipeline_anomaly_detection(client, config: PipelineConfig, contamination: float = 0.1) -> None:
    """
    Pipeline step 3: Anomaly Detection.
      - Loads vectorized data.
      - Trains the Isolation Forest model.
      - Detects anomalies and attaches scores.
      - Generates summaries for anomalies.
      - Saves the final dataset and updates metadata.
    """
    df = load_dataset(config.vectorized_output_file)
    df['embedding'] = df['embedding'].apply(parse_embedding)
    df = df[df['embedding'].notnull()]

    try:
        X = np.array(df['embedding'].tolist())
    except Exception as e:
        msg = "Embedding column not found. Ensure vectorization step is completed."
        logging.error(msg)
        raise ValueError(msg)
    
    model, anomaly_scores, anomaly_predictions = train_isolation_forest(X, contamination)
    df = detect_anomalies(df, model, anomaly_scores, anomaly_predictions)
    df.loc[df["is_anomaly"], "anomaly_summary"] = summarize_anomalies_parallel(client, df)
    visualize_anomaly_results(df, X, model, config.images_file)
    save_dataset(df, config.anomaly_detection_file)
    metadata = {
            "anomaly_count": int(df["is_anomaly"].sum()),
            "contamination_level": contamination,
            "visualization_files": [
                "anomaly_scatter.png",
                "anomaly_score_histogram.png",
                "isolation_tree_1.png",
                "feature_importance_heatmap.png"
        ]
    }
    update_metadata(metadata, config.metadata_anomaly_detection)
if __name__ == "__main__":
    config = PipelineConfig()
    client_text, _ = init_openai_client(config)
    main_pipeline_anomaly_detection(client_text, config)