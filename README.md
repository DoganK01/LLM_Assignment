# LLM_Assignment



Below is an advanced, detailed, and comprehensive documentation in Markdown (.md) format, tailored to your project. It covers how Python, AI, machine learning, and language models are implemented and used, with a focus on technical aspects. This documentation is designed to be exhaustive and serve as a reference for developers and data scientists working on or extending your project.

---

# Advanced Technical Documentation: Anomaly Detection Pipeline

This document provides an in-depth exploration of the implementation and utilization of Python, artificial intelligence (AI), machine learning (ML), and language models within an anomaly detection pipeline. The project integrates data preparation, feature extraction, anomaly detection, and evaluation into a cohesive system, leveraging modern Python libraries, OpenAI's GPT-4o for natural language processing (NLP), and scikit-learn for ML tasks. Below, we dissect each component, its technical underpinnings, and the interplay of AI/ML techniques.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Python Implementation](#python-implementation)
5. [AI and Language Models](#ai-and-language-models)
6. [Machine Learning Techniques](#machine-learning-techniques)
7. [Data Flow and Processing](#data-flow-and-processing)
8. [Visualization and Interpretability](#visualization-and-interpretability)
9. [API Integration](#api-integration)
10. [Scalability and Performance](#scalability-and-performance)
11. [Error Handling and Logging](#error-handling-and-logging)
12. [Future Enhancements](#future-enhancements)
13. [Comments](#comments)

---

## Project Overview

The anomaly detection pipeline is a multi-stage system designed to process text data, identify outliers (anomalies), and evaluate performance. It comprises four core stages:
- **Data Preparation**: Augments text data using GPT-4o.
- **Feature Extraction & Vectorization**: Extracts semantic features and converts text into numerical embeddings.
- **Anomaly Detection**: Applies an Isolation Forest model to detect anomalies in the vectorized data.
- **Evaluation**: Assesses the model's performance against known anomalies.

The pipeline is orchestrated via a FastAPI-based RESTful API, enabling asynchronous execution and status monitoring. This project exemplifies the fusion of AI-driven NLP and unsupervised ML for anomaly detection.

---

## Technology Stack

The project leverages a robust stack of Python libraries and tools:

- **Python 3.12**: Core programming language for scripting and orchestration.
- **Pandas & NumPy**: Data manipulation and numerical computation.
- **Scikit-learn**: Machine learning algorithms (Isolation Forest, PCA) and evaluation metrics.
- **OpenAI API (GPT-4o)**: Language model for text augmentation, feature extraction, and anomaly summarization.
- **Matplotlib & Seaborn**: Data visualization for anomaly analysis.
- **Graphviz**: Visualization of decision trees in Isolation Forest.
- **FastAPI & Uvicorn**: Asynchronous API framework and server.
- **Concurrent.futures**: Parallel processing for scalability.
- **Pydantic**: Data validation and type enforcement.
- **Logging**: Error tracking and debugging.

---

## Pipeline Architecture

The pipeline follows a modular, sequential architecture:

1. **Data Preparation (`data_preparation.py`)**:
   - Input: Raw text dataset (CSV).
   - Process: Augments data using GPT-4o paraphrasing.
   - Output: Augmented dataset (CSV).

2. **Feature Extraction & Vectorization (`feature_extraction.py`)**:
   - Input: Augmented dataset.
   - Process: Extracts features (entities, sentiment, theme) and vectorizes text (TF-IDF or OpenAI embeddings).
   - Output: Vectorized dataset with features (CSV).

3. **Anomaly Detection (`anomaly_detection.py`)**:
   - Input: Vectorized dataset.
   - Process: Trains an Isolation Forest model, detects anomalies, and summarizes them with GPT-4o.
   - Output: Dataset with anomaly scores and summaries (CSV), visualizations (PNG).

4. **Evaluation (`evaluation.py`)**:
   - Input: Anomaly-detected dataset, true anomaly labels.
   - Process: Computes performance metrics (precision, recall, F1, ROC-AUC).
   - Output: Metadata with evaluation results (JSON).

5. **API Layer (`app.py` & `instance.py`)**:
   - Orchestrates pipeline execution and provides status endpoints.

---

## Python Implementation

Python serves as the backbone of the project, enabling modular design, efficient data handling, and integration with AI/ML libraries.

### Key Python Features Used
- **Type Hints**: Functions use type annotations (e.g., `Tuple[IsolationForest, np.ndarray, np.ndarray]`) for clarity and static type checking.
- **Exception Handling**: Comprehensive `try-except` blocks ensure robustness (e.g., in `train_isolation_forest`).
- **Parallel Processing**: `concurrent.futures.ThreadPoolExecutor` accelerates tasks like feature extraction and anomaly summarization.
- **Object-Oriented Design**: `PipelineConfig` and `AugmentedText` (via Pydantic) encapsulate configuration and data models.
- **File I/O**: Custom utilities (`load_dataset`, `save_dataset`) handle CSV and JSON operations.

### Example: Parallel Processing in `summarize_anomalies_parallel`
```python
def summarize_anomalies_parallel(client, df: pd.DataFrame, num_workers: int = 5) -> List[str]:
    anomaly_texts = df[df["is_anomaly"]]["text"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        summaries = list(executor.map(lambda t: summarize_anomalies(client, t), anomaly_texts))
    return summaries
```
This leverages Python's threading capabilities to parallelize GPT-4o summarization, improving throughput for large datasets.

---

## AI and Language Models

The OpenAI GPT-4o model is a cornerstone of the pipeline, enhancing data augmentation, feature extraction, and anomaly interpretation.

### Integration with OpenAI API
- **Client Initialization**: `init_openai_client` (in `src.openai_client`) configures two clients: one for text generation (`client_text`) and one for embeddings (`client_embedding`).
- **API Calls**: Structured using `client.chat.completions.create` for text tasks and `client.embeddings.create` for vectorization.

### Use Cases
1. **Text Augmentation (`data_preparation.py`)**
   - GPT-4o paraphrases input texts with high creativity (`temperature=1.5`) to enrich the dataset.
   - Example:
     ```python
     response = client.chat.completions.create(
         model="gpt-4o",
         messages=[{"role": "user", "content": f"Paraphrase this: '{text}'"}],
         max_tokens=100,
         temperature=1.5,
     )
     ```
   - Output: Synthetic texts stored with unique IDs and timestamps.

2. **Feature Extraction (`feature_extraction.py`)**
   - GPT-4o extracts entities, sentiment, and themes via structured parsing (`response_format=Features`).
   - Warning: The current implementation uses `eval` for parsing, which is insecure in production; JSON parsing is recommended.
   - Example:
     ```python
     class Features(BaseModel):
         entities: List[str]
         sentiment: str
         theme: str

     
     response = client.beta.chat.completions.parse(
         model="gpt-4o",
         messages=[{"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": f"Extract entities, sentiment, and theme from by going through step by step: '{text}'"}],
         response_format=Features,
     )
     ```

3. **Anomaly Summarization (`anomaly_detection.py`)**
   - GPT-4o summarizes why flagged texts are anomalous, providing human-readable insights.
   - Example:
     ```python
     response = client.chat.completions.create(
         model="gpt-4o",
         messages=[{"role": "user", "content": f"Summarize why this text might be an anomaly: '{text}'"}],
     )
     ```

### Technical Considerations
- **Latency**: API calls introduce network latency; parallelization mitigates this.
- **Cost**: Frequent GPT-4o usage can be expensive; batching requests could optimize costs.
- **Error Handling**: Fallbacks return original text or default values if API calls fail.

---

## Machine Learning Techniques

The core ML component is the Isolation Forest algorithm, an unsupervised method for anomaly detection, implemented via scikit-learn.

### Isolation Forest (`anomaly_detection.py`)
- **Purpose**: Identifies anomalies by isolating data points in a feature space.
- **Implementation**:
  ```python
  model = IsolationForest(
      contamination=0.1,
      n_estimators=200,
      max_samples=50,   # 33 samples created
      max_features=100, # Because TFID vector has 100 feautres.
      random_state=42,
      n_jobs=-1
  )
  model.fit(embeddings)
  ```

### Dimensionality Reduction with PCA
- **Purpose**: Reduces high-dimensional embeddings to 2D for visualization.
- **Implementation**:
  ```python
  pca = PCA(n_components=2, random_state=42)
  embeddings_2d = pca.fit_transform(embeddings)
  ```
- **Usage**: Enables scatter plots of anomaly scores in 2D space.

### Evaluation Metrics (`evaluation.py`)
- **Metrics Computed**:
  - Precision, Recall, F1: Assess binary classification performance.
  - ROC-AUC: Measures ranking quality of anomaly scores.
  - PR-AUC & Average Precision: Focus on positive (anomaly) class performance.
  - Confusion Matrix: Detailed breakdown of TP, FP, TN, FN.
- **Implementation**:
  ```python
  precision = precision_score(y_true, y_pred, zero_division=0)
  roc_auc = roc_auc_score(y_true, y_scores)
  ```

### Experimentation
- **Parallel Contamination Testing**:
  - Tests multiple `contamination` values (e.g., `[0.05, 0.15]`) to analyze sensitivity.
  - Uses `concurrent.futures` for efficiency.

---

## Data Flow and Processing

### Data Structure
- **Input**: CSV with a `text` column.
- **Intermediate**:
  - Augmented data adds `id` and `timestamp`.
  - Feature extraction adds `entities`, `sentiment`, `theme`, and `embedding`.
  - Anomaly detection adds `anomaly_score`, `is_anomaly`, and `anomaly_summary`.
- **Output**: CSV with all columns, plus JSON metadata.

### Processing Steps
1. **Loading**: `load_dataset` reads CSVs into Pandas DataFrames.
2. **Embedding Parsing**: `parse_embedding` converts string embeddings to NumPy arrays.
3. **Saving**: `save_dataset` persists DataFrames to CSV.

## Visualization and Interpretability

Visualizations enhance understanding of anomaly detection results:

1. **Scatter Plot**: 2D PCA projection with anomaly scores and flagged anomalies.
2. **Histogram**: Distribution of anomaly scores with decision boundary.
3. **Decision Tree**: Graphviz rendering of an Isolation Forest tree.
4. **Heatmap**: Top 10 feature contributions for anomalous samples.

### Technical Details
- **PCA**: Reduces dimensionality for scatter plots.
- **Graphviz**: Custom `export_tree_to_dot` function recursively builds DOT format for tree visualization.
- **Seaborn**: Heatmaps and histograms leverage statistical plotting capabilities.

---

## API Integration

### FastAPI (`app.py`)
- **Endpoints**:
  - `GET /pipeline/status`: Returns current pipeline status.
  - `POST /pipeline/run`: Triggers pipeline execution with configurable parameters.
- **Asynchronous Execution**: `BackgroundTasks` runs the pipeline non-blocking.
- **CORS**: Enabled for cross-origin requests.

### Client Interaction (`instance.py`)
- **Polling**: `poll_until_pipeline_completion` checks status periodically.
- **Custom Config**: Allows overriding defaults via JSON payload.

---

## Scalability and Performance

- **Parallelization**: ThreadPoolExecutor speeds up GPT-4o calls and experiments.
- **Multiprocessing**: Isolation Forest uses `n_jobs=-1` for CPU parallelism.
- **Bottlenecks**: Network latency from OpenAI API calls; mitigated by batching and caching potential.
- **Memory**: Large embeddings may strain RAM; PCA or sparse formats could help.

---

## Error Handling and Logging

- **Try-Except**: Wraps all critical operations (e.g., model training, API calls).
- **Logging**: `logging.info` and `logging.error` track progress and issues.
- **Fallbacks**: Default values (e.g., "unknown" for sentiment) ensure pipeline continuity.

---

## Future Enhancements

Some limitations have been encountered since the synthetic data generated was not diverse and numerically large enough.

1. **Model Tuning**: Hyperparameter optimization for Isolation Forest. We can use search algorithms as GridSearchCV for this purpose if we had larger and diverse dataset.
2. **Pre-process**: If the dataset consisted of data that was not between 0 and 1, they would be scaled before fitting the model. Very high dimensionality in data could introduce noise or computational overhead. Techniques like PCA could reduce the number of dimensions while preserving most of the variance. In our case (100) dimensions (features) are big enough to handle for IsolationForest.
3. **Alternative Algorithms**: Experiment with other outlier detection algorithms.
4. **Real-Time Processing**: Stream data via WebSockets and better API implementation with cool UI.
5. **Distributed Computing**: Use a real dataset to work on.

---


# Installation and Usage


## 1. Clone the Repository

First, clone the Git repository to your local machine:

```bash
git clone https://github.com/DoganK01/LLM_Assignment.git
```

Navigate to the project directory:

```bash
cd LLM_Assignment
```

## 2. Configure Environment Variables

Copy the stub `.env.copy` file to `.env` and replace the placeholder values with the required credentials:

```bash
cp .env.copy .env
```

Make sure to update the values in `.env` with your actual credentials, such as API keys, endpoints or any other necessary information.

## 3. Install `uv` and Dependencies

### 3.1 Install `uv`

Install **`uv`** by following the instructions from the official documentation. For example, you can install `uv` via the following command:

```bash
pip install uv
```

For more details on installing `uv`, visit the [official documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

### 3.2 Install Dependencies

After installing `uv`, you can sync all dependencies and packages for the project by running:

```bash
uv sync --all-groups
```

This will install all the required dependencies specified for the project.

## 4. Running the Application Locally

Now that the dependencies are installed, you can run the application using `uv`. To start the app, run:

```bash
uv run instance.py
```

This will start the application, and it should be accessible locally.


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
