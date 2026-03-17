# Jargomatic 🤖

**Developed by Jayesh Muley**

Jargomatic is an end-to-end MLOps project that takes plain English and "operationalizes" it into enterprise-grade corporate jargon.

It is a full machine learning pipeline that generates its own training data, versions it, fine-tunes a local T5 model and serves it via a web UI.

## What it does

If you type: _"We need to fix this bug."_ The model outputs: _"We need to leverage our core competencies to operationalize a solution for this issue."_

## Architecture

This project is built to demonstrate a complete lifecycle using modern MLOps tools:

- **Orchestration (Dagster):** Manages the entire pipeline from data ingestion to model training.
- **Model (HuggingFace T5):** A local `google/flan-t5-small` model fine-tuned specifically on corporate speak.
- **Data Generation (Groq):** Uses a "Weak Labeler" asset to generate synthetic training pairs using the Groq API when new jargon is detected.
- **Versioning (DVC):** Tracks datasets (`training_data.jsonl`, `style_guide.jsonl`) so training is reproducible.
- **Tracking (MLflow):** Logs training runs, metrics (loss), and model artifacts.
- **Frontend (Streamlit):** A clean interface for interacting with the final model.

## The Pipeline

The logic is contained in `jargomatic_dagster` and runs through these steps:

1.  **Drift Detection:** Checks for new jargon or data shifts.
2.  **Weak Labeling:** If new data is needed, it hits the Groq API to generate new (Plain English -\> Jargon) pairs.
3.  **Data Versioning:** Updates DVC with the new dataset state.
4.  **Training:** Fine-tunes the T5 model on the updated data and logs the results to MLflow.
5.  **Quality Gate:** Checks if the model meets performance thresholds before promotion (optional).

## Setup

### Prerequisites

- Python 3.9+
- Git
- A Groq API Key (for data generation)

### Installation

1.  **Clone the repo**

    ```bash
    git clone https://github.com/your-username/jargomatic.git
    cd jargomatic
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull Data**
    Since data is versioned with DVC, you need to pull the actual files:

    ```bash
    dvc pull
    ```

4.  **Environment Variables**
    Create a `.env` file or export your API key for the weak labeler:

    ```bash
    export GROQ_API_KEY="your_groq_api_key"
    ```

## Usage

### 1\. Start MLflow Server

The pipeline logs metrics to a local MLflow server. You must start this before running the pipeline.

1.  **Create the storage directories:**

    ```bash
    mkdir -p mlflow_store/artifacts
    ```

2.  **Start the server:**

    ```bash
    mlflow server --backend-store-uri sqlite:///%cd%\mlflow_store\mlflow.db --default-artifact-root %cd%\mlflow_store\artifacts --host 127.0.0.1 --port 5000
    ```

    _Keep this terminal open._

### 2\. Run the Pipeline (Dagster)

To train the model or generate new data, open a new terminal and launch the Dagster UI:

```bash
dagster dev
```

Navigate to `localhost:3000` to visualize and materialize the assets (specifically `model_trainer` and `weak_labeler`).

### 3\. Run the App (Streamlit)

Once you have a trained model in `models/t5_jargon_v1`, you can launch the frontend:

```bash
streamlit run streamlit_app.py
```

_Note: If the app complains that the model is missing, run the training pipeline in Dagster first._

## Directory Structure

- `jargomatic_dagster/`: Contains the pipeline assets, sensors, and definitions.
- `models/`: Stores the fine-tuned T5 binaries (after training).
- `mlflow_store/`: Stores MLflow run data and artifacts.
- `data/`: Contains DVC-tracked JSONL files for training and style guides.
- `streamlit_app.py`: The user interface.
