"""
Constants for the Jargomatic MLOps Pipeline
Centralized configuration to avoid hardcoding values across the codebase.
"""
import os
from pathlib import Path

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DVC_REMOTE = PROJECT_ROOT / "dvc_remote"

# Model paths
MODEL_V1_DIR = MODELS_DIR / "t5_jargon_v1"
MODEL_CHECKPOINTS_DIR = MODELS_DIR / "t5_jargon_v1_checkpoints"

# DATA PATHS
TRAINING_DATA_FILE = DATA_DIR / "training_data.jsonl"
STYLE_GUIDE_FILE = DATA_DIR / "style_guide.jsonl"
NEW_JARGON_CANDIDATES_FILE = PROJECT_ROOT / "new_jargon_candidates.txt"
KNOWN_JARGON_FILE = DATA_DIR / "known_jargon.csv"
QA_GOLDEN_SET_FILE = DATA_DIR / "qa_golden_set.jsonl"

# MODEL CONFIG
BASE_MODEL_NAME = "google/flan-t5-small"
MODEL_MAX_LENGTH = 128
MODEL_MIN_LENGTH = 20

# --- MLFLOW CONFIG ---
MLFLOW_STORE = PROJECT_ROOT / "mlflow_store"
MLFLOW_ARTIFACT_ROOT = MLFLOW_STORE / "artifacts"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "jargomatic"

# Training parameters
TRAINING_BATCH_SIZE = 4
TRAINING_EPOCHS = 20
TRAINING_LEARNING_RATE = 3e-4
TRAINING_WARMUP_RATIO = 0.15

# Inference parameters
INFERENCE_NUM_BEAMS = 4

# GROQ API (Weak Labeler)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_key_here")
GROQ_MODEL = "llama-3.3-70b-versatile"
NUM_FEW_SHOT_EXAMPLES = 3
GROQ_WEAK_LABELER_PROMPT = """
You are a helpful assistant who generates synthetic training data for a machine learning model.
Your response MUST be a single, valid JSON object.
Your generated pairs MUST match the style, creativity, and tone of the examples provided.

---
EXAMPLES OF THE STYLE I WANT:
{style_examples_str}
---

Now, following that EXACT style, generate 10 new, high-quality input-output pairs for the given corporate jargon word/phrase: '{jargon}'.

Return a single JSON object in the following format:
{{
  "pairs": [
    {{"input": "simple phrase", "output": "jargon-filled phrase"}},
    {{"input": "...", "output":..."}}
  ]
}}
"""

# QA / VALIDATION
QA_JUDGE_MIN_SCORE = 4.0
QA_JUDGE_SLEEP_TIME = 5
QA_JUDGE_PROMPT = """
You are an expert evaluator for a text style transfer model.
Your goal is to evaluate if the "Model Output" is a good corporate jargon translation of the "Original Input".
Your evaluation MUST focus on CONTENT PRESERVATION.

---
EXAMPLES
- Original Input: "The project is delayed."
- Model Output: "We're seeing timeline slippage on this initiative."
- Evaluation: This is a 5/5. The meaning is preserved perfectly.

- Original Input: "The project is delayed."
- Model Output: "The project has been successfully completed."
- Evaluation: This is a 1/5. The meaning is the complete opposite.

- Original Input: "Let's start the project."
- Output: "We should convene a strategic synergy to synergize our efforts."
- Evaluation: This is a 1/5. This is a vague, repetitive hallucination that does not mean 'start'.
---

Now, evaluate this pair. On a scale of 1 to 5, how well did the model preserve the original meaning?
1 = Total meaning change or hallucination.
5 = Perfect meaning preservation.

Return ONLY a single, valid JSON object with your score and reasoning.

Original Input: "{input}"
Model Output: "{output}"

{{
  "score": <1-5>,
  "reasoning": "<Your brief reason for the score>"
}}
"""
# DVC CONFIG
DVC_REMOTE_NAME = "local_storage"
DVC_REMOTE_PATH = str(DVC_REMOTE)

# BENTOML CONFIG
BENTO_MODEL_NAME = "jargon_translator"
BENTO_SERVICE_NAME = "jargon_translator_service"
BENTO_PRODUCTION_TAG = "production"