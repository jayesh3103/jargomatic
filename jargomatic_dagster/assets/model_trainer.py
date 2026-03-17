"""
Dagster asset: trains the T5 model on training_data.jsonl and logs run + artifacts to MLflow.
"""
import os
import mlflow
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from dagster import asset, AssetExecutionContext
from mlflow.exceptions import MlflowException
from ..constants import (
    BASE_MODEL_NAME,
    TRAINING_DATA_FILE,
    MODEL_V1_DIR,
    MODEL_CHECKPOINTS_DIR,
    MODEL_MAX_LENGTH,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE,
    TRAINING_WARMUP_RATIO,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT
)
from .data_versioner import data_versioner


@asset(deps=[data_versioner])
def model_trainer(context: AssetExecutionContext, data_versioner: dict) -> dict:
    """Runs training + logs metrics, model artifacts, and params to MLflow."""
    if data_versioner.get("status") != "success":
        return {"status": "skipped", "run_id": None}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Ensure experiment exists with correct artifact location
    try:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri(),
        )
    except MlflowException:
        pass

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

            def preprocess(examples):
                inputs = ["Translate to corporate jargon: " + text for text in examples["input"]]
                targets = examples["output"]
                model_inputs = tokenizer(inputs, max_length=MODEL_MAX_LENGTH, truncation=True, padding="max_length")
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(targets, max_length=MODEL_MAX_LENGTH, truncation=True, padding="max_length")
                # Replace padding token IDs with -100 so loss ignores them
                model_inputs["labels"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in row]
                    for row in labels["input_ids"]
                ]
                return model_inputs

            raw_dataset = load_dataset("json", data_files=str(TRAINING_DATA_FILE), split="train")
            tokenized = raw_dataset.map(preprocess, batched=True)

            model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

            total_steps = (len(raw_dataset) // TRAINING_BATCH_SIZE) * TRAINING_EPOCHS
            warmup_steps = int(TRAINING_WARMUP_RATIO * total_steps)

            training_args = TrainingArguments(
                output_dir=str(MODEL_CHECKPOINTS_DIR),
                per_device_train_batch_size=TRAINING_BATCH_SIZE,
                num_train_epochs=TRAINING_EPOCHS,
                learning_rate=TRAINING_LEARNING_RATE,
                warmup_steps=warmup_steps,
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=3,
                report_to="none",
            )

            mlflow.log_params({
                "model_name": BASE_MODEL_NAME,
                "learning_rate": TRAINING_LEARNING_RATE,
                "num_train_epochs": TRAINING_EPOCHS,
                "batch_size": TRAINING_BATCH_SIZE,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
            })

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                tokenizer=tokenizer,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            )

            result = trainer.train()
            final_loss = result.metrics["train_loss"]
            mlflow.log_metric("final_loss", final_loss)

            os.makedirs(MODEL_V1_DIR, exist_ok=True)
            trainer.save_model(str(MODEL_V1_DIR))
            tokenizer.save_pretrained(str(MODEL_V1_DIR))
            mlflow.log_artifacts(str(MODEL_V1_DIR), artifact_path="model")

            return {"status": "success", "run_id": run_id, "final_loss": final_loss}

        except Exception as e:
            mlflow.log_param("status", "failed")
            return {"status": "failed", "run_id": run_id, "error": str(e)}
