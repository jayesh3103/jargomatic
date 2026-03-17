"""
It checks the result of the `model_qa_gate`. If the gate passed,
this asset:
1. Connects to MLflow using the MlflowClient.
2. Creates a new version of the registered model from the
   training run's ID.
3. Transitions this new model version to the "Staging" stage.
"""

import time
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from dagster import asset, AssetExecutionContext
from ..constants import MLFLOW_TRACKING_URI, BENTO_MODEL_NAME
from .model_qa_gate import model_qa_gate
from .model_trainer import model_trainer


@asset(deps=[model_qa_gate, model_trainer])
def model_registry_promoter(
    context: AssetExecutionContext, 
    model_qa_gate: dict,
    model_trainer: dict
) -> dict:
    """If QA passed, register a new model version from the training run and promote it to Staging."""
    if not model_qa_gate.get("qa_passed"):
        context.log.info("QA skipped or failed. Skipping promotion.")
        return {"status": "skipped", "model_version": None}

    run_id = model_trainer.get("run_id")
    if not run_id:
        context.log.error("Missing run_id from model_trainer.")
        return {"status": "failed", "error": "Missing run_id"}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model_name = BENTO_MODEL_NAME

    try:
        # Ensure the registered model exists
        try:
            client.create_registered_model(registered_model_name)
            context.log.info(f"Created registered model: {registered_model_name}")
        except MlflowException:
            context.log.info(f"Registered model {registered_model_name} already exists.")

        # Create a new version from the training run artifact
        model_uri = f"runs:/{run_id}/model"
        context.log.info(f"Creating model version from: {model_uri}")
        mv = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=run_id,
            description="Automated model from Dagster retraining pipeline.",
        )
        context.log.info(f"Created model version: {mv.name} v{mv.version}")

        # Wait for the version to be READY
        status = None
        for _ in range(10):
            status = client.get_model_version(mv.name, mv.version).status
            if status == "READY":
                break
            context.log.info("Model version not ready; waiting 10s...")
            time.sleep(10)

        if status != "READY":
            context.log.error("Model version failed to become READY.")
            return {"status": "failed", "error": "Model version not READY"}

        # Promote to Staging
        context.log.info(f"Transitioning version {mv.version} to 'Staging'...")
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        context.log.info("Model promotion complete.")
        return {"status": "success", "model_version": mv.version, "registered_model_name": mv.name}

    except Exception as e:
        context.log.error(f"Model promotion failed: {e}")
        return {"status": "failed", "error": str(e)}
