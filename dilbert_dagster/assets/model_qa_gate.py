"""
Runs a post-training QA evaluation using a "golden set" and an LLM (Groq) as a semantic judge.
Logs the result to the same MLflow run. The gate passes if the model meets the score threshold.
"""

import torch
import mlflow
import json
import time
from groq import Groq
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dagster import asset, AssetExecutionContext
from ..constants import (
    MODEL_V1_DIR,
    QA_GOLDEN_SET_FILE,
    MODEL_MAX_LENGTH,
    INFERENCE_NUM_BEAMS,
    MLFLOW_TRACKING_URI,
    GROQ_API_KEY,
    GROQ_MODEL,
    QA_JUDGE_PROMPT,
    QA_JUDGE_MIN_SCORE,
    QA_JUDGE_SLEEP_TIME,
)
from .model_trainer import model_trainer


@asset(deps=[model_trainer])
def model_qa_gate(context: AssetExecutionContext, model_trainer: dict) -> dict:
    """Run semantic QA on the newly trained model and log results to the same MLflow run."""
    if model_trainer.get("status") != "success":
        context.log.info("Upstream training failed or was skipped. Skipping QA.")
        return {"status": "skipped", "qa_passed": False}

    run_id = model_trainer.get("run_id")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_key_here":
        context.log.error("GROQ_API_KEY not set. Skipping QA.")
        return {"status": "error_api_key_missing", "qa_passed": False}
    
    client = Groq(api_key=GROQ_API_KEY)

    with mlflow.start_run(run_id=run_id):
        try:
            context.log.info(f"Loading golden QA set: {QA_GOLDEN_SET_FILE}")
            qa_df = load_dataset("json", data_files=str(QA_GOLDEN_SET_FILE), split="train").to_pandas()
            context.log.info(f"Loaded {len(qa_df)} examples.")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            context.log.info(f"Loading trained model from {MODEL_V1_DIR}")
            trained_model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_V1_DIR)).to(device)
            trained_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_V1_DIR))

            prefix = "Translate to corporate jargon: "
            prompts = [prefix + s for s in qa_df["input"]]
            batch = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)

            context.log.info("Running batch inference...")
            with torch.inference_mode():
                outputs = trained_model.generate(
                    **batch,
                    max_new_tokens=MODEL_MAX_LENGTH,
                    num_beams=INFERENCE_NUM_BEAMS,
                    early_stopping=True,
                )

            generated_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            expected_outputs = qa_df["expected_output"].tolist()
            
            context.log.info("Starting LLM-as-a-Judge Evaluation")
            all_scores = []
            
            for i in range(len(qa_df)):
                input_text = qa_df["input"][i]
                model_output = generated_outputs[i]
                
                context.log.info(f"IN:  {input_text}")
                context.log.info(f"OUT: {model_output}")
                context.log.info(f"EXP: {expected_outputs[i]}")

                prompt = QA_JUDGE_PROMPT.format(input=input_text, output=model_output)
                
                try:
                    result = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=GROQ_MODEL,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    
                    data = json.loads(result.choices[0].message.content)
                    score = int(data.get("score", 0))
                    reasoning = data.get("reasoning", "No reasoning provided.")
                    all_scores.append(score)
                    
                    context.log.info(f"SCORE: {score}/5 | REASON: {reasoning}")
                    context.log.info("-" * 20)
                    
                except Exception as e:
                    context.log.error(f"Failed to get QA score for item {i}: {e}")
                    context.log.info("-" * 20)
                
                time.sleep(QA_JUDGE_SLEEP_TIME)

            if not all_scores:
                context.log.error("No QA scores were recorded. Failing gate.")
                return {"status": "failed", "qa_passed": False, "error": "No scores recorded"}

            average_score = sum(all_scores) / len(all_scores)
            passed = average_score >= QA_JUDGE_MIN_SCORE

            context.log.info(f"QA Judge Score: {average_score:.4f} / Threshold: {QA_JUDGE_MIN_SCORE}")
            context.log.info(f"QA Gate Passed: {passed}")

            mlflow.log_metric("qa_judge_score", average_score)
            mlflow.log_param("qa_gate_passed", str(passed))

            return {"status": "success", "qa_passed": passed, "qa_score": average_score}

        except Exception as e:
            context.log.error(f"Model QA failed: {e}")
            mlflow.log_param("qa_gate_passed", "false_error")
            return {"status": "failed", "qa_passed": False, "error": str(e)}
