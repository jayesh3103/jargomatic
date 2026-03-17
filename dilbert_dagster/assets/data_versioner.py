"""
Versions all data assets (training data, known jargon) using DVC
and cleans up the candidate file to "close the loop".
"""
import subprocess
import os
import pandas as pd
from dagster import asset, AssetExecutionContext
from ..constants import (
    PROJECT_ROOT,
    TRAINING_DATA_FILE,
    KNOWN_JARGON_FILE,
    NEW_JARGON_CANDIDATES_FILE
)

@asset
def data_versioner(
    context: AssetExecutionContext, 
    weak_labeler: dict, 
    jargon_drift_detector: dict
) -> dict:
    """Versions new data with DVC, updates known jargon, and clears candidates file."""
    
    weak_labeler_status = weak_labeler.get("status")
    if weak_labeler_status != "success":
        context.log.info(f"Weak labeler status was '{weak_labeler_status}'. Skipping.")
        return {"status": "skipped_no_new_data"}
        
    new_pairs = weak_labeler.get("new_pairs_generated", 0)
    if new_pairs == 0:
        context.log.info("Weak labeler generated 0 pairs. Skipping.")
        return {"status": "skipped_no_new_data"}
        
    context.log.info(f"Weak labeler added {new_pairs} new pairs. Versioning all data assets...")

    new_jargon_list = jargon_drift_detector.get("new_jargon", [])
    if not new_jargon_list:
        context.log.warning("Weak labeler succeeded, but new jargon list was empty. Skipping.")
        return {"status": "skipped_no_new_jargon_list"}

    try:
        with open(KNOWN_JARGON_FILE, 'a+') as f:
            f.seek(0)
            if len(f.read(1)) == 0:
                f.write('jargon\n')
            
            f.seek(0, os.SEEK_END)
            
            if f.tell() > 0:
                f.seek(f.tell() - 1)
                if f.read(1) != '\n':
                    f.write('\n')

            for word in new_jargon_list:
                f.write(f"{word}\n")

        context.log.info(f"Appended {len(new_jargon_list)} words to {KNOWN_JARGON_FILE.name}")
    except Exception as e:
        context.log.error(f"Failed to append new jargon to {KNOWN_JARGON_FILE.name}: {e}")
        return {"status": "error_writing_known_jargon"}

    def run_dvc_command(cmd: list):
        context.log.info(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            context.log.info(f"DVC STDOUT: {result.stdout}")
            if result.stderr:
                context.log.warning(f"DVC STDERR: {result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            context.log.error(f"DVC command failed: {e.stderr}")
            return False
        except FileNotFoundError:
            context.log.error(f"DVC command not found. Is DVC installed?")
            return False

    try:
        rel_training_data_path = os.path.relpath(TRAINING_DATA_FILE, PROJECT_ROOT)
        rel_jargon_file_path = os.path.relpath(KNOWN_JARGON_FILE, PROJECT_ROOT)
    except Exception as e:
        context.log.error(f"Could not calculate relative paths: {e}. Aborting.")
        return {"status": "error_path_calculation"}

    if not run_dvc_command(["dvc", "add", rel_training_data_path]):
        return {"status": "error_dvc_add_training_data"}
    
    if not run_dvc_command(["dvc", "add", rel_jargon_file_path]):
        return {"status": "error_dvc_add_known_jargon"}
        
    if not run_dvc_command(["dvc", "push"]):
        return {"status": "error_dvc_push"}
        
    context.log.info(f"Successfully versioned and pushed {rel_training_data_path} and {rel_jargon_file_path}.")
    
    try:
        with open(NEW_JARGON_CANDIDATES_FILE, 'w') as f:
            f.write("")
        context.log.info(f"Cleared {NEW_JARGON_CANDIDATES_FILE.name} for next run.")
    except Exception as e:
        context.log.error(f"Failed to clear {NEW_JARGON_CANDIDATES_FILE.name}: {e}")
        return {"status": "error_clearing_candidates"}

    return {
        "status": "success",
        "versioned_files": [rel_training_data_path, rel_jargon_file_path]
    }