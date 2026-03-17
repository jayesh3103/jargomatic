"""
Dagster assets for the Jargomatic pipeline.
"""
from .drift_detection import jargon_drift_detector
from .weak_labeler import weak_labeler 
from .data_versioner import data_versioner

# --- TODO: Implement and uncomment next assets ---
# from .model_training import model_trainer
# from .model_qa import model_qa_gate
# from .model_registry import model_registry_promoter
# from .model_deployment import build_new_bento, retag_production

__all__ = [
    "jargon_drift_detector",
    "weak_labeler",
    "data_versioner",
    # --- TODO: Add next assets here ---
    # "model_trainer",
    # "model_qa_gate",
    # "model_registry_promoter",
    # "build_new_bento",
    # "retag_production",
]