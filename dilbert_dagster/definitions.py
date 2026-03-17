"""
Dagster Definitions - Entry Point
Defines all assets, jobs, and sensors.
"""
from dagster import Definitions, define_asset_job

# IMPORT ASSETS DIRECTLY
from .assets.drift_detection import jargon_drift_detector
from .assets.weak_labeler import weak_labeler
from .assets.data_versioner import data_versioner
from .assets.model_trainer import model_trainer
from .assets.model_qa_gate import model_qa_gate
from .assets.model_registry_promoter import model_registry_promoter

# IMPORT SENSORS
from .sensors.jargon_sensor import jargon_candidate_sensor

# LOAD ALL ASSETS
all_assets = [
    jargon_drift_detector,
    weak_labeler,
    data_versioner,
    model_trainer,
    model_qa_gate,
    model_registry_promoter,
]

# DEFINE JOBS
process_new_jargon_job = define_asset_job(
    name="process_new_jargon_job",
    selection="*",
)

all_jobs = [process_new_jargon_job]

# DEFINE SENSORS
all_sensors = [jargon_candidate_sensor]

# DAGSTER DEFINITIONS
defs = Definitions(
    assets=all_assets,
    sensors=all_sensors,
    jobs=all_jobs,
)