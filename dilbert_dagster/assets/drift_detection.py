"""
Detects new jargon by comparing candidate file to a known jargon list.
"""
import pandas as pd
from dagster import asset, AssetExecutionContext
from dilbert_dagster.constants import KNOWN_JARGON_FILE, NEW_JARGON_CANDIDATES_FILE


@asset
def jargon_drift_detector(context: AssetExecutionContext) -> dict:
    """
    Detects if new jargon words represent a data drift.
    Returns metadata about detected drift.
    """
    try:
        known_jargon_df = pd.read_csv(KNOWN_JARGON_FILE)
        known_jargon = set(known_jargon_df['jargon'].str.lower())
        
        with open(NEW_JARGON_CANDIDATES_FILE, 'r') as f:
            new_candidates = [line.strip().lower() for line in f if line.strip()]
        
        new_jargon = [word for word in new_candidates if word not in known_jargon]
        
        drift_detected = len(new_jargon) > 0
        
        context.log.info(f"Drift detection: {len(new_jargon)} new jargon words detected")
        context.log.info(f"New jargon: {new_jargon}")
        
        return {
            "drift_detected": drift_detected,
            "new_jargon": new_jargon,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    
    except FileNotFoundError as e:
        context.log.warning(f"File not found: {e}. Skipping drift check.")
        # CRITICAL: Create candidate file if missing to prevent repeated errors
        try:
            with open(NEW_JARGON_CANDIDATES_FILE, 'w') as f:
                pass
            context.log.info(f"Created empty {NEW_JARGON_CANDIDATES_FILE.name}")
        except Exception as create_e:
            context.log.error(f"Could not create file {NEW_JARGON_CANDIDATES_FILE.name}: {create_e}")

        return {"drift_detected": False, "new_jargon": [], "timestamp": pd.Timestamp.now().isoformat()}