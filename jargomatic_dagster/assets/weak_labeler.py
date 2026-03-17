"""
Weak labeling asset: uses Groq + few-shot examples (style_guide.jsonl) to generate new training pairs.
"""
import json
import random
from groq import Groq
from dagster import asset, AssetExecutionContext
from ..constants import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_WEAK_LABELER_PROMPT,
    TRAINING_DATA_FILE,
    STYLE_GUIDE_FILE,
    NUM_FEW_SHOT_EXAMPLES,
)


@asset
def weak_labeler(context: AssetExecutionContext, jargon_drift_detector: dict) -> dict:
    """Appends newly generated pairs to training data + style guide."""
    if not jargon_drift_detector.get("drift_detected"):
        return {"new_pairs_generated": 0, "status": "skipped"}

    new_jargon_list = jargon_drift_detector.get("new_jargon", [])
    if not new_jargon_list:
        return {"new_pairs_generated": 0, "status": "skipped"}

    if not GROQ_API_KEY or GROQ_API_KEY == "your_key_here":
        return {"new_pairs_generated": 0, "status": "error_api_key_missing"}

    
    try:
        if not STYLE_GUIDE_FILE.exists():
            context.log.error(f"Style guide file not found at: {STYLE_GUIDE_FILE}")
            return {"new_pairs_generated": 0, "status": "error_missing_style_guide"}

        with open(STYLE_GUIDE_FILE, "r") as f:
            style_pool = [json.loads(line) for line in f if line.strip()]
        
        if not style_pool:
            context.log.warning("Style guide file exists, but is empty.")
            return {"new_pairs_generated": 0, "status": "error_empty_style_guide"}
            
    except json.JSONDecodeError as e:
        context.log.error(
            f"Failed to decode JSON from {STYLE_GUIDE_FILE}: {e}. "
            "Is it a DVC pointer file? Run 'dvc pull'."
        )
        return {"new_pairs_generated": 0, "status": "error_corrupt_style_guide"}
        
    except Exception as e:
        context.log.error(f"An unexpected error occurred while reading style guide: {e}")
        return {"new_pairs_generated": 0, "status": "error_unknown_read_error"}
    
    client = Groq(api_key=GROQ_API_KEY)
    new_training_pairs = []

    for jargon_word in new_jargon_list:
        selected = (
            style_pool if len(style_pool) <= NUM_FEW_SHOT_EXAMPLES
            else random.sample(style_pool, NUM_FEW_SHOT_EXAMPLES)
        )

        style_examples_str = "\n".join(f"Example {i+1}:\n{json.dumps(s)}" for i, s in enumerate(selected))

        prompt = GROQ_WEAK_LABELER_PROMPT.format(
            jargon=jargon_word,
            style_examples_str=style_examples_str,
        )

        try:
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(result.choices[0].message.content)
            pairs = [p for p in data.get("pairs", []) if "input" in p and "output" in p]
            new_training_pairs.extend(pairs)
        except Exception:
            continue

    if not new_training_pairs:
        return {"new_pairs_generated": 0, "status": "no_pairs_generated"}

    try:
        with open(TRAINING_DATA_FILE, "a") as f, open(STYLE_GUIDE_FILE, "a") as f2:
            for pair in new_training_pairs:
                line = json.dumps(pair) + "\n"
                f.write(line)
                f2.write(line)

        return {
            "new_pairs_generated": len(new_training_pairs),
            "status": "success",
            "new_jargon_processed": new_jargon_list,
        }
    except Exception:
        return {"new_pairs_generated": 0, "status": "error_file_write"}
