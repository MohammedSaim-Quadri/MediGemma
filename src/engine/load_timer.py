import json
import os
import time
import logging

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../config")
LOAD_TIMES_FILE = os.path.join(_CONFIG_DIR, "model_load_times.json")

MODEL_KEY_MEDGEMMA_27B = "medgemma_27b"
MODEL_KEY_MEDGEMMA_4B = "medgemma_4b"
MODEL_KEY_HULUMED = "hulumed"


def get_estimated_load_time(model_key: str) -> float | None:
    try:
        if os.path.exists(LOAD_TIMES_FILE):
            with open(LOAD_TIMES_FILE, "r") as f:
                data = json.load(f)
            return data.get(model_key, {}).get("load_seconds")
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read load times file: {e}")
    return None


def record_load_time(model_key: str, elapsed_seconds: float):
    data = {}
    try:
        if os.path.exists(LOAD_TIMES_FILE):
            with open(LOAD_TIMES_FILE, "r") as f:
                data = json.load(f)
    except (json.JSONDecodeError, IOError):
        data = {}

    data[model_key] = {
        "load_seconds": round(elapsed_seconds, 1),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        os.makedirs(_CONFIG_DIR, exist_ok=True)
        with open(LOAD_TIMES_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Recorded load time for {model_key}: {elapsed_seconds:.1f}s")
    except IOError as e:
        logger.warning(f"Could not write load times file: {e}")
