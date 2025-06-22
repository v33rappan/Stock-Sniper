import os
import json
import numpy as np

OPTIMISED_FILE = "Output/optimised_strategies.json"

def load_optimised_thresholds():
    if not os.path.exists(OPTIMISED_FILE):
        return {}

    with open(OPTIMISED_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def apply_optimised_threshold(STRATEGIES):
    updated = False
    optimised = load_optimised_thresholds()
    for strategy, values in optimised.items():
        if strategy in STRATEGIES:
            STRATEGIES[strategy]['PRICE_THRESHOLD_PERCENT'] = values.get('PRICE_THRESHOLD_PERCENT', STRATEGIES[strategy]['PRICE_THRESHOLD_PERCENT'])
            STRATEGIES[strategy]['VOLUME_SPIKE_MULTIPLIER'] = values.get('VOLUME_SPIKE_MULTIPLIER', STRATEGIES[strategy]['VOLUME_SPIKE_MULTIPLIER'])
            updated = True
    return updated

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int_, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
