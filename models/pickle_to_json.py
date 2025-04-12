# conda activate DRL
import pickle
import json
import numpy as np

filename = "stage1_weights_ep55000_new"
with open(f"{filename}.pkl", "rb") as f:
    data = pickle.load(f)

# Convert numpy arrays and tuple keys to JSON-safe format
def serialize_weights(weights):
    converted = []
    for stage_weights in weights:
        new_dict = {}
        for key, value in stage_weights.items():
            # Convert key (tuple) to a string
            str_key = f"{key[0]}::{key[1]}"
            # Convert value (if ndarray) to float
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                value = float(value)
            new_dict[str_key] = value
        converted.append(new_dict)
    return converted

json_weights = serialize_weights(data)

# Save as JSON
with open(f"{filename}.json", "w") as f:
    json.dump(json_weights, f)

print("✅ Saved weights to JSON format.")
