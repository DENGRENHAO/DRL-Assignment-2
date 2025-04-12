# conda activate python310
import json
import pickle
from collections import defaultdict

# Load JSON
filename = "stage1_weights_ep55000_new"
with open(f"{filename}.json", "r") as f:
    json_weights = json.load(f)

# Convert string keys back to tuples
def deserialize_weights(json_weights):
    result = []
    for stage_weights in json_weights:
        restored = {}
        for key_str, value in stage_weights.items():
            # Split string back to tuple
            parts = key_str.split("::")
            restored[(parts[0], int(parts[1]))] = value
        result.append(defaultdict(float, restored))
    return result

converted_weights = deserialize_weights(json_weights)

# Save to pickle
with open(f"converted_{filename}.pkl", "wb") as f:
    pickle.dump(converted_weights, f)

print("✅ Saved as pickle format.")