#!/usr/bin/env python3
"""Test the merge logic for ASV results in pure Python."""

import json

# Mock ASV result files based on actual structure
result1 = {
    "benchmark_models.BenchmarkModels.track_val_outcome_rmse": {
        "code": "class BenchmarkModels...",
        "name": "benchmark_models.BenchmarkModels.track_val_outcome_rmse",
        "param_names": ["dataset", "model"],
        "params": [
            ["'synthetic'", "'synthetic_mixed'"],
            ["'cycle_dual'"]
        ],
        "timeout": 600,
        "type": "track",
        "unit": "unit",
        "version": "abc123"
    },
    "version": 2
}

result2 = {
    "benchmark_models.BenchmarkModels.track_val_outcome_rmse": {
        "code": "class BenchmarkModels...",
        "name": "benchmark_models.BenchmarkModels.track_val_outcome_rmse", 
        "param_names": ["dataset", "model"],
        "params": [
            ["'synthetic'", "'synthetic_mixed'"],
            ["'mean_teacher'"]
        ],
        "timeout": 600,
        "type": "track",
        "unit": "unit",
        "version": "abc123"
    },
    "version": 2
}

print("=== Original Results ===")
print("Result 1 models:", result1["benchmark_models.BenchmarkModels.track_val_outcome_rmse"]["params"][1])
print("Result 2 models:", result2["benchmark_models.BenchmarkModels.track_val_outcome_rmse"]["params"][1])

# Test merge logic
def merge_asv_results(base, merge_data):
    """Merge ASV result files correctly."""
    merged = base.copy()
    
    for key, value in merge_data.items():
        if key == "version":
            continue
            
        if key in merged:
            # Merge the params[1] arrays (model names)
            base_models = merged[key]["params"][1]
            merge_models = value["params"][1]
            
            # Combine and deduplicate
            combined_models = list(set(base_models + merge_models))
            combined_models.sort()
            
            merged[key]["params"][1] = combined_models
        else:
            merged[key] = value
    
    return merged

print("\n=== Testing Python merge logic ===")
merged = merge_asv_results(result1, result2)

bench_key = "benchmark_models.BenchmarkModels.track_val_outcome_rmse"
merged_models = merged[bench_key]["params"][1]
print(f"Merged models: {merged_models}")
print(f"Model count: {len(merged_models)}")

# Generate corrected jq command
print("\n=== Corrected jq command ===")
jq_command = '''
.[0] as $base | .[1] as $merge |
$base |
# For each non-version key, merge params[1] arrays
to_entries | map(
  if .key == "version" then
    .
  else
    .key as $bench_key |
    if ($merge | has($bench_key)) then
      .value.params[1] = (.value.params[1] + $merge[$bench_key].params[1] | unique | sort)
    else
      .
    end
  end
) | from_entries
'''

print("jq -s '" + jq_command.strip() + "' file1.json file2.json")

print("\n=== Expected output structure ===")
print(json.dumps(merged, indent=2))