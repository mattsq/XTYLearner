#!/usr/bin/env python3
"""Test the jq merge logic for ASV results."""

import json
import subprocess
import tempfile
import os

# Mock ASV result files
result1 = {
    "benchmark_models.BenchmarkModels.track_val_outcome_rmse": {
        "code": "...",
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
        "code": "...",
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

# Create temp files
with tempfile.TemporaryDirectory() as tmpdir:
    file1 = os.path.join(tmpdir, "result1.json")
    file2 = os.path.join(tmpdir, "result2.json")
    
    with open(file1, 'w') as f:
        json.dump(result1, f, indent=2)
    
    with open(file2, 'w') as f:
        json.dump(result2, f, indent=2)
    
    # Test current jq logic
    print("=== Testing current jq logic ===")
    jq_cmd = [
        "jq", "-s",
        """
        .[0] as $base | .[1] as $merge |
        $base | 
        # For each benchmark in the results, merge the params[1] arrays (model names)
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
        """,
        file1, file2
    ]
    
    try:
        result = subprocess.run(jq_cmd, capture_output=True, text=True, check=True)
        print("SUCCESS!")
        merged = json.loads(result.stdout)
        
        # Check the merged result
        bench_key = "benchmark_models.BenchmarkModels.track_val_outcome_rmse"
        models = merged[bench_key]["params"][1]
        print(f"Merged models: {models}")
        print(f"Model count: {len(models)}")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"STDERR: {e.stderr}")
        
        # Try alternative approach
        print("\n=== Testing alternative jq logic ===")
        alt_jq_cmd = [
            "jq", "-s",
            """
            .[0] as $base | .[1] as $merge |
            $base |
            # Direct merge approach
            .["benchmark_models.BenchmarkModels.track_val_outcome_rmse"].params[1] = 
              (.["benchmark_models.BenchmarkModels.track_val_outcome_rmse"].params[1] + 
               $merge["benchmark_models.BenchmarkModels.track_val_outcome_rmse"].params[1] | 
               unique | sort)
            """,
            file1, file2
        ]
        
        try:
            result = subprocess.run(alt_jq_cmd, capture_output=True, text=True, check=True)
            print("ALTERNATIVE SUCCESS!")
            merged = json.loads(result.stdout)
            
            bench_key = "benchmark_models.BenchmarkModels.track_val_outcome_rmse"
            models = merged[bench_key]["params"][1]
            print(f"Merged models: {models}")
            print(f"Model count: {len(models)}")
            
        except subprocess.CalledProcessError as e2:
            print(f"ALTERNATIVE ERROR: {e2}")
            print(f"STDERR: {e2.stderr}")