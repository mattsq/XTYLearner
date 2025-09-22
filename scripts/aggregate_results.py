#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
import os

def aggregate_results(input_dir, history_file, output_file):
    results_dir = Path(input_dir)
    all_results = []
    
    # Collect all result files
    for result_file in results_dir.rglob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                if "results" in data:
                    all_results.extend(data["results"])
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Load existing history
    history = []
    history_path = Path(history_file)
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
        except:
            history = []
    
    # Create current benchmark entry
    current_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "results": all_results,
        "metadata": {
            "total_benchmarks": len(all_results),
            "models": list(set(r["name"].split("_")[0] for r in all_results)),
            "datasets": list(set(r["name"].split("_")[1] for r in all_results))
        }
    }
    
    # Add to history
    history.append(current_entry)
    
    # Keep only last 100 entries
    history = history[-100:]
    
    # Save updated history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save current results
    with open(output_file, 'w') as f:
        json.dump(current_entry, f, indent=2)
    
    print(f"Aggregated {len(all_results)} results")
    print(f"History contains {len(history)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--history-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    
    aggregate_results(args.input_dir, args.history_file, args.output_file)