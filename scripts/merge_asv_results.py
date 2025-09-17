#!/usr/bin/env python3
"""
Merge ASV benchmark results from multiple chunks into a single result file.

This script combines benchmark results from parallel execution chunks,
ensuring the final result uses the real commit hash and proper ASV format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse


def merge_asv_results(chunk_dirs: List[Path], output_dir: Path, real_commit_hash: str) -> None:
    """
    Merge ASV results from multiple chunk directories into a single result file.
    
    Args:
        chunk_dirs: List of directories containing chunk results
        output_dir: Output directory for merged results
        real_commit_hash: The actual git commit hash to use
    """
    merged_results = {}
    merged_durations = {}
    base_metadata = None
    
    print(f"Merging results from {len(chunk_dirs)} chunks...")
    
    for chunk_dir in chunk_dirs:
        print(f"Processing chunk directory: {chunk_dir}")
        
        # Find result files in this chunk (excluding machine.json)
        result_files = list(chunk_dir.glob("github-actions/*.json"))
        result_files = [f for f in result_files if f.name != "machine.json"]
        
        for result_file in result_files:
            print(f"  Processing file: {result_file}")
            
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Use this file's metadata as base (they should all be similar)
                if base_metadata is None:
                    base_metadata = {k: v for k, v in data.items() 
                                   if k not in ['results', 'durations', 'commit_hash']}
                    # Ensure we use the real commit hash
                    base_metadata['commit_hash'] = real_commit_hash
                
                # Merge results
                if 'results' in data:
                    for benchmark_name, benchmark_data in data['results'].items():
                        if benchmark_name in merged_results:
                            # Combine results for the same benchmark
                            existing_results = merged_results[benchmark_name][0]
                            new_results = benchmark_data[0]
                            
                            # Merge the result arrays
                            if isinstance(existing_results, list) and isinstance(new_results, list):
                                merged_results[benchmark_name][0].extend(new_results)
                            else:
                                print(f"Warning: Unexpected result format for {benchmark_name}")
                        else:
                            merged_results[benchmark_name] = benchmark_data
                
                # Merge durations
                if 'durations' in data:
                    merged_durations.update(data['durations'])
                    
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    if not merged_results:
        print("No benchmark results found to merge!")
        return
    
    # Create the final merged result
    final_result = base_metadata.copy()
    final_result['results'] = merged_results
    final_result['durations'] = merged_durations
    
    # Ensure output directory exists
    output_path = output_dir / "github-actions"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on commit hash and environment
    env_name = base_metadata.get('env_name', 'unknown')
    output_file = output_path / f"{real_commit_hash}-{env_name}.json"
    
    # Write merged results
    with open(output_file, 'w') as f:
        json.dump(final_result, f, separators=(',', ':'))
    
    print(f"Merged results written to: {output_file}")
    print(f"Total benchmarks: {len(merged_results)}")
    
    # Show summary of merged results
    for benchmark_name, benchmark_data in merged_results.items():
        if isinstance(benchmark_data[0], list):
            result_count = len(benchmark_data[0])
            print(f"  {benchmark_name}: {result_count} results")


def main():
    parser = argparse.ArgumentParser(description="Merge ASV benchmark results from multiple chunks")
    parser.add_argument("--chunk-dirs", nargs="+", required=True, 
                       help="Directories containing chunk results")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for merged results")
    parser.add_argument("--commit-hash", required=True,
                       help="Real git commit hash to use")
    
    args = parser.parse_args()
    
    chunk_dirs = [Path(d) for d in args.chunk_dirs]
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            print(f"Error: Chunk directory does not exist: {chunk_dir}")
            sys.exit(1)
    
    merge_asv_results(chunk_dirs, output_dir, args.commit_hash)


if __name__ == "__main__":
    main()