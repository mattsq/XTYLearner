#!/usr/bin/env python3
"""
Test script for validating ASV workflow components locally.
Run this before pushing changes to verify the workflow will work.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture=True):
    """Run a command and return result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, 
        check=check, 
        capture_output=capture, 
        text=True,
        cwd=Path(__file__).parent.parent
    )
    if capture:
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
    return result


def test_asv_config():
    """Test ASV configuration is valid."""
    print("\n=== Testing ASV Configuration ===")
    
    config_path = Path("asv.conf.json")
    if not config_path.exists():
        print("❌ asv.conf.json not found")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = ["project", "repo", "benchmark_dir", "environment_type"]
        for key in required_keys:
            if key not in config:
                print(f"Missing required key in asv.conf.json: {key}")
                return False
        
        print("✅ ASV configuration valid")
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in asv.conf.json: {e}")
        return False


def test_benchmark_discovery():
    """Test that ASV can discover benchmarks."""
    print("\n=== Testing Benchmark Discovery ===")
    
    try:
        # Test minimal ASV command
        result = run_command(["python", "-m", "asv", "check", "--config", "asv.conf.json"])
        if result.returncode == 0:
            print("✅ ASV benchmark discovery successful")
            return True
        else:
            print("❌ ASV benchmark discovery failed")
            return False
    except Exception as e:
        print(f"❌ ASV benchmark discovery error: {e}")
        return False


def test_single_benchmark():
    """Test running a single benchmark."""
    print("\n=== Testing Single Benchmark Execution ===")
    
    # Set minimal environment
    env = os.environ.copy()
    env["BENCHMARK_MODELS"] = "cycle_dual"
    
    try:
        # Create machine config
        result = run_command([
            "python", "-m", "asv", "machine", 
            "--machine", "test-local",
            "--config", "asv.conf.json",
            "--yes"
        ], check=False)
        
        # Run single benchmark
        result = run_command([
            "python", "-m", "asv", "run",
            "HEAD^!",
            "--machine", "test-local", 
            "--config", "asv.conf.json",
            "--show-stderr"
        ], check=False)
        
        if result.returncode == 0:
            print("✅ Single benchmark execution successful")
            
            # Check result files
            results_dir = Path(".asv/results/test-local")
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                if json_files:
                    print(f"✅ Found {len(json_files)} result files")
                    return True
                else:
                    print("❌ No result files generated")
                    return False
            else:
                print("❌ Results directory not created")
                return False
        else:
            print("❌ Single benchmark execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Benchmark execution error: {e}")
        return False


def test_html_generation():
    """Test HTML generation from results."""
    print("\n=== Testing HTML Generation ===")
    
    try:
        result = run_command([
            "python", "-m", "asv", "publish",
            "--config", "asv.conf.json",
            "--html-dir", ".asv/html-test"
        ], check=False)
        
        if result.returncode == 0:
            html_path = Path(".asv/html-test/index.html")
            if html_path.exists():
                print("✅ HTML generation successful")
                
                # Check HTML content
                content = html_path.read_text()
                if "No benchmark data available" in content:
                    print("⚠️  HTML contains 'No benchmark data available'")
                    return False
                else:
                    print("✅ HTML content looks valid")
                    return True
            else:
                print("❌ index.html not generated")
                return False
        else:
            print("❌ HTML generation failed")
            return False
            
    except Exception as e:
        print(f"❌ HTML generation error: {e}")
        return False


def cleanup():
    """Clean up test artifacts."""
    print("\n=== Cleaning up test artifacts ===")
    
    cleanup_paths = [
        ".asv/results/test-local",
        ".asv/html-test",
        ".asv/env"
    ]
    
    for path in cleanup_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                import shutil
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
            print(f"Removed: {path}")


def main():
    """Run all tests."""
    print("ASV Workflow Test Suite")
    print("=" * 50)
    
    tests = [
        test_asv_config,
        test_benchmark_discovery,
        test_single_benchmark,
        test_html_generation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test.__name__}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! Workflow should work correctly.")
        exit_code = 0
    else:
        print("Some tests failed. Fix issues before deploying.")
        exit_code = 1
    
    # Always cleanup
    cleanup()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())