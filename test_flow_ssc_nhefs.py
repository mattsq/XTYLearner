"""Test script to debug flow_ssc NHEFS performance degradation."""

import torch
import sys

# Check if we can find the issue by inspecting model state
print("Testing flow_ssc configuration with NHEFS parameters...")
print()

# Simulate NHEFS dimensions
d_x = 65
d_y = 1
k = 2

print(f"NHEFS dimensions: d_x={d_x}, d_y={d_y}, k={k}")
print()

# Check beta parameter behavior
print("Beta parameter test:")
print(f"  Default beta should be: 1.0")
print(f"  Default adaptive_beta should be: False")
print()

# Simulate what happens with batch size
batch_size = 10
full_train = 783

print(f"Batch size analysis:")
print(f"  Batch size: {batch_size}")
print(f"  Full train dataset: {full_train}")
print(f"  Samples per dim (batch): {batch_size/d_x:.3f}")
print(f"  Samples per dim (full): {full_train/d_x:.2f}")
print()

# Test adaptive beta thresholds
print("If adaptive_beta were enabled (IT SHOULD NOT BE):")
samples_per_dim = batch_size / d_x
if samples_per_dim < 5:
    print(f"  Batch: beta would be 0.0 (BAD - disables flow_x)")
elif samples_per_dim < 10:
    print(f"  Batch: beta would be 0.1")
else:
    print(f"  Batch: beta would be full")

samples_per_dim = full_train / d_x
if samples_per_dim < 5:
    print(f"  Full: beta would be 0.0")
elif samples_per_dim < 10:
    print(f"  Full: beta would be 0.1")
elif samples_per_dim < 20:
    print(f"  Full: beta would be 0.5")
else:
    print(f"  Full: beta would be 1.0")

print()
print("=" * 60)
print("HYPOTHESIS:")
print("The beta parameter change is mathematically equivalent when:")
print("  beta=1.0 and adaptive_beta=False (defaults)")
print()
print("Old formula: -(ll_x + ll_y).mean()")
print("New formula: -(1.0 * ll_x + ll_y).mean()")
print("These are identical.")
print()
print("Therefore, the issue is NOT the beta parameter itself.")
print("Let me check if something else changed...")
print("=" * 60)
