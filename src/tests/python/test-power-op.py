"""
Test for power operations in Python DSL.

This test verifies the fix for the issue where a**3 and a**4 would cause:
    AttributeError: 'tuple' object has no attribute 'dtype'

The bug was in builtin.py where nested builtin_bin_op calls were not
properly wrapping intermediate results in SimpleNamespace objects.
"""
from luisa import *
from luisa.types import *
import numpy as np

init()

# Create a simple test for power operations
buffer_in = Buffer(4, float)
buffer_out = Buffer(4, float)

@func
def test_power():
    idx = dispatch_id().x
    val = buffer_in.read(idx)
    
    # Test power of 2 (this always worked)
    val2 = val ** 2
    
    # Test power of 3 (this was broken before the fix)
    val3 = val ** 3
    
    # Test power of 4 (this was broken before the fix)
    val4 = val ** 4
    
    # Store the result of power of 4 for verification
    buffer_out.write(idx, val4)

# Test data
test_data = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
buffer_in.copy_from(test_data)

# Run the kernel
test_power(dispatch_size=(4, 1, 1))
execute()
synchronize()

# Verify results
result = np.zeros(4, dtype=np.float32)
buffer_out.copy_to(result)

expected = test_data ** 4
print("Input:", test_data)
print("Output (a**4):", result)
print("Expected:", expected)

# Check if results match
tolerance = 1e-5
if np.allclose(result, expected, atol=tolerance):
    print("✓ Test passed! Power operations work correctly.")
else:
    print("✗ Test failed! Results don't match.")
    print("Difference:", np.abs(result - expected))
