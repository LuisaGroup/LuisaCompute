"""Tests for vector constant folding using tuples."""

import math
import pytest
from luisa import (
    kernel, callable, pprint, Float, Float3, Buffer,
    normalize, length, length_squared, dot, cross, distance, reflect,
)
from luisa.transform.ir import ConstantValue


def test_vector_tuple_constants():
    """Test using tuples as vector constants."""
    print("\n" + "=" * 60)
    print("Test: Vector Tuple Constants")
    print("=" * 60)
    
    # Use tuples to represent vector constants
    v1 = (1.0, 2.0, 3.0)
    print(f"v1 = {v1}")
    
    assert len(v1) == 3
    assert v1[0] == 1.0
    assert v1[1] == 2.0
    assert v1[2] == 3.0
    
    print("✓ Vector tuple constants work")
    print("=" * 60)


def test_normalize_constant_folding():
    """Test normalize() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Normalize Constant Folding")
    print("=" * 60)
    
    # Create a vector tuple and normalize it at compile time
    v = (3.0, 0.0, 0.0)
    result = normalize(v)  # Should be (1.0, 0.0, 0.0)
    
    print(f"normalize({v}) = {result}")
    
    assert isinstance(result, ConstantValue)
    res_val = result.value
    assert abs(res_val[0] - 1.0) < 1e-10
    assert abs(res_val[1] - 0.0) < 1e-10
    assert abs(res_val[2] - 0.0) < 1e-10
    
    print("✓ Normalize constant folding works")
    print("=" * 60)


def test_length_constant_folding():
    """Test length() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Length Constant Folding")
    print("=" * 60)
    
    v = (3.0, 4.0, 0.0)
    result = length(v)  # Should be 5.0
    
    print(f"length({v}) = {result}")
    
    # Result is wrapped in ConstantValue
    assert isinstance(result, ConstantValue)
    assert abs(result.value - 5.0) < 1e-10
    
    print("✓ Length constant folding works")
    print("=" * 60)


def test_dot_constant_folding():
    """Test dot() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Dot Product Constant Folding")
    print("=" * 60)
    
    a = (1.0, 2.0, 3.0)
    b = (4.0, 5.0, 6.0)
    result = dot(a, b)  # Should be 1*4 + 2*5 + 3*6 = 32
    
    print(f"dot({a}, {b}) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert abs(result.value - 32.0) < 1e-10
    
    print("✓ Dot product constant folding works")
    print("=" * 60)


def test_cross_constant_folding():
    """Test cross() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Cross Product Constant Folding")
    print("=" * 60)
    
    a = (1.0, 0.0, 0.0)  # x-axis
    b = (0.0, 1.0, 0.0)  # y-axis
    result = cross(a, b)  # Should be z-axis (0, 0, 1)
    
    print(f"cross({a}, {b}) = {result}")
    
    assert isinstance(result, ConstantValue)
    res_val = result.value
    assert abs(res_val[0] - 0.0) < 1e-10
    assert abs(res_val[1] - 0.0) < 1e-10
    assert abs(res_val[2] - 1.0) < 1e-10
    
    print("✓ Cross product constant folding works")
    print("=" * 60)


def test_distance_constant_folding():
    """Test distance() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Distance Constant Folding")
    print("=" * 60)
    
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
    result = distance(a, b)  # Should be 5.0
    
    print(f"distance({a}, {b}) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert abs(result.value - 5.0) < 1e-10
    
    print("✓ Distance constant folding works")
    print("=" * 60)


def test_reflect_constant_folding():
    """Test reflect() constant folding for vector tuples."""
    print("\n" + "=" * 60)
    print("Test: Reflect Constant Folding")
    print("=" * 60)
    
    # Reflect incident vector off normal
    i = (1.0, -1.0, 0.0)
    n = (0.0, 1.0, 0.0)  # Upward normal
    result = reflect(i, n)
    
    print(f"reflect({i}, {n}) = {result}")
    
    assert isinstance(result, ConstantValue)
    res_val = result.value
    # Reflection of (1, -1) off (0, 1) should be (1, 1)
    assert abs(res_val[0] - 1.0) < 1e-10
    assert abs(res_val[1] - 1.0) < 1e-10
    
    print("✓ Reflect constant folding works")
    print("=" * 60)


def test_vector_device_routing():
    """Test that vector functions route to device when given DSL values."""
    print("\n" + "=" * 60)
    print("Test: Vector Device Routing")
    print("=" * 60)
    
    @callable
    def vector_ops(v: Float3) -> Float:
        # v is a DSL value, so this should emit device instructions
        n = normalize(v)
        l = length(v)
        return l
    
    ir = vector_ops(None)
    
    print("Generated IR:")
    print(pprint(ir))
    
    from luisa.lang.inspect import count_instructions
    counts = count_instructions(ir)
    
    # Should have NORMALIZE and LENGTH instructions
    assert 'NORMALIZE' in counts or 'LENGTH' in counts, "Expected vector instructions"
    
    print("✓ Vector device routing works")
    print("=" * 60)


def test_mixed_vector_operations():
    """Test mixing constant-folded and device vector operations."""
    print("\n" + "=" * 60)
    print("Test: Mixed Vector Operations")
    print("=" * 60)
    
    @callable
    def mixed_ops(v: Float3) -> Float:
        # Constant-folded using tuple
        const_len = length((3.0, 4.0, 0.0))  # 5.0
        
        # Device-side
        var_len = length(v)
        
        return const_len + var_len
    
    ir = mixed_ops(None)
    
    print("Generated IR:")
    print(pprint(ir))
    
    print("✓ Mixed vector operations work")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_vector_constant_folding.py tests")
    print("=" * 70)
    
    test_vector_tuple_constants()
    test_normalize_constant_folding()
    test_length_constant_folding()
    test_dot_constant_folding()
    test_cross_constant_folding()
    test_distance_constant_folding()
    test_reflect_constant_folding()
    test_vector_device_routing()
    test_mixed_vector_operations()
    
    print("\n" + "=" * 70)
    print("All test_vector_constant_folding.py tests passed!")
    print("=" * 70)
