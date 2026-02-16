"""Tests for vector constant folding using tuples."""

import math
from luisa import (
    callable, Float, Float3,
    normalize, length, dot, cross, distance, reflect,
)


def test_vector_tuple_constants():
    """Test using tuples as vector constants."""
    # Use tuples to represent vector constants
    v1 = (1.0, 2.0, 3.0)
    assert len(v1) == 3
    assert v1[0] == 1.0
    assert v1[1] == 2.0
    assert v1[2] == 3.0


def test_normalize_constant_folding():
    """Test normalize() constant folding for vector tuples."""
    # Create a vector tuple and normalize it at compile time
    v = (3.0, 0.0, 0.0)
    result = normalize(v)  # Should be (1.0, 0.0, 0.0)
    
    assert isinstance(result, tuple)
    assert abs(result[0] - 1.0) < 1e-10
    assert abs(result[1] - 0.0) < 1e-10
    assert abs(result[2] - 0.0) < 1e-10


def test_length_constant_folding():
    """Test length() constant folding for vector tuples."""
    v = (3.0, 4.0, 0.0)
    result = length(v)  # Should be 5.0
    
    assert isinstance(result, float)
    assert abs(result - 5.0) < 1e-10


def test_dot_constant_folding():
    """Test dot() constant folding for vector tuples."""
    a = (1.0, 2.0, 3.0)
    b = (4.0, 5.0, 6.0)
    result = dot(a, b)  # Should be 1*4 + 2*5 + 3*6 = 32
    
    assert isinstance(result, float)
    assert abs(result - 32.0) < 1e-10


def test_cross_constant_folding():
    """Test cross() constant folding for vector tuples."""
    a = (1.0, 0.0, 0.0)  # x-axis
    b = (0.0, 1.0, 0.0)  # y-axis
    result = cross(a, b)  # Should be z-axis (0, 0, 1)
    
    assert isinstance(result, tuple)
    assert abs(result[0] - 0.0) < 1e-10
    assert abs(result[1] - 0.0) < 1e-10
    assert abs(result[2] - 1.0) < 1e-10


def test_distance_constant_folding():
    """Test distance() constant folding for vector tuples."""
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
    result = distance(a, b)  # Should be 5.0
    
    assert isinstance(result, float)
    assert abs(result - 5.0) < 1e-10


def test_reflect_constant_folding():
    """Test reflect() constant folding for vector tuples."""
    # Reflect incident vector off normal
    i = (1.0, -1.0, 0.0)
    n = (0.0, 1.0, 0.0)  # Upward normal
    result = reflect(i, n)
    
    assert isinstance(result, tuple)
    # Reflection of (1, -1) off (0, 1) should be (1, 1)
    assert abs(result[0] - 1.0) < 1e-10
    assert abs(result[1] - 1.0) < 1e-10


def test_vector_device_routing(verify_ir):
    """Test that vector functions route to device when given DSL values."""
    @callable
    def vector_ops(v: Float3) -> Float:
        # v is a DSL value, so this should emit device instructions
        n = normalize(v)
        l = length(v)
        return l
    
    expected = """
f32 vector_ops(<3 x f32> arg0) {
  <3 x f32> v0 = normalize(arg0);
  f32 v1 = length(arg0);
  return v1;
}
"""
    verify_ir(vector_ops, expected)


def test_mixed_vector_operations(verify_ir):
    """Test mixing constant-folded and device vector operations."""
    @callable
    def mixed_ops(v: Float3) -> Float:
        # Constant-folded using tuple
        const_len = length((3.0, 4.0, 0.0))  # 5.0
        
        # Device-side
        var_len = length(v)
        
        return const_len + var_len
    
    expected = """
f32 mixed_ops(<3 x f32> arg0) {
  f32 v0 = length(arg0);
  f32 v1 = add(5.0, v0);
  return v1;
}
"""
    verify_ir(mixed_ops, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
