"""Tests for new math functions (rsqrt, exp10, hyperbolic, etc.)."""

import math
from luisa import (
    callable, Float,
    rsqrt, exp10, sinh, cosh, tanh, asinh, acosh, atanh,
    isinf, isnan, copysign, fma,
    popcount,
)


def test_rsqrt_constant_folding():
    """Test rsqrt() constant folding."""
    result = rsqrt(4.0)  # Should be 0.5
    assert isinstance(result, float)
    assert abs(result - 0.5) < 1e-10


def test_exp10_constant_folding():
    """Test exp10() constant folding."""
    result = exp10(2.0)  # Should be 100.0
    assert isinstance(result, float)
    assert abs(result - 100.0) < 1e-10


def test_hyperbolic_constant_folding():
    """Test hyperbolic functions constant folding."""
    # sinh
    result = sinh(1.0)
    assert isinstance(result, float)
    assert abs(result - math.sinh(1.0)) < 1e-10
    
    # cosh
    result = cosh(1.0)
    assert isinstance(result, float)
    assert abs(result - math.cosh(1.0)) < 1e-10
    
    # tanh
    result = tanh(1.0)
    assert isinstance(result, float)
    assert abs(result - math.tanh(1.0)) < 1e-10


def test_inverse_hyperbolic_constant_folding():
    """Test inverse hyperbolic functions constant folding."""
    # asinh
    result = asinh(1.0)
    assert isinstance(result, float)
    assert abs(result - math.asinh(1.0)) < 1e-10
    
    # acosh (x >= 1)
    result = acosh(2.0)
    assert isinstance(result, float)
    assert abs(result - math.acosh(2.0)) < 1e-10
    
    # atanh (|x| < 1)
    result = atanh(0.5)
    assert isinstance(result, float)
    assert abs(result - math.atanh(0.5)) < 1e-10


def test_isinf_isnan_constant_folding():
    """Test isinf() and isnan() constant folding."""
    # isinf
    result = isinf(1.0)
    assert isinstance(result, int)  # Bool folded to Int 0/1 in current logic?
    assert result == 0
    
    result = isinf(float('inf'))
    assert result == 1
    
    # isnan
    result = isnan(1.0)
    assert result == 0
    
    result = isnan(float('nan'))
    assert result == 1


def test_copysign_constant_folding():
    """Test copysign() constant folding."""
    result = copysign(3.0, -1.0)  # Should be -3.0
    assert isinstance(result, float)
    assert result == -3.0


def test_fma_constant_folding():
    """Test fma() constant folding."""
    result = fma(2.0, 3.0, 4.0)  # Should be 2*3 + 4 = 10
    assert isinstance(result, float)
    assert result == 10.0


def test_bit_operations_constant_folding():
    """Test integer bit operations constant folding."""
    # popcount
    result = popcount(42)  # 0b101010 -> 3
    assert isinstance(result, int)
    assert result == 3


def test_new_functions_device_routing(verify_ir):
    """Test that new functions route to device when given DSL values."""
    @callable
    def test_funcs(x: Float) -> Float:
        a = rsqrt(x)
        b = exp10(x)
        c = sinh(x)
        return a + b + c
    
    expected = """
f32 test_funcs(f32 arg0) {
  f32 v0 = rsqrt(arg0);
  f32 v1 = exp10(arg0);
  f32 v2 = sinh(arg0);
  f32 v3 = add(v0, v1);
  f32 v4 = add(v3, v2);
  return v4;
}
"""
    verify_ir(test_funcs, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
