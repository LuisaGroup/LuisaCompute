"""Tests for constant folding and host/device routing of builtins."""

import math
from luisa import (
    kernel, callable,
    sin, cos, sqrt, exp,
    min, max, clamp, lerp, pow, atan2,
    Float, Buffer,
)
from luisa.transform.ir import ConstantValue
from luisa.lang.router import is_constant_value


def test_constant_folding_basic():
    """Test that constant expressions are folded at compile time."""
    # These should be constant-folded to plain Python values now
    result = sin(1.0 + 2.0)  # sin(3.0)
    
    assert isinstance(result, float)
    assert abs(result - math.sin(3.0)) < 1e-10
    
    # Test other math functions
    sqrt_result = sqrt(4.0)
    assert isinstance(sqrt_result, float)
    assert abs(sqrt_result - 2.0) < 1e-10
    
    exp_result = exp(1.0)
    assert isinstance(exp_result, float)
    assert abs(exp_result - math.e) < 1e-10


def test_constant_folding_arithmetic():
    """Test constant folding with arithmetic expressions."""
    # Complex expression that should be folded
    x = 1.5
    y = 2.5
    result = sin(x + y) * cos(x - y)
    
    expected = math.sin(4.0) * math.cos(-1.0)
    assert isinstance(result, float)
    assert abs(result - expected) < 1e-10


def test_device_routing_with_dsl_values(verify_ir):
    """Test that DSL values are routed to device."""
    @callable
    def use_sin(x: Float) -> Float:
        return sin(x)  # x is a DSL value, should emit device instruction
    
    expected = """
f32 use_sin(f32 arg0) {
  f32 v0 = sin(arg0);
  return v0;
}
"""
    verify_ir(use_sin, expected)


def test_mixed_constant_and_dsl(verify_ir):
    """Test mixing constants and DSL values."""
    @callable
    def mixed_ops(x: Float) -> Float:
        # x is DSL value, 0.5 is constant
        # sin(0.5) is computed at Python compile time
        # a becomes a DSL variable (for correct handling of potential reassignment)
        a = sin(0.5)  # constant-folded at Python level
        b = x * a     # device multiply with constant
        return b
    
    # a is now a DSL variable, b is also a DSL variable
    expected = """
f32 mixed_ops(f32 arg0) {
  f32 va = alloca();
  store(va, 0.479425538604203);
  f32 v2 = load(va);
  f32 v3 = mul(arg0, v2);
  f32 vb = alloca();
  store(vb, v3);
  f32 v6 = load(vb);
  return v6;
}
"""
    verify_ir(mixed_ops, expected)


def test_binary_ops_constant_folding():
    """Test binary operations with constant folding."""
    # min/max with constants
    result = min(5.0, 3.0)
    assert isinstance(result, float)
    assert result == 3.0
    
    result = max(5.0, 3.0)
    assert isinstance(result, float)
    assert result == 5.0
    
    # clamp with constants
    result = clamp(1.5, 0.0, 1.0)
    assert isinstance(result, float)
    assert result == 1.0
    
    result = clamp(-0.5, 0.0, 1.0)
    assert isinstance(result, float)
    assert result == 0.0
    
    # lerp with constants
    result = lerp(0.0, 10.0, 0.5)
    assert isinstance(result, float)
    assert result == 5.0
    
    # pow with constants
    result = pow(2.0, 10.0)
    assert isinstance(result, float)
    assert result == 1024.0
    
    # atan2 with constants
    result = atan2(1.0, 1.0)
    assert isinstance(result, float)
    assert abs(result - math.pi / 4) < 1e-10


def test_step_smoothstep_folding():
    """Test step and smoothstep constant folding."""
    from luisa import step, smoothstep
    
    # step
    result = step(0.5, 0.3)
    assert isinstance(result, float)
    assert result == 0.0
    
    result = step(0.5, 0.7)
    assert isinstance(result, float)
    assert result == 1.0
    
    # smoothstep
    result = smoothstep(0.0, 1.0, 0.5)
    assert isinstance(result, float)
    assert abs(result - 0.5) < 1e-10  # At midpoint, smoothstep = 0.5


def test_is_constant_value_helper():
    """Test the is_constant_value helper function."""
    # Python primitives are constants
    assert is_constant_value(1.0) == True
    assert is_constant_value(42) == True
    assert is_constant_value(True) == True
    assert is_constant_value(None) == True
    
    # ConstantValue is a constant
    const = ConstantValue(typ=Float, value=3.14)
    assert is_constant_value(const) == True


def test_constant_in_kernel(verify_ir):
    """Test constant folding in kernel context."""
    @kernel
    def const_fold_kernel(buf: Buffer[Float]):
        # idx is now a DSL variable (for correct handling of potential reassignment)
        idx = 0
        # sin(0.0) should be folded to 0.0
        buf[idx] = sin(0.0)
    
    # idx is now a DSL variable
    expected = """
kernel void const_fold_kernel(buffer<f32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  buffer_write(arg0, v2, 0.0);
}
"""
    verify_ir(const_fold_kernel, expected)


def test_routed_function_repr():
    """Test RoutedFunction representation."""
    # sin should now be a RoutedFunction
    assert hasattr(sin, '__class__')
    assert sin.__class__.__name__ == 'RoutedFunction'
    assert 'sin' in repr(sin)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
