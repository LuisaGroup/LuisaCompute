"""Tests for new math functions (rsqrt, exp10, hyperbolic, etc.)."""

import math
import pytest
from luisa import (
    kernel, callable, pprint, Float, Int, Buffer,
    rsqrt, exp10, sinh, cosh, tanh, asinh, acosh, atanh,
    isinf, isnan, copysign, fma,
    clz, ctz, popcount, reverse,
)
from luisa.lang.ir import ConstantValue


def test_rsqrt_constant_folding():
    """Test rsqrt() constant folding."""
    print("\n" + "=" * 60)
    print("Test: RSQRT Constant Folding")
    print("=" * 60)
    
    result = rsqrt(4.0)  # Should be 0.5
    print(f"rsqrt(4.0) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert abs(result.value - 0.5) < 1e-10
    
    print("✓ RSQRT constant folding works")
    print("=" * 60)


def test_exp10_constant_folding():
    """Test exp10() constant folding."""
    print("\n" + "=" * 60)
    print("Test: EXP10 Constant Folding")
    print("=" * 60)
    
    result = exp10(2.0)  # Should be 100.0
    print(f"exp10(2.0) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert abs(result.value - 100.0) < 1e-10
    
    print("✓ EXP10 constant folding works")
    print("=" * 60)


def test_hyperbolic_constant_folding():
    """Test hyperbolic functions constant folding."""
    print("\n" + "=" * 60)
    print("Test: Hyperbolic Functions Constant Folding")
    print("=" * 60)
    
    # sinh
    result = sinh(1.0)
    print(f"sinh(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.sinh(1.0)) < 1e-10
    
    # cosh
    result = cosh(1.0)
    print(f"cosh(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.cosh(1.0)) < 1e-10
    
    # tanh
    result = tanh(1.0)
    print(f"tanh(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.tanh(1.0)) < 1e-10
    
    print("✓ Hyperbolic functions constant folding works")
    print("=" * 60)


def test_inverse_hyperbolic_constant_folding():
    """Test inverse hyperbolic functions constant folding."""
    print("\n" + "=" * 60)
    print("Test: Inverse Hyperbolic Functions Constant Folding")
    print("=" * 60)
    
    # asinh
    result = asinh(1.0)
    print(f"asinh(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.asinh(1.0)) < 1e-10
    
    # acosh (x >= 1)
    result = acosh(2.0)
    print(f"acosh(2.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.acosh(2.0)) < 1e-10
    
    # atanh (|x| < 1)
    result = atanh(0.5)
    print(f"atanh(0.5) = {result}")
    assert isinstance(result, ConstantValue)
    assert abs(result.value - math.atanh(0.5)) < 1e-10
    
    print("✓ Inverse hyperbolic functions constant folding works")
    print("=" * 60)


def test_isinf_isnan_constant_folding():
    """Test isinf() and isnan() constant folding."""
    print("\n" + "=" * 60)
    print("Test: ISINF/ISNAN Constant Folding")
    print("=" * 60)
    
    # isinf
    result = isinf(1.0)
    print(f"isinf(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert result.value == False
    
    result = isinf(float('inf'))
    print(f"isinf(inf) = {result}")
    assert isinstance(result, ConstantValue)
    assert result.value == True
    
    # isnan
    result = isnan(1.0)
    print(f"isnan(1.0) = {result}")
    assert isinstance(result, ConstantValue)
    assert result.value == False
    
    result = isnan(float('nan'))
    print(f"isnan(nan) = {result}")
    assert isinstance(result, ConstantValue)
    assert result.value == True
    
    print("✓ ISINF/ISNAN constant folding works")
    print("=" * 60)


def test_copysign_constant_folding():
    """Test copysign() constant folding."""
    print("\n" + "=" * 60)
    print("Test: COPYSIGN Constant Folding")
    print("=" * 60)
    
    result = copysign(3.0, -1.0)  # Should be -3.0
    print(f"copysign(3.0, -1.0) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert result.value == -3.0
    
    print("✓ COPYSIGN constant folding works")
    print("=" * 60)


def test_fma_constant_folding():
    """Test fma() constant folding."""
    print("\n" + "=" * 60)
    print("Test: FMA Constant Folding")
    print("=" * 60)
    
    result = fma(2.0, 3.0, 4.0)  # Should be 2*3 + 4 = 10
    print(f"fma(2.0, 3.0, 4.0) = {result}")
    
    assert isinstance(result, ConstantValue)
    assert result.value == 10.0
    
    print("✓ FMA constant folding works")
    print("=" * 60)


def test_bit_operations_constant_folding():
    """Test integer bit operations constant folding."""
    print("\n" + "=" * 60)
    print("Test: Bit Operations Constant Folding")
    print("=" * 60)
    
    # popcount
    result = popcount(0b101010)  # Should be 3
    print(f"popcount(0b101010) = {result}")
    assert isinstance(result, ConstantValue)
    assert result.value == 3
    
    print("✓ Bit operations constant folding works")
    print("=" * 60)


def test_new_functions_device_routing():
    """Test that new functions route to device when given DSL values."""
    print("\n" + "=" * 60)
    print("Test: New Functions Device Routing")
    print("=" * 60)
    
    @callable
    def test_funcs(x: Float) -> Float:
        a = rsqrt(x)
        b = exp10(x)
        c = sinh(x)
        return a + b + c
    
    ir = test_funcs(1.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    from luisa.lang.inspect import count_instructions
    counts = count_instructions(ir)
    
    # Should have RSQRT, EXP10, SINH instructions
    assert 'RSQRT' in counts or 'EXP10' in counts or 'SINH' in counts
    
    print("✓ New functions device routing works")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_new_math_functions.py tests")
    print("=" * 70)
    
    test_rsqrt_constant_folding()
    test_exp10_constant_folding()
    test_hyperbolic_constant_folding()
    test_inverse_hyperbolic_constant_folding()
    test_isinf_isnan_constant_folding()
    test_copysign_constant_folding()
    test_fma_constant_folding()
    test_bit_operations_constant_folding()
    test_new_functions_device_routing()
    
    print("\n" + "=" * 70)
    print("All test_new_math_functions.py tests passed!")
    print("=" * 70)
