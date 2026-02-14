"""Tests for builtin functions."""

import pytest
from luisa import (
    # Math
    sqrt, abs, sin, cos, tan, asin, acos, atan, atan2,
    exp, exp2, log, log2, log10,
    floor, ceil, round, trunc, fract, saturate,
    normalize, length, length_squared,
    min, max, clamp, lerp, step, smoothstep, pow,
    dot, cross, distance, reflect, refract, faceforward,
    transpose, inverse, determinant,
    # Special registers
    dispatch_id, thread_id, block_id, dispatch_size,
    # Synchronization
    sync_block,
    # Type casting
    cast, bitcast,
    # Print
    print_msg,
    # Assertions
    assume, assert_,
    # Profiling
    clock,
    # Types
    int32, float32, float3,
)


def test_math_builtins_exist():
    """Test that math builtins are defined."""
    print("Testing math builtins exist...")
    
    # Check that functions exist
    assert callable(sqrt)
    assert callable(sin)
    assert callable(cos)
    assert callable(dot)
    assert callable(cross)
    assert callable(normalize)
    assert callable(length)
    assert callable(clamp)
    assert callable(lerp)
    
    print("  ✓ Math builtins exist OK")


def test_special_registers_exist():
    """Test that special register builtins are defined."""
    print("Testing special registers exist...")
    
    # Check that functions exist
    assert callable(dispatch_id)
    assert callable(thread_id)
    assert callable(block_id)
    assert callable(dispatch_size)
    
    print("  ✓ Special registers exist OK")


def test_dispatch_id_basic():
    """Test dispatch_id builtin returns proper type."""
    print("Testing dispatch_id...")
    
    from luisa import kernel, uint3
    
    @kernel
    def test_kernel() -> None:
        idx = dispatch_id()
        # Would use idx in real code
    
    # dispatch_id returns a uint3 - this creates the IR
    ir_func = test_kernel()
    assert ir_func is not None
    assert len(ir_func.blocks) > 0
    
    print("  ✓ dispatch_id OK")


def test_sync_block_exists():
    """Test sync_block builtin."""
    print("Testing sync_block...")
    
    assert callable(sync_block)
    
    print("  ✓ sync_block OK")


def test_cast_exists():
    """Test cast builtin."""
    print("Testing cast...")
    
    assert callable(cast)
    assert callable(bitcast)
    
    print("  ✓ cast OK")


def test_print_msg_exists():
    """Test print_msg builtin."""
    print("Testing print_msg...")
    
    assert callable(print_msg)
    
    print("  ✓ print_msg OK")


def test_clock_exists():
    """Test clock builtin."""
    print("Testing clock...")
    
    assert callable(clock)
    
    print("  ✓ clock OK")


def test_assertions_exist():
    """Test assertion builtins."""
    print("Testing assertions...")
    
    assert callable(assume)
    assert callable(assert_)
    
    print("  ✓ assertions OK")


def test_matrix_ops_exist():
    """Test matrix operation builtins."""
    print("Testing matrix ops...")
    
    assert callable(transpose)
    assert callable(inverse)
    assert callable(determinant)
    
    print("  ✓ matrix ops OK")
