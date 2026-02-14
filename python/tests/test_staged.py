"""Tests for staged functions."""

import pytest
from luisa import (
    kernel, callable, StagedFunction,
    int32, float32,
)


def test_staged_function_basic():
    """Test basic staged function."""
    print("Testing staged function basic...")
    
    @callable
    def add(a: float32, b: float32) -> float32:
        return a + b
    
    assert isinstance(add, StagedFunction)
    assert add.name == 'add'
    assert not add.is_kernel
    
    # Generate IR
    ir_func = add(1.0, 2.0)
    
    assert ir_func.name == 'add'
    assert len(ir_func.blocks) > 0
    assert len(ir_func.blocks[0].instructions) > 0
    
    # Check caching
    ir_func2 = add(1.0, 2.0)
    assert ir_func is ir_func2
    
    print("  ✓ Staged function basic OK")


def test_staged_function_with_kernel():
    """Test staged function marked as kernel."""
    print("Testing staged function kernel...")
    
    @kernel
    def simple_kernel(x: int32) -> None:
        pass
    
    assert isinstance(simple_kernel, StagedFunction)
    assert simple_kernel.is_kernel
    
    # Generate IR
    ir_func = simple_kernel(42)
    assert ir_func.is_kernel
    
    print("  ✓ Staged function kernel OK")


def test_staged_function_control_flow():
    """Test staged function with control flow."""
    print("Testing staged function control flow...")
    
    @callable
    def abs_value(x: float32) -> float32:
        if x > 0.0:
            return x
        else:
            return -x
    
    # Generate IR
    ir_func = abs_value(1.0)
    
    assert ir_func.name == 'abs_value'
    # Should have created multiple blocks for if-else
    assert len(ir_func.blocks) >= 2
    
    print("  ✓ Staged function control flow OK")


def test_staged_function_captured_vars():
    """Test staged function with captured variables."""
    print("Testing staged function captured vars...")
    
    threshold = 0.5  # Captured variable
    
    @callable
    def threshold_check(x: float32) -> int32:
        if x > threshold:
            return 1
        else:
            return 0
    
    # Generate IR
    ir_func = threshold_check(1.0)
    
    assert ir_func.name == 'threshold_check'
    # The threshold value should be folded as a constant
    
    print("  ✓ Staged function captured vars OK")


def test_staged_function_while_loop():
    """Test staged function with while loop."""
    print("Testing staged function while loop...")
    
    @callable
    def count_up() -> int32:
        i = 0
        while i < 10:
            i = i + 1
        return i
    
    # Generate IR
    ir_func = count_up()
    
    assert ir_func.name == 'count_up'
    # Should have loop blocks
    
    print("  ✓ Staged function while loop OK")


def test_staged_function_for_range():
    """Test staged function with for-range loop."""
    print("Testing staged function for-range loop...")
    
    @callable
    def sum_range(n: int32) -> int32:
        total = 0
        for i in range(n):
            total = total + i
        return total
    
    # Generate IR
    ir_func = sum_range(10)
    
    assert ir_func.name == 'sum_range'
    # Should have loop blocks
    
    print("  ✓ Staged function for-range loop OK")
