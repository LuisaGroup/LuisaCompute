"""Tests for unrolled loops."""

import pytest
from luisa import kernel, callable, float32, int32, Buffer, unrolled
from luisa.lang.inspect import count_instructions

from luisa.lang.inspect import get_ir_ast
import ast as python_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")



def test_unrolled_simple():
    """Test simple unrolled loop."""
    @callable
    def unrolled_sum(buf: Buffer[float32]) -> None:
        total = 0.0
        for i in unrolled(range(4)):
            total = total + float32(i)
        buf[0] = total
    
    ir = unrolled_sum(0)
    print_ast(unrolled_sum, "AST: unrolled_sum")
    
    # Should have 4 ADD operations (unrolled)
    counts = count_instructions(ir)
    assert counts['ADD'] == 4


def test_unrolled_with_captured_constant():
    """Test unrolled loop with captured constant."""
    UNROLL_COUNT = 3
    
    @callable
    def unrolled_with_capture(buf: Buffer[float32]) -> None:
        for i in unrolled(range(UNROLL_COUNT)):
            buf[i] = float32(i)
    
    ir = unrolled_with_capture(0)
    
    # Should have 3 BUFFER_WRITE and 3 CAST operations
    counts = count_instructions(ir)
    assert counts['BUFFER_WRITE'] == 3
    assert counts['CAST'] == 3


def test_unrolled_with_computation():
    """Test unrolled loop with computation."""
    @callable
    def unrolled_compute(buf: Buffer[float32]) -> None:
        for i in unrolled(range(4)):
            buf[i] = float32(i) * 2.0 + 1.0
    
    ir = unrolled_compute(0)
    
    counts = count_instructions(ir)
    # Each iteration: CAST, MUL, ADD, BUFFER_WRITE
    assert counts['BUFFER_WRITE'] == 4
    assert counts['MUL'] == 4
    assert counts['ADD'] == 4
    assert counts['CAST'] == 4


def test_unrolled_with_step():
    """Test unrolled loop with step."""
    @callable
    def unrolled_step(buf: Buffer[float32]) -> None:
        for i in unrolled(range(0, 8, 2)):  # 0, 2, 4, 6
            buf[i // 2] = float32(i)
    
    ir = unrolled_step(0)
    
    counts = count_instructions(ir)
    # Should have 4 iterations
    assert counts['BUFFER_WRITE'] == 4


def test_nested_unrolled():
    """Test nested unrolled loops."""
    @callable
    def nested_unrolled(buf: Buffer[float32]) -> None:
        for i in unrolled(range(2)):
            for j in unrolled(range(2)):
                idx = i * 2 + j
                buf[idx] = float32(i + j)
    
    ir = nested_unrolled(0)
    
    counts = count_instructions(ir)
    # Should have 4 total iterations (2x2)
    assert counts['BUFFER_WRITE'] == 4


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_unrolled_loops.py tests")
    print("="*70)
    
    test_unrolled_simple()
    test_unrolled_with_captured_constant()
    test_unrolled_with_computation()
    test_unrolled_with_step()
    test_nested_unrolled()
    
    print("\n" + "="*70)
    print("All test_unrolled_loops.py tests passed!")
    print("="*70)
