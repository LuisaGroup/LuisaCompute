"""Tests for kernel calling callable functions."""

import pytest
import ast as python_ast
from luisa import kernel, callable, float32, int32, Buffer, pprint
from luisa.lang.inspect import count_instructions, get_ir_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


def test_kernel_calls_simple_callable():
    """Test kernel calling a simple callable function."""
    print("\n" + "="*60)
    print("Test: kernel calls simple callable")
    print("="*60)
    
    @callable
    def square(x: float32) -> float32:
        return x * x
    
    @kernel
    def compute_squares(buf: Buffer[float32]):
        idx = int32(0)  # Simplified for test
        val = buf[idx]
        result = square(val)
        buf[idx] = result
    
    ir = compute_squares(None)
    
    print("\nCallable 'square' AST:")
    print_ast(square, "AST: square")
    
    print("\nGenerated Kernel IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    counts = count_instructions(ir)
    
    # Should have a CALL instruction
    assert 'CALL' in counts or 'MUL' in counts, "Should have CALL or inlined MUL"
    
    print(f"✓ Kernel calling callable works! {len(ir.blocks)} blocks")
    print(f"  Instructions: {dict(counts)}")
    print("="*60)


def test_kernel_calls_math_callable():
    """Test kernel calling a callable with math operations."""
    print("\n" + "="*60)
    print("Test: kernel calls math callable")
    print("="*60)
    
    @callable
    def normalize_value(x: float32) -> float32:
        if x < 0.0:
            return 0.0
        elif x > 1.0:
            return 1.0
        return x
    
    @kernel
    def process_buffer(buf: Buffer[float32]):
        idx = int32(0)
        val = buf[idx]
        normalized = normalize_value(val)
        buf[idx] = normalized
    
    ir = process_buffer(None)
    
    print("\nCallable 'normalize_value' AST:")
    print_ast(normalize_value, "AST: normalize_value")
    
    print("\nGenerated Kernel IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    
    print(f"✓ Kernel calling math callable works! {len(ir.blocks)} blocks")
    print("="*60)


def test_kernel_calls_callable_with_multiple_args():
    """Test kernel calling callable with multiple arguments."""
    print("\n" + "="*60)
    print("Test: kernel calls callable with multiple args")
    print("="*60)
    
    @callable
    def lerp(a: float32, b: float32, t: float32) -> float32:
        return a + (b - a) * t
    
    @kernel
    def interpolate(buf: Buffer[float32]):
        idx = int32(0)
        result = lerp(0.0, 1.0, buf[idx])
        buf[idx] = result
    
    ir = interpolate(None)
    
    print("\nCallable 'lerp' AST:")
    print_ast(lerp, "AST: lerp")
    
    print("\nGenerated Kernel IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    
    print(f"✓ Kernel calling multi-arg callable works! {len(ir.blocks)} blocks")
    print("="*60)


def test_kernel_calls_nested_callable():
    """Test kernel calling a callable that calls another callable."""
    print("\n" + "="*60)
    print("Test: kernel calls nested callable")
    print("="*60)
    
    @callable
    def square(x: float32) -> float32:
        return x * x
    
    @callable
    def sum_of_squares(a: float32, b: float32) -> float32:
        return square(a) + square(b)
    
    @kernel
    def compute(buf: Buffer[float32]):
        idx = int32(0)
        result = sum_of_squares(buf[idx], buf[idx + 1])
        buf[idx] = result
    
    ir = compute(None)
    
    print("\nCallable 'sum_of_squares' AST:")
    print_ast(sum_of_squares, "AST: sum_of_squares")
    
    print("\nGenerated Kernel IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    
    print(f"✓ Kernel calling nested callable works! {len(ir.blocks)} blocks")
    print("="*60)


def test_kernel_calls_callable_with_loop():
    """Test kernel calling a callable that contains a loop."""
    print("\n" + "="*60)
    print("Test: kernel calls callable with loop")
    print("="*60)
    
    @callable
    def factorial(n: int32) -> int32:
        result = 1
        i = 1
        while i <= n:
            result = result * i
            i = i + 1
        return result
    
    @kernel
    def compute_factorials(buf: Buffer[int32]):
        idx = int32(0)
        n = buf[idx]
        result = factorial(n)
        buf[idx] = result
    
    ir = compute_factorials(None)
    
    print("\nCallable 'factorial' AST:")
    print_ast(factorial, "AST: factorial")
    
    print("\nGenerated Kernel IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    
    print(f"✓ Kernel calling callable with loop works! {len(ir.blocks)} blocks")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_kernel_call_callable.py tests")
    print("="*70)
    
    test_kernel_calls_simple_callable()
    test_kernel_calls_math_callable()
    test_kernel_calls_callable_with_multiple_args()
    test_kernel_calls_nested_callable()
    test_kernel_calls_callable_with_loop()
    
    print("\n" + "="*70)
    print("All test_kernel_call_callable.py tests passed!")
    print("="*70)
