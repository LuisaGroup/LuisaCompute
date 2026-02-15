"""Tests for staged functions - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable, StagedFunction,
    int32, float32,
    pprint,
)
from luisa.lang.inspect import get_ir_ast, get_ir_source
from luisa.lang.inspect import count_instructions


def test_staged_function_basic():
    """Test basic staged function."""
    print("\n" + "=" * 60)
    print("Test: staged function basic")
    print("=" * 60)

    @callable
    def add(a: float32, b: float32) -> float32:
        return a + b

    assert isinstance(add, StagedFunction)
    assert add.name == 'add'
    assert not add.is_kernel

    ir_func = add(1.0, 2.0)

    # Print AST
    print("\nParsed AST:")
    import ast
    ast_tree = get_ir_ast(add)
    print(ast.dump(ast_tree, indent=2))

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.name == 'add'
    assert len(ir_func.blocks) > 0
    assert len(ir_func.blocks[0].instructions) > 0

    # Check caching
    ir_func2 = add(1.0, 2.0)
    assert ir_func is ir_func2

    print(f"✓ Staged function built with {len(ir_func.blocks)} blocks")
    print("=" * 60)


def test_staged_function_with_kernel():
    """Test staged function marked as kernel."""
    print("\n" + "=" * 60)
    print("Test: staged function kernel")
    print("=" * 60)

    @kernel
    def simple_kernel(x: int32) -> None:
        pass

    assert isinstance(simple_kernel, StagedFunction)
    assert simple_kernel.is_kernel

    ir_func = simple_kernel(42)

    # Print AST
    print("\nParsed AST:")
    import ast
    ast_tree = get_ir_ast(simple_kernel)
    print(ast.dump(ast_tree, indent=2))

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.is_kernel

    print(f"✓ Kernel built with {len(ir_func.blocks)} blocks")
    print("=" * 60)


def test_staged_function_control_flow():
    """Test staged function with control flow."""
    print("\n" + "=" * 60)
    print("Test: staged function control flow")
    print("=" * 60)

    @callable
    def abs_value(x: float32) -> float32:
        if x > 0.0:
            return x
        else:
            return -x

    from luisa.lang.ir import ArgumentValue
    ir_func = abs_value(ArgumentValue(typ=float32, index=0))

    # Print AST
    print("\nParsed AST:")
    import ast
    ast_tree = get_ir_ast(abs_value)
    print(ast.dump(ast_tree, indent=2))

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.name == 'abs_value'
    assert len(ir_func.blocks) >= 2

    print(f"✓ Control flow built with {len(ir_func.blocks)} blocks")
    print("=" * 60)


def test_staged_function_captured_vars():
    """Test staged function with captured variables."""
    print("\n" + "=" * 60)
    print("Test: staged function captured vars")
    print("=" * 60)

    threshold = 0.5

    @callable
    def threshold_check(x: float32) -> int32:
        if x > threshold:
            return 1
        else:
            return 0

    from luisa.lang.ir import ArgumentValue
    ir_func = threshold_check(ArgumentValue(typ=float32, index=0))

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.name == 'threshold_check'

    print(f"✓ Captured vars built with {len(ir_func.blocks)} blocks")
    print("=" * 60)


def test_staged_function_while_loop():
    """Test staged function with while loop."""
    print("\n" + "=" * 60)
    print("Test: staged function while loop")
    print("=" * 60)

    @callable
    def count_up() -> int32:
        i = int32(0)
        while i < 10:
            i = i + 1
        return i

    ir_func = count_up()

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.name == 'count_up'

    counts = count_instructions(ir_func)
    print(f"✓ While loop built with {len(ir_func.blocks)} blocks")
    print(f"  Instructions: {dict(counts)}")
    print("=" * 60)


def test_staged_function_for_range():
    """Test staged function with for-range loop."""
    print("\n" + "=" * 60)
    print("Test: staged function for-range loop")
    print("=" * 60)

    @callable
    def sum_range(n: int32) -> int32:
        total = int32(0)
        for i in range(n):
            total = total + i
        return total

    from luisa.lang.ir import ArgumentValue
    ir_func = sum_range(ArgumentValue(typ=int32, index=0))

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.name == 'sum_range'

    print(f"✓ For-range loop built with {len(ir_func.blocks)} blocks")
    print("=" * 60)


def test_staged_function_complex():
    """Test complex staged function with multiple operations."""
    print("\n" + "=" * 60)
    print("Test: staged function complex")
    print("=" * 60)

    @callable
    def compute(x: float32, y: float32) -> float32:
        # Compute x^2 + y^2
        x2 = x * x
        y2 = y * y
        sum_sq = x2 + y2

        # Apply some conditions
        if sum_sq > 0.0:
            return sum_sq
        else:
            return float32(0.0)

    ir_func = compute(3.0, 4.0)

    print("\nGenerated IR:")
    print(pprint(ir_func))

    counts = count_instructions(ir_func)

    assert 'MUL' in counts
    assert 'ADD' in counts

    print(f"✓ Complex function built with {len(ir_func.blocks)} blocks")
    print(f"  Instructions: {dict(counts)}")
    print("=" * 60)


def test_kernel_with_dispatch_id():
    """Test kernel using dispatch_id."""
    print("\n" + "=" * 60)
    print("Test: kernel with dispatch_id")
    print("=" * 60)

    from luisa import Buffer, dispatch_id

    @kernel
    def index_kernel(buf: Buffer[float32]):
        idx = dispatch_id().x
        buf[idx] = float32(idx)

    ir_func = index_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir_func))

    assert ir_func.is_kernel

    counts = count_instructions(ir_func)
    assert 'DISPATCH_ID' in counts

    print(f"✓ Kernel with dispatch_id built, {len(ir_func.blocks)} blocks")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_staged.py tests")
    print("=" * 70)

    test_staged_function_basic()
    test_staged_function_with_kernel()
    test_staged_function_control_flow()
    test_staged_function_captured_vars()
    test_staged_function_while_loop()
    test_staged_function_for_range()
    test_staged_function_complex()
    test_kernel_with_dispatch_id()

    print("\n" + "=" * 70)
    print("All test_staged.py tests passed!")
    print("=" * 70)
