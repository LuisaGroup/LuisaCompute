"""Tests for introspection utilities - with IR building and pretty printing."""

import pytest
from luisa import kernel, callable, int32, float32, pprint
from luisa.lang.inspect import (
    get_ir_source, get_ir_ast, get_ir_types,
    count_instructions, get_basic_block_count, get_instruction_count,
    find_operations, analyze_control_flow, is_kernel, format_ir_summary
)
from luisa.lang.ir import Op


def test_get_ir_source():
    """Test getting IR source from staged function."""
    print("\n" + "=" * 60)
    print("Test: get_ir_source")
    print("=" * 60)

    @callable
    def simple_func(a: float32) -> float32:
        return a + 1.0

    source = get_ir_source(simple_func)
    print(f"\nSource:\n{source}")

    assert source is not None
    assert 'simple_func' in source

    print("✓ Source extracted successfully")
    print("=" * 60)


def test_get_ir_types():
    """Test getting type information."""
    print("\n" + "=" * 60)
    print("Test: get_ir_types")
    print("=" * 60)

    @callable
    def typed_func(a: float32, b: int32) -> float32:
        return a + float32(b)

    types = get_ir_types(typed_func)
    print(f"\nTypes: {types}")

    assert types is not None
    assert len(types['arg_types']) == 2
    assert types['ret_type'] is not None

    print("✓ Types extracted successfully")
    print("=" * 60)


def test_count_instructions():
    """Test instruction counting."""
    print("\n" + "=" * 60)
    print("Test: count_instructions")
    print("=" * 60)

    from luisa import Builder

    builder = Builder('test', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.constant(float32, 1.0)
    c = builder.add(a, b)
    builder.return_(c)
    ir = builder.build()

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    print(f"\nInstruction counts: {dict(counts)}")

    assert 'ADD' in counts
    assert 'RETURN' in counts

    total = get_instruction_count(ir)
    assert total >= 2

    print(f"✓ Total instructions: {total}")
    print("=" * 60)


def test_get_basic_block_count():
    """Test basic block counting."""
    print("\n" + "=" * 60)
    print("Test: get_basic_block_count")
    print("=" * 60)

    from luisa import Builder

    builder = Builder('test', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    builder.return_()
    ir = builder.build()

    print("\nGenerated IR:")
    print(pprint(ir))

    count = get_basic_block_count(ir)
    print(f"\nBlock count: {count}")

    assert count == 1
    print("✓ Block count correct")
    print("=" * 60)


def test_find_operations():
    """Test finding specific operations."""
    print("\n" + "=" * 60)
    print("Test: find_operations")
    print("=" * 60)

    from luisa import Builder

    builder = Builder('test', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    c = builder.add(a, b)
    builder.return_(c)
    ir = builder.build()

    print("\nGenerated IR:")
    print(pprint(ir))

    adds = find_operations(ir, Op.ADD)
    returns = find_operations(ir, Op.RETURN)

    print(f"\nFound {len(adds)} ADD operations")
    print(f"Found {len(returns)} RETURN operations")

    assert len(adds) == 1
    assert len(returns) == 1
    print("✓ Operations found correctly")
    print("=" * 60)


def test_analyze_control_flow():
    """Test control flow analysis."""
    print("\n" + "=" * 60)
    print("Test: analyze_control_flow")
    print("=" * 60)

    from luisa import Builder

    # Simple function
    builder = Builder('test', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    builder.return_()
    ir = builder.build()

    print("\nGenerated IR:")
    print(pprint(ir))

    cf = analyze_control_flow(ir)
    print(f"\nControl flow analysis: {cf}")

    assert cf['blocks'] == 1
    assert cf['returns'] == 1
    print("✓ Control flow analyzed correctly")
    print("=" * 60)


def test_is_kernel():
    """Test kernel detection."""
    print("\n" + "=" * 60)
    print("Test: is_kernel")
    print("=" * 60)

    from luisa import Builder

    builder = Builder('test', (), None)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    builder.return_()
    ir = builder.build()

    print(f"\nis_kernel (before flag): {is_kernel(ir)}")
    assert not is_kernel(ir)

    ir.is_kernel = True
    print(f"is_kernel (after flag): {is_kernel(ir)}")
    assert is_kernel(ir)

    print("✓ Kernel detection works")
    print("=" * 60)


def test_format_ir_summary():
    """Test IR summary formatting."""
    print("\n" + "=" * 60)
    print("Test: format_ir_summary")
    print("=" * 60)

    from luisa import Builder

    builder = Builder('summary_test', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.constant(float32, 1.0)
    c = builder.add(a, b)
    builder.return_(c)
    ir = builder.build()

    summary = format_ir_summary(ir)
    print(f"\nIR Summary:\n{summary}")

    assert 'summary_test' in summary
    assert 'Callable' in summary

    print("✓ Summary formatted correctly")
    print("=" * 60)


def test_inspect_staged_function():
    """Test inspecting a staged function after compilation."""
    print("\n" + "=" * 60)
    print("Test: inspect staged function")
    print("=" * 60)

    @callable
    def compute(a: float32, b: float32) -> float32:
        x = a * a
        y = b * b
        return x + y

    ir = compute(3.0, 4.0)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    total = get_instruction_count(ir)
    summary = format_ir_summary(ir)

    print(f"\nInstruction counts: {dict(counts)}")
    print(f"Total instructions: {total}")
    print(f"\nSummary:\n{summary}")

    assert 'MUL' in counts
    assert 'ADD' in counts
    assert total >= 4

    print("✓ Staged function inspection works")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_inspect.py tests")
    print("=" * 70)

    test_get_ir_source()
    test_get_ir_types()
    test_count_instructions()
    test_get_basic_block_count()
    test_find_operations()
    test_analyze_control_flow()
    test_is_kernel()
    test_format_ir_summary()
    test_inspect_staged_function()

    print("\n" + "=" * 70)
    print("All test_inspect.py tests passed!")
    print("=" * 70)
