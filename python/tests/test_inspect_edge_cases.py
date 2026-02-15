"""Edge case tests for the inspect module - with IR building and pretty printing."""

import pytest
from luisa import kernel, callable, Int, Float, pprint
from luisa.lang.inspect import (
    get_ir_source, get_ir_ast, get_ir_types,
    count_instructions, get_basic_block_count,
    find_operations, analyze_control_flow,
    is_kernel, format_ir_summary, get_type_size
)
from luisa.lang.types import Int as int32_type, Float as float32_type


def test_get_ir_source_non_staged():
    """Test get_ir_source with a non-staged function."""
    print("\n" + "=" * 60)
    print("Test: get_ir_source non-staged")
    print("=" * 60)

    def regular_func():
        pass

    result = get_ir_source(regular_func)
    assert result is None

    print("✓ Non-staged function returns None")
    print("=" * 60)


def test_get_ir_ast_non_staged():
    """Test get_ir_ast with a non-staged function."""
    print("\n" + "=" * 60)
    print("Test: get_ir_ast non-staged")
    print("=" * 60)

    def regular_func():
        pass

    result = get_ir_ast(regular_func)
    assert result is None

    print("✓ Non-staged function returns None")
    print("=" * 60)


def test_get_ir_types_non_staged():
    """Test get_ir_types with a non-staged function."""
    print("\n" + "=" * 60)
    print("Test: get_ir_types non-staged")
    print("=" * 60)

    def regular_func():
        pass

    result = get_ir_types(regular_func)
    assert result is None

    print("✓ Non-staged function returns None")
    print("=" * 60)


def test_count_instructions_empty():
    """Test count_instructions with a simple function."""
    print("\n" + "=" * 60)
    print("Test: count_instructions empty")
    print("=" * 60)

    @callable
    def simple() -> Int:
        return 0

    ir = simple()

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert isinstance(counts, dict)

    print(f"✓ Instructions counted: {dict(counts)}")
    print("=" * 60)


def test_get_basic_block_count():
    """Test get_basic_block_count."""
    print("\n" + "=" * 60)
    print("Test: get_basic_block_count")
    print("=" * 60)

    @callable
    def simple() -> Int:
        return 0

    ir = simple()

    print("\nGenerated IR:")
    print(pprint(ir))

    count = get_basic_block_count(ir)
    assert count >= 1

    print(f"✓ Block count: {count}")
    print("=" * 60)


def test_find_operations_no_match():
    """Test find_operations with no matching operations."""
    print("\n" + "=" * 60)
    print("Test: find_operations no match")
    print("=" * 60)

    from luisa.lang.ir import Op

    @callable
    def simple() -> Int:
        return 0

    ir = simple()

    print("\nGenerated IR:")
    print(pprint(ir))

    results = find_operations(ir, Op.ATOMIC_ADD)
    assert len(results) == 0

    print("✓ No ATOMIC_ADD operations found (as expected)")
    print("=" * 60)


def test_analyze_control_flow_simple():
    """Test analyze_control_flow with simple function."""
    print("\n" + "=" * 60)
    print("Test: analyze_control_flow simple")
    print("=" * 60)

    @callable
    def simple() -> Int:
        return 0

    ir = simple()

    print("\nGenerated IR:")
    print(pprint(ir))

    cf = analyze_control_flow(ir)

    assert 'blocks' in cf
    assert 'branches' in cf
    assert 'conditional_branches' in cf
    assert 'returns' in cf
    assert 'has_loops' in cf

    print(f"✓ Control flow: {cf}")
    print("=" * 60)


def test_analyze_control_flow_with_if():
    """Test analyze_control_flow with conditional."""
    print("\n" + "=" * 60)
    print("Test: analyze_control_flow with if")
    print("=" * 60)

    @callable
    def with_conditional(x: Int) -> Int:
        if x > 0:
            return x
        return -x

    ir = with_conditional(1)

    print("\nGenerated IR:")
    print(pprint(ir))

    cf = analyze_control_flow(ir)
    assert cf['conditional_branches'] >= 1

    print(f"✓ Control flow with if: {cf}")
    print("=" * 60)


def test_format_ir_summary():
    """Test format_ir_summary."""
    print("\n" + "=" * 60)
    print("Test: format_ir_summary")
    print("=" * 60)

    @callable
    def simple() -> Int:
        return 0

    ir = simple()

    summary = format_ir_summary(ir)

    print("\nIR Summary:")
    print(summary)

    assert 'Function:' in summary
    assert 'simple' in summary
    assert 'Type:' in summary
    assert 'Arguments:' in summary

    print("✓ Summary formatted")
    print("=" * 60)


def test_is_kernel_true():
    """Test is_kernel with a kernel function."""
    print("\n" + "=" * 60)
    print("Test: is_kernel true")
    print("=" * 60)

    @kernel
    def my_kernel():
        pass

    ir = my_kernel()

    print("\nGenerated IR:")
    print(pprint(ir))

    assert is_kernel(ir)

    print("✓ Kernel detected correctly")
    print("=" * 60)


def test_is_kernel_false():
    """Test is_kernel with a callable function."""
    print("\n" + "=" * 60)
    print("Test: is_kernel false")
    print("=" * 60)

    @callable
    def my_callable() -> Int:
        return 0

    ir = my_callable()

    print("\nGenerated IR:")
    print(pprint(ir))

    assert not is_kernel(ir)

    print("✓ Callable detected correctly")
    print("=" * 60)


def test_get_type_size_scalar():
    """Test get_type_size for scalar types."""
    print("\n" + "=" * 60)
    print("Test: get_type_size scalar")
    print("=" * 60)

    assert get_type_size(int32_type) == 4
    assert get_type_size(float32_type) == 4

    print(f"✓ Int: {get_type_size(int32_type)} bytes")
    print(f"✓ Float: {get_type_size(float32_type)} bytes")
    print("=" * 60)


def test_get_type_size_vector():
    """Test get_type_size for vector types."""
    print("\n" + "=" * 60)
    print("Test: get_type_size vector")
    print("=" * 60)

    from luisa.lang.types import Float3
    assert get_type_size(Float3) == 12  # 4 * 3

    print(f"✓ Float3: {get_type_size(Float3)} bytes")
    print("=" * 60)


def test_get_ir_types_content():
    """Test that get_ir_types returns correct content."""
    print("\n" + "=" * 60)
    print("Test: get_ir_types content")
    print("=" * 60)

    @callable
    def typed_func(x: Int, y: Float) -> Int:
        return x

    types = get_ir_types(typed_func)

    print(f"\nTypes: {types}")

    assert types is not None
    assert 'arg_types' in types
    assert 'ret_type' in types
    assert 'captured_vars' in types

    print("✓ Types structure correct")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_inspect_edge_cases.py tests")
    print("=" * 70)

    test_get_ir_source_non_staged()
    test_get_ir_ast_non_staged()
    test_get_ir_types_non_staged()
    test_count_instructions_empty()
    test_get_basic_block_count()
    test_find_operations_no_match()
    test_analyze_control_flow_simple()
    test_analyze_control_flow_with_if()
    test_format_ir_summary()
    test_is_kernel_true()
    test_is_kernel_false()
    test_get_type_size_scalar()
    test_get_type_size_vector()
    test_get_ir_types_content()

    print("\n" + "=" * 70)
    print("All test_inspect_edge_cases.py tests passed!")
    print("=" * 70)
