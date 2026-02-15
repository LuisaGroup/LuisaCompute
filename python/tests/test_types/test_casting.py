"""Tests for type casting - with IR building and pretty printing."""

import pytest
import ast as python_ast
from luisa import kernel, callable, Float, Int, Buffer, pprint
from luisa.lang.inspect import count_instructions, get_ir_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


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


def test_int_to_float_cast():
    """Test casting int to float - builds and prints IR."""
    print("\n" + "=" * 60)
    print("Test: Int to Float cast")
    print("=" * 60)

    @callable
    def cast_int_to_float(x: Int) -> Float:
        return Float(x)

    ir = cast_int_to_float(42)
    print_ast(cast_int_to_float, "AST: cast_int_to_float")

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CAST' in counts
    assert counts['CAST'] == 1

    print(f"✓ CAST instruction count: {counts['CAST']}")
    print("=" * 60)


def test_float_to_int_cast():
    """Test casting float to int."""
    print("\n" + "=" * 60)
    print("Test: Float to Int cast")
    print("=" * 60)

    @callable
    def cast_float_to_int(x: Float) -> Int:
        return Int(x)

    ir = cast_float_to_int(3.14)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CAST' in counts

    print(f"✓ CAST instruction count: {counts.get('CAST', 0)}")
    print("=" * 60)


def test_cast_in_computation():
    """Test cast in the middle of computation."""
    print("\n" + "=" * 60)
    print("Test: cast in computation")
    print("=" * 60)

    @callable
    def mixed_computation(i: Int, f: Float) -> Float:
        return Float(i) + f

    ir = mixed_computation(10, 2.5)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CAST' in counts
    assert 'ADD' in counts

    print(f"✓ CAST={counts.get('CAST', 0)}, ADD={counts.get('ADD', 0)}")
    print("=" * 60)


def test_cast_with_buffer():
    """Test cast with buffer operations."""
    print("\n" + "=" * 60)
    print("Test: cast with buffer")
    print("=" * 60)

    @callable
    def store_index_as_float(buf: Buffer[Float], idx: Int) -> None:
        buf[idx] = Float(idx) * 2.0

    ir = store_index_as_float(0, 5)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CAST' in counts
    assert 'MUL' in counts
    assert 'BUFFER_WRITE' in counts

    print(f"✓ CAST={counts.get('CAST', 0)}, MUL={counts.get('MUL', 0)}, "
          f"BUFFER_WRITE={counts.get('BUFFER_WRITE', 0)}")
    print("=" * 60)


def test_chained_casts():
    """Test multiple chained casts."""
    print("\n" + "=" * 60)
    print("Test: chained casts")
    print("=" * 60)

    @callable
    def chain_cast(x: Int) -> Int:
        f = Float(x)
        i = Int(f)
        return i

    ir = chain_cast(42)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    # Should have 2 casts
    assert counts.get('CAST', 0) == 2

    print(f"✓ Chained casts: {counts.get('CAST', 0)} CAST instructions")
    print("=" * 60)


def test_cast_in_kernel():
    """Test cast in a kernel context."""
    print("\n" + "=" * 60)
    print("Test: cast in kernel")
    print("=" * 60)

    from luisa import dispatch_id

    @kernel
    def cast_kernel(out: Buffer[Float]):
        idx = dispatch_id().x
        out[idx] = Float(idx) * 1.5

    ir = cast_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir.is_kernel
    counts = count_instructions(ir)

    print(f"✓ Kernel with {len(ir.blocks)} blocks, CAST={counts.get('CAST', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_casting.py tests")
    print("=" * 70)

    test_int_to_float_cast()
    test_float_to_int_cast()
    test_cast_in_computation()
    test_cast_with_buffer()
    test_chained_casts()
    test_cast_in_kernel()

    print("\n" + "=" * 70)
    print("All test_casting.py tests passed!")
    print("=" * 70)
