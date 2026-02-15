"""Tests for atomic operations - with IR building and pretty printing."""

import pytest
import ast as python_ast
from luisa import (
    kernel, callable,
    Int, UInt, Float,
    Buffer,
    atomic_exchange, atomic_add, atomic_sub,
    atomic_and, atomic_or, atomic_xor,
    atomic_min, atomic_max,
    dispatch_id,
    pprint,
)
from luisa.lang.inspect import get_ir_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


def test_atomic_add_builds_ir():
    """Test atomic_add actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: atomic_add builds IR")
    print("=" * 60)

    @kernel
    def atomic_add_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_add(buf, idx, 1)

    # Build IR with a dummy buffer (None for compilation only)
    ir = atomic_add_kernel(None)

    print_ast(atomic_add_kernel, "AST: atomic_add_kernel")

    # Print the generated IR
    print("\nGenerated IR:")
    print(pprint(ir))

    # Verify it actually built something
    assert ir is not None
    assert ir.is_kernel
    assert len(ir.blocks) > 0

    # Count instructions
    total_inst = sum(len(b.instructions) for b in ir.blocks)
    assert total_inst > 0, "Should have generated instructions"
    print(f"✓ Generated {len(ir.blocks)} blocks with {total_inst} instructions")
    print("=" * 60)


def test_atomic_exchange_builds_ir():
    """Test atomic_exchange actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: atomic_exchange builds IR")
    print("=" * 60)

    @kernel
    def atomic_exchange_kernel(buf: Buffer[Int], val: Int) -> Int:
        idx = dispatch_id().x
        return atomic_exchange(buf, idx, val)

    ir = atomic_exchange_kernel(None, 42)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir is not None
    assert ir.is_kernel
    total_inst = sum(len(b.instructions) for b in ir.blocks)
    assert total_inst > 0
    print(f"✓ Generated {len(ir.blocks)} blocks with {total_inst} instructions")
    print("=" * 60)


def test_atomic_sub_builds_ir():
    """Test atomic_sub actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: atomic_sub builds IR")
    print("=" * 60)

    @kernel
    def atomic_sub_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_sub(buf, idx, 1)

    ir = atomic_sub_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir is not None
    assert len(ir.blocks) > 0
    print(f"✓ Generated {len(ir.blocks)} blocks")
    print("=" * 60)


def test_atomic_bitwise_builds_ir():
    """Test atomic bitwise operations build IR."""
    print("\n" + "=" * 60)
    print("Test: atomic bitwise builds IR")
    print("=" * 60)

    @kernel
    def atomic_bitwise_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_and(buf, idx, 0xFF)
        atomic_or(buf, idx, 0x01)
        atomic_xor(buf, idx, 0x02)

    ir = atomic_bitwise_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir is not None
    assert len(ir.blocks) > 0
    total_inst = sum(len(b.instructions) for b in ir.blocks)
    assert total_inst >= 3, "Should have at least 3 atomic instructions"
    print(f"✓ Generated {len(ir.blocks)} blocks with {total_inst} instructions")
    print("=" * 60)


def test_atomic_min_max_builds_ir():
    """Test atomic min/max actually build IR."""
    print("\n" + "=" * 60)
    print("Test: atomic_min/max builds IR")
    print("=" * 60)

    @kernel
    def atomic_minmax_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_min(buf, idx, 100)
        atomic_max(buf, idx, 0)

    ir = atomic_minmax_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir is not None
    assert len(ir.blocks) > 0
    print(f"✓ Generated {len(ir.blocks)} blocks")
    print("=" * 60)


def test_multiple_atomics_in_kernel():
    """Test multiple atomic operations in one kernel."""
    print("\n" + "=" * 60)
    print("Test: multiple atomics in kernel")
    print("=" * 60)

    @kernel
    def multi_atomic_kernel(counter: Buffer[Int], sum_buf: Buffer[Int]):
        idx = dispatch_id().x
        # Increment counter
        old_val = atomic_add(counter, idx, 1)
        # Add to sum
        atomic_add(sum_buf, idx, old_val)

    ir = multi_atomic_kernel(None, None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir is not None
    total_inst = sum(len(b.instructions) for b in ir.blocks)
    assert total_inst >= 2, "Should have at least 2 atomic instructions"
    print(f"✓ Generated {len(ir.blocks)} blocks with {total_inst} instructions")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_atomics.py tests")
    print("=" * 70)

    test_atomic_add_builds_ir()
    test_atomic_exchange_builds_ir()
    test_atomic_sub_builds_ir()
    test_atomic_bitwise_builds_ir()
    test_atomic_min_max_builds_ir()
    test_multiple_atomics_in_kernel()

    print("\n" + "=" * 70)
    print("All test_atomics.py tests passed!")
    print("=" * 70)
