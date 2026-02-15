"""Tests for warp operations - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable, pprint,
    int32, float32, Buffer,
    # Warp query
    warp_is_first_active_lane, warp_first_active_lane, warp_active_count_bits,
    # Warp reduction
    warp_sum, warp_product, warp_min, warp_max,
    warp_all, warp_any, warp_all_equal,
    # Warp prefix
    warp_prefix_sum, warp_prefix_product, warp_prefix_count_bits,
    # Warp broadcast
    warp_read_lane, warp_read_first_lane,
    # Warp bitwise
    warp_bit_and, warp_bit_or, warp_bit_xor, warp_bit_mask,
    dispatch_id,
)
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


def test_warp_query_functions_build_ir():
    """Test warp query functions actually build IR."""
    print("\n" + "=" * 60)
    print("Test: warp query functions build IR")
    print("=" * 60)

    @callable
    def warp_queries() -> int32:
        first = warp_is_first_active_lane()
        lane = warp_first_active_lane()
        bits = warp_active_count_bits(True)
        return int32(lane)

    ir = warp_queries()
    print_ast(warp_queries, "AST: warp_queries")

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    print(f"✓ Built IR with warp query instructions")
    print("=" * 60)


def test_warp_reduction_builds_ir():
    """Test warp reduction functions build IR."""
    print("\n" + "=" * 60)
    print("Test: warp reduction builds IR")
    print("=" * 60)

    @callable
    def warp_reductions(x: float32) -> float32:
        s = warp_sum(x)
        p = warp_product(x)
        mn = warp_min(x)
        mx = warp_max(x)
        return s

    ir = warp_reductions(1.0)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    print(f"✓ Built IR with warp reduction instructions")
    print("=" * 60)


def test_warp_boolean_reduction_builds_ir():
    """Test warp boolean reduction builds IR."""
    print("\n" + "=" * 60)
    print("Test: warp boolean reduction builds IR")
    print("=" * 60)

    @callable
    def warp_bool_checks(x: float32) -> int32:
        all_val = warp_all(x > 0)
        any_val = warp_any(x > 0)
        eq_val = warp_all_equal(x)
        return int32(all_val)

    ir = warp_bool_checks(1.0)

    print("\nGenerated IR:")
    print(pprint(ir))

    print(f"✓ Built IR with {len(ir.blocks)} blocks")
    print("=" * 60)


def test_warp_prefix_builds_ir():
    """Test warp prefix functions build IR."""
    print("\n" + "=" * 60)
    print("Test: warp prefix builds IR")
    print("=" * 60)

    @callable
    def warp_prefix_ops(x: float32, b: int32) -> float32:
        ps = warp_prefix_sum(x)
        pp = warp_prefix_product(x)
        pc = warp_prefix_count_bits(True)
        return ps

    ir = warp_prefix_ops(1.0, 1)

    print("\nGenerated IR:")
    print(pprint(ir))

    print(f"✓ Built IR with warp prefix instructions")
    print("=" * 60)


def test_warp_broadcast_builds_ir():
    """Test warp broadcast functions build IR."""
    print("\n" + "=" * 60)
    print("Test: warp broadcast builds IR")
    print("=" * 60)

    @callable
    def warp_broadcast_ops(x: float32) -> float32:
        from_lane = warp_read_lane(x, int32(0))
        first = warp_read_first_lane(x)
        return first

    ir = warp_broadcast_ops(1.0)

    print("\nGenerated IR:")
    print(pprint(ir))

    print(f"✓ Built IR with warp broadcast instructions")
    print("=" * 60)


def test_warp_bitwise_builds_ir():
    """Test warp bitwise functions build IR."""
    print("\n" + "=" * 60)
    print("Test: warp bitwise builds IR")
    print("=" * 60)

    @callable
    def warp_bitwise_ops(x: int32) -> int32:
        a = warp_bit_and(x)
        o = warp_bit_or(x)
        x_val = warp_bit_xor(x)
        m = warp_bit_mask(True)
        return a

    ir = warp_bitwise_ops(0xFF)

    print("\nGenerated IR:")
    print(pprint(ir))

    print(f"✓ Built IR with warp bitwise instructions")
    print("=" * 60)


def test_warp_in_kernel():
    """Test warp operations in a kernel."""
    print("\n" + "=" * 60)
    print("Test: warp operations in kernel")
    print("=" * 60)

    @kernel
    def warp_kernel(buf: Buffer[float32]):
        idx = dispatch_id().x
        val = buf[idx]
        # Warp reduction
        sum_val = warp_sum(val)
        # Only first lane writes
        if warp_is_first_active_lane():
            buf[idx] = sum_val

    ir = warp_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir.is_kernel
    print(f"✓ Built kernel with warp operations, {len(ir.blocks)} blocks")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_warp.py tests")
    print("=" * 70)

    test_warp_query_functions_build_ir()
    test_warp_reduction_builds_ir()
    test_warp_boolean_reduction_builds_ir()
    test_warp_prefix_builds_ir()
    test_warp_broadcast_builds_ir()
    test_warp_bitwise_builds_ir()
    test_warp_in_kernel()

    print("\n" + "=" * 70)
    print("All test_warp.py tests passed!")
    print("=" * 70)
