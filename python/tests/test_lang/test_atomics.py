"""Tests for atomic operations - with IR building and pretty printing."""

from luisa import (
    kernel,
    Int,
    Buffer,
    atomic_exchange, atomic_add, atomic_sub,
    atomic_and, atomic_or, atomic_xor,
    atomic_min, atomic_max,
    dispatch_id,
)


def test_atomic_add_builds_ir(print_ir, verify_ir):
    """Test atomic_add actually builds IR."""
    @kernel
    def atomic_add_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_add(buf, idx, 1)

    print_ir(atomic_add_kernel, "atomic_add_kernel")

    assert atomic_add_kernel.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel void atomic_add_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_add(arg0, v4, 1);
}
"""
    verify_ir(atomic_add_kernel, expected)


def test_atomic_exchange_builds_ir(print_ir, verify_ir):
    """Test atomic_exchange actually builds IR."""
    @kernel
    def atomic_exchange_kernel(buf: Buffer[Int], val: Int) -> Int:
        idx = dispatch_id().x
        return atomic_exchange(buf, idx, val)

    print_ir(atomic_exchange_kernel, "atomic_exchange_kernel")

    assert atomic_exchange_kernel.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel i32 atomic_exchange_kernel(buffer<i32> arg0, i32 arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_exchange(arg0, v4, arg1);
  return v5;
}
"""
    verify_ir(atomic_exchange_kernel, expected)


def test_atomic_sub_builds_ir(print_ir, verify_ir):
    """Test atomic_sub actually builds IR."""
    @kernel
    def atomic_sub_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_sub(buf, idx, 1)

    print_ir(atomic_sub_kernel, "atomic_sub_kernel")

    # idx is now a DSL variable
    expected = """
kernel void atomic_sub_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_sub(arg0, v4, 1);
}
"""
    verify_ir(atomic_sub_kernel, expected)


def test_atomic_bitwise_builds_ir(print_ir, verify_ir):
    """Test atomic bitwise operations build IR."""
    @kernel
    def atomic_bitwise_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_and(buf, idx, 255)
        atomic_or(buf, idx, 1)
        atomic_xor(buf, idx, 2)

    print_ir(atomic_bitwise_kernel, "atomic_bitwise_kernel")

    # idx is now a DSL variable
    expected = """
kernel void atomic_bitwise_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_and(arg0, v4, 255);
  u32 v6 = load(vidx);
  i32 v7 = atomic_or(arg0, v6, 1);
  u32 v8 = load(vidx);
  i32 v9 = atomic_xor(arg0, v8, 2);
}
"""
    verify_ir(atomic_bitwise_kernel, expected)


def test_atomic_min_max_builds_ir(print_ir, verify_ir):
    """Test atomic min/max actually build IR."""
    @kernel
    def atomic_minmax_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_min(buf, idx, 100)
        atomic_max(buf, idx, 0)

    print_ir(atomic_minmax_kernel, "atomic_minmax_kernel")

    # idx is now a DSL variable
    expected = """
kernel void atomic_minmax_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_min(arg0, v4, 100);
  u32 v6 = load(vidx);
  i32 v7 = atomic_max(arg0, v6, 0);
}
"""
    verify_ir(atomic_minmax_kernel, expected)


def test_multiple_atomics_in_kernel(print_ir, verify_ir):
    """Test multiple atomic operations in one kernel."""
    @kernel
    def multi_atomic_kernel(counter: Buffer[Int], sum_buf: Buffer[Int]):
        idx = dispatch_id().x
        # Increment counter
        old_val = atomic_add(counter, idx, 1)
        # Add to sum
        atomic_add(sum_buf, idx, old_val)

    print_ir(multi_atomic_kernel, "multi_atomic_kernel")

    # idx and old_val are now DSL variables
    expected = """
kernel void multi_atomic_kernel(buffer<i32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = atomic_add(arg0, v4, 1);
  i32 vold_val = alloca();
  store(vold_val, v5);
  u32 v8 = load(vidx);
  i32 v9 = load(vold_val);
  i32 v10 = atomic_add(arg1, v8, v9);
}
"""
    verify_ir(multi_atomic_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
