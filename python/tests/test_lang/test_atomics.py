"""Tests for atomic operations - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel,
    Int,
    Buffer,
    atomic_exchange, atomic_add, atomic_sub,
    atomic_and, atomic_or, atomic_xor,
    atomic_min, atomic_max,
    dispatch_id,
)


def test_atomic_add_builds_ir(verify_ir):
    """Test atomic_add actually builds IR."""
    @kernel
    def atomic_add_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_add(buf, idx, 1)

    ir = atomic_add_kernel(None)
    assert ir.is_kernel
    
    expected = """
kernel void atomic_add_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_add(arg0, v1, 1);
}
"""
    verify_ir(ir, expected)


def test_atomic_exchange_builds_ir(verify_ir):
    """Test atomic_exchange actually builds IR."""
    @kernel
    def atomic_exchange_kernel(buf: Buffer[Int], val: Int) -> Int:
        idx = dispatch_id().x
        return atomic_exchange(buf, idx, val)

    ir = atomic_exchange_kernel(None, 42)
    assert ir.is_kernel
    
    expected = """
kernel i32 atomic_exchange_kernel(buffer<i32> arg0, i32 arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_exchange(arg0, v1, arg1);
  return v2;
}
"""
    verify_ir(ir, expected)


def test_atomic_sub_builds_ir(verify_ir):
    """Test atomic_sub actually builds IR."""
    @kernel
    def atomic_sub_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_sub(buf, idx, 1)

    ir = atomic_sub_kernel(None)
    
    expected = """
kernel void atomic_sub_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_sub(arg0, v1, 1);
}
"""
    verify_ir(ir, expected)


def test_atomic_bitwise_builds_ir(verify_ir):
    """Test atomic bitwise operations build IR."""
    @kernel
    def atomic_bitwise_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_and(buf, idx, 255)
        atomic_or(buf, idx, 1)
        atomic_xor(buf, idx, 2)

    ir = atomic_bitwise_kernel(None)
    
    expected = """
kernel void atomic_bitwise_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_and(arg0, v1, 255);
  i32 v3 = atomic_or(arg0, v1, 1);
  i32 v4 = atomic_xor(arg0, v1, 2);
}
"""
    verify_ir(ir, expected)


def test_atomic_min_max_builds_ir(verify_ir):
    """Test atomic min/max actually build IR."""
    @kernel
    def atomic_minmax_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        atomic_min(buf, idx, 100)
        atomic_max(buf, idx, 0)

    ir = atomic_minmax_kernel(None)
    
    expected = """
kernel void atomic_minmax_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_min(arg0, v1, 100);
  i32 v3 = atomic_max(arg0, v1, 0);
}
"""
    verify_ir(ir, expected)


def test_multiple_atomics_in_kernel(verify_ir):
    """Test multiple atomic operations in one kernel."""
    @kernel
    def multi_atomic_kernel(counter: Buffer[Int], sum_buf: Buffer[Int]):
        idx = dispatch_id().x
        # Increment counter
        old_val = atomic_add(counter, idx, 1)
        # Add to sum
        atomic_add(sum_buf, idx, old_val)

    ir = multi_atomic_kernel(None, None)
    
    expected = """
kernel void multi_atomic_kernel(buffer<i32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = atomic_add(arg0, v1, 1);
  i32 v3 = atomic_add(arg1, v1, v2);
}
"""
    verify_ir(ir, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
