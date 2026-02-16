"""Tests for warp operations - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable, pprint,
    Int, Float, Buffer,
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


def test_warp_query_functions_build_ir(verify_ir):
    """Test warp query functions actually build IR."""
    @callable
    def warp_queries() -> Int:
        first = warp_is_first_active_lane()
        lane = warp_first_active_lane()
        bits = warp_active_count_bits(True)
        return Int(lane)

    ir = warp_queries()
    
    expected = """
i32 warp_queries() {
  i1 v0 = warp_is_first_active_lane();
  u32 v1 = warp_first_active_lane();
  u32 v2 = warp_active_count_bits(True);
  i32 v3 = cast(v1);
  return v3;
}
"""
    verify_ir(ir, expected)


def test_warp_reduction_builds_ir(verify_ir):
    """Test warp reduction functions build IR."""
    @callable
    def warp_reductions(x: Float) -> Float:
        s = warp_sum(x)
        p = warp_product(x)
        mn = warp_min(x)
        mx = warp_max(x)
        return s

    ir = warp_reductions(1.0)
    
    expected = """
f32 warp_reductions(f32 arg0) {
  f32 v0 = warp_sum(arg0);
  f32 v1 = warp_product(arg0);
  f32 v2 = warp_min(arg0);
  f32 v3 = warp_max(arg0);
  return v0;
}
"""
    verify_ir(ir, expected)


def test_warp_boolean_reduction_builds_ir(verify_ir):
    """Test warp boolean reduction builds IR."""
    @callable
    def warp_bool_checks(x: Float) -> Int:
        all_val = warp_all(x > 0.0)
        any_val = warp_any(x > 0.0)
        eq_val = warp_all_equal(x)
        return Int(all_val)

    ir = warp_bool_checks(1.0)
    
    expected = """
i32 warp_bool_checks(f32 arg0) {
  i1 v0 = gt(arg0, 0.0);
  i1 v1 = warp_all(v0);
  i1 v2 = gt(arg0, 0.0);
  i1 v3 = warp_any(v2);
  i1 v4 = warp_active_all_equal(arg0);
  i32 v5 = cast(v1);
  return v5;
}
"""
    verify_ir(ir, expected)


def test_warp_prefix_builds_ir(verify_ir):
    """Test warp prefix functions build IR."""
    @callable
    def warp_prefix_ops(x: Float, b: Int) -> Float:
        ps = warp_prefix_sum(x)
        pp = warp_prefix_product(x)
        pc = warp_prefix_count_bits(True)
        return ps

    ir = warp_prefix_ops(1.0, 1)
    
    expected = """
f32 warp_prefix_ops(f32 arg0, i32 arg1) {
  f32 v0 = warp_prefix_sum(arg0);
  f32 v1 = warp_prefix_product(arg0);
  u32 v2 = warp_prefix_count_bits(True);
  return v0;
}
"""
    verify_ir(ir, expected)


def test_warp_broadcast_builds_ir(verify_ir):
    """Test warp broadcast functions build IR."""
    @callable
    def warp_broadcast_ops(x: Float) -> Float:
        from_lane = warp_read_lane(x, Int(0))
        first = warp_read_first_lane(x)
        return first

    ir = warp_broadcast_ops(1.0)
    
    expected = """
f32 warp_broadcast_ops(f32 arg0) {
  f32 v0 = warp_read_lane(arg0, 0);
  f32 v1 = warp_read_first_active_lane(arg0);
  return v1;
}
"""
    verify_ir(ir, expected)


def test_warp_bitwise_builds_ir(verify_ir):
    """Test warp bitwise functions build IR."""
    @callable
    def warp_bitwise_ops(x: Int) -> Int:
        a = warp_bit_and(x)
        o = warp_bit_or(x)
        x_val = warp_bit_xor(x)
        m = warp_bit_mask(True)
        return a

    ir = warp_bitwise_ops(255)
    
    expected = """
i32 warp_bitwise_ops(i32 arg0) {
  i32 v0 = warp_active_bit_and(arg0);
  i32 v1 = warp_active_bit_or(arg0);
  i32 v2 = warp_active_bit_xor(arg0);
  <4 x u32> v3 = warp_active_bit_mask(True);
  return v0;
}
"""
    verify_ir(ir, expected)


def test_warp_in_kernel(verify_ir):
    """Test warp operations in a kernel."""
    @kernel
    def warp_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        val = buf[idx]
        # Warp reduction
        sum_val = warp_sum(val)
        # Only first lane writes
        if warp_is_first_active_lane():
            buf[idx] = sum_val

    ir = warp_kernel(None)
    assert ir.is_kernel
    
    expected = """
kernel void warp_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = buffer_read(arg0, v1);
  f32 val = alloca();
  store(val, v2);
  f32 v5 = load(val);
  f32 v6 = warp_sum(v5);
  i1 v7 = warp_is_first_active_lane();
  if (v7) { 
    buffer_write(arg0, v1, v6);
  } else {
    (empty)
  }
}
"""
    verify_ir(ir, expected)
