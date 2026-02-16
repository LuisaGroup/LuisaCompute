"""Tests for warp operations - with IR building and pretty printing."""

from luisa import (
    kernel, callable,
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

    # first, lane, bits are now DSL variables
    expected = """
i32 warp_queries() {
  i1 v0 = warp_is_first_active_lane();
  i1 vfirst = alloca();
  store(vfirst, v0);
  u32 v3 = warp_first_active_lane();
  u32 vlane = alloca();
  store(vlane, v3);
  u32 v6 = warp_active_count_bits(True);
  u32 vbits = alloca();
  store(vbits, v6);
  u32 v9 = load(vlane);
  i32 v10 = cast(v9);
  return v10;
}
"""
    verify_ir(warp_queries, expected)


def test_warp_reduction_builds_ir(verify_ir):
    """Test warp reduction functions build IR."""
    @callable
    def warp_reductions(x: Float) -> Float:
        s = warp_sum(x)
        p = warp_product(x)
        mn = warp_min(x)
        mx = warp_max(x)
        return s

    # s, p, mn, mx are now DSL variables
    expected = """
f32 warp_reductions(f32 arg0) {
  f32 v0 = warp_sum(arg0);
  f32 vs = alloca();
  store(vs, v0);
  f32 v3 = warp_product(arg0);
  f32 vp = alloca();
  store(vp, v3);
  f32 v6 = warp_min(arg0);
  f32 vmn = alloca();
  store(vmn, v6);
  f32 v9 = warp_max(arg0);
  f32 vmx = alloca();
  store(vmx, v9);
  f32 v12 = load(vs);
  return v12;
}
"""
    verify_ir(warp_reductions, expected)


def test_warp_boolean_reduction_builds_ir(verify_ir):
    """Test warp boolean reduction builds IR."""
    @callable
    def warp_bool_checks(x: Float) -> Int:
        all_val = warp_all(x > 0.0)
        any_val = warp_any(x > 0.0)
        eq_val = warp_all_equal(x)
        return Int(all_val)

    # all_val, any_val, eq_val are now DSL variables
    expected = """
i32 warp_bool_checks(f32 arg0) {
  i1 v0 = gt(arg0, 0.0);
  i1 v1 = warp_all(v0);
  i1 vall_val = alloca();
  store(vall_val, v1);
  i1 v4 = gt(arg0, 0.0);
  i1 v5 = warp_any(v4);
  i1 vany_val = alloca();
  store(vany_val, v5);
  i1 v8 = warp_active_all_equal(arg0);
  i1 veq_val = alloca();
  store(veq_val, v8);
  i1 v11 = load(vall_val);
  i32 v12 = cast(v11);
  return v12;
}
"""
    verify_ir(warp_bool_checks, expected)


def test_warp_prefix_builds_ir(verify_ir):
    """Test warp prefix functions build IR."""
    @callable
    def warp_prefix_ops(x: Float, b: Int) -> Float:
        ps = warp_prefix_sum(x)
        pp = warp_prefix_product(x)
        pc = warp_prefix_count_bits(True)
        return ps

    # ps, pp, pc are now DSL variables
    expected = """
f32 warp_prefix_ops(f32 arg0, i32 arg1) {
  f32 v0 = warp_prefix_sum(arg0);
  f32 vps = alloca();
  store(vps, v0);
  f32 v3 = warp_prefix_product(arg0);
  f32 vpp = alloca();
  store(vpp, v3);
  u32 v6 = warp_prefix_count_bits(True);
  u32 vpc = alloca();
  store(vpc, v6);
  f32 v9 = load(vps);
  return v9;
}
"""
    verify_ir(warp_prefix_ops, expected)


def test_warp_broadcast_builds_ir(verify_ir):
    """Test warp broadcast functions build IR."""
    @callable
    def warp_broadcast_ops(x: Float) -> Float:
        from_lane = warp_read_lane(x, Int(0))
        first = warp_read_first_lane(x)
        return first

    # from_lane, first are now DSL variables
    expected = """
f32 warp_broadcast_ops(f32 arg0) {
  f32 v0 = warp_read_lane(arg0, 0);
  f32 vfrom_lane = alloca();
  store(vfrom_lane, v0);
  f32 v3 = warp_read_first_active_lane(arg0);
  f32 vfirst = alloca();
  store(vfirst, v3);
  f32 v6 = load(vfirst);
  return v6;
}
"""
    verify_ir(warp_broadcast_ops, expected)


def test_warp_bitwise_builds_ir(verify_ir):
    """Test warp bitwise functions build IR."""
    @callable
    def warp_bitwise_ops(x: Int) -> Int:
        a = warp_bit_and(x)
        o = warp_bit_or(x)
        x_val = warp_bit_xor(x)
        m = warp_bit_mask(True)
        return a

    # a, o, x_val, m are now DSL variables
    expected = """
i32 warp_bitwise_ops(i32 arg0) {
  i32 v0 = warp_active_bit_and(arg0);
  i32 va = alloca();
  store(va, v0);
  i32 v3 = warp_active_bit_or(arg0);
  i32 vo = alloca();
  store(vo, v3);
  i32 v6 = warp_active_bit_xor(arg0);
  i32 vx_val = alloca();
  store(vx_val, v6);
  <4 x u32> v9 = warp_active_bit_mask(True);
  <4 x u32> vm = alloca();
  store(vm, v9);
  i32 v12 = load(va);
  return v12;
}
"""
    verify_ir(warp_bitwise_ops, expected)


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

    assert warp_kernel.ir.is_kernel
    
    # idx, val, sum_val are now DSL variables
    expected = """
kernel void warp_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  f32 v5 = buffer_read(arg0, v4);
  f32 val = alloca();
  store(val, v5);
  f32 v8 = load(val);
  f32 v9 = warp_sum(v8);
  f32 vsum_val = alloca();
  store(vsum_val, v9);
  i1 v12 = warp_is_first_active_lane();
  if (v12) { 
    u32 v14 = load(vidx);
    f32 v15 = load(vsum_val);
    buffer_write(arg0, v14, v15);
  } else {
    (empty)
  }
}
"""
    verify_ir(warp_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
