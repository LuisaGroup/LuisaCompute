"""Tests for JIT compilation and staged functions."""

from luisa import kernel, callable, Float, Int, Buffer


def test_staged_function_basic(verify_ir):
    """Test building a basic staged function."""
    @callable
    def add(a: Float, b: Float) -> Float:
        return a + b

    expected = """
f32 add(f32 arg0, f32 arg1) {
  f32 v0 = add(arg0, arg1);
  return v0;
}
"""
    verify_ir(add, expected)


def test_staged_function_with_kernel(verify_ir):
    """Test staged function marked as kernel."""
    @kernel
    def simple_kernel(x: Int) -> None:
        pass

    assert simple_kernel.ir.is_kernel
    
    expected = """
kernel void simple_kernel(i32 arg0) {
  (empty)
}
"""
    verify_ir(simple_kernel, expected)


def test_staged_function_control_flow(verify_ir):
    """Test staged function with control flow."""
    @callable
    def abs_value(x: Float) -> Float:
        if x > 0.0:
            return x
        else:
            return -x

    expected = """
f32 abs_value(f32 arg0) {
  i1 v0 = gt(arg0, 0.0);
  if (v0) {
    return arg0;
  } else {
    f32 v3 = neg(arg0);
    return v3;
  }
}
"""
    verify_ir(abs_value, expected)


def test_staged_function_captured_vars(verify_ir):
    """Test staged function with captured variables."""
    threshold = 0.5

    @callable
    def threshold_check(x: Float) -> Int:
        if x > threshold:
            return 1
        else:
            return 0

    expected = """
i32 threshold_check(f32 arg0) {
  i1 v0 = gt(arg0, 0.5);
  if (v0) {
    return 1;
  } else {
    return 0;
  }
}
"""
    verify_ir(threshold_check, expected)


def test_staged_function_while_loop(verify_ir):
    """Test staged function with while loop."""
    @callable
    def count_up(n: Int) -> Int:
        i = Int(0)
        while i < n:
            i = i + 1
        return i

    expected = """
i32 count_up(i32 arg0) {
  i32 vi = alloca();
  store(vi, 0);
  i32 v2 = load(vi);
  i1 v3 = lt(v2, arg0);
  while (true) { 
    i1 v5 = logical_not(v3);
    if (v5) { 
      break;
    } else {
      (empty)
    }
    i32 v8 = load(vi);
    i32 v9 = add(v8, 1);
    store(vi, v9);
  }
  i32 v11 = load(vi);
  return v11;
}
"""
    verify_ir(count_up, expected)


def test_staged_function_for_range(verify_ir):
    """Test staged function with for-range loop."""
    @callable
    def sum_range(n: Int, start_val: Int) -> Int:
        total = start_val
        for i in range(n):
            total = total + i
        return total

    expected = """
i32 sum_range(i32 arg0, i32 arg1) {
  i32 vtotal = alloca();
  store(vtotal, arg1);
  i32 vi = alloca();
  store(vi, 0);
  while (true) {
    i32 v5 = load(vi);
    i1 v6 = lt(v5, arg0);
    i1 v7 = logical_not(v6);
    if (v7) {
      break;
    } else {
      (empty)
    }
    i32 v10 = load(vtotal);
    i32 v11 = add(v10, v5);
    store(vtotal, v11);
    i32 v13 = add(v5, 1);
    store(vi, v13);
  }
  i32 v15 = load(vtotal);
  return v15;
}
"""
    verify_ir(sum_range, expected)


def test_staged_function_complex(verify_ir):
    """Test staged function with complex logic."""
    @callable
    def compute(x: Float, y: Float) -> Float:
        x2 = x * x
        y2 = y * y
        sum_sq = x2 + y2
        
        if sum_sq > 0.0:
            return sum_sq
        else:
            return 0.0

    expected = """
f32 compute(f32 arg0, f32 arg1) {
  f32 v0 = mul(arg0, arg0);
  f32 vx2 = alloca();
  store(vx2, v0);
  f32 v3 = mul(arg1, arg1);
  f32 vy2 = alloca();
  store(vy2, v3);
  f32 v6 = load(vx2);
  f32 v7 = load(vy2);
  f32 v8 = add(v6, v7);
  f32 vsum_sq = alloca();
  store(vsum_sq, v8);
  f32 v11 = load(vsum_sq);
  i1 v12 = gt(v11, 0.0);
  if (v12) {
    f32 v14 = load(vsum_sq);
    return v14;
  } else {
    return 0.0;
  }
}
"""
    verify_ir(compute, expected)


def test_kernel_with_dispatch_id(verify_ir):
    """Test kernel using dispatch_id register."""
    from luisa import dispatch_id
    
    @kernel
    def index_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = Float(idx)

    # idx is now a DSL variable
    expected = """
kernel void index_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  u32 v5 = load(vidx);
  f32 v6 = cast(v5);
  buffer_write(arg0, v4, v6);
}
"""
    verify_ir(index_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
