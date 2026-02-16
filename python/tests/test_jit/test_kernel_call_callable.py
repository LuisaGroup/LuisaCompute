"""Tests for kernel calling callable functions."""

from luisa import kernel, callable, Float, Int, Buffer


def test_kernel_calls_simple_callable(verify_ir):
    """Test kernel calling a simple callable function."""
    @callable
    def square(x: Float) -> Float:
        return x * x

    @kernel
    def compute_squares(buf: Buffer[Float]):
        idx = Int(0)
        val = buf[idx]
        result = square(val)
        buf[idx] = result

    assert compute_squares.ir.is_kernel
    
    expected = """
kernel void compute_squares(buffer<f32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  f32 v3 = buffer_read(arg0, v2);
  f32 val = alloca();
  store(val, v3);
  f32 v6 = load(val);
  f32 v7 = call(@square, v6);
  i32 v8 = load(vidx);
  buffer_write(arg0, v8, v7);
}

f32 square(f32 arg0) {
  f32 v0 = mul(arg0, arg0);
  return v0;
}
"""
    verify_ir(compute_squares, expected)


def test_kernel_calls_math_callable(verify_ir):
    """Test kernel calling a callable with math operations."""
    @callable
    def normalize_value(x: Float) -> Float:
        if x < 0.0:
            return 0.0
        elif x > 1.0:
            return 1.0
        return x

    @kernel
    def process_buffer(buf: Buffer[Float]):
        idx = Int(0)
        val = buf[idx]
        normalized = normalize_value(val)
        buf[idx] = normalized

    assert process_buffer.ir.is_kernel
    
    expected = """
kernel void process_buffer(buffer<f32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  f32 v3 = buffer_read(arg0, v2);
  f32 val = alloca();
  store(val, v3);
  f32 v6 = load(val);
  f32 v7 = call(@normalize_value, v6);
  i32 v8 = load(vidx);
  buffer_write(arg0, v8, v7);
}

f32 normalize_value(f32 arg0) {
  i1 v0 = lt(arg0, 0.0);
  if (v0) { 
    return 0.0;
  } else {
    i1 v3 = gt(arg0, 1.0);
    if (v3) { 
      return 1.0;
    } else {
      (empty)
    }
  }
  return arg0;
}
"""
    verify_ir(process_buffer, expected)


def test_kernel_calls_callable_with_multiple_args(verify_ir):
    """Test kernel calling callable with multiple arguments."""
    @callable
    def lerp_func(a: Float, b: Float, t: Float) -> Float:
        return a + (b - a) * t

    @kernel
    def interpolate(buf: Buffer[Float]):
        idx = Int(0)
        result = lerp_func(0.0, 1.0, buf[idx])
        buf[idx] = result

    expected = """
kernel void interpolate(buffer<f32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  f32 v3 = buffer_read(arg0, v2);
  f32 v4 = call(@lerp_func, 0.0, 1.0, v3);
  i32 v5 = load(vidx);
  buffer_write(arg0, v5, v4);
}

f32 lerp_func(f32 arg0, f32 arg1, f32 arg2) {
  f32 v0 = sub(arg1, arg0);
  f32 v1 = mul(v0, arg2);
  f32 v2 = add(arg0, v1);
  return v2;
}
"""
    verify_ir(interpolate, expected)


def test_kernel_calls_nested_callable(verify_ir):
    """Test kernel calling a callable that calls another callable."""
    @callable
    def square(x: Float) -> Float:
        return x * x

    @callable
    def sum_of_squares(a: Float, b: Float) -> Float:
        return square(a) + square(b)

    @kernel
    def compute(buf: Buffer[Float]):
        idx = Int(0)
        result = sum_of_squares(buf[idx], buf[idx + 1])
        buf[idx] = result

    expected = """
kernel void compute(buffer<f32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  f32 v3 = buffer_read(arg0, v2);
  i32 v4 = load(vidx);
  i32 v5 = add(v4, 1);
  f32 v6 = buffer_read(arg0, v5);
  f32 v7 = call(@sum_of_squares, v3, v6);
  i32 v8 = load(vidx);
  buffer_write(arg0, v8, v7);
}

f32 sum_of_squares(f32 arg0, f32 arg1) {
  f32 v0 = call(@square, arg0);
  f32 v1 = call(@square, arg1);
  f32 v2 = add(v0, v1);
  return v2;
}

f32 square(f32 arg0) {
  f32 v0 = mul(arg0, arg0);
  return v0;
}
"""
    verify_ir(compute, expected)


def test_kernel_calls_callable_with_loop(verify_ir):
    """Test kernel calling a callable that contains a loop."""
    @callable
    def factorial(n: Int) -> Int:
        result = Int(1)
        i = Int(1)
        while i <= n:
            result = result * i
            i = i + 1
        return result

    @kernel
    def compute_factorials(buf: Buffer[Int]):
        idx = Int(0)
        n = buf[idx]
        result = factorial(n)
        buf[idx] = result

    expected = """
kernel void compute_factorials(buffer<i32> arg0) {
  i32 vidx = alloca();
  store(vidx, 0);
  i32 v2 = load(vidx);
  i32 v3 = buffer_read(arg0, v2);
  i32 vn = alloca();
  store(vn, v3);
  i32 v6 = load(vn);
  i32 v7 = call(@factorial, v6);
  i32 v8 = load(vidx);
  buffer_write(arg0, v8, v7);
}

i32 factorial(i32 arg0) {
  i32 vresult = alloca();
  store(vresult, 1);
  i32 vi = alloca();
  store(vi, 1);
  i32 v4 = load(vi);
  i1 v5 = le(v4, arg0);
  while (true) { 
    i1 v7 = logical_not(v5);
    if (v7) { 
      break;
    } else {
      (empty)
    }
    i32 v10 = load(vresult);
    i32 v11 = load(vi);
    i32 v12 = mul(v10, v11);
    store(vresult, v12);
    i32 v14 = load(vi);
    i32 v15 = add(v14, 1);
    store(vi, v15);
  }
  i32 v17 = load(vresult);
  return v17;
}
"""
    verify_ir(compute_factorials, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
