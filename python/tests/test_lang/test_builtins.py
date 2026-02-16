"""Tests for builtin functions - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable,
    # Math
    sqrt, sin, cos, exp,
    floor, ceil,
    normalize,
    clamp, lerp, step, smoothstep,
    dot, cross,
    transpose,
    # Special registers
    dispatch_id, thread_id, block_id, dispatch_size,
    # Synchronization
    sync_block,
    # Type casting
    cast,
    # Print
    device_print,
    # Assertions
    assume, device_assert, unreachable,
    # Profiling
    clock,
    # Types
    Int, Float, Float3, Buffer,
)


def test_math_builtins_build_ir(verify_ir):
    """Test math builtins actually build IR."""
    @callable
    def math_ops(x: Float) -> Float:
        a = sqrt(x)
        b = sin(a)
        c = cos(b)
        d = exp(c)
        e = log(d)
        f = floor(e)
        g = ceil(f)
        return g

    ir = math_ops(1.0)
    
    expected = """
f32 math_ops(f32 arg0) {
  f32 v0 = sqrt(arg0);
  f32 v1 = sin(v0);
  f32 v2 = cos(v1);
  f32 v3 = exp(v2);
  f32 v4 = log(v3);
  f32 v5 = floor(v4);
  f32 v6 = ceil(v5);
  return v6;
}
"""
    verify_ir(ir, expected)


def test_special_registers_build_ir(verify_ir):
    """Test special registers actually build IR."""
    @kernel
    def special_reg_kernel():
        did = dispatch_id()
        tid = thread_id()
        bid = block_id()
        dsize = dispatch_size()

    ir = special_reg_kernel()
    
    expected = """
kernel void special_reg_kernel() {
  <3 x u32> v0 = dispatch_id();
  <3 x u32> v1 = thread_id();
  <3 x u32> v2 = block_id();
  <3 x u32> v3 = dispatch_size();
}
"""
    verify_ir(ir, expected)


def test_dispatch_id_in_computation(verify_ir):
    """Test dispatch_id used in actual computation."""
    @kernel
    def index_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = Float(idx)

    ir = index_kernel(None)
    
    expected = """
kernel void index_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = cast(v1);
  buffer_write(arg0, v1, v2);
}
"""
    verify_ir(ir, expected)


def test_sync_block_builds_ir(verify_ir):
    """Test sync_block actually builds IR."""
    @kernel
    def sync_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = 1.0
        sync_block()
        buf[idx] = buf[idx] + 1.0

    ir = sync_kernel(None)
    
    expected = """
kernel void sync_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  buffer_write(arg0, v1, 1.0);
  sync_block();
  f32 v4 = buffer_read(arg0, v1);
  f32 v5 = add(v4, 1.0);
  buffer_write(arg0, v1, v5);
}
"""
    verify_ir(ir, expected)


def test_cast_builds_ir(verify_ir):
    """Test cast/bitcast actually build IR."""
    @callable
    def cast_ops(x: Int) -> Float:
        f = Float(x)
        i = Int(f)
        return Float(i)

    ir = cast_ops(42)
    
    expected = """
f32 cast_ops(i32 arg0) {
  f32 v0 = cast(arg0);
  f32 vf = alloca();
  store(vf, v0);
  f32 v3 = load(vf);
  i32 v4 = cast(v3);
  i32 vi = alloca();
  store(vi, v4);
  i32 v7 = load(vi);
  f32 v8 = cast(v7);
  return v8;
}
"""
    verify_ir(ir, expected)


def test_device_print_builds_ir(verify_ir):
    """Test device_print actually builds IR."""
    @kernel
    def print_kernel(x: Int):
        device_print("Value: {}", x)

    ir = print_kernel(42)
    
    expected = """
kernel void print_kernel(i32 arg0) {
  print('Value: {}', arg0);
}
"""
    verify_ir(ir, expected)


def test_clock_builds_ir(verify_ir):
    """Test clock actually builds IR."""
    @callable
    def timed_function() -> Int:
        start = clock()
        # Some computation
        x = Int(0)
        i = Int(0)
        while i < 10:
            x = x + i
            i = i + 1
        end = clock()
        return Int(end - start)

    ir = timed_function()
    
    expected = """
i32 timed_function() {
  u64 v0 = clock();
  i32 vx = alloca();
  store(vx, 0);
  i32 vi = alloca();
  store(vi, 0);
  i32 v5 = load(vi);
  i1 v6 = lt(v5, 10);
  while (true) { 
    i1 v8 = logical_not(v6);
    if (v8) { 
      break;
    } else {
      (empty)
    }
    i32 v11 = load(vx);
    i32 v12 = load(vi);
    i32 v13 = add(v11, v12);
    store(vx, v13);
    i32 v15 = load(vi);
    i32 v16 = add(v15, 1);
    store(vi, v16);
  }
  u64 v18 = clock();
  u64 v19 = sub(v18, v0);
  i32 v20 = cast(v19);
  return v20;
}
"""
    verify_ir(ir, expected)


def test_assertions_build_ir(verify_ir):
    """Test assume/device_assert actually build IR."""
    @callable
    def checked_function(x: Int) -> Int:
        assume(x > 0, "x must be positive")
        result = x * 2
        device_assert(result > x, "result should be greater than x")
        return result

    ir = checked_function(5)
    
    expected = """
i32 checked_function(i32 arg0) {
  i1 v0 = gt(arg0, 0);
  assume(v0, 'x must be positive');
  i32 v2 = mul(arg0, 2);
  i32 vresult = alloca();
  store(vresult, v2);
  i32 v5 = load(vresult);
  i1 v6 = gt(v5, arg0);
  assert(v6, 'result should be greater than x');
  i32 v8 = load(vresult);
  return v8;
}
"""
    verify_ir(ir, expected)


def test_matrix_ops_build_ir(verify_ir):
    """Test matrix operations actually build IR."""
    from luisa import Float4x4

    @callable
    def matrix_ops(m: Float4x4) -> Float:
        t = transpose(m)
        return float(0.0)

    # Use ArgumentValue to avoid constant folding
    from luisa.transform.ir import ArgumentValue
    ir = matrix_ops(ArgumentValue(typ=Float4x4, index=0))
    
    expected = """
f32 matrix_ops([4 x <4 x f32>] arg0) {
  [4 x <4 x f32>] v0 = matrix_transpose(arg0);
  return 0.0;
}
"""
    verify_ir(ir, expected)


def test_vector_math_builds_ir(verify_ir):
    """Test vector math operations build IR."""
    @callable
    def vector_ops(a: Float3, b: Float3) -> Float3:
        d = dot(a, b)
        c = cross(a, b)
        n = normalize(a)
        return c

    # Use ArgumentValue to avoid constant folding
    from luisa.transform.ir import ArgumentValue
    ir = vector_ops(ArgumentValue(typ=Float3, index=0), ArgumentValue(typ=Float3, index=1))
    
    expected = """
<3 x f32> vector_ops(<3 x f32> arg0, <3 x f32> arg1) {
  f32 v0 = dot(arg0, arg1);
  <3 x f32> v1 = cross(arg0, arg1);
  <3 x f32> v2 = normalize(arg0);
  return v1;
}
"""
    verify_ir(ir, expected)


def test_clamp_lerp_build_ir(verify_ir):
    """Test clamp and lerp build IR."""
    @callable
    def utility_ops(x: Float) -> Float:
        c = clamp(x, 0.0, 1.0)
        l = lerp(0.0, 1.0, c)
        s = step(0.5, l)
        return smoothstep(0.0, 1.0, s)

    ir = utility_ops(0.5)
    
    expected = """
f32 utility_ops(f32 arg0) {
  f32 v0 = clamp(arg0, 0.0, 1.0);
  f32 v1 = lerp(0.0, 1.0, v0);
  f32 v2 = step(0.5, v1);
  f32 v3 = smoothstep(0.0, 1.0, v2);
  return v3;
}
"""
    verify_ir(ir, expected)


def test_unreachable_builds_ir(verify_ir):
    """Test unreachable actually builds IR."""
    @kernel
    def unreachable_kernel():
        unreachable("this should not happen")

    ir = unreachable_kernel()
    
    expected = """
kernel void unreachable_kernel() {
  unreachable('this should not happen');
}
"""
    verify_ir(ir, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
