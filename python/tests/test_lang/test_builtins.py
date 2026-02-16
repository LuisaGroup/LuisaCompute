"""Tests for builtin functions - with IR building and pretty printing."""

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


def test_math_builtins_build_ir(print_ir, verify_ir):
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

    print_ir(math_ops, "math_ops")

    # All variables are now DSL variables
    expected = """
f32 math_ops(f32 arg0) {
  f32 v0 = sqrt(arg0);
  f32 va = alloca();
  store(va, v0);
  f32 v3 = load(va);
  f32 v4 = sin(v3);
  f32 vb = alloca();
  store(vb, v4);
  f32 v7 = load(vb);
  f32 v8 = cos(v7);
  f32 vc = alloca();
  store(vc, v8);
  f32 v11 = load(vc);
  f32 v12 = exp(v11);
  f32 vd = alloca();
  store(vd, v12);
  f32 v15 = load(vd);
  f32 v16 = log(v15);
  f32 ve = alloca();
  store(ve, v16);
  f32 v19 = load(ve);
  f32 v20 = floor(v19);
  f32 vf = alloca();
  store(vf, v20);
  f32 v23 = load(vf);
  f32 v24 = ceil(v23);
  f32 vg = alloca();
  store(vg, v24);
  f32 v27 = load(vg);
  return v27;
}
"""
    verify_ir(math_ops, expected)


def test_special_registers_build_ir(print_ir, verify_ir):
    """Test special registers actually build IR."""
    @kernel
    def special_reg_kernel():
        did = dispatch_id()
        tid = thread_id()
        bid = block_id()
        dsize = dispatch_size()

    print_ir(special_reg_kernel, "special_reg_kernel")

    # All variables are now DSL variables
    expected = """
kernel void special_reg_kernel() {
  <3 x u32> v0 = dispatch_id();
  <3 x u32> vdid = alloca();
  store(vdid, v0);
  <3 x u32> v3 = thread_id();
  <3 x u32> vtid = alloca();
  store(vtid, v3);
  <3 x u32> v6 = block_id();
  <3 x u32> vbid = alloca();
  store(vbid, v6);
  <3 x u32> v9 = dispatch_size();
  <3 x u32> vdsize = alloca();
  store(vdsize, v9);
}
"""
    verify_ir(special_reg_kernel, expected)


def test_dispatch_id_in_computation(print_ir, verify_ir):
    """Test dispatch_id used in actual computation."""
    @kernel
    def index_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = Float(idx)

    print_ir(index_kernel, "index_kernel")

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


def test_sync_block_builds_ir(print_ir, verify_ir):
    """Test sync_block actually builds IR."""
    @kernel
    def sync_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = 1.0
        sync_block()
        buf[idx] = buf[idx] + 1.0

    print_ir(sync_kernel, "sync_kernel")

    # idx is now a DSL variable
    expected = """
kernel void sync_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  buffer_write(arg0, v4, 1.0);
  sync_block();
  u32 v7 = load(vidx);
  u32 v8 = load(vidx);
  f32 v9 = buffer_read(arg0, v8);
  f32 v10 = add(v9, 1.0);
  buffer_write(arg0, v7, v10);
}
"""
    verify_ir(sync_kernel, expected)


def test_cast_builds_ir(print_ir, verify_ir):
    """Test cast/bitcast actually build IR."""
    @callable
    def cast_ops(x: Int) -> Float:
        f = Float(x)
        i = Int(f)
        return Float(i)

    print_ir(cast_ops, "cast_ops")

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
    verify_ir(cast_ops, expected)


def test_device_print_builds_ir(print_ir, verify_ir):
    """Test device_print actually builds IR."""
    @kernel
    def print_kernel(x: Int):
        device_print("Value: {}", x)

    print_ir(print_kernel, "print_kernel")

    expected = """
kernel void print_kernel(i32 arg0) {
  print('Value: {}', arg0);
}
"""
    verify_ir(print_kernel, expected)


def test_clock_builds_ir(print_ir, verify_ir):
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

    print_ir(timed_function, "timed_function")

    # start, x, i, end are all now DSL variables
    expected = """
i32 timed_function() {
  u64 v0 = clock();
  u64 vstart = alloca();
  store(vstart, v0);
  i32 vx = alloca();
  store(vx, 0);
  i32 vi = alloca();
  store(vi, 0);
  i32 v7 = load(vi);
  i1 v8 = lt(v7, 10);
  while (true) { 
    i1 v10 = logical_not(v8);
    if (v10) { 
      break;
    } else {
      (empty)
    }
    i32 v13 = load(vx);
    i32 v14 = load(vi);
    i32 v15 = add(v13, v14);
    store(vx, v15);
    i32 v17 = load(vi);
    i32 v18 = add(v17, 1);
    store(vi, v18);
  }
  u64 v20 = clock();
  u64 vend = alloca();
  store(vend, v20);
  u64 v23 = load(vend);
  u64 v24 = load(vstart);
  u64 v25 = sub(v23, v24);
  i32 v26 = cast(v25);
  return v26;
}
"""
    verify_ir(timed_function, expected)


def test_assertions_build_ir(print_ir, verify_ir):
    """Test assume/device_assert actually build IR."""
    @callable
    def checked_function(x: Int) -> Int:
        assume(x > 0, "x must be positive")
        result = x * 2
        device_assert(result > x, "result should be greater than x")
        return result

    print_ir(checked_function, "checked_function")

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
    verify_ir(checked_function, expected)


def test_matrix_ops_build_ir(print_ir, verify_ir):
    """Test matrix operations actually build IR."""
    from luisa import Float4x4

    @callable
    def matrix_ops(m: Float4x4) -> Float:
        t = transpose(m)
        return float(0.0)

    print_ir(matrix_ops, "matrix_ops")

    # t is now a DSL variable
    expected = """
f32 matrix_ops([4 x <4 x f32>] arg0) {
  [4 x <4 x f32>] v0 = matrix_transpose(arg0);
  [4 x <4 x f32>] vt = alloca();
  store(vt, v0);
  return 0.0;
}
"""
    verify_ir(matrix_ops, expected)


def test_vector_math_builds_ir(print_ir, verify_ir):
    """Test vector math operations build IR."""
    @callable
    def vector_ops(a: Float3, b: Float3) -> Float3:
        d = dot(a, b)
        c = cross(a, b)
        n = normalize(a)
        return c

    print_ir(vector_ops, "vector_ops")

    # d, c, n are all now DSL variables
    expected = """
<3 x f32> vector_ops(<3 x f32> arg0, <3 x f32> arg1) {
  f32 v0 = dot(arg0, arg1);
  f32 vd = alloca();
  store(vd, v0);
  <3 x f32> v3 = cross(arg0, arg1);
  <3 x f32> vc = alloca();
  store(vc, v3);
  <3 x f32> v6 = normalize(arg0);
  <3 x f32> vn = alloca();
  store(vn, v6);
  <3 x f32> v9 = load(vc);
  return v9;
}
"""
    verify_ir(vector_ops, expected)


def test_clamp_lerp_build_ir(print_ir, verify_ir):
    """Test clamp and lerp build IR."""
    @callable
    def utility_ops(x: Float) -> Float:
        c = clamp(x, 0.0, 1.0)
        l = lerp(0.0, 1.0, c)
        s = step(0.5, l)
        return smoothstep(0.0, 1.0, s)

    print_ir(utility_ops, "utility_ops")

    # c, l, s are all now DSL variables
    expected = """
f32 utility_ops(f32 arg0) {
  f32 v0 = clamp(arg0, 0.0, 1.0);
  f32 vc = alloca();
  store(vc, v0);
  f32 v3 = load(vc);
  f32 v4 = lerp(0.0, 1.0, v3);
  f32 vl = alloca();
  store(vl, v4);
  f32 v7 = load(vl);
  f32 v8 = step(0.5, v7);
  f32 vs = alloca();
  store(vs, v8);
  f32 v11 = load(vs);
  f32 v12 = smoothstep(0.0, 1.0, v11);
  return v12;
}
"""
    verify_ir(utility_ops, expected)


def test_unreachable_builds_ir(print_ir, verify_ir):
    """Test unreachable actually builds IR."""
    @kernel
    def unreachable_kernel():
        unreachable("this should not happen")

    print_ir(unreachable_kernel, "unreachable_kernel")

    expected = """
kernel void unreachable_kernel() {
  unreachable('this should not happen');
}
"""
    verify_ir(unreachable_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
