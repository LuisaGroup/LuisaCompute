"""Tests for type casting - with IR building and pretty printing."""

from luisa import kernel, callable, Float, Int, Buffer, dispatch_id


def test_int_to_float_cast(verify_ir):
    """Test casting int to float - builds and prints IR."""
    @callable
    def cast_int_to_float(x: Int) -> Float:
        return Float(x)

    expected = """
f32 cast_int_to_float(i32 arg0) {
  f32 v0 = cast(arg0);
  return v0;
}
"""
    verify_ir(cast_int_to_float, expected)


def test_float_to_int_cast(verify_ir):
    """Test casting float to int."""
    @callable
    def cast_float_to_int(x: Float) -> Int:
        return Int(x)

    expected = """
i32 cast_float_to_int(f32 arg0) {
  i32 v0 = cast(arg0);
  return v0;
}
"""
    verify_ir(cast_float_to_int, expected)


def test_cast_in_computation(verify_ir):
    """Test cast in the middle of computation."""
    @callable
    def mixed_computation(i: Int, f: Float) -> Float:
        return Float(i) + f

    expected = """
f32 mixed_computation(i32 arg0, f32 arg1) {
  f32 v0 = cast(arg0);
  f32 v1 = add(v0, arg1);
  return v1;
}
"""
    verify_ir(mixed_computation, expected)


def test_cast_with_buffer(verify_ir):
    """Test cast with buffer operations."""
    @callable
    def store_index_as_float(buf: Buffer[Float], idx: Int) -> None:
        buf[idx] = Float(idx) * 2.0

    expected = """
void store_index_as_float(buffer<f32> arg0, i32 arg1) {
  f32 v0 = cast(arg1);
  f32 v1 = mul(v0, 2.0);
  buffer_write(arg0, arg1, v1);
}
"""
    verify_ir(store_index_as_float, expected)


def test_chained_casts(verify_ir):
    """Test multiple chained casts."""
    @callable
    def chain_cast(x: Int) -> Int:
        f = Float(x)
        i = Int(f)
        return i

    expected = """
i32 chain_cast(i32 arg0) {
  f32 v0 = cast(arg0);
  f32 vf = alloca();
  store(vf, v0);
  f32 v3 = load(vf);
  i32 v4 = cast(v3);
  i32 vi = alloca();
  store(vi, v4);
  i32 v7 = load(vi);
  return v7;
}
"""
    verify_ir(chain_cast, expected)


def test_cast_in_kernel(verify_ir):
    """Test cast in a kernel context."""
    @kernel
    def cast_kernel(out: Buffer[Float]):
        idx = dispatch_id().x
        out[idx] = Float(idx) * 1.5

    assert cast_kernel.ir.is_kernel
    
    expected = """
kernel void cast_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = cast(v1);
  f32 v3 = mul(v2, 1.5);
  buffer_write(arg0, v1, v3);
}
"""
    verify_ir(cast_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
