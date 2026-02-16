"""Tests for buffer operations - with IR building and pretty printing."""

import pytest
from luisa import kernel, callable, Float, Int, Buffer, dispatch_id


def test_buffer_write(verify_ir):
    """Test buffer write operation - builds and prints IR."""
    @callable
    def write_to_buffer(buf: Buffer[Float]) -> None:
        buf[0] = 1.0

    ir = write_to_buffer(0)
    
    expected = """
void write_to_buffer(buffer<f32> arg0) {
  buffer_write(arg0, 0, 1.0);
}
"""
    verify_ir(ir, expected)


def test_buffer_read(verify_ir):
    """Test buffer read operation - builds and prints IR."""
    @callable
    def read_from_buffer(buf: Buffer[Float]) -> Float:
        return buf[0]

    ir = read_from_buffer(0)
    
    expected = """
f32 read_from_buffer(buffer<f32> arg0) {
  f32 v0 = buffer_read(arg0, 0);
  return v0;
}
"""
    verify_ir(ir, expected)


def test_buffer_read_write(verify_ir):
    """Test buffer read and write in same function."""
    @callable
    def copy_buffer(src: Buffer[Float], dst: Buffer[Float]) -> None:
        dst[0] = src[0]

    ir = copy_buffer(0, 0)
    
    expected = """
void copy_buffer(buffer<f32> arg0, buffer<f32> arg1) {
  f32 v0 = buffer_read(arg0, 0);
  buffer_write(arg1, 0, v0);
}
"""
    verify_ir(ir, expected)


def test_saxpy_kernel(verify_ir):
    """Test SAXPY kernel pattern - Single-precision A*X Plus Y."""
    @kernel
    def saxpy(result: Buffer[Float], a: Float, x: Buffer[Float], y: Buffer[Float]) -> None:
        idx = dispatch_id().x
        result[idx] = a * x[idx] + y[idx]

    ir = saxpy(0, 2.0, 0, 0)
    assert ir.is_kernel
    
    expected = """
kernel void saxpy(buffer<f32> arg0, f32 arg1, buffer<f32> arg2, buffer<f32> arg3) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = buffer_read(arg2, v1);
  f32 v3 = mul(arg1, v2);
  f32 v4 = buffer_read(arg3, v1);
  f32 v5 = add(v3, v4);
  buffer_write(arg0, v1, v5);
}
"""
    verify_ir(ir, expected)


def test_buffer_with_dynamic_index(verify_ir):
    """Test buffer access with dynamic index."""
    @callable
    def dynamic_access(buf: Buffer[Float], idx: Int) -> Float:
        return buf[idx]

    ir = dynamic_access(0, 5)
    
    expected = """
f32 dynamic_access(buffer<f32> arg0, i32 arg1) {
  f32 v0 = buffer_read(arg0, arg1);
  return v0;
}
"""
    verify_ir(ir, expected)


def test_buffer_multiple_writes(verify_ir):
    """Test multiple buffer writes."""
    @callable
    def fill_buffer(buf: Buffer[Float]) -> None:
        buf[0] = 0.0
        buf[1] = 1.0
        buf[2] = 2.0

    ir = fill_buffer(0)
    
    expected = """
void fill_buffer(buffer<f32> arg0) {
  buffer_write(arg0, 0, 0.0);
  buffer_write(arg0, 1, 1.0);
  buffer_write(arg0, 2, 2.0);
}
"""
    verify_ir(ir, expected)


def test_buffer_2d_kernel(verify_ir):
    """Test 2D buffer access pattern."""
    @kernel
    def matrix_transpose(out: Buffer[Float], inp: Buffer[Float], width: Int, height: Int):
        x = dispatch_id().x
        y = dispatch_id().y
        if x < width and y < height:
            out[y * width + x] = inp[x * height + y]

    ir = matrix_transpose(None, None, 64, 64)
    assert ir.is_kernel
    
    expected = """
kernel void matrix_transpose(buffer<f32> arg0, buffer<f32> arg1, i32 arg2, i32 arg3) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  <3 x u32> v2 = dispatch_id();
  u32 v3 = swizzle(v2, 'y');
  i1 v4 = lt(v1, arg2);
  i1 v5 = alloca();
  store(v5, v4);
  if (v4) { 
    i1 v8 = lt(v3, arg3);
    store(v5, v8);
  } else {
    (empty)
  }
  i1 v10 = load(v5);
  if (v10) { 
    u32 v12 = mul(v3, arg2);
    u32 v13 = add(v12, v1);
    u32 v14 = mul(v1, arg3);
    u32 v15 = add(v14, v3);
    f32 v16 = buffer_read(arg1, v15);
    buffer_write(arg0, v13, v16);
  } else {
    (empty)
  }
}
"""
    verify_ir(ir, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
