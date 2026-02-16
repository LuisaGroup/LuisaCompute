"""Tests for buffer operations - with IR building and pretty printing."""

from luisa import kernel, callable, Float, Int, Buffer, dispatch_id


def test_buffer_write(print_ir, verify_ir):
    """Test buffer write operation - builds and prints IR."""
    @callable
    def write_to_buffer(buf: Buffer[Float]) -> None:
        buf[0] = 1.0

    print_ir(write_to_buffer, "write_to_buffer")

    expected = """
void write_to_buffer(buffer<f32> arg0) {
  buffer_write(arg0, 0, 1.0);
}
"""
    verify_ir(write_to_buffer, expected)


def test_buffer_read(print_ir, verify_ir):
    """Test buffer read operation - builds and prints IR."""
    @callable
    def read_from_buffer(buf: Buffer[Float]) -> Float:
        return buf[0]

    print_ir(read_from_buffer, "read_from_buffer")

    expected = """
f32 read_from_buffer(buffer<f32> arg0) {
  f32 v0 = buffer_read(arg0, 0);
  return v0;
}
"""
    verify_ir(read_from_buffer, expected)


def test_buffer_read_write(print_ir, verify_ir):
    """Test buffer read and write in same function."""
    @callable
    def copy_buffer(src: Buffer[Float], dst: Buffer[Float]) -> None:
        dst[0] = src[0]

    print_ir(copy_buffer, "copy_buffer")

    expected = """
void copy_buffer(buffer<f32> arg0, buffer<f32> arg1) {
  f32 v0 = buffer_read(arg0, 0);
  buffer_write(arg1, 0, v0);
}
"""
    verify_ir(copy_buffer, expected)


def test_saxpy_kernel(print_ir, verify_ir):
    """Test SAXPY kernel pattern - Single-precision A*X Plus Y."""
    @kernel
    def saxpy(result: Buffer[Float], a: Float, x: Buffer[Float], y: Buffer[Float]) -> None:
        idx = dispatch_id().x
        result[idx] = a * x[idx] + y[idx]

    print_ir(saxpy, "saxpy")

    assert saxpy.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel void saxpy(buffer<f32> arg0, f32 arg1, buffer<f32> arg2, buffer<f32> arg3) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  u32 v5 = load(vidx);
  f32 v6 = buffer_read(arg2, v5);
  f32 v7 = mul(arg1, v6);
  u32 v8 = load(vidx);
  f32 v9 = buffer_read(arg3, v8);
  f32 v10 = add(v7, v9);
  buffer_write(arg0, v4, v10);
}
"""
    verify_ir(saxpy, expected)


def test_buffer_with_dynamic_index(print_ir, verify_ir):
    """Test buffer access with dynamic index."""
    @callable
    def dynamic_access(buf: Buffer[Float], idx: Int) -> Float:
        return buf[idx]

    print_ir(dynamic_access, "dynamic_access")

    expected = """
f32 dynamic_access(buffer<f32> arg0, i32 arg1) {
  f32 v0 = buffer_read(arg0, arg1);
  return v0;
}
"""
    verify_ir(dynamic_access, expected)


def test_buffer_multiple_writes(print_ir, verify_ir):
    """Test multiple buffer writes."""
    @callable
    def fill_buffer(buf: Buffer[Float]) -> None:
        buf[0] = 0.0
        buf[1] = 1.0
        buf[2] = 2.0

    print_ir(fill_buffer, "fill_buffer")

    expected = """
void fill_buffer(buffer<f32> arg0) {
  buffer_write(arg0, 0, 0.0);
  buffer_write(arg0, 1, 1.0);
  buffer_write(arg0, 2, 2.0);
}
"""
    verify_ir(fill_buffer, expected)


def test_buffer_2d_kernel(print_ir, verify_ir):
    """Test 2D buffer access pattern."""
    @kernel
    def matrix_transpose(out: Buffer[Float], inp: Buffer[Float], width: Int, height: Int):
        x = dispatch_id().x
        y = dispatch_id().y
        if x < width and y < height:
            out[y * width + x] = inp[x * height + y]

    print_ir(matrix_transpose, "matrix_transpose")

    assert matrix_transpose.ir.is_kernel
    
    # x and y are now DSL variables
    expected = """
kernel void matrix_transpose(buffer<f32> arg0, buffer<f32> arg1, i32 arg2, i32 arg3) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vx = alloca();
  store(vx, v1);
  <3 x u32> v4 = dispatch_id();
  u32 v5 = swizzle(v4, 'y');
  u32 vy = alloca();
  store(vy, v5);
  u32 v8 = load(vx);
  i1 v9 = lt(v8, arg2);
  i1 v10 = alloca();
  store(v10, v9);
  if (v9) { 
    u32 v13 = load(vy);
    i1 v14 = lt(v13, arg3);
    store(v10, v14);
  } else {
    (empty)
  }
  i1 v16 = load(v10);
  if (v16) { 
    u32 v18 = load(vy);
    u32 v19 = mul(v18, arg2);
    u32 v20 = load(vx);
    u32 v21 = add(v19, v20);
    u32 v22 = load(vx);
    u32 v23 = mul(v22, arg3);
    u32 v24 = load(vy);
    u32 v25 = add(v23, v24);
    f32 v26 = buffer_read(arg1, v25);
    buffer_write(arg0, v21, v26);
  } else {
    (empty)
  }
}
"""
    verify_ir(matrix_transpose, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
