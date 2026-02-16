"""Tests for resource types (buffers, textures, etc.)."""

from luisa import (
    kernel, callable,
    Buffer, Texture2D, Texture3D, BindlessArray, Accel,
    Int, Float, Float3, UInt,
    dispatch_id,
)


def test_buffer_type():
    """Test Buffer type creation."""
    b1 = Buffer(Float)
    assert str(b1) == "buffer<f32>"
    
    b2 = Buffer(Float3)
    assert str(b2) == "buffer<<3 x f32>>"


def test_buffer_in_kernel_builds_ir(print_ir, verify_ir):
    """Test Buffer in kernel actually builds IR."""
    @kernel
    def fill_buffer(buf: Buffer(Float), value: Float) -> None:
        idx = dispatch_id().x
        buf[idx] = value

    print_ir(fill_buffer, "fill_buffer")

    assert fill_buffer.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel void fill_buffer(buffer<f32> arg0, f32 arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  buffer_write(arg0, v4, arg1);
}
"""
    verify_ir(fill_buffer, expected)


def test_buffer_vector_type_kernel(print_ir, verify_ir):
    """Test Buffer of vectors in kernel."""
    @kernel
    def process_vectors(buf: Buffer(Float3)):
        idx = dispatch_id().x
        val = buf[idx]
        buf[idx] = val

    print_ir(process_vectors, "process_vectors")

    assert process_vectors.ir.is_kernel
    
    # idx and val are now DSL variables
    expected = """
kernel void process_vectors(buffer<<3 x f32>> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  <3 x f32> v5 = buffer_read(arg0, v4);
  <3 x f32> val = alloca();
  store(val, v5);
  u32 v8 = load(vidx);
  <3 x f32> v9 = load(val);
  buffer_write(arg0, v8, v9);
}
"""
    verify_ir(process_vectors, expected)


def test_texture2d_type():
    """Test Texture2D type creation."""
    t = Texture2D(Float)
    assert str(t) == "texture2d<f32>"


def test_texture2d_in_kernel(print_ir, verify_ir):
    """Test Texture2D in kernel builds IR."""
    @kernel
    def sample_texture(tex: Texture2D(Float), output: Buffer(Float)):
        idx = dispatch_id().x
        # Note: full texture sampling would need more support
        output[idx] = 0.0

    print_ir(sample_texture, "sample_texture")

    assert sample_texture.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel void sample_texture(texture2d<f32> arg0, buffer<f32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  buffer_write(arg1, v4, 0.0);
}
"""
    verify_ir(sample_texture, expected)


def test_texture3d_type():
    """Test Texture3D type creation."""
    t = Texture3D(Float)
    assert str(t) == "texture3d<f32>"


def test_bindless_array_type():
    """Test BindlessArray type."""
    t = BindlessArray()
    assert str(t) == "bindless_array"


def test_accel_type():
    """Test Accel type."""
    t = Accel()
    assert str(t) == "accel"


def test_multiple_resources_in_kernel(print_ir, verify_ir):
    """Test multiple resource types in one kernel."""
    @kernel
    def multi_resource_kernel(
            buf: Buffer(Float),
            tex: Texture2D(Float),
            accel: Accel
    ):
        idx = dispatch_id().x
        buf[idx] = Float(idx)

    multi_resource_kernel(
        Buffer(Float),
        Texture2D(Float),
        Accel()
    )

    print_ir(multi_resource_kernel, "multi_resource_kernel")

    assert multi_resource_kernel.ir.is_kernel
    
    # idx is now a DSL variable
    expected = """
kernel void multi_resource_kernel(buffer<f32> arg0, texture2d<f32> arg1, accel arg2) {
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
    verify_ir(multi_resource_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
