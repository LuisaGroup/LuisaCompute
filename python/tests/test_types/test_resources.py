"""Tests for resource types (Buffer, Texture, etc.) - with IR building."""

import pytest
from luisa import (
    kernel,
    Float, Float3,
    Buffer, Texture2D, Texture3D,
    BindlessArray, Accel,
    dispatch_id,
)


def test_buffer_type():
    """Test buffer resource type."""
    buf_f32 = Buffer(Float)
    assert buf_f32.element == Float

    buf_float3 = Buffer(Float3)
    assert buf_float3.element == Float3


def test_buffer_in_kernel_builds_ir(verify_ir):
    """Test Buffer in kernel actually builds IR."""
    @kernel
    def fill_buffer(buf: Buffer(Float), value: Float) -> None:
        idx = dispatch_id().x
        buf[idx] = value

    fill_buffer(Buffer(Float), 1.0)
    assert fill_buffer.ir.is_kernel
    
    expected = """
kernel void fill_buffer(buffer<f32> arg0, f32 arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  buffer_write(arg0, v1, arg1);
}
"""
    verify_ir(fill_buffer, expected)


def test_buffer_vector_type_kernel(verify_ir):
    """Test Buffer of vectors in kernel."""
    @kernel
    def process_vectors(buf: Buffer(Float3)):
        idx = dispatch_id().x
        val = buf[idx]
        buf[idx] = val

    process_vectors(Buffer(Float3))
    assert process_vectors.ir.is_kernel
    
    expected = """
kernel void process_vectors(buffer<<3 x f32>> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  <3 x f32> v2 = buffer_read(arg0, v1);
  <3 x f32> val = alloca();
  store(val, v2);
  <3 x f32> v5 = load(val);
  buffer_write(arg0, v1, v5);
}
"""
    verify_ir(process_vectors, expected)


def test_texture2d_type():
    """Test 2D texture resource type."""
    tex_f32 = Texture2D(Float)
    assert tex_f32.element == Float


def test_texture2d_in_kernel(verify_ir):
    """Test Texture2D in kernel builds IR."""
    @kernel
    def sample_texture(tex: Texture2D(Float), output: Buffer(Float)):
        idx = dispatch_id().x
        # Note: full texture sampling would need more support
        output[idx] = 0.0

    sample_texture(Texture2D(Float), Buffer(Float))
    assert sample_texture.ir.is_kernel
    
    expected = """
kernel void sample_texture(texture2d<f32> arg0, buffer<f32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  buffer_write(arg1, v1, 0.0);
}
"""
    verify_ir(sample_texture, expected)


def test_texture3d_type():
    """Test 3D texture resource type."""
    tex_f32 = Texture3D(Float)
    assert tex_f32.element == Float


def test_bindless_array_type():
    """Test bindless array resource type."""
    bindless = BindlessArray()
    assert isinstance(bindless, BindlessArray)


def test_accel_type():
    """Test acceleration structure resource type."""
    accel = Accel()
    assert isinstance(accel, Accel)


def test_multiple_resources_in_kernel(verify_ir):
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
    assert multi_resource_kernel.ir.is_kernel
    
    expected = """
kernel void multi_resource_kernel(buffer<f32> arg0, texture2d<f32> arg1, Accel arg2) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = cast(v1);
  buffer_write(arg0, v1, v2);
}
"""
    verify_ir(multi_resource_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
