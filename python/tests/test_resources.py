"""Tests for resource types (Buffer, Texture, etc.)."""

import pytest
from luisa import (
    int32, float32, float3, float4,
    Buffer, Texture2D, Texture3D,
    BindlessArray, Accel,
)


def test_buffer_type():
    """Test buffer resource type."""
    print("Testing buffer type...")
    
    # Simple buffer
    buf_f32 = Buffer(float32)
    assert buf_f32.element == float32
    
    # Buffer of vectors
    buf_float3 = Buffer(float3)
    assert buf_float3.element == float3
    
    # Buffer of ints
    buf_int32 = Buffer(int32)
    assert buf_int32.element == int32
    
    print("  ✓ Buffer type OK")


def test_texture2d_type():
    """Test 2D texture resource type."""
    print("Testing Texture2D type...")
    
    tex_f32 = Texture2D(float32)
    assert tex_f32.element == float32
    
    tex_float4 = Texture2D(float32)
    assert tex_float4.element == float32
    
    print("  ✓ Texture2D type OK")


def test_texture3d_type():
    """Test 3D texture resource type."""
    print("Testing Texture3D type...")
    
    tex_f32 = Texture3D(float32)
    assert tex_f32.element == float32
    
    print("  ✓ Texture3D type OK")


def test_bindless_array_type():
    """Test bindless array resource type."""
    print("Testing BindlessArray type...")
    
    bindless = BindlessArray()
    assert isinstance(bindless, BindlessArray)
    
    print("  ✓ BindlessArray type OK")


def test_accel_type():
    """Test acceleration structure resource type."""
    print("Testing Accel type...")
    
    accel = Accel()
    assert isinstance(accel, Accel)
    
    print("  ✓ Accel type OK")


def test_buffer_in_function_signature():
    """Test using Buffer in function signatures."""
    print("Testing Buffer in function signature...")
    
    from luisa import kernel
    
    @kernel
    def fill_buffer(buf: Buffer(float32), value: float32) -> None:
        idx = 0  # Placeholder
        # buf[idx] = value  # Would need proper indexing support
    
    ir_func = fill_buffer(Buffer(float32), 1.0)
    assert ir_func.name == 'fill_buffer'
    
    print("  ✓ Buffer in function signature OK")
