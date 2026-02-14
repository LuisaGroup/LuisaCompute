"""Tests for the type system."""

import pytest
from luisa import (
    # Scalar types
    bool_, int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, float16, float32, float64,
    # Vector types
    int2, int3, int4, uint2, uint3, uint4,
    float2, float3, float4, bool2, bool3, bool4,
    # Matrix types
    float2x2, float3x3, float4x4,
    # Resource types
    Buffer, Texture2D, Texture3D,
    # Utilities
    get_element_type, get_length,
    is_scalar_type, is_vector_type, is_integer_type, is_float_type,
    promote_types,
    Scalar, Vector, Matrix,
)


def test_scalar_types():
    """Test scalar types."""
    print("Testing scalar types...")
    
    # Check basic properties
    assert is_scalar_type(bool_)
    assert is_scalar_type(int32)
    assert is_scalar_type(float32)
    
    assert is_integer_type(int32)
    assert is_integer_type(uint32)
    assert not is_integer_type(float32)
    
    assert is_float_type(float32)
    assert is_float_type(float64)
    assert not is_float_type(int32)
    
    print("  ✓ Scalar types OK")


def test_vector_types():
    """Test vector types."""
    print("Testing vector types...")
    
    # Check vector properties
    assert is_vector_type(float3)
    assert not is_vector_type(float32)
    
    # Check element types
    assert get_element_type(float3) == float32
    assert get_element_type(int4) == int32
    
    # Check lengths
    assert get_length(float2) == 2
    assert get_length(float3) == 3
    assert get_length(float4) == 4
    
    # Create custom vector
    custom_vec = Vector(float32, 3)
    assert custom_vec.size == 3
    assert custom_vec.element == float32
    
    print("  ✓ Vector types OK")


def test_matrix_types():
    """Test matrix types."""
    print("Testing matrix types...")
    
    # Check matrix properties
    assert isinstance(float3x3, Matrix)
    assert float3x3.size == 3
    assert float3x3.element == float32
    
    assert float4x4.size == 4
    assert float2x2.size == 2
    
    print("  ✓ Matrix types OK")


def test_buffer_type():
    """Test buffer type."""
    print("Testing buffer type...")
    
    buf_type = Buffer(float32)
    assert isinstance(buf_type, Buffer)
    assert buf_type.element == float32
    
    buf_int3 = Buffer(int3)
    assert buf_int3.element == int3
    
    print("  ✓ Buffer type OK")


def test_type_promotion():
    """Test type promotion."""
    print("Testing type promotion...")
    
    # Same type
    assert promote_types(int32, int32) == int32
    
    # Scalar to vector broadcasting
    result = promote_types(float32, float3)
    assert isinstance(result, Vector)
    assert result.size == 3
    assert result.element == float32
    
    # Vector to scalar broadcasting
    result = promote_types(float3, float32)
    assert isinstance(result, Vector)
    assert result.size == 3
    
    print("  ✓ Type promotion OK")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_types.py tests")
    print("="*70)
    
    test_scalar_types()
    test_vector_types()
    test_matrix_types()
    test_buffer_type()
    test_type_promotion()
    
    print("\n" + "="*70)
    print("All test_types.py tests passed!")
    print("="*70)
