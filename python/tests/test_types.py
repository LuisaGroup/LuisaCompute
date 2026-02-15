"""Tests for the type system."""

import pytest
from luisa import (
    # Scalar types
    Bool, Short, UShort, Int, UInt, Long, ULong, Half, Float, Double,
    # Vector types
    Int2, Int3, Int4, UInt2, UInt3, UInt4,
    Float2, Float3, Float4, Bool2, Bool3, Bool4,
    # Matrix types
    Float2x2, Float3x3, Float4x4,
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
    assert is_scalar_type(Bool)
    assert is_scalar_type(Int)
    assert is_scalar_type(Float)

    assert is_integer_type(Int)
    assert is_integer_type(UInt)
    assert not is_integer_type(Float)

    assert is_float_type(Float)
    assert is_float_type(Double)
    assert not is_float_type(Int)

    print("  ✓ Scalar types OK")


def test_vector_types():
    """Test vector types."""
    print("Testing vector types...")

    # Check vector properties
    assert is_vector_type(Float3)
    assert not is_vector_type(Float)

    # Check element types
    assert get_element_type(Float3) == Float
    assert get_element_type(Int4) == Int

    # Check lengths
    assert get_length(Float2) == 2
    assert get_length(Float3) == 3
    assert get_length(Float4) == 4

    # Create custom vector
    custom_vec = Vector(Float, 3)
    assert custom_vec.size == 3
    assert custom_vec.element == Float

    print("  ✓ Vector types OK")


def test_matrix_types():
    """Test matrix types."""
    print("Testing matrix types...")

    # Check matrix properties
    assert isinstance(Float3x3, Matrix)
    assert Float3x3.size == 3
    assert Float3x3.element == Float

    assert Float4x4.size == 4
    assert Float2x2.size == 2

    print("  ✓ Matrix types OK")


def test_buffer_type():
    """Test buffer type."""
    print("Testing buffer type...")

    buf_type = Buffer(Float)
    assert isinstance(buf_type, Buffer)
    assert buf_type.element == Float

    buf_int3 = Buffer(Int3)
    assert buf_int3.element == Int3

    print("  ✓ Buffer type OK")


def test_type_promotion():
    """Test type promotion."""
    print("Testing type promotion...")

    # Same type
    assert promote_types(Int, Int) == Int

    # Scalar to vector broadcasting
    result = promote_types(Float, Float3)
    assert isinstance(result, Vector)
    assert result.size == 3
    assert result.element == Float

    # Vector to scalar broadcasting
    result = promote_types(Float3, Float)
    assert isinstance(result, Vector)
    assert result.size == 3

    print("  ✓ Type promotion OK")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_types.py tests")
    print("=" * 70)

    test_scalar_types()
    test_vector_types()
    test_matrix_types()
    test_buffer_type()
    test_type_promotion()

    print("\n" + "=" * 70)
    print("All test_types.py tests passed!")
    print("=" * 70)
