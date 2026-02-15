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
    promote_types, get_alignment, is_data_type,
    Scalar, Vector, Matrix, Array, Struct, struct,
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

    # Create custom vector via class
    custom_vec = Vector(Float, 3)
    assert custom_vec.size == 3
    assert custom_vec.element == Float

    # Test subscripting
    assert Vector[Float, 3] == Float3
    assert Vector[Int, 4] == Int4
    assert Vector[Bool, 2] == Bool2

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

    # Test subscripting
    assert Matrix[Float, 3] == Float3x3
    assert Matrix[Float, 4] == Float4x4

    print("  ✓ Matrix types OK")


def test_array_types():
    """Test array types."""
    print("Testing array types...")

    # Test subscripting
    arr_type = Array[Float, 10]
    assert isinstance(arr_type, Array)
    assert arr_type.element == Float
    assert arr_type.size == 10

    arr_vec_type = Array[Float3, 5]
    assert arr_vec_type.element == Float3
    assert arr_vec_type.size == 5

    print("  ✓ Array types OK")


def test_buffer_type():
    """Test buffer type."""
    print("Testing buffer type...")

    buf_type = Buffer(Float)
    assert isinstance(buf_type, Buffer)
    assert buf_type.element == Float

    buf_int3 = Buffer(Int3)
    assert buf_int3.element == Int3

    print("  ✓ Buffer type OK")


def test_alignment():
    """Test alignment calculation."""
    print("Testing alignment...")

    assert get_alignment(Bool) == 1
    assert get_alignment(Short) == 2
    assert get_alignment(Int) == 4
    assert get_alignment(Long) == 8
    assert get_alignment(Float) == 4
    
    assert get_alignment(Float2) == 8
    assert get_alignment(Float3) == 16
    assert get_alignment(Float4) == 16
    
    assert get_alignment(Float3x3) == 16
    assert get_alignment(Array[Int, 10]) == 4
    
    print("  ✓ Alignment OK")


def test_struct_types():
    """Test struct types and decorator."""
    print("Testing struct types...")

    @struct
    class MyStruct:
        x: Int
        y: Float3
        
    dsl_type = MyStruct.get_dsl_type()
    assert isinstance(dsl_type, Struct)
    assert dsl_type.name == "MyStruct"
    assert dsl_type.fields[0] == ("x", Int)
    assert dsl_type.fields[1] == ("y", Float3)
    # Max alignment of Int(4) and Float3(16) is 16
    assert dsl_type.alignment == 16

    @struct(align=32)
    class AlignedStruct:
        x: Int
        
    assert AlignedStruct.get_dsl_type().alignment == 32

    # Test anonymous struct subscripting
    anon = Struct[Int, Float3]
    assert anon.name == "anonymous_struct"
    assert anon.fields[0] == ("_0", Int)
    assert anon.fields[1] == ("_1", Float3)
    assert anon.alignment == 16

    anon_aligned = Struct[Int, 64]
    assert anon_aligned.fields[0] == ("_0", Int)
    assert anon_aligned.alignment == 64

    # Test validation
    with pytest.raises(TypeError):
        @struct
        class BadStruct:
            b: Buffer[Float]  # Resource type not allowed

    print("  ✓ Struct types OK")


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
    test_array_types()
    test_buffer_type()
    test_alignment()
    test_struct_types()
    test_type_promotion()

    print("\n" + "=" * 70)
    print("All test_types.py tests passed!")
    print("=" * 70)
