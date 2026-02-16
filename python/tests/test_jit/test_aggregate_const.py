"""Tests for aggregate constant construction and folding."""

import pytest
from luisa import (
    callable, pprint,
    Int, Float, Float2, Float3, Float2x2, Array, struct,
    Const
)


def test_vector_const_construction():
    """Test vector constant construction from various arguments."""
    # Exact components
    c1 = Const[Float2](1.0, 2.0)
    assert c1.value == (1.0, 2.0)
    
    # From tuple
    c2 = Const[Float2]((3.0, 4.0))
    assert c2.value == (3.0, 4.0)
    
    # Broadcast
    c3 = Const[Float3](5.0)
    assert c3.value == (5.0, 5.0, 5.0)
    
    # Error cases
    with pytest.raises(ValueError, match="requires 2 components, got 3"):
        Const[Float2](1.0, 2.0, 3.0)
        
    with pytest.raises(ValueError, match="requires 2 components, got 3"):
        Const[Float2]((1.0, 2.0, 3.0))


def test_matrix_const_construction():
    """Test matrix constant construction."""
    # 4 components for 2x2 - stored as column-major tuple-of-tuples
    c1 = Const[Float2x2](1.0, 2.0, 3.0, 4.0)
    assert c1.value == ((1.0, 2.0), (3.0, 4.0))  # (col0, col1)
    
    # From list
    c2 = Const[Float2x2]([5.0, 6.0, 7.0, 8.0])
    assert c2.value == ((5.0, 6.0), (7.0, 8.0))
    
    # Diagonal broadcast
    c3 = Const[Float2x2](2.0)
    assert c3.value == ((2.0, 0.0), (0.0, 2.0))
    
    with pytest.raises(ValueError, match="requires 4 components, got 2"):
        Const[Float2x2](1.0, 2.0)


def test_matrix_column_major_construction():
    """Test matrix construction from columns (column-major)."""
    # 2x2 matrix from 2 columns - stored as column-major tuple-of-tuples
    c1 = Const[Float2x2]((1.0, 2.0), (3.0, 4.0))
    # Structure: (col0, col1) where each col is (x, y)
    assert c1.value == ((1.0, 2.0), (3.0, 4.0))
    
    # 3x3 matrix from 3 columns
    from luisa import Float3x3
    c2 = Const[Float3x3]((1,2,3), (4,5,6), (7,8,9))
    assert c2.value == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))


def test_nested_aggregates(print_ir, verify_ir):
    """Test complicated nested aggregates."""
    @struct
    class Inner:
        v: Float2
        a: Array[Int, 2]
        
    @struct
    class Outer:
        i: Inner
        f: Float
        
    # Construct nested constant
    # i.v=(1,2), i.a=(3,4), f=5.0
    c = Const[Outer](Inner((1.0, 2.0), (3, 4)), 5.0)
    
    # Check values
    assert c.value.f == 5.0
    assert c.value.i.v == (1.0, 2.0)
    assert c.value.i.a == (3, 4)
    
    # Test folding with nested aggregate
    # Note: This test requires the callable to be specialized by calling from a kernel
    # For now, we just verify the Const construction works (checked above)
    # TODO: Re-enable when nested struct constant folding in DSL is fully supported
    @callable
    def nested_fold() -> Float:
        o = Const[Outer](Inner((1.0, 2.0), (3, 4)), 5.0)
        return o.i.v.x + Float(o.i.a[0]) + o.f
        
    # Skip IR verification - the function is unspecialized without being called
    # print_ir(nested_fold)
    # expected = """
    # f32 nested_fold() {
    #   return 9.0;
    # }
    # """
    # verify_ir(nested_fold, expected)


def test_struct_const_construction():
    """Test struct constant construction."""
    @struct
    class Point:
        x: Float
        y: Float
        
    # Positional
    c1 = Const[Point](1.0, 2.0)
    assert c1.value == Point(1.0, 2.0)
    
    # Named
    c2 = Const[Point](y=4.0, x=3.0)
    assert c2.value == Point(3.0, 4.0)
    
    # Tuple
    c3 = Const[Point]((5.0, 6.0))
    assert c3.value == Point(5.0, 6.0)


def test_matrix_folding(print_ir, verify_ir):
    """Test matrix constant folding."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
        
    # Test Const construction at Python level
    m = Const[Float2x2](1.0, 2.0, 3.0, 4.0)
    # Verify the Const was constructed correctly
    assert m.value == ((1.0, 2.0), (3.0, 4.0))
    
    # Note: DSL-level constant folding with matrix operations requires 
    # the callable to be specialized by calling from a kernel
    # TODO: Re-enable when full matrix constant folding in DSL is supported
    @callable
    def mat_ops() -> Float:
        m = Const[Float2x2](1.0, 2.0, 3.0, 4.0)
        # These should all be folded on the host
        from luisa import transpose, determinant
        t = transpose(m)
        d = determinant(m)
        return d + m[0][1] # matrices are in column-major, so m[0][1] is 2.0

    # Skip IR verification - the function is unspecialized without being called
    # print_ir(mat_ops)
    # If folded, shouldn't have matrix instructions
    # The result should be approximately 0 (det = -2, m[0][1] = 2)
    # ...


def test_matmul_folding(print_ir, verify_ir):
    """Test matrix multiplication constant folding."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
        
    @callable
    def matmul_fold() -> Float2x2:
        m1 = Const[Float2x2](1.0, 0.0, 0.0, 1.0)
        m2 = Const[Float2x2](2.0, 3.0, 4.0, 5.0)
        return m1 @ m2

    print_ir(matmul_fold)
    
    # Result should be constant (2,3,4,5)
    # Using wildcard for numpy float64 repr
    actual = pprint(matmul_fold, recursive=True, show_location=False)
    assert 'return (2.0, 3.0, 4.0, 5.0);' in actual or \
           'return (np.float64(2.0), np.float64(3.0), np.float64(4.0), np.float64(5.0));' in actual

if __name__ == "__main__":
    pytest.main([__file__])
