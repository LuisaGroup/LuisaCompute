"""Tests for aggregate constant construction and folding."""

import pytest
from luisa import (
    kernel, callable, pprint,
    Int, Float, Bool, Float2, Float3, Float2x2, Array, struct,
    Const, static
)
from luisa.lang.inspect import count_instructions


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
    # 4 components for 2x2
    c1 = Const[Float2x2](1.0, 2.0, 3.0, 4.0)
    assert c1.value == (1.0, 2.0, 3.0, 4.0)
    
    # From list
    c2 = Const[Float2x2]([5.0, 6.0, 7.0, 8.0])
    assert c2.value == (5.0, 6.0, 7.0, 8.0)
    
    # Diagonal broadcast
    c3 = Const[Float2x2](2.0)
    assert c3.value == (2.0, 0.0, 0.0, 2.0)
    
    with pytest.raises(ValueError, match="requires 4 components, got 2"):
        Const[Float2x2](1.0, 2.0)


def test_matrix_column_major_construction():
    """Test matrix construction from columns (column-major)."""
    # 2x2 matrix from 2 columns
    c1 = Const[Float2x2]((1.0, 2.0), (3.0, 4.0))
    # Elements should be in column-major order: c0.x, c0.y, c1.x, c1.y
    assert c1.value == (1.0, 2.0, 3.0, 4.0)
    
    # 3x3 matrix from 3 columns
    from luisa import Float3x3
    c2 = Const[Float3x3]((1,2,3), (4,5,6), (7,8,9))
    assert c2.value == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)


def test_nested_aggregates():
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
    @callable
    def nested_fold() -> Float:
        o = Const[Outer](Inner((1.0, 2.0), (3, 4)), 5.0)
        return o.i.v.x + Float(o.i.a[0]) + o.f
        
    ir = nested_fold()
    # Should fold to 1.0 + 3.0 + 5.0 = 9.0
    counts = count_instructions(ir)
    assert 'ADD' not in counts
    assert 'RETURN' in counts


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


def test_matrix_folding():
    """Test matrix constant folding."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
        
    @callable
    def mat_ops() -> Float:
        m = Const[Float2x2](1.0, 2.0, 3.0, 4.0)
        # These should all be folded on the host
        t = transpose(m)
        d = determinant(m)
        return d

    ir = mat_ops()
    counts = count_instructions(ir)
    
    # If folded, shouldn't have matrix instructions
    assert 'MATRIX_TRANSPOSE' not in counts
    assert 'MATRIX_DETERMINANT' not in counts
    # It will just be a RETURN of a constant
    assert 'RETURN' in counts


def test_matmul_folding():
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

    ir = matmul_fold()
    counts = count_instructions(ir)
    
    # Result should be constant (2,3,4,5)
    assert 'MUL' not in counts
    assert 'RETURN' in counts
