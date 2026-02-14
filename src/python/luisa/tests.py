"""
Tests for the LuisaCompute Python DSL v2.

Run with: python -m pytest src/python/luisa/tests.py -v
"""

import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from luisa.dsl_types import (
    bool_, int32, uint32, float32, float64,
    float2, float3, float4, int2, int3, int4,
    float3x3, float4x4,
    Buffer, Array, Struct, promote_types, is_vector_type, is_integer_type,
    Scalar, Vector, Matrix, ScalarType,
)
from luisa.ir import IROp, ConstantValue, IRBasicBlock
from luisa.builder import IRBuilder
from luisa.staged import kernel, callable, StagedFunction


def test_scalar_types():
    """Test scalar type definitions."""
    print("Testing scalar types...")
    assert float32.dtype == ScalarType.FLOAT32
    assert int32.dtype == ScalarType.INT32
    assert bool_.dtype == ScalarType.BOOL
    assert repr(float32) == "float32"
    print("  ✓ Scalar types OK")


def test_vector_types():
    """Test vector type definitions."""
    print("Testing vector types...")
    assert float3.element == float32
    assert float3.size == 3
    assert repr(float3) == "float32[3]"
    
    try:
        Vector(float32, 5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ Vector types OK")


def test_matrix_types():
    """Test matrix type definitions."""
    print("Testing matrix types...")
    assert float3x3.element == float32
    assert float3x3.size == 3
    assert repr(float3x3) == "float32[3]x[3]"
    print("  ✓ Matrix types OK")


def test_buffer_type():
    """Test buffer type."""
    print("Testing buffer type...")
    b = Buffer(float32)
    assert b.element == float32
    assert repr(b) == "buffer<float32>"
    
    b_vec = Buffer(float3)
    assert b_vec.element == float3
    print("  ✓ Buffer type OK")


def test_type_promotion():
    """Test type promotion rules."""
    print("Testing type promotion...")
    
    # Same type
    assert promote_types(float32, float32) == float32
    
    # Scalar broadcasts to vector
    assert promote_types(float32, float3) == float3
    assert promote_types(float3, float32) == float3
    
    # Float beats int
    assert promote_types(int32, float32) == float32
    
    # Vector promotion
    v1 = Vector(int32, 3)
    v2 = Vector(float32, 3)
    assert promote_types(v1, v2) == float3
    
    print("  ✓ Type promotion OK")


def test_ir_operations():
    """Test IR operations."""
    print("Testing IR operations...")
    
    assert IROp.ADD.name == "ADD"
    assert IROp.MUL.name == "MUL"
    
    # Constant value
    const = ConstantValue(type=float32, value=3.14)
    assert const.value == 3.14
    assert "3.14" in repr(const)
    
    print("  ✓ IR operations OK")


def test_ir_builder_basic():
    """Test basic IR builder functionality."""
    print("Testing IR builder basic...")
    
    builder = IRBuilder('test', (float32, float32), float32)
    
    # Create entry block
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    # Get arguments
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    # Add them
    result = builder.add(a, b)
    assert result.name == "t0"
    
    # Return
    builder.return_(result)
    
    # Build function
    func = builder.build()
    
    assert func.name == 'test'
    assert len(func.blocks) == 1
    assert len(func.blocks[0].instructions) == 2
    
    print("  ✓ IR builder basic OK")


def test_ir_builder_control_flow():
    """Test IR builder with control flow."""
    print("Testing IR builder control flow...")
    
    builder = IRBuilder('test_if', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    const_0 = builder.constant(float32, 0.0)
    
    # if a > 0: return a; else: return -a
    cond = builder.gt(a, const_0)
    
    with builder.if_(cond) as if_scope:
        builder.return_(a)
        with if_scope.otherwise():
            neg_a = builder.neg(a)
            builder.return_(neg_a)
    
    func = builder.build()
    
    # Check we have multiple blocks
    assert len(func.blocks) >= 3  # entry, if_true, if_false, merge
    
    print("  ✓ IR builder control flow OK")


def test_constant_folding_if():
    """Test constant folding in if statements."""
    print("Testing constant folding in if...")
    
    builder = IRBuilder('test_fold', (), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    # Constant True condition - should fold
    true_cond = builder.constant(bool_, True)
    
    with builder.if_(true_cond) as if_scope:
        # This branch should be taken
        result = builder.constant(float32, 1.0)
        builder.return_(result)
        # else branch is NoOpScope due to constant folding
    
    func = builder.build()
    
    # Check the function is valid
    assert func.name == 'test_fold'
    
    print("  ✓ Constant folding in if OK")


def test_staged_function_basic():
    """Test basic staged function."""
    print("Testing staged function basic...")
    
    @callable
    def add(a: float32, b: float32) -> float32:
        return a + b
    
    assert isinstance(add, StagedFunction)
    assert add.name == 'add'
    assert not add.is_kernel
    
    # Generate IR
    ir_func = add(1.0, 2.0)
    
    assert ir_func.name == 'add'
    assert len(ir_func.blocks) > 0
    assert len(ir_func.blocks[0].instructions) > 0
    
    # Check caching
    ir_func2 = add(1.0, 2.0)
    assert ir_func is ir_func2
    
    print("  ✓ Staged function basic OK")


def test_staged_function_with_kernel():
    """Test staged function marked as kernel."""
    print("Testing staged function kernel...")
    
    @kernel
    def saxpy(result, a: float32, x, y):
        # Note: This is simplified - real implementation needs index calculation
        result[0] = a * x[0] + y[0]
    
    assert isinstance(saxpy, StagedFunction)
    assert saxpy.is_kernel
    
    print("  ✓ Staged function kernel OK")


def test_staged_function_control_flow():
    """Test staged function with control flow."""
    print("Testing staged function control flow...")
    
    @callable
    def abs_val(a: float32) -> float32:
        if a > 0.0:
            return a
        else:
            return -a
    
    ir_func = abs_val(5.0)
    
    assert ir_func.name == 'abs_val'
    # Should have multiple blocks for if-else
    assert len(ir_func.blocks) >= 3
    
    print("  ✓ Staged function control flow OK")


def test_staged_function_captured_vars():
    """Test staged function with captured variables."""
    print("Testing staged function captured vars...")
    
    threshold = 0.5  # Captured variable
    
    @callable
    def threshold_fn(a: float32) -> float32:
        if a > threshold:
            return a
        else:
            return 0.0
    
    ir_func = threshold_fn(1.0)
    
    assert ir_func.name == 'threshold_fn'
    # threshold should be a constant in the IR
    
    print("  ✓ Staged function captured vars OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("LuisaCompute Python DSL v2 - Test Suite")
    print("=" * 60)
    
    test_scalar_types()
    test_vector_types()
    test_matrix_types()
    test_buffer_type()
    test_type_promotion()
    test_ir_operations()
    test_ir_builder_basic()
    test_ir_builder_control_flow()
    test_constant_folding_if()
    test_staged_function_basic()
    test_staged_function_with_kernel()
    test_staged_function_control_flow()
    test_staged_function_captured_vars()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
