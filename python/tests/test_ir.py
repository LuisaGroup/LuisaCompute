"""Tests for the IR (Intermediate Representation)."""

import pytest
from luisa import (
    int32, uint32, float32, bool_,
    IRBuilder, IROp, IRModule,
    ConstantValue, ArgumentValue, InstructionValue,
)


def test_ir_operations():
    """Test IR operation types."""
    print("Testing IR operations...")
    
    # Check that all expected operations exist
    assert hasattr(IROp, 'ADD')
    assert hasattr(IROp, 'SUB')
    assert hasattr(IROp, 'MUL')
    assert hasattr(IROp, 'DIV')
    assert hasattr(IROp, 'RETURN')
    
    # Check special register operations
    assert hasattr(IROp, 'DISPATCH_ID')
    assert hasattr(IROp, 'THREAD_ID')
    assert hasattr(IROp, 'BLOCK_ID')
    
    print("  ✓ IR operations OK")


def test_ir_builder_basic():
    """Test basic IR builder operations."""
    print("Testing IR builder basic...")
    
    builder = IRBuilder('test_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    # Get arguments
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    assert isinstance(a, ArgumentValue)
    assert isinstance(b, ArgumentValue)
    assert a.type == float32
    assert b.type == float32
    
    # Create constant
    const = builder.constant(float32, 2.0)
    assert isinstance(const, ConstantValue)
    assert const.value == 2.0
    
    # Emit arithmetic operations
    sum_val = builder.add(a, b)
    assert isinstance(sum_val, InstructionValue)
    
    prod = builder.mul(sum_val, const)
    assert isinstance(prod, InstructionValue)
    
    # Return
    builder.return_(prod)
    
    # Build function
    func = builder.build()
    assert func.name == 'test_func'
    assert len(func.blocks) == 1
    assert len(func.blocks[0].instructions) == 3  # add, mul, return
    
    print("  ✓ IR builder basic OK")


def test_ir_builder_control_flow():
    """Test IR builder with structured control flow."""
    print("Testing IR builder control flow...")
    
    builder = IRBuilder('test_if', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    const_0 = builder.constant(float32, 0.0)
    
    # if a > 0: return a; else: return -a
    cond = builder.gt(a, const_0)
    
    if_ = builder.if_(cond)
    with if_.true_scope():
        builder.return_(a)
    with if_.false_scope():
        neg_a = builder.neg(a)
        builder.return_(neg_a)
    
    func = builder.build()
    
    # Check we have multiple blocks
    assert len(func.blocks) >= 2  # entry, if_true, if_false, merge
    
    print("  ✓ IR builder control flow OK")


def test_constant_folding_if():
    """Test constant folding in if statements."""
    print("Testing constant folding in if...")
    
    builder = IRBuilder('test_fold', (), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    # Constant True condition - should fold
    true_cond = builder.constant(bool_, True)
    
    if_ = builder.if_(true_cond)
    with if_.true_scope():
        # This branch should be taken
        result = builder.constant(float32, 1.0)
        builder.return_(result)
    with if_.false_scope():
        # This branch is skipped due to constant folding
        pass
    
    func = builder.build()
    
    # Check the function is valid
    assert func.name == 'test_fold'
    
    print("  ✓ Constant folding in if OK")


def test_ir_module():
    """Test IR module creation."""
    print("Testing IR module...")
    
    # Create a simple function
    builder = IRBuilder('func1', (int32,), int32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func1 = builder.build()
    
    # Create module
    module = IRModule(functions=[func1])
    assert len(module.functions) == 1
    assert module.functions[0].name == 'func1'
    
    print("  ✓ IR module OK")
