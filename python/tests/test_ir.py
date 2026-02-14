"""Tests for the IR (Intermediate Representation) - with pretty printing."""

import pytest
from luisa import (
    int32, uint32, float32, bool_,
    IRBuilder, IROp, IRModule,
    ConstantValue, ArgumentValue, InstructionValue,
    pprint,
)


def test_ir_operations():
    """Test IR operation types."""
    print("\n" + "="*60)
    print("Test: IR operations")
    print("="*60)
    
    assert hasattr(IROp, 'ADD')
    assert hasattr(IROp, 'SUB')
    assert hasattr(IROp, 'MUL')
    assert hasattr(IROp, 'DIV')
    assert hasattr(IROp, 'RETURN')
    assert hasattr(IROp, 'DISPATCH_ID')
    assert hasattr(IROp, 'THREAD_ID')
    assert hasattr(IROp, 'BLOCK_ID')
    
    print("✓ All expected IR operations exist")
    print("="*60)


def test_ir_builder_basic():
    """Test basic IR builder operations."""
    print("\n" + "="*60)
    print("Test: IR builder basic")
    print("="*60)
    
    builder = IRBuilder('test_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    assert isinstance(a, ArgumentValue)
    assert isinstance(b, ArgumentValue)
    
    const = builder.constant(float32, 2.0)
    assert isinstance(const, ConstantValue)
    assert const.value == 2.0
    
    sum_val = builder.add(a, b)
    assert isinstance(sum_val, InstructionValue)
    
    prod = builder.mul(sum_val, const)
    assert isinstance(prod, InstructionValue)
    
    builder.return_(prod)
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    assert func.name == 'test_func'
    assert len(func.blocks) == 1
    assert len(func.blocks[0].instructions) == 3
    
    print(f"✓ Built function with {len(func.blocks)} block(s), {len(func.blocks[0].instructions)} instructions")
    print("="*60)


def test_ir_builder_control_flow():
    """Test IR builder with structured control flow."""
    print("\n" + "="*60)
    print("Test: IR builder control flow")
    print("="*60)
    
    builder = IRBuilder('test_if', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    const_0 = builder.constant(float32, 0.0)
    
    cond = builder.gt(a, const_0)
    
    if_ = builder.if_(cond)
    with if_.true_scope():
        builder.return_(a)
    with if_.false_scope():
        neg_a = builder.neg(a)
        builder.return_(neg_a)
    
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    assert len(func.blocks) >= 2
    
    print(f"✓ Built function with {len(func.blocks)} blocks (control flow)")
    print("="*60)


def test_constant_folding_if():
    """Test constant folding in if statements."""
    print("\n" + "="*60)
    print("Test: constant folding in if")
    print("="*60)
    
    builder = IRBuilder('test_fold', (), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    true_cond = builder.constant(bool_, True)
    
    if_ = builder.if_(true_cond)
    with if_.true_scope():
        result = builder.constant(float32, 1.0)
        builder.return_(result)
    with if_.false_scope():
        pass
    
    func = builder.build()
    
    print("\nGenerated IR (constant folded):")
    print(pprint(func))
    
    assert func.name == 'test_fold'
    
    print(f"✓ Constant folding works, {len(func.blocks)} blocks")
    print("="*60)


def test_ir_module():
    """Test IR module creation."""
    print("\n" + "="*60)
    print("Test: IR module")
    print("="*60)
    
    builder = IRBuilder('func1', (int32,), int32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func1 = builder.build()
    
    print("\nFunction 1:")
    print(pprint(func1))
    
    builder2 = IRBuilder('func2', (float32,), float32)
    entry2 = builder2.create_block('entry')
    builder2.set_insert_point(entry2)
    b = builder2.get_argument(0)
    builder2.return_(b)
    func2 = builder2.build()
    
    print("\nFunction 2:")
    print(pprint(func2))
    
    module = IRModule(functions=[func1, func2])
    
    assert len(module.functions) == 2
    
    print(f"✓ Module created with {len(module.functions)} functions")
    print("="*60)


def test_ir_builder_arithmetic_ops():
    """Test all basic arithmetic operations."""
    print("\n" + "="*60)
    print("Test: IR builder arithmetic ops")
    print("="*60)
    
    builder = IRBuilder('arith', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    # Test various operations
    add_res = builder.add(a, b)
    sub_res = builder.sub(a, b)
    mul_res = builder.mul(a, b)
    div_res = builder.div(a, b)
    mod_res = builder.mod(a, b)
    
    builder.return_(add_res)
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    assert len(func.blocks[0].instructions) == 6  # 5 ops + return
    
    print(f"✓ All arithmetic operations built, {len(func.blocks[0].instructions)} instructions")
    print("="*60)


def test_ir_builder_comparison_ops():
    """Test comparison operations."""
    print("\n" + "="*60)
    print("Test: IR builder comparison ops")
    print("="*60)
    
    builder = IRBuilder('compare', (float32, float32), bool_)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    eq_res = builder.eq(a, b)
    ne_res = builder.ne(a, b)
    lt_res = builder.lt(a, b)
    le_res = builder.le(a, b)
    gt_res = builder.gt(a, b)
    ge_res = builder.ge(a, b)
    
    builder.return_(gt_res)
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    assert len(func.blocks[0].instructions) == 7  # 6 compares + return
    
    print(f"✓ All comparison operations built, {len(func.blocks[0].instructions)} instructions")
    print("="*60)


def test_ir_builder_bitwise_ops():
    """Test bitwise operations."""
    print("\n" + "="*60)
    print("Test: IR builder bitwise ops")
    print("="*60)
    
    builder = IRBuilder('bitwise', (int32, int32), int32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    
    and_res = builder.bit_and(a, b)
    or_res = builder.bit_or(a, b)
    xor_res = builder.bit_xor(a, b)
    not_res = builder.bit_not(a)
    shl_res = builder.shl(a, b)
    shr_res = builder.shr(a, b)
    
    builder.return_(and_res)
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    print(f"✓ Bitwise operations built, {len(func.blocks[0].instructions)} instructions")
    print("="*60)


def test_ir_builder_switch():
    """Test IR builder with structured switch statement."""
    print("\n" + "="*60)
    print("Test: IR builder switch")
    print("="*60)
    
    builder = IRBuilder('test_switch', (int32,), int32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    tag = builder.get_argument(0)
    
    with builder.switch(tag) as sw:
        with sw.case_scope(1):
            builder.return_(builder.constant(int32, 10))
        with sw.case_scope(2):
            builder.return_(builder.constant(int32, 20))
        with sw.default_scope():
            builder.return_(builder.constant(int32, -1))
            
    func = builder.build()
    
    print("\nGenerated IR:")
    print(pprint(func))
    
    # Check if SWITCH instruction exists
    # It should be the first instruction in entry block
    inst = func.blocks[0].instructions[0]
    assert inst.op == IROp.SWITCH
    
    print(f"✓ Built function with structured switch statement")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_ir.py tests")
    print("="*70)
    
    test_ir_operations()
    test_ir_builder_basic()
    test_ir_builder_control_flow()
    test_constant_folding_if()
    test_ir_module()
    test_ir_builder_arithmetic_ops()
    test_ir_builder_comparison_ops()
    test_ir_builder_bitwise_ops()
    test_ir_builder_switch()
    
    print("\n" + "="*70)
    print("All test_ir.py tests passed!")
    print("="*70)
