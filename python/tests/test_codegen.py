"""Tests for code generation (JSON and pretty printing)."""

import json
import pytest
from luisa import (
    int32, float32, bool_,
    IRBuilder,
    serialize_function,
    serialize_module,
    pprint,
    IRModule,
)


def test_json_serialization_basic():
    """Test basic JSON serialization."""
    print("\n" + "="*60)
    print("Test: JSON serialization basic")
    print("="*60)
    
    builder = IRBuilder('simple_func', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func = builder.build()
    
    print("\nPretty printed IR:")
    print(pprint(func))
    
    json_str = serialize_function(func)
    data = json.loads(json_str)
    
    assert data['name'] == 'simple_func'
    assert len(data['blocks']) == 1
    
    print(f"✓ JSON serialization works, {len(data['blocks'])} blocks")
    print("="*60)


def test_json_serialization_with_ops():
    """Test JSON serialization with operations."""
    print("\n" + "="*60)
    print("Test: JSON serialization with ops")
    print("="*60)
    
    builder = IRBuilder('math_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    sum_val = builder.add(a, b)
    builder.return_(sum_val)
    func = builder.build()
    
    print("\nPretty printed IR:")
    print(pprint(func))
    
    json_str = serialize_function(func, indent=None)
    data = json.loads(json_str)
    
    assert data['name'] == 'math_func'
    assert len(data['arg_types']) == 2
    
    print(f"✓ JSON with ops works, {len(data['blocks'])} blocks")
    print("="*60)


def test_json_serialization_control_flow():
    """Test JSON serialization with control flow."""
    print("\n" + "="*60)
    print("Test: JSON serialization control flow")
    print("="*60)
    
    builder = IRBuilder('if_func', (float32,), float32)
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
    
    print("\nPretty printed IR (with control flow):")
    print(pprint(func))
    
    json_str = serialize_function(func)
    data = json.loads(json_str)
    
    assert data['name'] == 'if_func'
    assert len(data['blocks']) >= 2
    
    print(f"✓ JSON with control flow works, {len(data['blocks'])} blocks")
    print("="*60)


def test_pprint_basic():
    """Test pretty printing basic function."""
    print("\n" + "="*60)
    print("Test: pprint basic")
    print("="*60)
    
    builder = IRBuilder('simple_func', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func = builder.build()
    
    output = pprint(func)
    
    print("\nPretty printed output:")
    print(output)
    
    assert 'float32 simple_func(float32 arg0)' in output
    assert 'entry:' in output
    assert 'return' in output
    
    print("✓ pprint basic works")
    print("="*60)


def test_pprint_with_ops():
    """Test pretty printing with operations."""
    print("\n" + "="*60)
    print("Test: pprint with ops")
    print("="*60)
    
    builder = IRBuilder('math_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    sum_val = builder.add(a, b)
    builder.return_(sum_val)
    func = builder.build()
    
    output = pprint(func)
    
    print("\nPretty printed output:")
    print(output)
    
    assert 'float32 math_func(float32 arg0, float32 arg1)' in output
    assert 'add' in output.lower()
    
    print("✓ pprint with ops works")
    print("="*60)


def test_pprint_control_flow():
    """Test pretty printing with control flow."""
    print("\n" + "="*60)
    print("Test: pprint control flow")
    print("="*60)
    
    builder = IRBuilder('if_func', (float32,), float32)
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
    
    output = pprint(func)
    
    print("\nPretty printed output (control flow):")
    print(output)
    
    assert 'float32 if_func(float32 arg0)' in output
    assert 'if_true' in output or 'entry' in output
    
    print("✓ pprint control flow works")
    print("="*60)


def test_serialize_module():
    """Test JSON serialization of a module."""
    print("\n" + "="*60)
    print("Test: serialize module")
    print("="*60)
    
    builder1 = IRBuilder('func1', (float32,), float32)
    entry1 = builder1.create_block('entry')
    builder1.set_insert_point(entry1)
    a = builder1.get_argument(0)
    builder1.return_(a)
    func1 = builder1.build()
    
    builder2 = IRBuilder('func2', (int32,), int32)
    entry2 = builder2.create_block('entry')
    builder2.set_insert_point(entry2)
    b = builder2.get_argument(0)
    builder2.return_(b)
    func2 = builder2.build()
    
    module = IRModule(functions=[func1, func2])
    
    print("\nModule functions:")
    for func in module.functions:
        print(f"  - {func.name}")
    
    json_str = serialize_module(module)
    data = json.loads(json_str)
    
    assert len(data['functions']) == 2
    
    print(f"✓ Module serialization works, {len(data['functions'])} functions")
    print("="*60)


def test_complex_ir_pprint():
    """Test pretty printing complex IR."""
    print("\n" + "="*60)
    print("Test: pprint complex IR")
    print("="*60)
    
    builder = IRBuilder('complex_func', (float32, float32, int32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    
    x = builder.get_argument(0)
    y = builder.get_argument(1)
    n = builder.get_argument(2)
    
    # Compute x*x + y*y
    x2 = builder.mul(x, x)
    y2 = builder.mul(y, y)
    sum_sq = builder.add(x2, y2)
    
    # Loop condition
    const_0 = builder.constant(int32, 0)
    cond = builder.gt(n, const_0)
    
    while_ = builder.while_(cond)
    with while_.body_scope():
        # Decrement n
        const_1 = builder.constant(int32, 1)
        new_n = builder.sub(n, const_1)
        n = new_n
        # Update condition
        cond = builder.gt(n, const_0)
    
    builder.return_(sum_sq)
    func = builder.build()
    
    output = pprint(func)
    
    print("\nComplex IR pretty printed:")
    print(output)
    
    assert 'float32 complex_func(float32 arg0, float32 arg1, int32 arg2)' in output
    assert len(func.blocks) >= 3  # entry, while_header, while_body, etc.
    
    print(f"✓ Complex IR pprint works, {len(func.blocks)} blocks")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_codegen.py tests")
    print("="*70)
    
    test_json_serialization_basic()
    test_json_serialization_with_ops()
    test_json_serialization_control_flow()
    test_pprint_basic()
    test_pprint_with_ops()
    test_pprint_control_flow()
    test_serialize_module()
    test_complex_ir_pprint()
    
    print("\n" + "="*70)
    print("All test_codegen.py tests passed!")
    print("="*70)
