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
    print("Testing JSON serialization basic...")
    
    # Create a simple function
    builder = IRBuilder('simple_func', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func = builder.build()
    
    # Serialize to JSON
    json_str = serialize_function(func)
    
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data['name'] == 'simple_func'
    assert len(data['blocks']) == 1
    
    print("  ✓ JSON serialization basic OK")


def test_json_serialization_with_ops():
    """Test JSON serialization with operations."""
    print("Testing JSON serialization with ops...")
    
    # Create a function with operations
    builder = IRBuilder('math_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    sum_val = builder.add(a, b)
    builder.return_(sum_val)
    func = builder.build()
    
    # Serialize to JSON
    json_str = serialize_function(func, indent=None)  # Compact
    
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data['name'] == 'math_func'
    assert len(data['arg_types']) == 2
    
    print("  ✓ JSON serialization with ops OK")


def test_json_serialization_control_flow():
    """Test JSON serialization with control flow."""
    print("Testing JSON serialization control flow...")
    
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
    
    # Serialize to JSON
    json_str = serialize_function(func)
    
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data['name'] == 'if_func'
    assert len(data['blocks']) >= 2
    
    print("  ✓ JSON serialization control flow OK")


def test_pprint_basic():
    """Test pretty printing basic function."""
    print("Testing pprint basic...")
    
    # Create a simple function
    builder = IRBuilder('simple_func', (float32,), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    builder.return_(a)
    func = builder.build()
    
    # Pretty print
    output = pprint(func)
    
    # Verify output contains expected elements
    assert 'func simple_func' in output
    assert 'entry:' in output
    assert 'return' in output
    
    print("  ✓ pprint basic OK")
    print(f"    Output:\n{output}")


def test_pprint_with_ops():
    """Test pretty printing with operations."""
    print("Testing pprint with ops...")
    
    builder = IRBuilder('math_func', (float32, float32), float32)
    entry = builder.create_block('entry')
    builder.set_insert_point(entry)
    a = builder.get_argument(0)
    b = builder.get_argument(1)
    sum_val = builder.add(a, b)
    builder.return_(sum_val)
    func = builder.build()
    
    # Pretty print
    output = pprint(func)
    
    # Verify output
    assert 'func math_func' in output
    assert 'add' in output
    
    print("  ✓ pprint with ops OK")
    print(f"    Output:\n{output}")


def test_pprint_control_flow():
    """Test pretty printing with control flow."""
    print("Testing pprint control flow...")
    
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
    
    # Pretty print
    output = pprint(func)
    
    # Verify output contains multiple blocks
    assert 'func if_func' in output
    assert 'if_true' in output or 'entry' in output
    
    print("  ✓ pprint control flow OK")
    print(f"    Output:\n{output}")


def test_serialize_module():
    """Test JSON serialization of a module."""
    print("Testing serialize module...")
    
    # Create functions
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
    
    # Create module
    module = IRModule(functions=[func1, func2])
    
    # Serialize
    json_str = serialize_module(module)
    
    # Verify
    data = json.loads(json_str)
    assert len(data['functions']) == 2
    
    print("  ✓ serialize module OK")
