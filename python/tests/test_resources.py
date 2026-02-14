"""Tests for resource types (Buffer, Texture, etc.) - with IR building."""

import pytest
import ast as python_ast
from luisa import (
    kernel, callable, pprint,
    int32, float32, float3, float4,
    Buffer, Texture2D, Texture3D,
    BindlessArray, Accel,
    dispatch_id,
)
from luisa.lang.inspect import count_instructions, get_ir_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")

from luisa.lang.inspect import get_ir_ast
import ast as python_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")



def test_buffer_type():
    """Test buffer resource type."""
    print("\n" + "="*60)
    print("Test: Buffer type")
    print("="*60)
    
    buf_f32 = Buffer(float32)
    assert buf_f32.element == float32
    
    buf_float3 = Buffer(float3)
    assert buf_float3.element == float3
    
    print("✓ Buffer types created correctly")
    print("="*60)


def test_buffer_in_kernel_builds_ir():
    """Test Buffer in kernel actually builds IR."""
    print("\n" + "="*60)
    print("Test: Buffer in kernel builds IR")
    print("="*60)
    
    @kernel
    def fill_buffer(buf: Buffer(float32), value: float32) -> None:
        idx = dispatch_id().x
        buf[idx] = value
    
    ir = fill_buffer(Buffer(float32), 1.0)
    print_ast(fill_buffer, "AST: fill_buffer")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.name == 'fill_buffer'
    assert ir.is_kernel
    assert len(ir.blocks) > 0
    
    counts = count_instructions(ir)
    assert 'BUFFER_WRITE' in counts
    
    print(f"✓ Kernel built with {len(ir.blocks)} blocks, BUFFER_WRITE={counts.get('BUFFER_WRITE',0)}")
    print("="*60)


def test_buffer_vector_type_kernel():
    """Test Buffer of vectors in kernel."""
    print("\n" + "="*60)
    print("Test: Buffer<float3> kernel")
    print("="*60)
    
    @kernel
    def process_vectors(buf: Buffer(float3)):
        idx = dispatch_id().x
        val = buf[idx]
        buf[idx] = val
    
    ir = process_vectors(Buffer(float3))
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    print(f"✓ Vector buffer kernel built with {len(ir.blocks)} blocks")
    print("="*60)


def test_texture2d_type():
    """Test 2D texture resource type."""
    print("\n" + "="*60)
    print("Test: Texture2D type")
    print("="*60)
    
    tex_f32 = Texture2D(float32)
    assert tex_f32.element == float32
    
    print("✓ Texture2D types created correctly")
    print("="*60)


def test_texture2d_in_kernel():
    """Test Texture2D in kernel builds IR."""
    print("\n" + "="*60)
    print("Test: Texture2D in kernel")
    print("="*60)
    
    @kernel
    def sample_texture(tex: Texture2D(float32), output: Buffer(float32)):
        idx = dispatch_id().x
        # Note: full texture sampling would need more support
        output[idx] = 0.0
    
    ir = sample_texture(Texture2D(float32), Buffer(float32))
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    print(f"✓ Texture kernel built with {len(ir.blocks)} blocks")
    print("="*60)


def test_texture3d_type():
    """Test 3D texture resource type."""
    print("\n" + "="*60)
    print("Test: Texture3D type")
    print("="*60)
    
    tex_f32 = Texture3D(float32)
    assert tex_f32.element == float32
    
    print("✓ Texture3D types created correctly")
    print("="*60)


def test_bindless_array_type():
    """Test bindless array resource type."""
    print("\n" + "="*60)
    print("Test: BindlessArray type")
    print("="*60)
    
    bindless = BindlessArray()
    assert isinstance(bindless, BindlessArray)
    
    print("✓ BindlessArray type created correctly")
    print("="*60)


def test_accel_type():
    """Test acceleration structure resource type."""
    print("\n" + "="*60)
    print("Test: Accel type")
    print("="*60)
    
    accel = Accel()
    assert isinstance(accel, Accel)
    
    print("✓ Accel type created correctly")
    print("="*60)


def test_multiple_resources_in_kernel():
    """Test multiple resource types in one kernel."""
    print("\n" + "="*60)
    print("Test: multiple resources in kernel")
    print("="*60)
    
    @kernel
    def multi_resource_kernel(
        buf: Buffer(float32),
        tex: Texture2D(float32),
        accel: Accel
    ):
        idx = dispatch_id().x
        buf[idx] = float32(idx)
    
    ir = multi_resource_kernel(
        Buffer(float32),
        Texture2D(float32),
        Accel()
    )
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    print(f"✓ Multi-resource kernel built with {len(ir.blocks)} blocks")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_resources.py tests")
    print("="*70)
    
    test_buffer_type()
    test_buffer_in_kernel_builds_ir()
    test_buffer_vector_type_kernel()
    test_texture2d_type()
    test_texture2d_in_kernel()
    test_texture3d_type()
    test_bindless_array_type()
    test_accel_type()
    test_multiple_resources_in_kernel()
    
    print("\n" + "="*70)
    print("All test_resources.py tests passed!")
    print("="*70)
