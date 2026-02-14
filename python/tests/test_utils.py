"""Tests for utility functions - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable, pprint,
    unrolled, UnrolledRange,
    struct,
    int32, float32, float3,
    Buffer, dispatch_id,
)
from luisa.lang.inspect import count_instructions

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



def test_unrolled_range():
    """Test UnrolledRange class."""
    print("\n" + "="*60)
    print("Test: UnrolledRange")
    print("="*60)
    
    ur = UnrolledRange(4)
    assert ur.start == 0
    assert ur.stop == 4
    assert ur.step == 1
    assert len(ur) == 4
    
    ur = UnrolledRange(1, 10, 2)
    assert ur.start == 1
    assert ur.stop == 10
    assert ur.step == 2
    
    print("✓ UnrolledRange works correctly")
    print("="*60)


def test_unrolled_builds_ir():
    """Test unrolled loop actually builds IR."""
    print("\n" + "="*60)
    print("Test: unrolled loop builds IR")
    print("="*60)
    
    @callable
    def sum_unrolled() -> int32:
        total = int32(0)
        for i in unrolled(range(4)):
            total = total + int32(i)
        return total
    
    ir = sum_unrolled()
    print_ast(sum_unrolled, "AST: sum_unrolled")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    # Should have multiple ADDs (unrolled)
    assert 'ADD' in counts
    assert counts['ADD'] >= 3  # At least 3 adds for 4 iterations
    
    print(f"✓ Unrolled loop built with {len(ir.blocks)} blocks, ADD={counts.get('ADD',0)}")
    print("="*60)


def test_struct_decorator():
    """Test @struct decorator."""
    print("\n" + "="*60)
    print("Test: @struct decorator")
    print("="*60)
    
    @struct
    class Particle:
        position: float3
        mass: float32
    
    assert hasattr(Particle, '_dsl_type')
    assert Particle._dsl_type.name == 'Particle'
    assert 'position' in Particle._dsl_fields
    assert 'mass' in Particle._dsl_fields
    
    print("✓ @struct decorator works correctly")
    print("="*60)


def test_struct_with_buffer_kernel():
    """Test using struct with buffer in kernel."""
    print("\n" + "="*60)
    print("Test: struct with buffer kernel")
    print("="*60)
    
    @struct
    class Particle:
        position: float3
        velocity: float3
        mass: float32
    
    @kernel
    def update_particles(particles: Buffer(Particle)) -> None:
        idx = dispatch_id().x
        # Read particle
        p = particles[idx]
        # Simple update (in real code would do physics)
        particles[idx] = p
    
    buf_type = Buffer(Particle)
    ir = update_particles(buf_type)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.name == 'update_particles'
    assert ir.is_kernel
    assert len(ir.blocks) > 0
    
    print(f"✓ Struct buffer kernel built with {len(ir.blocks)} blocks")
    print("="*60)


def test_nested_unrolled():
    """Test nested unrolled loops."""
    print("\n" + "="*60)
    print("Test: nested unrolled loops")
    print("="*60)
    
    @callable
    def nested_sum() -> int32:
        total = int32(0)
        for i in unrolled(range(3)):
            for j in unrolled(range(3)):
                total = total + int32(i) + int32(j)
        return total
    
    ir = nested_sum()
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    # Should have many ADDs from unrolled nested loops
    assert 'ADD' in counts
    
    print(f"✓ Nested unrolled loops: {len(ir.blocks)} blocks, {counts.get('ADD',0)} ADDs")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_utils.py tests")
    print("="*70)
    
    test_unrolled_range()
    test_unrolled_builds_ir()
    test_struct_decorator()
    test_struct_with_buffer_kernel()
    test_nested_unrolled()
    
    print("\n" + "="*70)
    print("All test_utils.py tests passed!")
    print("="*70)
