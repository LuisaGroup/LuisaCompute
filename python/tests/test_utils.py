"""Tests for utility functions."""

import pytest
from luisa import (
    unrolled, UnrolledRange,
    struct,
    int32, float32, float3,
)


def test_unrolled_range():
    """Test UnrolledRange class."""
    print("Testing UnrolledRange...")
    
    # Test with single argument
    ur = UnrolledRange(4)
    assert ur.start == 0
    assert ur.stop == 4
    assert ur.step == 1
    assert len(ur) == 4
    
    # Test with all arguments
    ur = UnrolledRange(1, 10, 2)
    assert ur.start == 1
    assert ur.stop == 10
    assert ur.step == 2
    
    print("  ✓ UnrolledRange OK")


def test_unrolled_helper():
    """Test unrolled() helper function."""
    print("Testing unrolled helper...")
    
    r = range(0, 8, 2)
    ur = unrolled(r)
    
    assert isinstance(ur, UnrolledRange)
    assert ur.start == 0
    assert ur.stop == 8
    assert ur.step == 2
    
    print("  ✓ unrolled helper OK")


def test_struct_decorator():
    """Test @struct decorator."""
    print("Testing @struct decorator...")
    
    @struct
    class Particle:
        position: float3
        mass: float32
    
    # Check that struct was created
    assert hasattr(Particle, '_dsl_type')
    assert Particle._dsl_type.name == 'Particle'
    
    # Check fields
    assert 'position' in Particle._dsl_fields
    assert 'mass' in Particle._dsl_fields
    
    print("  ✓ @struct decorator OK")


def test_struct_with_buffer():
    """Test using struct with buffer."""
    print("Testing struct with buffer...")
    
    from luisa import Buffer, kernel
    
    @struct
    class Particle:
        position: float3
        velocity: float3
        mass: float32
    
    @kernel
    def update_particles(particles: Buffer(Particle)) -> None:
        # Just a placeholder - actual indexing would need full implementation
        pass
    
    # Should be able to create kernel
    buf_type = Buffer(Particle)
    ir_func = update_particles(buf_type)
    
    assert ir_func.name == 'update_particles'
    
    print("  ✓ Struct with buffer OK")
