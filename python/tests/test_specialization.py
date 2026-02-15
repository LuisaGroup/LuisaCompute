"""
Test demonstrating DSL specialization (generics/templates).
"""

import pytest
from luisa import (
    kernel, callable,
    Int, Float, Buffer, dispatch_id,
    pprint
)
from luisa.lang.inspect import analyze_control_flow


def test_callable_specialization():
    """Test specialized callable functions."""
    print("\n" + "=" * 60)
    print("Test: Callable Specialization")
    print("=" * 60)

    # Define a specialized callable
    @callable['T', 'i']
    def add_offset(a: T):
        # T will be replaced by Int/Float
        # i will be replaced by the constant value
        return a + i

    # Test with Int and offset 5
    ir_int = add_offset[Int, 5](10)
    print("\nGenerated IR (Int, 5):")
    print(pprint(ir_int))

    # Verify IR contains addition
    from luisa.lang.inspect import count_instructions
    counts = count_instructions(ir_int)
    assert 'ADD' in counts

    # Test with Float and offset 1.5
    ir_float = add_offset[Float, 1.5](10.0)
    print("\nGenerated IR (Float, 1.5):")
    print(pprint(ir_float))

    counts = count_instructions(ir_float)
    assert 'ADD' in counts

    print("✓ Specialized callables built successfully")
    print("=" * 60)


def test_kernel_specialization():
    """Test specialized kernels."""
    print("\n" + "=" * 60)
    print("Test: Kernel Specialization")
    print("=" * 60)

    @kernel['BLOCK_SIZE']
    def tiled_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        if idx < BLOCK_SIZE:
            buf[idx] = Float(idx)

    # Compile with BLOCK_SIZE = 64
    ir_64 = tiled_kernel[64](None)
    print("\nGenerated IR (BLOCK_SIZE=64):")
    print(pprint(ir_64))

    # Compile with BLOCK_SIZE = 128
    ir_128 = tiled_kernel[128](None)
    print("\nGenerated IR (BLOCK_SIZE=128):")
    print(pprint(ir_128))

    print("✓ Specialized kernels built successfully")
    print("=" * 60)


if __name__ == "__main__":
    test_callable_specialization()
    test_kernel_specialization()
