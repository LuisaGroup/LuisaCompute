"""
Test demonstrating reference argument support in the LuisaCompute Python DSL v2.
"""

import pytest
import sys
import os

# Ensure local luisa package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from luisa import (
    kernel, callable,
    int32, Buffer, dispatch_id, Ref,
    pprint
)

def test_reference_argument_basic():
    """Test basic reference argument support."""
    print("\n" + "="*60)
    print("Test: Reference Argument Basic")
    print("="*60)
    
    @callable
    def increment(x: Ref[int32]):
        # x is a Ref[int32]
        x = x + 1
        
    @kernel
    def ref_kernel(buf: Buffer[int32]):
        idx = dispatch_id().x
        val = buf[idx]
        increment(val)
        buf[idx] = val

    # Provide a typed buffer so indexing works
    ir = ref_kernel(Buffer[int32])
    
    # Get the increment IR from cache
    inc_ir = list(increment._cache.values())[0]
    
    print("\nGenerated IR for increment:")
    print(pprint(inc_ir))
    
    print("\nGenerated IR for ref_kernel:")
    print(pprint(ir))
    
    print("✓ Reference argument IR generated")
    print("="*60)

def test_swap_references():
    """Test swapping values using reference arguments."""
    print("\n" + "="*60)
    print("Test: Swap via References")
    print("="*60)
    
    @callable
    def swap(a: Ref[int32], b: Ref[int32]):
        tmp = a
        a = b
        b = tmp
        
    @kernel
    def swap_kernel(buf: Buffer[int32]):
        idx = dispatch_id().x
        a = buf[idx*2]
        b = buf[idx*2 + 1]
        swap(a, b)
        buf[idx*2] = a
        buf[idx*2 + 1] = b

    ir = swap_kernel(Buffer[int32])
    
    # Get swap IR from cache
    swap_ir = list(swap._cache.values())[0]
    
    print("\nGenerated IR for swap:")
    print(pprint(swap_ir))
    
    print("\nGenerated IR for swap_kernel:")
    print(pprint(ir))
    
    print("✓ Swap via references built successfully")
    print("="*60)

if __name__ == "__main__":
    test_reference_argument_basic()
    test_swap_references()
