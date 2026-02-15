"""
Test demonstrating multistage programming with a polymorphic dispatch system.
Inspired by luisa/dsl/polymorphic.h
"""

import pytest
from luisa import (
    kernel, callable, static_range,
    int32, float32, Buffer, dispatch_id,
    pprint
)
from luisa.lang.inspect import analyze_control_flow

class Polymorphic:
    """
    A host-side helper that manages multiple DSL implementations
    and generates a structured SWITCH for dispatch.
    """
    def __init__(self):
        self.impls = []
    
    def register(self, func):
        tag = len(self.impls)
        # Ensure it's compiled so builder_func is available
        func._do_compile()
        self.impls.append(func)
        return tag
    
    def dispatch(self, builder, tag_value, *args):
        """
        Multistage dispatch:
        Loop over registered implementations on the host
        and generate a case for each in the IR.
        """
        sw = builder.switch(tag_value)
        # Host-side loop: expanded during IR generation
        for i, impl in enumerate(self.impls):
            with sw.case_scope(i):
                # Call the internal builder_func to add instructions to this case
                impl.builder_func(builder, *args)
        
        with sw.default_scope():
            # Optional: handle invalid tags
            pass

def test_multistage_polymorphic_dispatch():
    """Test polymorphic dispatch using multistage programming."""
    print("\n" + "="*60)
    print("Test: Multistage Polymorphic Dispatch")
    print("="*60)
    
    poly = Polymorphic()
    
    @callable
    def add_one(x: Buffer[float32], idx: int32):
        x[idx] = x[idx] + 1.0
        
    @callable
    def multiply_two(x: Buffer[float32], idx: int32):
        x[idx] = x[idx] * 2.0
        
    @callable
    def square(x: Buffer[float32], idx: int32):
        val = x[idx]
        x[idx] = val * val

    # Register implementations on the host
    tag_add = poly.register(add_one)
    tag_mul = poly.register(multiply_two)
    tag_square = poly.register(square)
    
    assert tag_add == 0
    assert tag_mul == 1
    assert tag_square == 2

    @kernel
    def dispatch_kernel(buf: Buffer[float32], tags: Buffer[int32]):
        idx = dispatch_id().x
        tag = tags[idx]
        
        # Use the host-side helper to generate IR dispatch
        # We need access to the builder. In the rewritten AST, 
        # the builder is passed as the first argument to the built function.
        # But we can also use a helper that knows how to get the builder.
        from luisa.lang.builder import get_current_builder
        
        poly.dispatch(get_current_builder(), tag, buf, idx)

    # Build IR
    ir = dispatch_kernel(None, None)
    
    print("\nGenerated IR Summary:")
    cf = analyze_control_flow(ir)
    print(f"  Blocks: {cf['blocks']}")
    print(f"  Switches: {cf['switches']}")
    print(f"  Has Loops: {cf['has_loops']}")
    
    # Check if we have a switch with 3 cases
    assert cf['switches'] == 1
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    print("✓ Multistage polymorphic dispatch built successfully")
    print("="*60)

def test_python_match_to_switch():
    """Test that Python's match statement is translated to IR SWITCH."""
    print("\n" + "="*60)
    print("Test: Python match to IR SWITCH")
    print("="*60)
    
    @callable
    def match_test(tag: int32) -> int32:
        res = int32(0)
        match tag:
            case 0:
                res = int32(10)
            case 1:
                res = int32(20)
            case 2:
                res = int32(30)
            case _:
                res = int32(-1)
        return res

    ir = match_test(0)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['switches'] == 1
    
    print("✓ Python match translated to IR SWITCH")
    print("="*60)

def test_nested_polymorphic_callables():
    """Test defining polymorphic callables nested inside a kernel."""
    print("\n" + "="*60)
    print("Test: Nested Polymorphic Callables")
    print("="*60)

    @kernel
    def nested_dispatch_kernel(buf: Buffer[float32], tags: Buffer[int32]):
        idx = dispatch_id().x
        tag = tags[idx]

        @callable
        def add_one(x: float32) -> float32:
            return x + 1.0
            
        @callable
        def multiply_two(x: float32) -> float32:
            return x * 2.0
            
        @callable
        def square(x: float32) -> float32:
            return x * x

        # Simple dispatch logic using host-side loop
        val = buf[idx]
        from luisa.lang.builder import get_current_builder
        builder = get_current_builder()
        
        sw = builder.switch(tag)
        impls = [add_one, multiply_two, square]
        for i, impl in enumerate(impls):
            with sw.case_scope(i):
                # Use the new call() method which handles compilation and emitting the call
                res = impl.call(builder, val)
                builder.buffer_write(buf, idx, res)
        
        with sw.default_scope():
            pass

    # Build IR
    ir = nested_dispatch_kernel(None, None)
    
    print("\nGenerated IR Summary:")
    cf = analyze_control_flow(ir)
    print(f"  Blocks: {cf['blocks']}")
    print(f"  Switches: {cf['switches']}")
    
    # Check if we have a switch
    assert cf['switches'] == 1
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    print("✓ Nested polymorphic callables built successfully")
    print("="*60)

if __name__ == "__main__":
    test_multistage_polymorphic_dispatch()
    test_python_match_to_switch()
    test_nested_polymorphic_callables()
