"""
Test demonstrating multistage programming with a polymorphic dispatch system.
Inspired by luisa/dsl/polymorphic.h
"""

import pytest
from luisa import (
    kernel, callable, static_range,
    Int, Float, Buffer, dispatch_id,
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

    def dispatch(self, tag_value, *args):
        """
        Multistage dispatch:
        Loop over registered implementations on the host
        and generate a case for each in the IR.
        """
        from luisa.lang.builder import get_current_builder
        builder = get_current_builder()
        sw = builder.switch(tag_value)
        # Host-side loop: expanded during IR generation
        for i, impl in enumerate(self.impls):
            with sw.case_scope(i):
                # Call the internal builder_func to add instructions to this case
                impl.builder_func(*args)

        with sw.default_scope():
            # Optional: handle invalid tags
            pass


def test_multistage_polymorphic_dispatch():
    """Test polymorphic dispatch using multistage programming."""
    print("\n" + "=" * 60)
    print("Test: Multistage Polymorphic Dispatch")
    print("=" * 60)

    poly = Polymorphic()

    @callable
    def add_one(x: Buffer[Float], idx: Int):
        x[idx] = x[idx] + 1.0

    @callable
    def multiply_two(x: Buffer[Float], idx: Int):
        x[idx] = x[idx] * 2.0

    @callable
    def square(x: Buffer[Float], idx: Int):
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
    def dispatch_kernel(buf: Buffer[Float], tags: Buffer[Int]):
        idx = dispatch_id().x
        tag = tags[idx]

        # Use the host-side helper to generate IR dispatch
        # But we can also use a helper that knows how to get the builder.
        poly.dispatch(tag, buf, idx)

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
    print("=" * 60)


def test_nested_polymorphic_callables():
    """Test defining polymorphic callables nested inside a kernel."""
    print("\n" + "=" * 60)
    print("Test: Nested Polymorphic Callables")
    print("=" * 60)

    @kernel
    def nested_dispatch_kernel(buf: Buffer[Float], tags: Buffer[Int]):
        idx = dispatch_id().x
        tag = tags[idx]

        @callable
        def add_one(x: Float) -> Float:
            return x + 1.0

        @callable
        def multiply_two(x: Float) -> Float:
            return x * 2.0

        @callable
        def square(x: Float) -> Float:
            return x * x

        # Simple dispatch logic using host-side loop
        val = buf[idx]
        from luisa.lang.builder import get_current_builder
        builder = get_current_builder()

        sw = builder.switch(tag)
        impls = [add_one, multiply_two, square]
        for i, impl in enumerate(impls):
            with sw.case_scope(i):
                # Use the builder.call() method which now handles StagedFunction objects
                res = builder.call(impl, val)
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
    print("=" * 60)


if __name__ == "__main__":
    test_multistage_polymorphic_dispatch()
    test_nested_polymorphic_callables()
