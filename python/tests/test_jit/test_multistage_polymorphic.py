"""
Test demonstrating multistage programming with a polymorphic dispatch system.
Inspired by luisa/dsl/polymorphic.h
"""

import pytest
from luisa import (
    kernel, callable,
    Int, Float, Buffer, dispatch_id
)
from luisa.lang.builtins.math import sin


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
        from luisa.transform.builder import get_current_builder
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


def test_multistage_polymorphic_dispatch(verify_ir):
    """Test polymorphic dispatch using multistage programming."""
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
    poly.register(add_one)
    poly.register(multiply_two)
    poly.register(square)

    @kernel
    def dispatch_kernel(buf: Buffer[Float], tags: Buffer[Int]):
        idx = dispatch_id().x
        tag = tags[idx]

        # Use the host-side helper to generate IR dispatch
        poly.dispatch(tag, buf, idx)

    expected = """
kernel void dispatch_kernel(buffer<f32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = buffer_read(arg1, v1);
  i32 vtag = alloca();
  store(vtag, v2);
  i32 v5 = load(vtag);
  switch (v5) { 
    case 0: {
      f32 v7 = buffer_read(arg0, v1);
      f32 v8 = add(v7, 1.0);
      buffer_write(arg0, v1, v8);
    }
    case 1: {
      f32 v10 = buffer_read(arg0, v1);
      f32 v11 = mul(v10, 2.0);
      buffer_write(arg0, v1, v11);
    }
    case 2: {
      f32 v13 = buffer_read(arg0, v1);
      f32 val = alloca();
      store(val, v13);
      f32 v16 = load(val);
      f32 v17 = load(val);
      f32 v18 = mul(v16, v17);
      buffer_write(arg0, v1, v18);
    }
  }
}
"""
    verify_ir(dispatch_kernel, expected)


def test_nested_polymorphic_callables(verify_ir):
    """Test defining polymorphic callables nested inside a kernel."""
    @kernel
    def nested_dispatch_kernel(buf: Buffer[Float], tags: Buffer[Int]):
        idx = dispatch_id().x
        tag = tags[idx]

        @callable
        def add_one(x: Float) -> Float:
            return x + sin(1.0 + 2.0)

        @callable
        def multiply_two(x: Float) -> Float:
            return x * 2.0

        @callable
        def square(x: Float) -> Float:
            return x * x

        # Simple dispatch logic using host-side loop
        val = buf[idx]
        from luisa.transform.builder import get_current_builder
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

    expected = """
kernel void nested_dispatch_kernel(buffer<f32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = buffer_read(arg1, v1);
  i32 vtag = alloca();
  store(vtag, v2);
  f32 v5 = buffer_read(arg0, v1);
  f32 val = alloca();
  store(val, v5);
  i32 v8 = load(vtag);
  switch (v8) { 
    case 0: {
      f32 v10 = load(val);
      f32 v11 = call(@add_one, v10);
      buffer_write(arg0, v1, v11);
    }
    case 1: {
      f32 v13 = load(val);
      f32 v14 = call(@multiply_two, v13);
      buffer_write(arg0, v1, v14);
    }
    case 2: {
      f32 v16 = load(val);
      f32 v17 = call(@square, v16);
      buffer_write(arg0, v1, v17);
    }
  }
}

f32 add_one(f32 arg0) {
  f32 v0 = add(arg0, 0.1411200080598672);
  return v0;
}

f32 multiply_two(f32 arg0) {
  f32 v0 = mul(arg0, 2.0);
  return v0;
}

f32 square(f32 arg0) {
  f32 v0 = mul(arg0, arg0);
  return v0;
}
"""
    verify_ir(nested_dispatch_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
