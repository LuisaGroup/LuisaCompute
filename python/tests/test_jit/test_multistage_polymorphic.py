"""
Test demonstrating multistage programming with a polymorphic dispatch system.
Inspired by luisa/dsl/polymorphic.h
"""

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

    # idx is now a DSL variable (for correct handling of potential reassignment)
    expected = """
kernel void dispatch_kernel(buffer<f32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = buffer_read(arg1, v4);
  i32 vtag = alloca();
  store(vtag, v5);
  i32 v8 = load(vtag);
  u32 v9 = load(vidx);
  switch (v8) { 
    case 0: {
      f32 v11 = buffer_read(arg0, v9);
      f32 v12 = add(v11, 1.0);
      buffer_write(arg0, v9, v12);
    }
    case 1: {
      f32 v14 = buffer_read(arg0, v9);
      f32 v15 = mul(v14, 2.0);
      buffer_write(arg0, v9, v15);
    }
    case 2: {
      f32 v17 = buffer_read(arg0, v9);
      f32 val = alloca();
      store(val, v17);
      f32 v20 = load(val);
      f32 v21 = load(val);
      f32 v22 = mul(v20, v21);
      buffer_write(arg0, v9, v22);
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

    # idx is now a DSL variable, val is a DSL variable, res is a DSL variable
    expected = """
kernel void nested_dispatch_kernel(buffer<f32> arg0, buffer<i32> arg1) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = buffer_read(arg1, v4);
  i32 vtag = alloca();
  store(vtag, v5);
  u32 v8 = load(vidx);
  f32 v9 = buffer_read(arg0, v8);
  f32 val = alloca();
  store(val, v9);
  i32 v12 = load(vtag);
  switch (v12) { 
    case 0: {
      f32 v14 = load(val);
      f32 v15 = call(@add_one, v14);
      f32 vres = alloca();
      store(vres, v15);
      u32 v18 = load(vidx);
      f32 v19 = load(vres);
      buffer_write(arg0, v18, v19);
    }
    case 1: {
      f32 v21 = load(val);
      f32 v22 = call(@multiply_two, v21);
      f32 vres = alloca();
      store(vres, v22);
      u32 v25 = load(vidx);
      f32 v26 = load(vres);
      buffer_write(arg0, v25, v26);
    }
    case 2: {
      f32 v28 = load(val);
      f32 v29 = call(@square, v28);
      f32 vres = alloca();
      store(vres, v29);
      u32 v32 = load(vidx);
      f32 v33 = load(vres);
      buffer_write(arg0, v32, v33);
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
