"""
Test demonstrating reference argument support in the LuisaCompute Python DSL v2.
"""

import pytest
from luisa import (
    kernel, callable,
    Int, Buffer, dispatch_id, Ref
)


def test_reference_argument_basic(verify_ir):
    """Test basic reference argument support."""
    @callable
    def increment(x: Ref[Int]):
        # x is a Ref[Int]
        x = x + 1

    @kernel
    def ref_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        val = buf[idx]
        increment(val)
        buf[idx] = val

    expected = """
kernel void ref_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i32 v2 = buffer_read(arg0, v1);
  i32 val = alloca();
  store(val, v2);
  i32 v5 = load(val);
  call(@increment, v5);
  i32 v7 = load(val);
  buffer_write(arg0, v1, v7);
}

void increment(i32 arg0) {
  i32 v0 = load(arg0);
  i32 v1 = add(v0, 1);
  store(arg0, v1);
}
"""
    verify_ir(ref_kernel, expected)


def test_swap_references(verify_ir):
    """Test swapping values using reference arguments."""
    @callable
    def swap(a: Ref[Int], b: Ref[Int]):
        tmp = a
        a = b
        b = tmp

    @kernel
    def swap_kernel(buf: Buffer[Int]):
        idx = dispatch_id().x
        a = buf[idx * 2]
        b = buf[idx * 2 + 1]
        swap(a, b)
        buf[idx * 2] = a
        buf[idx * 2 + 1] = b

    expected = """
kernel void swap_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 v2 = mul(v1, 2);
  i32 v3 = buffer_read(arg0, v2);
  i32 va = alloca();
  store(va, v3);
  u32 v6 = mul(v1, 2);
  u32 v7 = add(v6, 1);
  i32 v8 = buffer_read(arg0, v7);
  i32 vb = alloca();
  store(vb, v8);
  i32 v11 = load(va);
  i32 v12 = load(vb);
  call(@swap, v11, v12);
  u32 v14 = mul(v1, 2);
  i32 v15 = load(va);
  buffer_write(arg0, v14, v15);
  u32 v17 = mul(v1, 2);
  u32 v18 = add(v17, 1);
  i32 v19 = load(vb);
  buffer_write(arg0, v18, v19);
}

void swap(i32 arg0, i32 arg1) {
  i32 v0 = load(arg0);
  i32 vtmp = alloca();
  store(vtmp, v0);
  i32 v3 = load(arg1);
  store(arg0, v3);
  i32 v5 = load(vtmp);
  store(arg1, v5);
}
"""
    verify_ir(swap_kernel, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
