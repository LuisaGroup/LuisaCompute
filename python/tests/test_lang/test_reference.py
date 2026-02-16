"""
Test demonstrating reference argument support in the LuisaCompute Python DSL v2.
"""

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

    # idx is now a DSL variable
    expected = """
kernel void ref_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  i32 v5 = buffer_read(arg0, v4);
  i32 val = alloca();
  store(val, v5);
  i32 v8 = load(val);
  call(@increment, v8);
  u32 v10 = load(vidx);
  i32 v11 = load(val);
  buffer_write(arg0, v10, v11);
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

    # idx is now a DSL variable
    expected = """
kernel void swap_kernel(buffer<i32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  u32 vidx = alloca();
  store(vidx, v1);
  u32 v4 = load(vidx);
  u32 v5 = mul(v4, 2);
  i32 v6 = buffer_read(arg0, v5);
  i32 va = alloca();
  store(va, v6);
  u32 v9 = load(vidx);
  u32 v10 = mul(v9, 2);
  u32 v11 = add(v10, 1);
  i32 v12 = buffer_read(arg0, v11);
  i32 vb = alloca();
  store(vb, v12);
  i32 v15 = load(va);
  i32 v16 = load(vb);
  call(@swap, v15, v16);
  u32 v18 = load(vidx);
  u32 v19 = mul(v18, 2);
  i32 v20 = load(va);
  buffer_write(arg0, v19, v20);
  u32 v22 = load(vidx);
  u32 v23 = mul(v22, 2);
  u32 v24 = add(v23, 1);
  i32 v25 = load(vb);
  buffer_write(arg0, v24, v25);
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
