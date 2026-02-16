"""
Test demonstrating DSL specialization (generics/templates).
"""

from luisa import (
    kernel, callable,
    Int, Float, Buffer, dispatch_id
)


def test_callable_specialization(verify_ir):
    """Test specialized callable functions."""
    # Define a specialized callable
    @callable['T', 'i']
    def add_offset(a: T):
        return a + i

    # Test with Int and offset 5
    ir_int = add_offset[Int, 5]
    
    expected_int = """
void add_offset(i32 arg0) {
  i32 v0 = add(arg0, 5);
  return v0;
}
"""
    verify_ir(ir_int, expected_int)

    # Test with Float and offset 1.5
    ir_float = add_offset[Float, 1.5]
    
    expected_float = """
void add_offset(f32 arg0) {
  f32 v0 = add(arg0, 1.5);
  return v0;
}
"""
    verify_ir(ir_float, expected_float)


def test_kernel_specialization(verify_ir):
    """Test specialized kernels."""
    @kernel['BLOCK_SIZE']
    def tiled_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        if idx < BLOCK_SIZE:
            buf[idx] = Float(idx)

    # Compile with BLOCK_SIZE = 64
    ir_64 = tiled_kernel[64]
    
    expected_64 = """
kernel void tiled_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i1 v2 = lt(v1, 64);
  if (v2) { 
    f32 v4 = cast(v1);
    buffer_write(arg0, v1, v4);
  } else {
    (empty)
  }
}
"""
    verify_ir(ir_64, expected_64)

    # Compile with BLOCK_SIZE = 128
    ir_128 = tiled_kernel[128]
    
    expected_128 = """
kernel void tiled_kernel(buffer<f32> arg0) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  i1 v2 = lt(v1, 128);
  if (v2) { 
    f32 v4 = cast(v1);
    buffer_write(arg0, v1, v4);
  } else {
    (empty)
  }
}
"""
    verify_ir(ir_128, expected_128)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
