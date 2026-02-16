"""Tests for unrolled loops."""

import pytest
from luisa import kernel, callable, Float, Int, Buffer, unrolled, static_range
from luisa.lang.inspect import count_instructions


def test_unrolled_simple(verify_ir):
    """Test simple unrolled loop."""
    @callable
    def unrolled_sum(buf: Buffer[Float], vals: Buffer[Float]) -> None:
        total = 0.0
        for i in static_range(4):
            # Using vals[i] prevents host-side constant folding of the addition
            total = total + vals[i]
        buf[0] = total

    ir = unrolled_sum(0, 0)
    
    expected = """
void unrolled_sum(buffer<f32> arg0, buffer<f32> arg1) {
  f32 v0 = buffer_read(arg1, 0);
  f32 v1 = add(0.0, v0);
  f32 vtotal = alloca();
  store(vtotal, v1);
  f32 v4 = load(vtotal);
  f32 v5 = buffer_read(arg1, 1);
  f32 v6 = add(v4, v5);
  f32 vtotal = alloca();
  store(vtotal, v6);
  f32 v9 = load(vtotal);
  f32 v10 = buffer_read(arg1, 2);
  f32 v11 = add(v9, v10);
  f32 vtotal = alloca();
  store(vtotal, v11);
  f32 v14 = load(vtotal);
  f32 v15 = buffer_read(arg1, 3);
  f32 v16 = add(v14, v15);
  f32 vtotal = alloca();
  store(vtotal, v16);
  f32 v19 = load(vtotal);
  buffer_write(arg0, 0, v19);
}
"""
    verify_ir(ir, expected)


def test_unrolled_with_captured_constant(verify_ir):
    """Test unrolled loop with captured constant."""
    UNROLL_COUNT = 3

    @callable
    def unrolled_with_capture(buf: Buffer[Float], val: Float) -> None:
        for i in static_range(UNROLL_COUNT):
            # Using dynamic val ensures ADD/BUFFER_WRITE are in IR
            buf[i] = val + Float(i)

    ir = unrolled_with_capture(0, 1.0)
    
    expected = """
void unrolled_with_capture(buffer<f32> arg0, f32 arg1) {
  f32 v0 = add(arg1, 0.0);
  buffer_write(arg0, 0, v0);
  f32 v2 = add(arg1, 1.0);
  buffer_write(arg0, 1, v2);
  f32 v4 = add(arg1, 2.0);
  buffer_write(arg0, 2, v4);
}
"""
    verify_ir(ir, expected)


def test_unrolled_with_computation(verify_ir):
    """Test unrolled loop with computation."""
    @callable
    def unrolled_compute(buf: Buffer[Float], val: Float) -> None:
        for i in static_range(4):
            buf[i] = val * Float(i) + 1.0

    ir = unrolled_compute(0, 1.0)
    
    expected = """
void unrolled_compute(buffer<f32> arg0, f32 arg1) {
  f32 v0 = mul(arg1, 0.0);
  f32 v1 = add(v0, 1.0);
  buffer_write(arg0, 0, v1);
  f32 v3 = mul(arg1, 1.0);
  f32 v4 = add(v3, 1.0);
  buffer_write(arg0, 1, v4);
  f32 v6 = mul(arg1, 2.0);
  f32 v7 = add(v6, 1.0);
  buffer_write(arg0, 2, v7);
  f32 v9 = mul(arg1, 3.0);
  f32 v10 = add(v9, 1.0);
  buffer_write(arg0, 3, v10);
}
"""
    verify_ir(ir, expected)


def test_unrolled_with_step(verify_ir):
    """Test unrolled loop with step."""
    @callable
    def unrolled_step(buf: Buffer[Float], val: Float) -> None:
        for i in static_range(0, 8, 2):  # 0, 2, 4, 6
            buf[i // 2] = val + Float(i)

    ir = unrolled_step(0, 1.0)
    
    expected = """
void unrolled_step(buffer<f32> arg0, f32 arg1) {
  f32 v0 = add(arg1, 0.0);
  buffer_write(arg0, 0, v0);
  f32 v2 = add(arg1, 2.0);
  buffer_write(arg0, 1, v2);
  f32 v4 = add(arg1, 4.0);
  buffer_write(arg0, 2, v4);
  f32 v6 = add(arg1, 6.0);
  buffer_write(arg0, 3, v6);
}
"""
    verify_ir(ir, expected)


def test_nested_unrolled(verify_ir):
    """Test nested unrolled loops."""
    @callable
    def nested_unrolled(buf: Buffer[Float], val: Float) -> None:
        for i in static_range(2):
            for j in static_range(2):
                idx = i * 2 + j
                buf[idx] = val + Float(i + j)

    ir = nested_unrolled(0, 1.0)
    
    expected = """
void nested_unrolled(buffer<f32> arg0, f32 arg1) {
  f32 v0 = add(arg1, 0.0);
  buffer_write(arg0, 0, v0);
  f32 v2 = add(arg1, 1.0);
  buffer_write(arg0, 1, v2);
  f32 v4 = add(arg1, 1.0);
  buffer_write(arg0, 2, v4);
  f32 v6 = add(arg1, 2.0);
  buffer_write(arg0, 3, v6);
}
"""
    verify_ir(ir, expected)
