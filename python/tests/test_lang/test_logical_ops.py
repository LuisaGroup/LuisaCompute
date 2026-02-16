"""Tests for logical operations and chained comparisons."""

import pytest
from luisa import (
    kernel, callable,
    Int, Bool, Buffer
)


def test_logical_and_short_circuit(verify_ir):
    """Test logical AND with short-circuiting."""
    @callable
    def logic_and(a: Bool, b: Bool) -> Bool:
        return a and b

    logic_and(True, True)
    
    expected = """
i1 logic_and(i1 arg0, i1 arg1) {
  i1 v0 = alloca();
  store(v0, arg0);
  if (arg0) {
    store(v0, arg1);
  } else {
    (empty)
  }
  i1 v4 = load(v0);
  return v4;
}
"""
    verify_ir(logic_and, expected)


def test_logical_or_short_circuit(verify_ir):
    """Test logical OR with short-circuiting."""
    @callable
    def logic_or(a: Bool, b: Bool) -> Bool:
        return a or b

    logic_or(True, True)
    
    expected = """
i1 logic_or(i1 arg0, i1 arg1) {
  i1 v0 = alloca();
  store(v0, arg0);
  if (arg0) {
    (empty)
  } else {
    store(v0, arg1);
  }
  i1 v4 = load(v0);
  return v4;
}
"""
    verify_ir(logic_or, expected)


def test_chained_comparisons(verify_ir):
    """Test chained comparisons like x < a < y."""
    @callable
    def chain_comp(x: Int, a: Int, y: Int) -> Bool:
        return x < a < y

    chain_comp(0, 5, 10)
    
    expected = """
i1 chain_comp(i32 arg0, i32 arg1, i32 arg2) {
  i1 v0 = lt(arg0, arg1);
  i1 v1 = alloca();
  store(v1, v0);
  if (v0) {
    i1 v4 = lt(arg1, arg2);
    store(v1, v4);
  } else {
    (empty)
  }
  i1 v6 = load(v1);
  return v6;
}
"""
    verify_ir(chain_comp, expected)


def test_complex_logic(verify_ir):
    """Test complex logical expressions."""
    @callable
    def complex_logic(a: Bool, b: Bool, c: Bool) -> Bool:
        return (a and b) or (not a and c)

    complex_logic(True, False, True)
    
    expected = """
i1 complex_logic(i1 arg0, i1 arg1, i1 arg2) {
  i1 v0 = alloca();
  store(v0, arg0);
  if (arg0) {
    store(v0, arg1);
  } else {
    (empty)
  }
  i1 v4 = load(v0);
  i1 v5 = alloca();
  store(v5, v4);
  if (v4) {
    (empty)
  } else {
    i1 v8 = logical_not(arg0);
    i1 v9 = alloca();
    store(v9, v8);
    if (v8) {
      store(v9, arg2);
    } else {
      (empty)
    }
    i1 v13 = load(v9);
    store(v5, v13);
  }
  i1 v15 = load(v5);
  return v15;
}
"""
    verify_ir(complex_logic, expected)


def test_logic_with_side_effects(verify_ir):
    """Test logical ops where the RHS has 'side effects' (buffer write)."""
    @callable
    def effect(buf: Buffer[Int], idx: Int) -> Bool:
        buf[idx] = 1
        return True

    @kernel
    def logic_kernel(a: Bool, buf: Buffer[Int]):
        if a and effect(buf, 0):
            pass

    logic_kernel(True, None)
    
    expected = """
kernel void logic_kernel(i1 arg0, buffer<i32> arg1) {
  i1 v0 = alloca();
  store(v0, arg0);
  if (arg0) {
    i1 v3 = call(@effect, arg1, 0);
    store(v0, v3);
  } else {
    (empty)
  }
  i1 v5 = load(v0);
  if (v5) {
    (empty)
  } else {
    (empty)
  }
}

i1 effect(buffer<i32> arg0, i32 arg1) {
  buffer_write(arg0, arg1, 1);
  return True;
}
"""
    verify_ir(logic_kernel, expected)


def test_chained_comparison_mixed(verify_ir):
    """Test chained comparisons with different operators."""
    @callable
    def mixed_chain(x: Int, a: Int, y: Int) -> Bool:
        return x <= a < y != 10

    mixed_chain(0, 5, 10)
    
    expected = """
i1 mixed_chain(i32 arg0, i32 arg1, i32 arg2) {
  i1 v0 = le(arg0, arg1);
  i1 v1 = alloca();
  store(v1, v0);
  if (v0) {
    i1 v4 = lt(arg1, arg2);
    i1 v5 = alloca();
    store(v5, v4);
    if (v4) {
      i1 v8 = ne(arg2, 10);
      store(v5, v8);
    } else {
      (empty)
    }
    i1 v10 = load(v5);
    store(v1, v10);
  } else {
    (empty)
  }
  i1 v12 = load(v1);
  return v12;
}
"""
    verify_ir(mixed_chain, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
