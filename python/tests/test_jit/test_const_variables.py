"""Tests for const variables and DSL variable assignment."""

from luisa import (
    kernel, callable, Float, Int, Buffer, Const, static,
    sin
)


def test_dsl_variable_reassignment(verify_ir):
    """Test that DSL variables can be reassigned."""
    @callable
    def test_reassign(x: Float) -> Float:
        # a is a DSL variable (not const)
        # Using the argument x (dynamic) ensures 'a' becomes a DSL variable
        a = x         # alloca + store
        # This should work - reassigning DSL variable
        a = a + 1.0   # load + add + store
        return a
    
    expected = """
f32 test_reassign(f32 arg0) {
  f32 va = alloca();
  store(va, arg0);
  f32 v2 = load(va);
  f32 v3 = add(v2, 1.0);
  store(va, v3);
  f32 v5 = load(va);
  return v5;
}
"""
    verify_ir(test_reassign, expected)


def test_const_variable(verify_ir):
    """Test that const variables are kept as Python values."""
    @callable
    def test_const_var(x: Float) -> Float:
        # b is a compile-time constant using static()
        b = static(sin(1.0))  # Should be Python float
        # This computes at compile time
        return x + b  # b is loaded as constant
    
    expected = """
f32 test_const_var(f32 arg0) {
  f32 v0 = add(arg0, 0.8414709848078965);
  return v0;
}
"""
    verify_ir(test_const_var, expected)


def test_mixed_const_and_dsl_vars(verify_ir):
    """Test mixing const and DSL variables."""
    @callable
    def test_mixed(x: Float) -> Float:
        # DSL variable (created from any assignment, for potential reassignment)
        a = sin(1.0)
        # Const variable using static()
        b = static(cos(1.0))
        # Or using Const
        c = Const[Float](0.5)
        # Use both
        a = a + b + c  # a is DSL var, b and c are Python const
        return a * 2.0
    
    # a is now a DSL variable (for correct handling of potential reassignment)
    # b and c are still Python constants (inlined)
    expected = """
f32 test_mixed(f32 arg0) {
  f32 va = alloca();
  store(va, 0.8414709848078965);
  f32 v2 = load(va);
  f32 v3 = add(v2, 0.5403023058681398);
  f32 v4 = add(v3, 0.5);
  store(va, v4);
  f32 v6 = load(va);
  f32 v7 = mul(v6, 2.0);
  return v7;
}
"""
    verify_ir(test_mixed, expected)


def test_const_with_arithmetic(verify_ir):
    """Test const variables in arithmetic expressions."""
    @callable
    def test_arith(x: Float) -> Float:
        # Multiple const values using static()
        c1 = static(1.0)
        c2 = static(2.0)
        # Arithmetic on consts happens at compile time
        c3 = c1 + c2  # This is 3.0 at Python compile time
        return x + c3
    
    # c3 is assigned from Python constant arithmetic, but still creates a DSL variable
    # for correct handling of potential reassignment
    expected = """
f32 test_arith(f32 arg0) {
  f32 vc3 = alloca();
  store(vc3, 3.0);
  f32 v2 = load(vc3);
  f32 v3 = add(arg0, v2);
  return v3;
}
"""
    verify_ir(test_arith, expected)


def test_const_typed_syntax(verify_ir):
    """Test Const[Type](value) syntax."""
    @callable
    def test_typed_const(x: Float) -> Float:
        # Using Const[Type](value) syntax - these are Python constants
        c1 = Const[Float](1.5)
        c2 = Const[Int](10)
        # Arithmetic: c1 is Python, Float(c2) is detected as DSL value
        # So c3 becomes a DSL variable
        c3 = c1 + Float(c2)
        return x + c3
    
    # c3 is a DSL variable because Float(c2) is detected as a DSL value
    expected = """
f32 test_typed_const(f32 arg0) {
  f32 vc3 = alloca();
  store(vc3, 11.5);
  f32 v2 = load(vc3);
  f32 v3 = add(arg0, v2);
  return v3;
}
"""
    verify_ir(test_typed_const, expected)


def test_const_multiple_values(verify_ir):
    """Test Const with multiple values."""
    @callable
    def test_multi(x: Float) -> Float:
        # Multiple values using static()
        a, b, c = static(1.0, 2.0, 3.0)
        # Or using Const with a vector type
        from luisa import Float3
        vals = Const[Float3](1.0, 2.0, 3.0)
        return x + a + vals.x
    
    expected = """
f32 test_multi(f32 arg0) {
  f32 v0 = add(arg0, 1.0);
  f32 v1 = add(v0, 1.0);
  return v1;
}
"""
    verify_ir(test_multi, expected)


def test_dsl_var_in_kernel(verify_ir):
    """Test DSL variables in kernel context."""
    @kernel
    def test_kernel(buf: Buffer[Float]):
        # DSL variable that gets reassigned
        # Reading from a buffer ensures it's a dynamic DSL value
        val = buf[0]      # Creates DSL variable (alloca)
        val = val + 1.0   # Reassign (load + add + store)
        buf[0] = val
    
    expected = """
kernel void test_kernel(buffer<f32> arg0) {
  f32 v0 = buffer_read(arg0, 0);
  f32 val = alloca();
  store(val, v0);
  f32 v3 = load(val);
  f32 v4 = add(v3, 1.0);
  store(val, v4);
  f32 v6 = load(val);
  buffer_write(arg0, 0, v6);
}
"""
    verify_ir(test_kernel, expected)


def test_multiple_reassignments(verify_ir):
    """Test multiple reassignments of DSL variable."""
    @callable
    def test_multi(x: Float) -> Float:
        a = 0.0
        a = a + 1.0
        a = a + 2.0
        a = a * 2.0
        return a
    
    # a is a DSL variable, so each assignment is a store/load cycle
    expected = """
f32 test_multi(f32 arg0) {
  f32 va = alloca();
  store(va, 0.0);
  f32 v2 = load(va);
  f32 v3 = add(v2, 1.0);
  store(va, v3);
  f32 v5 = load(va);
  f32 v6 = add(v5, 2.0);
  store(va, v6);
  f32 v8 = load(va);
  f32 v9 = mul(v8, 2.0);
  store(va, v9);
  f32 v11 = load(va);
  return v11;
}
"""
    verify_ir(test_multi, expected)


def test_const_init_and_reassign(verify_ir):
    """Test initializing with a constant and then reassigning with a DSL value."""
    @callable
    def test_init_reassign(x: Float) -> Float:
        # Initialized from a constant - now always creates a DSL variable
        # This ensures correct handling when reassigned in loops
        a = 1.0
        # Reassigned with a DSL value
        a = a + x
        return a
        
    # a is now always a DSL variable (for correct handling of potential reassignment)
    expected = """
f32 test_init_reassign(f32 arg0) {
  f32 va = alloca();
  store(va, 1.0);
  f32 v2 = load(va);
  f32 v3 = add(v2, arg0);
  store(va, v3);
  f32 v5 = load(va);
  return v5;
}
"""
    verify_ir(test_init_reassign, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
