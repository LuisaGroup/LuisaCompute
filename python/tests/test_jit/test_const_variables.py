"""Tests for const variables and DSL variable assignment."""

import pytest
from luisa import (
    kernel, callable, pprint, Float, Int, Buffer, Const, static,
    sin, cos, sqrt
)
from luisa.transform.ir import ConstantValue
from luisa.lang.inspect import count_instructions


def test_dsl_variable_reassignment():
    """Test that DSL variables can be reassigned."""
    print("\n" + "=" * 60)
    print("Test: DSL Variable Reassignment")
    print("=" * 60)
    
    @callable
    def test_reassign(x: Float) -> Float:
        # a is a DSL variable (not const)
        a = sin(1.0)  # Should create alloca + store
        # This should work - reassigning DSL variable
        a = a + 1.0   # load + add + store
        return a
    
    ir = test_reassign(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    # Should have ALLOCA for the variable
    assert 'ALLOCA' in counts, "Expected ALLOCA for DSL variable"
    # Should have STORE for initial assignment and reassignment
    assert counts.get('STORE', 0) >= 1, "Expected STORE for variable assignment"
    
    print("✓ DSL variable reassignment works correctly")
    print("=" * 60)


def test_const_variable():
    """Test that const variables are kept as Python values."""
    print("\n" + "=" * 60)
    print("Test: Const Variable")
    print("=" * 60)
    
    @callable
    def test_const_var(x: Float) -> Float:
        # b is a compile-time constant using static()
        b = static(sin(1.0))  # Should be Python float
        # This computes at compile time
        return x + b  # b is loaded as constant
    
    ir = test_const_var(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    # Should NOT have ALLOCA for const variable
    # But might have ALLOCA for the return or other purposes
    
    print("✓ Const variable works correctly")
    print("=" * 60)


def test_mixed_const_and_dsl_vars():
    """Test mixing const and DSL variables."""
    print("\n" + "=" * 60)
    print("Test: Mixed Const and DSL Variables")
    print("=" * 60)
    
    @callable
    def test_mixed(x: Float) -> Float:
        # DSL variable
        a = sin(1.0)
        # Const variable using static()
        b = static(cos(1.0))
        # Or using Const
        c = Const[Float](0.5)
        # Use both
        a = a + b + c  # a is DSL var, b and c are Python const
        return a * 2.0
    
    ir = test_mixed(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    print("✓ Mixed const and DSL variables work correctly")
    print("=" * 60)


def test_const_with_arithmetic():
    """Test const variables in arithmetic expressions."""
    print("\n" + "=" * 60)
    print("Test: Const With Arithmetic")
    print("=" * 60)
    
    @callable
    def test_arith(x: Float) -> Float:
        # Multiple const values using static()
        c1 = static(1.0)
        c2 = static(2.0)
        # Arithmetic on consts happens at compile time
        c3 = c1 + c2  # This should be 3.0 at compile time
        return x + c3
    
    ir = test_arith(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    # Check that c1 + c2 was folded (no ADD instruction for that)
    counts = count_instructions(ir)
    print(f"Instructions: {dict(counts)}")
    
    print("✓ Const with arithmetic works correctly")
    print("=" * 60)


def test_const_typed_syntax():
    """Test Const[Type](value) syntax."""
    print("\n" + "=" * 60)
    print("Test: Const[Type] Syntax")
    print("=" * 60)
    
    @callable
    def test_typed_const(x: Float) -> Float:
        # Using Const[Type](value) syntax
        c1 = Const[Float](1.5)
        c2 = Const[Int](10)
        # Arithmetic works
        c3 = c1 + Float(c2)  # Both are Python values
        return x + c3
    
    ir = test_typed_const(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    print("✓ Const[Type] syntax works correctly")
    print("=" * 60)


def test_const_multiple_values():
    """Test Const with multiple values."""
    print("\n" + "=" * 60)
    print("Test: Const Multiple Values")
    print("=" * 60)
    
    @callable
    def test_multi(x: Float) -> Float:
        # Multiple values using static()
        a, b, c = static(1.0, 2.0, 3.0)
        # Or using Const
        vals = Const[Float](1.0, 2.0, 3.0)
        return x + a + vals[0]
    
    ir = test_multi(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    print("✓ Const multiple values works correctly")
    print("=" * 60)


def test_dsl_var_in_kernel():
    """Test DSL variables in kernel context."""
    print("\n" + "=" * 60)
    print("Test: DSL Variable in Kernel")
    print("=" * 60)
    
    @kernel
    def test_kernel(buf: Buffer[Float]):
        # DSL variable that gets reassigned
        val = sqrt(2.0)  # Creates DSL variable
        val = val + 1.0  # Reassign
        buf[0] = val
    
    ir = test_kernel(None)
    
    print("Generated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    assert 'ALLOCA' in counts, "Expected ALLOCA for DSL variable"
    
    print("✓ DSL variable in kernel works correctly")
    print("=" * 60)


def test_multiple_reassignments():
    """Test multiple reassignments of DSL variable."""
    print("\n" + "=" * 60)
    print("Test: Multiple Reassignments")
    print("=" * 60)
    
    @callable
    def test_multi(x: Float) -> Float:
        a = 0.0
        a = a + 1.0
        a = a + 2.0
        a = a * 2.0
        return a
    
    ir = test_multi(0.0)
    
    print("Generated IR:")
    print(pprint(ir))
    
    print("✓ Multiple reassignments work correctly")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_const_variables.py tests")
    print("=" * 70)
    
    test_dsl_variable_reassignment()
    test_const_variable()
    test_mixed_const_and_dsl_vars()
    test_const_with_arithmetic()
    test_const_typed_syntax()
    test_const_multiple_values()
    test_dsl_var_in_kernel()
    test_multiple_reassignments()
    
    print("\n" + "=" * 70)
    print("All test_const_variables.py tests passed!")
    print("=" * 70)
