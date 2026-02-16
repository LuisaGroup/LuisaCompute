"""Tests for implicit template parameters (arguments without type annotations)."""

import pytest
from luisa import kernel, callable, Float, Int, Bool, Buffer
from luisa.lang.jit import StagedFunction, TemplatedFunction


# ============================================================================
# Pure Implicit Template Tests
# ============================================================================

def test_pure_implicit_template():
    """Test function with all implicit template params (no type annotations)."""
    @callable
    def identity(x):  # No type annotation -> implicit template
        return x
    
    # Should be a TemplatedFunction with implicit params
    assert isinstance(identity, TemplatedFunction)
    assert len(identity.explicit_params) == 0
    assert len(identity.implicit_params) == 1  # __impl_x
    
    # Can be called from a kernel - type inferred from argument
    @kernel
    def test_kernel():
        result = identity(Int(42))
    
    assert test_kernel.ir is not None
    
    # Check cache has the inferred type
    assert (Int,) in identity._cache
    staged = identity._cache[(Int,)]
    assert isinstance(staged, StagedFunction)
    assert staged.arg_types == (Int,)


def test_pure_implicit_multiple_args():
    """Test function with multiple implicit template params."""
    @callable
    def add(a, b):  # Both implicit templates
        return a + b
    
    assert isinstance(add, TemplatedFunction)
    assert len(add.implicit_params) == 2  # __impl_a, __impl_b
    
    @kernel
    def test_kernel():
        result = add(Int(1), Float(2.0))
    
    assert test_kernel.ir is not None
    
    # Check cache
    assert (Int, Float) in add._cache
    staged = add._cache[(Int, Float)]
    assert staged.arg_types == (Int, Float)


def test_pure_implicit_no_explicit_specialization():
    """Test that implicit-only templates cannot be explicitly specialized."""
    @callable
    def func(x, y):
        return x + y
    
    # Trying to use [] should raise an error
    with pytest.raises(TypeError, match="no explicit template parameters"):
        func[Int]


# ============================================================================
# Mixed Explicit and Implicit Template Tests
# ============================================================================

def test_mixed_explicit_implicit():
    """Test mixing explicit and implicit template params."""
    @callable['T']
    def func(a: T, b):  # T explicit, b implicit
        return a + T(b)
    
    assert isinstance(func, TemplatedFunction)
    assert func.explicit_params == ('T',)
    assert len(func.implicit_params) == 1  # __impl_b
    
    # Can specialize explicit param
    int_func = func[Int]
    assert isinstance(int_func, TemplatedFunction)
    assert int_func.specialization_values == (Int,)
    
    # Call from kernel - implicit param inferred from argument
    @kernel
    def test_kernel():
        result = int_func(Int(1), Float(2.0))
    
    assert test_kernel.ir is not None


def test_mixed_two_explicit_one_implicit():
    """Test the user's specific example: @callable['T', 'U'] def f(a: T, b, c: U)."""
    @callable['T', 'U']
    def f(a: T, b, c: U):  # T and U explicit, b implicit
        return T(a) + U(b) + c
    
    # Can specialize explicit params
    int_float_f = f[Int, Float]
    assert isinstance(int_float_f, TemplatedFunction)
    assert int_float_f.specialization_values == (Int, Float)
    
    # Call with different types for implicit param
    @kernel
    def test_kernel1():
        result = int_float_f(Int(1), Float(2.0), Float(3.0))  # b is Float
    
    assert test_kernel1.ir is not None
    
    @kernel
    def test_kernel2():
        result = int_float_f(Int(1), Int(2), Float(3.0))  # b is Int
    
    assert test_kernel2.ir is not None
    
    # Should have 2 cache entries for different implicit types
    assert len(int_float_f._cache) == 2


def test_mixed_partial_explicit_specialization():
    """Test partial explicit specialization with implicit param."""
    @callable['T', 'U']
    def func(a: T, b, c: U):  # T, U explicit; b implicit
        return a + T(b) + U(c)
    
    # Partial specialization: T=Int, U=?
    partial = func[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete specialization: T=Int, U=Float
    full = partial[Float]
    assert isinstance(full, TemplatedFunction)
    assert full.specialization_values == (Int, Float)
    
    # Call - implicit param b inferred
    @kernel
    def test_kernel():
        result = full(Int(1), Int(2), Float(3.0))
    
    assert test_kernel.ir is not None


# ============================================================================
# Implicit Template with Return Type Tests
# ============================================================================

def test_implicit_with_return_type():
    """Test implicit template with explicit return type."""
    @callable
    def to_float(x) -> Float:  # x is implicit template
        return Float(x)
    
    @kernel
    def test_kernel():
        result = to_float(Int(42))
    
    assert test_kernel.ir is not None
    
    # Check return type is Float
    staged = to_float._cache[(Int,)]
    assert staged.ret_type == Float


# ============================================================================
# Implicit Template Caching Tests
# ============================================================================

def test_implicit_template_caching():
    """Test that implicit template specializations are cached."""
    @callable
    def double(x):
        return x * 2
    
    # First call creates cache entry
    @kernel
    def test_kernel1():
        result = double(Int(1))
    
    assert test_kernel1.ir is not None
    cache_size = len(double._cache)
    assert cache_size == 1
    
    # Second call with same type reuses cache
    @kernel
    def test_kernel2():
        result = double(Int(2))
    
    assert test_kernel2.ir is not None
    assert len(double._cache) == cache_size
    
    # Call with different type creates new entry
    @kernel
    def test_kernel3():
        result = double(Float(3.0))
    
    assert test_kernel3.ir is not None
    assert len(double._cache) == cache_size + 1


# ============================================================================
# Kernel with Implicit Template Tests
# ============================================================================

def test_kernel_with_implicit_template():
    """Test kernel with implicit template param in callable."""
    @callable
    def scale(x, factor: Float):
        return x * factor
    
    @kernel
    def process(buf: Buffer[Float], n: Int):
        for i in range(n):
            val = scale(Float(i), 2.0)
            buf[i] = val
    
    assert process.ir is not None


# ============================================================================
# Edge Cases
# ============================================================================

def test_all_annotated_no_implicit():
    """Test that fully annotated function has no implicit params."""
    @callable
    def add(a: Int, b: Float) -> Float:
        return Float(a) + b
    
    # Should be StagedFunction immediately (no templates)
    assert isinstance(add, StagedFunction)


def test_empty_function():
    """Test function with no args."""
    @callable
    def get_const():
        return Int(42)
    
    # Should be StagedFunction (no templates at all)
    assert isinstance(get_const, StagedFunction)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
