"""Tests for callable template specialization and partial specialization."""

import pytest
from luisa import kernel, callable, Float, Int, Bool, Buffer
from luisa.lang.jit import StagedFunction, TemplatedFunction, KernelInvoke


# ============================================================================
# Basic Template Specialization Tests
# ============================================================================

def test_single_template_param_specialization():
    """Test specialization with a single template parameter."""
    @callable['T']
    def identity(x: T):
        return x
    
    # Explicit specialization returns StagedFunction
    int_identity = identity[Int]
    assert isinstance(int_identity, StagedFunction)
    assert int_identity.arg_types == (Int,)
    assert int_identity.specialization_values == (Int,)
    
    float_identity = identity[Float]
    assert isinstance(float_identity, StagedFunction)
    assert float_identity.arg_types == (Float,)
    assert float_identity.specialization_values == (Float,)


def test_multiple_template_param_specialization():
    """Test specialization with multiple template parameters."""
    @callable['T', 'U']
    def pair(first: T, second: U):
        return first, second
    
    # Full specialization
    int_float_pair = pair[Int, Float]
    assert isinstance(int_float_pair, StagedFunction)
    assert int_float_pair.arg_types == (Int, Float)
    assert int_float_pair.specialization_values == (Int, Float)


def test_same_type_constraint():
    """Test template where multiple args must be the same type."""
    @callable['T']
    def add(a: T, b: T):
        return a + b
    
    add_int = add[Int]
    assert isinstance(add_int, StagedFunction)
    assert add_int.arg_types == (Int, Int)
    
    add_float = add[Float]
    assert isinstance(add_float, StagedFunction)
    assert add_float.arg_types == (Float, Float)


# ============================================================================
# Partial Specialization Tests
# ============================================================================

def test_partial_specialization_single_param():
    """Test partial specialization with one of multiple params."""
    @callable['T', 'U']
    def combine(a: T, b: U):
        return a + b
    
    # Partial specialization - only T is bound
    partial = combine[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete the specialization
    full = partial[Float]
    assert isinstance(full, StagedFunction)
    assert full.arg_types == (Int, Float)
    assert full.specialization_values == (Int, Float)


def test_partial_specialization_with_non_type_params():
    """Test partial specialization mixing type and non-type params."""
    @callable['T', 'N']
    def scaled(x: T):
        return x * 2
    
    # Partial - bind type only
    partial = scaled[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete with non-type param
    full = partial[10]
    assert isinstance(full, StagedFunction)
    assert full.specialization_values == (Int, 10)


def test_chained_partial_specialization():
    """Test chaining multiple partial specializations."""
    @callable['A', 'B', 'C']
    def triple(a: A, b: B, c: C):
        return a + b + c
    
    # First partial
    step1 = triple[Int]
    assert isinstance(step1, TemplatedFunction)
    assert step1.specialization_values == (Int,)
    
    # Second partial
    step2 = step1[Float]
    assert isinstance(step2, TemplatedFunction)
    assert step2.specialization_values == (Int, Float)
    
    # Complete
    full = step2[Bool]
    assert isinstance(full, StagedFunction)
    assert full.arg_types == (Int, Float, Bool)
    assert full.specialization_values == (Int, Float, Bool)


# ============================================================================
# Implicit Specialization Tests
# ============================================================================

def test_implicit_specialization_via_call():
    """Test that calling a templated callable infers types from arguments."""
    @callable['T']
    def negate(x: T):
        return -x
    
    # Before implicit specialization, cache is empty
    assert len(negate._cache) == 0
    
    # Call with Int argument - should create StagedFunction in cache
    with pytest.raises(RuntimeError, match="can only be called from within a kernel or another callable"):
        negate(Int(5))
    
    # Check cache has the inferred type
    assert (Int,) in negate._cache
    cached = negate._cache[(Int,)]
    assert isinstance(cached, StagedFunction)
    assert cached.arg_types == (Int,)


def test_implicit_specialization_reuses_cache():
    """Test that implicit specialization reuses cached StagedFunctions."""
    @callable['T']
    def double(x: T):
        return x * 2
    
    # First call creates cache entry
    with pytest.raises(RuntimeError):
        double(Int(1))
    
    cache_size_after_first = len(double._cache)
    assert cache_size_after_first == 1
    
    # Second call with same type reuses cache
    with pytest.raises(RuntimeError):
        double(Int(2))
    
    assert len(double._cache) == cache_size_after_first


# ============================================================================
# Template with Return Type Tests
# ============================================================================

def test_template_with_return_type_annotation():
    """Test template callable with explicit return type."""
    @callable['T']
    def to_float(x: T) -> Float:
        return Float(x)
    
    specialized = to_float[Int]
    assert isinstance(specialized, StagedFunction)
    assert specialized.arg_types == (Int,)
    assert specialized.ret_type == Float


def test_template_with_generic_return_type():
    """Test template callable with return type depending on template param."""
    @callable['T']
    def wrap(x: T) -> T:
        return x
    
    int_wrap = wrap[Int]
    assert isinstance(int_wrap, StagedFunction)
    assert int_wrap.ret_type == Int
    
    float_wrap = wrap[Float]
    assert isinstance(float_wrap, StagedFunction)
    assert float_wrap.ret_type == Float


# ============================================================================
# Kernel Template Specialization Tests
# ============================================================================

def test_kernel_explicit_specialization():
    """Test explicit specialization of templated kernel with type-only template param."""
    @kernel['T']
    def fill_value(value: T):
        x = value
    
    # Explicit specialization returns KernelInvoke when called
    invoke = fill_value[Int](Int(42))
    assert isinstance(invoke, KernelInvoke)
    assert invoke.kernel.name == "fill_value"


def test_kernel_partial_specialization():
    """Test partial specialization of templated kernel."""
    @kernel['T', 'N']
    def process(value: T):
        x = value + 1
    
    # Partial specialization
    partial = process[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete specialization and call
    invoke = partial[10](Int(5))
    assert isinstance(invoke, KernelInvoke)
    assert invoke.kernel.name == "process"


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_too_many_template_arguments():
    """Test error when providing too many template arguments."""
    @callable['T']
    def single(x: T):
        return x
    
    with pytest.raises(TypeError, match="Too many template arguments"):
        single[Int, Float]


def test_specialization_on_non_templated_function():
    """Test error when trying to specialize non-templated function."""
    @callable
    def regular(x: Int):
        return x
    
    # When a function is fully annotated and not templated, it becomes a StagedFunction
    # immediately, so trying to subscript it raises a different error
    with pytest.raises(TypeError, match="not subscriptable"):
        regular[Float]


# ============================================================================
# Complex Template Tests
# ============================================================================

def test_template_with_buffer_type():
    """Test template with buffer type parameter."""
    # Note: Buffer[T] with template param T is not fully supported yet
    # This test verifies the basic template mechanism works with complex types
    @callable['T']
    def process_element(x: T) -> T:
        return x
    
    specialized = process_element[Float]
    assert isinstance(specialized, StagedFunction)
    assert specialized.arg_types[0] == Float


def test_nested_template_in_kernel():
    """Test calling templated callable from kernel."""
    @callable['T']
    def scale(x: T, factor: Float) -> T:
        return x * factor
    
    @kernel
    def apply_scale(buf: Buffer[Float]):
        idx = 0  # dispatch_id().x
        val = buf[idx]
        # Call templated callable
        buf[idx] = scale[Float](val, 2.0)
    
    # Should build without error
    # Note: Buffer argument is a placeholder since we only test IR building
    invoke = apply_scale(None)
    assert isinstance(invoke, KernelInvoke)
    assert invoke.kernel.name == "apply_scale"


def test_multiple_specializations_same_template():
    """Test creating multiple specializations from same template."""
    @callable['T']
    def square(x: T):
        return x * x
    
    # Create multiple specializations
    int_square = square[Int]
    float_square = square[Float]
    bool_square = square[Bool]
    
    assert isinstance(int_square, StagedFunction)
    assert isinstance(float_square, StagedFunction)
    assert isinstance(bool_square, StagedFunction)
    
    assert int_square.arg_types == (Int,)
    assert float_square.arg_types == (Float,)
    assert bool_square.arg_types == (Bool,)
    
    # They should be different objects
    assert int_square is not float_square
    assert float_square is not bool_square


def test_specialization_preserves_function_name():
    """Test that specialization preserves the original function name."""
    @callable['T']
    def my_function(x: T):
        return x
    
    specialized = my_function[Int]
    assert specialized.name == "my_function"
    assert specialized.templated.name == "my_function"


if __name__ == "__main__":
    # Run tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
