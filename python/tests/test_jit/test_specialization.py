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
    
    # Call with Int argument from within a kernel - should create StagedFunction in cache
    @kernel
    def test_kernel():
        result = negate(Int(5))
    
    # Trigger compilation to populate cache
    _ = test_kernel.ir
    
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
    @kernel
    def test_kernel1():
        result = double(Int(1))
    
    _ = test_kernel1.ir
    
    cache_size_after_first = len(double._cache)
    assert cache_size_after_first == 1
    
    # Second call with same type reuses cache (no new entry)
    @kernel
    def test_kernel2():
        result = double(Int(2))
    
    _ = test_kernel2.ir
    
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
    """Test explicit specialization of templated kernel."""
    @kernel['T']
    def fill_buffer(buf: Buffer[T], value: T):
        buf[0] = value
    
    # Explicit specialization returns KernelInvoke when called
    # Note: Buffer arguments are placeholders since we only test IR building
    invoke = fill_buffer[Int](None, Int(42))
    assert isinstance(invoke, KernelInvoke)
    assert invoke.kernel.name == "fill_buffer"


def test_kernel_partial_specialization():
    """Test partial specialization of templated kernel."""
    @kernel['T', 'N']
    def process(buf: Buffer[T]):
        buf[0] = buf[0] + 1
    
    # Partial specialization
    partial = process[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete specialization and call
    # Note: Buffer arguments are placeholders since we only test IR building
    invoke = partial[10](None)
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
    @callable['T']
    def buffer_sum(buf: Buffer[T]) -> T:
        return buf[0] + buf[1]
    
    specialized = buffer_sum[Float]
    assert isinstance(specialized, StagedFunction)
    # Buffer element type should be Float
    from luisa.lang.types import Buffer as BufferType
    assert isinstance(specialized.arg_types[0], BufferType)


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


# ============================================================================
# Challenging Specialization Tests
# ============================================================================

def test_mixed_explicit_and_implicit_specialization():
    """Test mixture of explicit template params and implicitly deduced from args.
    
    Some template params are explicitly provided via [...], while others
    are inferred from the types of arguments passed at call time.
    """
    @callable['T', 'U']
    def mixed_template(a: T, b: U) -> T:
        return a + b
    
    # Explicitly specialize T=Int, let U be deduced from argument
    int_mixed = mixed_template[Int]  # Partial specialization: T=Int, U=?
    assert isinstance(int_mixed, TemplatedFunction)
    assert int_mixed.specialization_values == (Int,)
    
    # Test that the staged function is created correctly when called from a kernel
    @kernel
    def test_kernel():
        # Call with Float argument - U should be deduced as Float
        result = int_mixed(Int(10), Float(5.5))
    
    ir = test_kernel.ir
    assert ir is not None


def test_mixed_explicit_and_implicit_kernel():
    """Test mixture of explicit and implicit specialization in kernels."""
    @kernel['T', 'N']
    def process_with_stride(buf: Buffer[T], stride: Int, default: T):
        for i in range(N):
            if i * stride < 10:
                buf[i] = default
    
    # Partially specialize with T=Float, leaving N to be specialized later
    float_processor = process_with_stride[Float]
    assert isinstance(float_processor, TemplatedFunction)
    assert float_processor.specialization_values == (Float,)
    
    # Now fully specialize with N=4
    processor_4 = float_processor[4]
    assert isinstance(processor_4, StagedFunction)
    assert processor_4.specialization_values == (Float, 4)


def test_nested_template_function_captures_outer_tparams():
    """Test that nested template functions correctly capture outer template params.
    
    The inner callable should be able to use T from the outer scope.
    """
    @callable['T']
    def outer_transform(x: T):
        # Inner callable defined inside outer - captures T
        @callable
        def inner_scale(factor: Float):
            # Can use T from outer scope through captured value
            return x * T(factor)
        
        return inner_scale(2.0)
    
    # Specialize outer with T=Float
    float_transform = outer_transform[Float]
    assert isinstance(float_transform, StagedFunction)
    
    # Test from within a kernel
    @kernel
    def test_kernel():
        result = float_transform(Float(5.0))
    
    ir = test_kernel.ir
    assert ir is not None


def test_nested_template_with_inner_template():
    """Test nested definition where inner is also a template.
    
    Inner template should be able to use outer template params
    plus its own template params.
    """
    @callable['T']
    def outer_with_inner_template(x: T):
        # Inner template that also uses T from outer
        @callable['U']
        def inner_combined(y: U):
            # Combine outer's T and inner's U
            return T(x) + U(y)
        
        # Call inner with specific U
        return inner_combined[Float](1.5)
    
    # Specialize outer with T=Int
    int_outer = outer_with_inner_template[Int]
    assert isinstance(int_outer, StagedFunction)


def test_inner_specialized_by_parent_template_args():
    """Test that inner template is specialized using parent template args.
    
    The parent's template arguments are used to specialize the inner template.
    """
    @callable['T', 'U']
    def outer_multi(x: T, y: U):
        # Inner template uses both T and U from parent
        @callable
        def inner_process(a: T, b: U):
            return a + b
        
        return inner_process(x, y)
    
    # Specialize outer with T=Int, U=Float
    specialized = outer_multi[Int, Float]
    assert isinstance(specialized, StagedFunction)
    assert specialized.specialization_values == (Int, Float)
    
    # Test from within a kernel
    @kernel
    def test_kernel():
        result = specialized(Int(10), Float(2.5))
    
    ir = test_kernel.ir
    assert ir is not None


def test_deeply_nested_templates():
    """Test deeply nested template definitions."""
    @callable['T']
    def level1(x: T):
        @callable['U']
        def level2(y: U):
            @callable
            def level3(z: T):  # Uses T from level1
                return z + T(y)
            return level3(x)
        return level2[Float](1.0)
    
    specialized = level1[Int]
    assert isinstance(specialized, StagedFunction)


def test_partial_specialization_with_nested():
    """Test partial specialization with nested operations.
    
    Tests that partial specialization correctly propagates to function body.
    """
    @callable['T', 'U']
    def combine_values(a: T, b: U) -> Float:
        # Cast both to Float and combine
        return Float(a) + Float(b)
    
    # Partially specialize: T=Int
    partial = combine_values[Int]
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete specialization: U=Float
    full = partial[Float]
    assert isinstance(full, StagedFunction)
    assert full.arg_types == (Int, Float)
    
    # Test from within a kernel
    @kernel
    def test_kernel():
        result = combine_values[Int, Float](Int(1), Float(2.0))
    
    ir = test_kernel.ir
    assert ir is not None


def test_template_in_kernel_with_capture():
    """Test template callable defined inside kernel capturing outer template params."""
    @kernel['T']
    def process_buffer(buf: Buffer[T], count: Int):
        # Inner callable template that uses T
        @callable
        def compute_value(idx: Int):
            # Use T from kernel scope
            return T(idx) * T(2.0)
        
        for i in range(count):
            val = compute_value(i)
            buf[i] = val
    
    specialized = process_buffer[Float]
    assert isinstance(specialized, StagedFunction)


def test_nested_explicit_and_implicit_mix():
    """Complex case: mixture of explicit/implicit in both outer and inner."""
    @callable['T', 'U']
    def outer_partial(x: T, y: U):
        @callable
        def inner_mixed(a: T, b: U):
            # T comes from outer's explicit specialization
            # U comes from outer's explicit specialization
            return a + T(b)
        
        # Call inner with the provided args
        return inner_mixed(x, y)
    
    # Fully specialize outer with T=Float, U=Int
    fully_specialized = outer_partial[Float, Int]
    assert isinstance(fully_specialized, StagedFunction)
    
    # Test from within a kernel
    @kernel
    def test_kernel():
        result = fully_specialized(Float(1.5), Int(10))
    
    ir = test_kernel.ir
    assert ir is not None


def test_template_param_shadowing():
    """Test that inner template params can shadow outer ones (if supported)."""
    @callable['T']
    def outer_shadow(x: T):
        # Inner with same param name T - should be independent
        @callable['T']
        def inner_shadow(y: T):
            # This T refers to inner's T, not outer's
            return y
        
        # Return inner specialized with different type
        return inner_shadow[Float](1.0)
    
    specialized = outer_shadow[Int]
    assert isinstance(specialized, StagedFunction)


def test_captured_template_in_loop():
    """Test template params captured in loops within nested functions."""
    @callable['T']
    def sum_template(values: T, count: Int):
        @callable
        def accumulate():
            result = T(0)
            for i in range(count):
                result = result + values
            return result
        
        return accumulate()
    
    specialized = sum_template[Float]
    assert isinstance(specialized, StagedFunction)
    
    # Test from within a kernel
    @kernel
    def test_kernel():
        result = specialized(Float(1.5), 4)
    
    ir = test_kernel.ir
    assert ir is not None


def test_kernel_nested_callable_with_parent_tparam():
    """Test kernel with nested callable using parent's template param."""
    @kernel['T']
    def transform_kernel(buf: Buffer[T], n: Int):
        @callable
        def transform_element(val: T) -> T:
            return val * T(2.0) + T(1.0)
        
        for i in range(n):
            buf[i] = transform_element(buf[i])
    
    specialized = transform_kernel[Int]
    assert isinstance(specialized, StagedFunction)
    
    # Also test Float specialization
    float_specialized = transform_kernel[Float]
    assert isinstance(float_specialized, StagedFunction)


if __name__ == "__main__":
    # Run tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
