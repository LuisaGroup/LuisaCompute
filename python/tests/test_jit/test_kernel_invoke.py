"""Tests for KernelInvoke and updated call logic."""

import pytest
from luisa import kernel, callable, Float, Int, Buffer
from luisa.lang.jit import KernelInvoke

def test_kernel_invoke_creation():
    """Test that calling a kernel from host returns a KernelInvoke object."""
    @kernel
    def my_kernel(x: Int):
        pass
    
    invoke = my_kernel(10)
    assert isinstance(invoke, KernelInvoke)
    assert invoke.kernel.name == "my_kernel"
    assert invoke.kernel.ir.name == "my_kernel"
    assert invoke.args == (10,)

def test_callable_from_host_works_in_kernel():
    """Test that calling a callable from within a kernel works correctly."""
    @callable
    def my_callable(x: Int):
        return x
    
    @kernel
    def test_kernel():
        result = my_callable(Int(10))
    
    # Should compile without error
    ir = test_kernel.ir
    assert ir is not None

def test_kernel_from_kernel_error(verify_ir):
    """Test that calling a kernel from another kernel raises an error."""
    @kernel
    def inner_kernel(x: Int):
        pass
        
    @kernel
    def outer_kernel(x: Int):
        inner_kernel(x)
    
    with pytest.raises(RuntimeError, match="Cannot call kernel 'inner_kernel' from within another kernel/callable"):
        outer_kernel(10)

def test_templated_kernel_invoke():
    """Test that calling a templated kernel from host works and uses cache."""
    @kernel['T']
    def my_templated_kernel(x: T):
        pass
    
    invoke1 = my_templated_kernel[Int](10)
    assert isinstance(invoke1, KernelInvoke)
    assert invoke1.args == (10,)
    
    invoke2 = my_templated_kernel[Float](1.0)
    assert isinstance(invoke2, KernelInvoke)
    assert invoke2.args == (1.0,)

    # Test implicit specialization
    invoke3 = my_templated_kernel(20)
    assert isinstance(invoke3, KernelInvoke)
    assert invoke3.args == (20,)
    
    # Check that they share the same TemplatedFunction but different StagedFunctions
    # (implicitly tested by their success)

def test_templated_callable_specialization():
    """Test explicit and implicit specialization of callables."""
    @callable['T']
    def add(a: T, b: T):
        return a + b
    
    # Explicit specialization
    add_int = add[Int]
    from luisa.lang.jit import StagedFunction
    assert isinstance(add_int, StagedFunction)
    assert add_int.arg_types == (Int, Int)
    
    add_float = add[Float]
    assert isinstance(add_float, StagedFunction)
    assert add_float.arg_types == (Float, Float)
    
    # Implicit specialization (via _get_or_create_staged)
    # Call from within a kernel to trigger cache population
    from luisa.lang.jit import TemplatedFunction
    assert isinstance(add, TemplatedFunction)
    
    @kernel
    def test_kernel():
        result = add(Int(1), Int(2))
    
    # Trigger compilation which populates cache
    _ = test_kernel.ir
    
    # Check cache
    assert (Int, Int) in add._cache
    assert isinstance(add._cache[(Int, Int)], StagedFunction)

def test_multi_param_specialization():
    """Test specialization with multiple template parameters."""
    @callable['T', 'N']
    def multi_param(x: T):
        return x + N
    
    # Partial specialization
    partial = multi_param[Int]
    from luisa.lang.jit import TemplatedFunction, StagedFunction
    assert isinstance(partial, TemplatedFunction)
    assert partial.specialization_values == (Int,)
    
    # Complete specialization
    final = partial[10]
    assert isinstance(final, StagedFunction)
    assert final.specialization_values == (Int, 10)
    
    # All at once
    direct = multi_param[Float, 20]
    assert isinstance(direct, StagedFunction)
    assert direct.specialization_values == (Float, 20)


if __name__ == "__main__":
    # Run tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
