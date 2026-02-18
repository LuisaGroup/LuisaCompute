
import pytest
from luisa import *
from luisa.lang.types import *

def test_implicit_deduction(verify_execution):
    @callable
    def add(x, y):
        return x + y
    
    # Implicit int deduction
    # Need to call from kernel to trigger compilation
    @kernel
    def k_int():
        res = add(1, 2)
    
    verify_execution(k_int)
    
    # Implicit float deduction
    @kernel
    def k_float():
        res = add(1.0, 2.0)
    
    verify_execution(k_float)

def test_explicit_deduction_from_args(verify_execution):
    @callable['T']
    def add_generic(x: T, y: T):
        return x + y

    @kernel
    def k():
        # Deduce T=int
        res = add_generic(1, 2)
        # Deduce T=float
        res2 = add_generic(1.0, 2.0)
        
    verify_execution(k)

def test_mixed_deduction(verify_execution):
    @callable['T', 'U']
    def mixed(a: T, b, c: U):
        # b is implicit
        return a + c

    @kernel
    def k():
        # Explicit T=int provided, U deduced from 2 (int), b implicit (float)
        res = mixed[int](0, 1.0, 2)
        
        # All deduced: T=int, b=float, U=int
        res2 = mixed(0, 1.0, 2)
    
    verify_execution(k)

def test_static_if_in_template(verify_execution):
    @callable['T']
    def check_type(x: T):
        if T == int or T == Int:
            return x
        else:
            return x + 1.0

    @kernel
    def k():
        # T=int. if True. return x (int).
        r1 = check_type(1)
        # T=float. if False. return x+1.0 (float).
        r2 = check_type(1.0)
        
    verify_execution(k)

def test_deduction_failure():
    @callable['T']
    def fails(x: T):
        return x

    @kernel
    def k():
        # Argument mismatch?
        # x is deduced as float, so T=float. This works.
        pass

    # However, if we specialize explicitly with wrong type?
    # fails[int](1.0) -> T=int. x: int. Arg: float.
    # Builder might cast float->int or raise error?
    # In DSL v2, it usually casts.
    
    # What if we have un-deducible param?
    @callable['T']
    def undecidable():
        return 0
    
    # Error is raised at definition time (when IR is built), not call time
    with pytest.raises(TypeError, match="Could not deduce template parameter 'T'"):
        @kernel
        def k_fail():
            # T cannot be deduced
            undecidable()

def test_implicit_conflicts():
    # If implicit params conflict?
    # Actually implicit params are per-argument.
    # def f(x, y). x -> __impl_x, y -> __impl_y.
    # So they never conflict.
    pass

def test_explicit_deduction_conflict():
    """Test that conflicting template parameter deductions raise an error."""
    @callable['T']
    def conflict(x: T, y: T):
        return x
    
    # Error is raised at definition time (when IR is built), not call time
    with pytest.raises(TypeError, match="Template parameter 'T' deduction conflict"):
        @kernel
        def k():
            # T=int from x=1, but T=float from y=2.0
            # This should raise a TypeError due to conflict
            conflict(1, 2.0)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
