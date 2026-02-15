"""
Special DSL type markers for const variables and shared memory.

This module provides Const and Shared types that can be used in DSL code
to mark variables with special semantics.
"""

from __future__ import annotations
from typing import Any, Optional

from .type import Type, value_to_type
from .ir import Value, ConstantValue


class Const:
    """
    Mark a value as a compile-time constant.
    
    Usage:
        # With explicit type
        a = Const[Float](1.0)
        b = Const[Int](42)
        
        # Without explicit type (inferred from value)
        c = Const(sin(1.0))
        
        # Multiple values (creates a tuple)
        d = Const[Float](1.0, 2.0, 3.0)
    
    Variables marked with Const are kept as Python values during DSL execution
    and are not converted to DSL variables (no alloca/store).
    """
    
    def __class_getitem__(cls, item):
        """Support Const[Type] syntax - returns a typed Const constructor."""
        return _TypedConst(item)
    
    def __new__(cls, *values):
        """
        Create a const value with type inferred from the value.
        
        If multiple values are provided, they are wrapped in a tuple.
        """
        if len(values) == 1:
            return _ConstValue(values[0])
        else:
            return _ConstValue(values)


class _TypedConst:
    """A const constructor with an explicit type."""
    
    def __init__(self, typ: Type):
        self.type = typ
    
    def __call__(self, *values):
        """Create a const value with the specified type."""
        if len(values) == 1:
            return _ConstValue(values[0], explicit_type=self.type)
        else:
            return _ConstValue(values, explicit_type=self.type)


class _ConstValue:
    """
    Wrapper for a compile-time constant value.
    
    This is used internally to mark values that should not be converted
to DSL variables.
    """
    
    def __init__(self, value: Any, explicit_type: Optional[Type] = None):
        self._raw_value = value
        self._explicit_type = explicit_type
        
        # Unwrap nested ConstValue
        if isinstance(value, _ConstValue):
            self._raw_value = value._raw_value
            if explicit_type is None:
                self._explicit_type = value._explicit_type
        
        # Unwrap ConstantValue
        elif isinstance(value, ConstantValue):
            self._raw_value = value.value
            if explicit_type is None:
                self._explicit_type = value.type
    
    @property
    def value(self) -> Any:
        """Get the raw Python value."""
        return self._raw_value
    
    @property
    def dsl_type(self) -> Optional[Type]:
        """Get the explicit DSL type if specified."""
        return self._explicit_type
    
    def __repr__(self) -> str:
        if self._explicit_type:
            return f"Const[{self._explicit_type}]({self._raw_value!r})"
        return f"Const({self._raw_value!r})"
    
    # Delegate arithmetic to the raw value
    def __add__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value + other)
    
    def __radd__(self, other):
        return _ConstValue(other + self._raw_value)
    
    def __sub__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value - other)
    
    def __rsub__(self, other):
        return _ConstValue(other - self._raw_value)
    
    def __mul__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value * other)
    
    def __rmul__(self, other):
        return _ConstValue(other * self._raw_value)
    
    def __truediv__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value / other)
    
    def __rtruediv__(self, other):
        return _ConstValue(other / self._raw_value)
    
    def __neg__(self):
        return _ConstValue(-self._raw_value)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        return _ConstValue(abs(self._raw_value))
    
    # Comparisons
    def __eq__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value == other._raw_value
        return self._raw_value == other
    
    def __ne__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value != other._raw_value
        return self._raw_value != other
    
    def __lt__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value < other._raw_value
        return self._raw_value < other
    
    def __le__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value <= other._raw_value
        return self._raw_value <= other
    
    def __gt__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value > other._raw_value
        return self._raw_value > other
    
    def __ge__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value >= other._raw_value
        return self._raw_value >= other
    
    def __bool__(self):
        return bool(self._raw_value)
    
    def __float__(self):
        return float(self._raw_value)
    
    def __int__(self):
        return int(self._raw_value)
    
    def __getitem__(self, index):
        """Allow indexing into const tuples/arrays."""
        return _ConstValue(self._raw_value[index])


def static(*values):
    """
    Mark value(s) as statically evaluated (compile-time constants).
    
    This is a shorthand for Const() that makes it explicit the value is
    evaluated at compile time (statically) rather than on the device.
    
    Usage:
        # Single value
        a = static(sin(1.0))
        
        # Multiple values
        b, c = static(1.0, 2.0)
    
    Variables marked with static() are kept as Python values and are not
    converted to DSL variables (no alloca/store).
    """
    if len(values) == 1:
        return _ConstValue(values[0])
    else:
        return tuple(_ConstValue(v) for v in values)


class Shared:
    """
    Mark a variable as shared memory (GPU threadgroup memory).
    
    Usage:
        # Shared memory array
        shared_buf = Shared[Float, 256]  # 256 floats of shared memory
        
        # Shared memory with initialization
        shared_val = Shared[Float](0.0)
    
    This is a placeholder for future implementation of GPU shared memory.
    Currently, it creates a normal DSL variable.
    """
    
    def __class_getitem__(cls, params):
        """
        Support Shared[Type] or Shared[Type, size] syntax.
        
        Returns a Shared memory constructor.
        """
        if isinstance(params, tuple):
            if len(params) == 2:
                typ, size = params
                return _SharedConstructor(typ, size)
            else:
                raise TypeError("Shared[...] expects 1 or 2 parameters: Shared[Type] or Shared[Type, size]")
        else:
            # Just the type, no size
            return _SharedConstructor(params, None)


class _SharedConstructor:
    """Constructor for shared memory variables."""
    
    def __init__(self, typ: Type, size: Optional[int] = None):
        self.type = typ
        self.size = size
    
    def __call__(self, *initial_values):
        """
        Create a shared memory variable.
        
        For now, this creates a normal DSL variable (placeholder implementation).
        In the future, this will allocate GPU shared memory.
        """
        # TODO: Implement actual shared memory allocation
        # For now, just mark it as a shared variable
        return _SharedValue(self.type, self.size, initial_values)


class _SharedValue:
    """Wrapper for a shared memory variable."""
    
    def __init__(self, typ: Type, size: Optional[int], initial_values: tuple):
        self.type = typ
        self.size = size
        self.initial_values = initial_values
        self._is_shared = True
    
    def __repr__(self) -> str:
        if self.size:
            return f"Shared[{self.type}, {self.size}]"
        return f"Shared[{self.type}]"
    
    def __getitem__(self, index):
        """Allow indexing into shared arrays."""
        # TODO: Implement shared memory access
        pass


def is_const_value(val: Any) -> bool:
    """Check if a value is a compile-time constant (Const or _ConstValue)."""
    return isinstance(val, (_ConstValue, Const))


def extract_const_value(val: Any) -> Any:
    """Extract the raw Python value from a const wrapper."""
    if isinstance(val, _ConstValue):
        return val.value
    return val
