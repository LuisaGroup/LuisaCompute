"""
Compile-time constant marker for the LuisaCompute Python DSL v2.

This module provides the `@const` decorator and `const()` function to mark
variables that should be treated as compile-time constants rather than
DSL variables.
"""

from __future__ import annotations
from typing import Any


class ConstMarker:
    """
    Marks a value as a compile-time constant.
    
    When a variable is marked with @const or const(), it will be kept as a
    Python value during DSL execution rather than being converted to a DSL variable.
    """
    
    def __init__(self, value: Any):
        self.value = value
    
    def __repr__(self) -> str:
        return f"const({self.value!r})"
    
    # Delegate attribute access to the wrapped value
    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)


def const(value: Any) -> Any:
    """
    Mark a value as a compile-time constant.
    
    When used in a kernel or callable, variables assigned with const()
    will be kept as Python values rather than being converted to DSL variables.
    
    Example:
        @kernel
        def my_kernel(buf: Buffer[Float]):
            # 'a' is a DSL variable (can be reassigned in DSL)
            a = sin(1.0)
            
            # 'b' is a compile-time constant (Python value)
            b = const(sin(1.0))
            
            # This works: reassigning DSL variable
            a = a + 1.0
            
            # This also works: b is just a Python float
            c = b + 1.0  # computed at compile time
    """
    # If it's already a ConstMarker, return it
    if isinstance(value, ConstMarker):
        return value
    
    # For ConstantValue, extract the underlying value
    from .ir import ConstantValue
    if isinstance(value, ConstantValue):
        return ConstMarker(value.value)
    
    return ConstMarker(value)


# Registry to track which variable names are marked as const at compile time
# This is populated by the AST rewriter and checked at runtime
_const_var_names: set[str] = set()


def _mark_const(name: str) -> None:
    """Mark a variable name as const (called by AST rewriter)."""
    _const_var_names.add(name)


def _is_const_var(name: str) -> bool:
    """Check if a variable name is marked as const."""
    return name in _const_var_names


def _clear_const_vars() -> None:
    """Clear the const variable registry."""
    _const_var_names.clear()


def _get_const_vars() -> set[str]:
    """Get the set of const variable names."""
    return _const_var_names.copy()
