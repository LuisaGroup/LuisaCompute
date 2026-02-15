"""
Host/Device Router for the LuisaCompute Python DSL v2.

This module provides the @router decorator that intelligently routes builtin
calls to either host (constant folding) or device (IR generation) execution
based on argument types.

Example:
    @router(math.sin, Op.SIN)
    def sin(x):
        # Automatically folds: sin(Float(1.0)) -> ConstantValue
        # Automatically routes: sin(dsl_var) -> device-side SIN instruction
        pass
"""

from __future__ import annotations
import math
from typing import Callable, Any, Optional, TYPE_CHECKING, Union
from functools import wraps

if TYPE_CHECKING:
    from .ir import Op, Value, InstructionValue, ConstantValue
    from .type import Type

from .ir import Op, ConstantValue, InstructionValue
from .builder import get_current_builder
from .type import (
    Type, Scalar, Vector, Matrix, 
    Bool, Int, Float, Double,
    value_to_type, is_data_type
)


# ============================================================================
# Constant Value Extraction and Checking
# ============================================================================

def is_constant_value(val: Any) -> bool:
    """Check if a value is a compile-time constant."""
    if isinstance(val, ConstantValue):
        return True
    # Python literals are also constants
    if val is None:
        return True
    if isinstance(val, (bool, int, float)):
        return True
    # Tuples/lists of constants are also constants (for vectors)
    if isinstance(val, (list, tuple)):
        return all(is_constant_value(v) for v in val)
    return False


def extract_constant_value(val: Any) -> Any:
    """Extract the Python value from a constant."""
    if isinstance(val, ConstantValue):
        return val.value
    if isinstance(val, (bool, int, float, type(None))):
        return val
    # Tuples/lists are passed through for vector constants
    if isinstance(val, (list, tuple)):
        return type(val)(extract_constant_value(v) for v in val)
    raise ValueError(f"Cannot extract constant value from {type(val)}")


def get_constant_type(val: Any) -> Optional[Type]:
    """Get the DSL type of a constant value."""
    if isinstance(val, ConstantValue):
        return val.type
    # Infer from Python type
    return value_to_type(val)


# ============================================================================
# Vector Constant Helpers
# ============================================================================

def is_vector_constant(val: Any) -> bool:
    """Check if a value is a constant vector (tuple of constants)."""
    if isinstance(val, (list, tuple)) and len(val) in (2, 3, 4):
        return all(is_constant_value(v) for v in val)
    return False


def extract_vector_components(val: Any) -> tuple:
    """Extract components from a vector constant."""
    if isinstance(val, (list, tuple)):
        return tuple(extract_constant_value(v) for v in val)
    raise ValueError(f"Cannot extract vector components from {type(val)}")


def vector_swizzle(components: tuple, pattern: str) -> Union[tuple, float]:
    """
    Perform swizzle operation on a vector constant.
    
    Args:
        components: Tuple of vector components
        pattern: Swizzle pattern like 'x', 'xy', 'xyz', 'xyzw', 'rgba'
    
    Returns:
        Scalar value for single-component patterns,
        Tuple for multi-component patterns
    """
    # Map component names to indices
    component_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3,
                     'r': 0, 'g': 1, 'b': 2, 'a': 3}
    
    result_components = []
    for ch in pattern:
        idx = component_map.get(ch)
        if idx is None or idx >= len(components):
            raise ValueError(f"Invalid swizzle pattern '{pattern}' for vector of size {len(components)}")
        result_components.append(components[idx])
    
    if len(result_components) == 1:
        return result_components[0]
    
    return tuple(result_components)


# ============================================================================
# Type Conversion for Routing
# ============================================================================

def to_python_value(val: Any) -> Any:
    """Convert a DSL value to a Python value for host execution."""
    if isinstance(val, ConstantValue):
        return val.value
    if isinstance(val, (list, tuple)):
        return val  # Vector constants are tuples
    if isinstance(val, (bool, int, float, type(None))):
        return val
    raise ValueError(f"Cannot convert {type(val)} to Python value")


def is_foldable_to_scalar(val: Any) -> bool:
    """Check if value can be folded to a scalar Python value."""
    if isinstance(val, ConstantValue):
        return isinstance(val.type, Scalar)
    if isinstance(val, (bool, int, float)):
        return True
    return False


def is_foldable_to_vector(val: Any) -> bool:
    """Check if value can be folded to a vector Python value."""
    if isinstance(val, (list, tuple)) and len(val) in (2, 3, 4):
        return True
    if isinstance(val, ConstantValue):
        from .type import Vector
        return isinstance(val.type, Vector)
    return False


# ============================================================================
# The Router Decorator
# ============================================================================

class RoutedFunction:
    """
    A function that can route calls to host (constant folding) or device (IR).
    
    This is returned by the @router decorator and provides intelligent dispatch.
    """
    
    def __init__(
        self,
        host_impl: Callable,
        device_op: Optional[Op],
        device_wrapper: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        self.host_impl = host_impl
        self.device_op = device_op
        self.device_wrapper = device_wrapper
        self.name = name or host_impl.__name__
    
    def __call__(self, *args, **kwargs):
        """
        Route the call based on argument types.
        
        - If all arguments are constants -> fold on host
        - If any argument is a DSL value -> emit device instruction
        """
        # Check if we can constant fold
        can_fold = all(is_constant_value(arg) for arg in args)
        
        if can_fold:
            # Extract Python values and call host implementation
            py_args = [extract_constant_value(arg) for arg in args]
            result = self.host_impl(*py_args, **kwargs)
            
            # Wrap result in appropriate DSL value
            return self._wrap_host_result(result)
        else:
            # Route to device
            return self._emit_device_call(*args, **kwargs)
    
    def _wrap_host_result(self, result: Any) -> Any:
        """Wrap a Python result value in the appropriate DSL value."""
        # Handle tuple of constants (for vector results or multiple return values)
        if isinstance(result, tuple):
            # If it's a small tuple (2-4 elements), it might be a vector
            if len(result) in (2, 3, 4):
                # Check if all elements are numeric constants
                if all(isinstance(r, (int, float)) for r in result):
                    # Return as tuple for vector constant
                    return result
            # Otherwise wrap each element
            return tuple(self._wrap_host_result(r) for r in result)
        
        # Handle list (convert to tuple for consistency)
        if isinstance(result, list):
            return self._wrap_host_result(tuple(result))
        
        # Determine result type
        result_type = value_to_type(result)
        if result_type is None:
            # Can't determine type, return as-is (e.g., for special types)
            return result
        
        # Create constant value
        return ConstantValue(typ=result_type, value=result)
    
    def _emit_device_call(self, *args, **kwargs) -> InstructionValue:
        """Emit a device-side instruction call."""
        builder = get_current_builder()
        
        # Convert all args to IR values
        from .ops import to_ir_value
        ir_args = [to_ir_value(arg) for arg in args]
        
        # Determine result type from first argument (common pattern)
        # For most math ops, result type matches the first argument
        if ir_args:
            first_arg_type = ir_args[0].type
            # Handle the case where first arg might be a string (for format in print)
            if hasattr(first_arg_type, 'element') or hasattr(first_arg_type, 'dtype'):
                result_type = first_arg_type
            else:
                result_type = Float  # Default for unknown types
        else:
            result_type = Float  # Default
        
        # If we have a custom device wrapper, use it
        if self.device_wrapper is not None:
            return self.device_wrapper(builder, *ir_args, **kwargs)
        
        # Otherwise emit standard op
        if self.device_op is not None:
            return builder._emit(self.device_op, result_type, ir_args)
        
        raise RuntimeError(f"No device implementation for {self.name}")
    
    def __repr__(self) -> str:
        return f"RoutedFunction({self.name})"


_UNSET = object()


def router(
    host_impl: Optional[Callable] = _UNSET,
    device_op: Optional[Op] = None,
    device_wrapper: Optional[Callable] = None
):
    """
    Decorator that creates a host/device routed function with constant folding.
    
    Args:
        host_impl: The Python function to use for constant folding.
                   Should accept and return Python values (not DSL values).
        device_op: The IR Op to emit for device execution.
        device_wrapper: Optional custom function for device-side emission.
                       Signature: (builder, *args) -> InstructionValue
    
    Returns:
        A RoutedFunction that intelligently dispatches calls.
    
    Example:
        # Using the math module for host folding
        @router(host_impl=math.sin, device_op=Op.SIN)
        def sin(x):
            pass
        
        # With custom device wrapper
        def dot_device_wrapper(builder, a, b):
            return builder._emit(Op.DOT, Float, [a, b])
        
        @router(host_impl=lambda a, b: sum(x*y for x, y in zip(a, b)),
                device_wrapper=dot_device_wrapper)
        def dot(a, b):
            pass
    """
    def decorator(func):
        # If host_impl not provided, try to use func as host impl
        impl = host_impl if host_impl is not _UNSET else func
        return RoutedFunction(
            host_impl=impl,
            device_op=device_op,
            device_wrapper=device_wrapper,
            name=func.__name__
        )
    
    # Check if called with bare @router (func passed as first positional arg)
    # In this case, host_impl is the function and no other args are provided
    if host_impl is not _UNSET and callable(host_impl) and device_op is None and device_wrapper is None:
        # Could be @router (bare) or @router(my_func) 
        # We distinguish by checking if this looks like a configuration call
        # @router bare usage is not supported - always use @router(...)
        pass
    
    # If host_impl is an Op, treat it as device_op (for @router(Op.SIN) syntax)
    if host_impl is not _UNSET and isinstance(host_impl, Op):
        device_op = host_impl
        host_impl = _UNSET
    
    return decorator


# ============================================================================
# Utility: Create routed math functions
# ============================================================================

def make_routed_math_func(name: str, op: Op, host_func: Optional[Callable] = None):
    """
    Create a routed math function using Python's math module for folding.
    
    Args:
        name: Function name
        op: IR operation
        host_func: Optional custom host function (defaults to getattr(math, name))
    """
    if host_func is None:
        host_func = getattr(math, name, None)
    
    if host_func is None:
        # Fallback for functions not in math module
        def host_func(x):
            raise NotImplementedError(f"No host implementation for {name}")
    
    return RoutedFunction(
        host_impl=host_func,
        device_op=op,
        name=name
    )
