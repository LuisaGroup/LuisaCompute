"""
Runtime support for the LuisaCompute Python DSL v2.

This module provides the core logic for operations within the DSL,
handling both IR building and host-side execution.
"""

from __future__ import annotations
import ast
from typing import Callable, Any, Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from ..transform.builder import Builder
    from ..transform.ir import Value

from ..transform.builder import get_current_builder
from ..transform.op import Op
from ..transform.ir import Value
from .types import Type, value_to_type, Bool, Int, Float, Scalar, Vector, Buffer, Array


# ============================================================================
# Value Management
# ============================================================================

def is_ir_value(val: Any) -> bool:
    """Check if a value is an IR Value (excluding ConstantValue)."""
    from ..transform.ir import ConstantValue
    return isinstance(val, Value) and not isinstance(val, ConstantValue)


def to_ir_value(val: Any) -> Value:
    """Ensure a value is an IR Value, converting literals if necessary."""
    from ..transform.ir import ConstantValue
    if isinstance(val, Value):
        if isinstance(val, ConstantValue):
            inner_val = val.value
            if hasattr(inner_val, 'to_tuple') and hasattr(inner_val, 'get_dsl_type'):
                return get_current_builder().constant(val.type, inner_val.to_tuple())
        return val
    if isinstance(val, str):
        return val
    
    # Unwrap _ConstValue
    from .types import _ConstValue
    if isinstance(val, _ConstValue):
        # Prefer the explicit type if provided
        if val.dsl_type is not None:
            inner_val = val.value
            if hasattr(inner_val, 'to_tuple'):
                inner_val = inner_val.to_tuple()
            return get_current_builder().constant(val.dsl_type, inner_val)
        val = val.value

    typ = value_to_type(val)
    if typ is None:
        if isinstance(val, (list, tuple)):
            # Infer type for tuples (Vector or Matrix)
            length = len(val)
            from .types import Vector, Matrix, Float
            if length in (2, 3, 4):
                typ = Vector(Float, length)
            elif length in (4, 9, 16):
                typ = Matrix(Float, int(math.sqrt(length)))
        elif hasattr(val, 'to_tuple') and hasattr(val, 'get_dsl_type'):
            # It's a Struct object
            typ = val.get_dsl_type()
            # Convert to tuple for the IR builder
            val = val.to_tuple()
            
    if typ is None:
        raise TypeError(f"Cannot convert {type(val)} to Luisa type")
    return get_current_builder().constant(typ, val)


def try_to_ir_value(val: Any) -> Any:
    """Try to convert to IR value, return original if not possible."""
    try:
        return to_ir_value(val)
    except TypeError:
        return val


# ============================================================================
# Operators
# ============================================================================

def binop(op: ast.operator, left: Any, right: Any) -> Any:
    """Handle binary operations."""
    # Check if both operands are constants - if so, do host-side computation
    from ..transform.ir import ConstantValue
    from .types import _ConstValue
    left_is_const = isinstance(left, (ConstantValue, _ConstValue)) or left is None or isinstance(left, (bool, int, float, str))
    right_is_const = isinstance(right, (ConstantValue, _ConstValue)) or right is None or isinstance(right, (bool, int, float, str))
    
    # If both are constants, do host-side computation
    if left_is_const and right_is_const:
        # Extract Python values from ConstantValue or _ConstValue
        if isinstance(left, ConstantValue):
            left = left.value
        elif isinstance(left, _ConstValue):
            left = left.value
        if isinstance(right, ConstantValue):
            right = right.value
        elif isinstance(right, _ConstValue):
            right = right.value
        
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
        if isinstance(op, ast.MatMult):
            from .builtins.math import _matmul_host
            return _matmul_host(left, right)
        raise NotImplementedError(f"Unsupported binary operator: {type(op)}")
    
    if is_ir_value(left) or is_ir_value(right):
        left = to_ir_value(left)
        right = to_ir_value(right)
        builder = get_current_builder()
        if isinstance(op, ast.Add):
            return builder.add(left, right)
        if isinstance(op, ast.Sub):
            return builder.sub(left, right)
        if isinstance(op, ast.Mult):
            return builder.mul(left, right)
        if isinstance(op, ast.Div):
            return builder.div(left, right)
        if isinstance(op, ast.Mod):
            return builder.mod(left, right)
        if isinstance(op, ast.Pow):
            return builder.pow(left, right)
        if isinstance(op, ast.FloorDiv):
            return builder.floor(builder.div(left, right))
        if isinstance(op, ast.BitAnd):
            return builder.bit_and(left, right)
        if isinstance(op, ast.BitOr):
            return builder.bit_or(left, right)
        if isinstance(op, ast.BitXor):
            return builder.bit_xor(left, right)
        if isinstance(op, ast.LShift):
            return builder.shl(left, right)
        if isinstance(op, ast.RShift):
            return builder.shr(left, right)
        if isinstance(op, ast.MatMult):
            # Luisa uses Op.MUL for matrix-matrix and matrix-vector multiplication
            return builder.mul(left, right)
        raise NotImplementedError(f"Unsupported binary operator for IR: {type(op)}")
    else:
        # Host side
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
        if isinstance(op, ast.MatMult):
            from .builtins.math import _matmul_host
            return _matmul_host(left, right)
        raise NotImplementedError(f"Unsupported binary operator: {type(op)}")


def unaryop(op: ast.unaryop, operand: Any) -> Any:
    """Handle unary operations."""
    if is_ir_value(operand):
        operand = to_ir_value(operand)
        builder = get_current_builder()
        if isinstance(op, ast.USub):
            return builder.neg(operand)
        if isinstance(op, ast.Not):
            return builder.logical_not(operand)
        if isinstance(op, ast.Invert):
            return builder.bit_not(operand)
        raise NotImplementedError(f"Unsupported unary operator for IR: {type(op)}")
    else:
        if isinstance(op, ast.USub):
            return -operand
        if isinstance(op, ast.Not):
            return not operand
        if isinstance(op, ast.Invert):
            return ~operand
        raise NotImplementedError(f"Unsupported unary operator: {type(op)}")


def compare(op: ast.cmpop, left: Any, right: Any) -> Any:
    """Handle comparison operations."""
    if is_ir_value(left) or is_ir_value(right):
        left = to_ir_value(left)
        right = to_ir_value(right)
        builder = get_current_builder()
        if isinstance(op, ast.Eq):
            return builder.eq(left, right)
        if isinstance(op, ast.NotEq):
            return builder.ne(left, right)
        if isinstance(op, ast.Lt):
            return builder.lt(left, right)
        if isinstance(op, ast.LtE):
            return builder.le(left, right)
        if isinstance(op, ast.Gt):
            return builder.gt(left, right)
        if isinstance(op, ast.GtE):
            return builder.ge(left, right)
        raise NotImplementedError(f"Unsupported comparison operator for IR: {type(op)}")
    else:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        raise NotImplementedError(f"Unsupported comparison operator: {type(op)}")


def boolop(op: ast.boolop, values: list[Any]) -> Any:
    """Handle boolean operations (and/or)."""
    # Note: rewriter now uses and_ and or_ for short-circuiting
    # This is kept for backward compatibility or direct calls
    is_ir = any(is_ir_value(v) for v in values)
    if is_ir:
        ir_values = [to_ir_value(v) for v in values]
        builder = get_current_builder()
        result = ir_values[0]
        if isinstance(op, ast.And):
            for v in ir_values[1:]:
                result = builder.logical_and(result, v)
        elif isinstance(op, ast.Or):
            for v in ir_values[1:]:
                result = builder.logical_or(result, v)
        return result
    else:
        if isinstance(op, ast.And):
            res = values[0]
            for v in values[1:]:
                res = res and v
            return res
        elif isinstance(op, ast.Or):
            res = values[0]
            for v in values[1:]:
                res = res or v
            return res
        raise NotImplementedError(f"Unsupported boolean operator: {type(op)}")


def and_(lhs_func: Callable[[], Any], rhs_func: Callable[[], Any]) -> Any:
    """Logical AND with short-circuiting."""
    lhs = lhs_func()
    
    # Constant folding for IR ConstantValue or Python literals
    from ..transform.ir import ConstantValue
    if isinstance(lhs, ConstantValue):
        if lhs.value:
            return rhs_func()
        return lhs
        
    if is_ir_value(lhs):
        builder = get_current_builder()
        # Create a variable for the result
        from .types import Bool
        res_ptr = builder.alloca(Bool)
        builder.store(res_ptr, lhs)
        
        if_ = builder.if_(lhs)
        with if_.true_scope():
            rhs = rhs_func()
            builder.store(res_ptr, to_ir_value(rhs))
        # false branch: already has lhs (which is false)
        
        return builder.load(res_ptr)
    else:
        if lhs:
            return rhs_func()
        return lhs


def or_(lhs_func: Callable[[], Any], rhs_func: Callable[[], Any]) -> Any:
    """Logical OR with short-circuiting."""
    lhs = lhs_func()
    
    # Constant folding for IR ConstantValue or Python literals
    from ..transform.ir import ConstantValue
    if isinstance(lhs, ConstantValue):
        if lhs.value:
            return lhs
        return rhs_func()

    if is_ir_value(lhs):
        builder = get_current_builder()
        # Create a variable for the result
        from .types import Bool
        res_ptr = builder.alloca(Bool)
        builder.store(res_ptr, lhs)
        
        if_ = builder.if_(lhs)
        # true branch: already has lhs (which is true)
        with if_.false_scope():
            rhs = rhs_func()
            builder.store(res_ptr, to_ir_value(rhs))
            
        return builder.load(res_ptr)
    else:
        if lhs:
            return lhs
        return rhs_func()


# ============================================================================
# Control Flow
# ============================================================================

def if_(cond_func: Callable[[], Any]) -> Any:
    """Handle if statements."""
    cond = cond_func()
    if is_ir_value(cond):
        return get_current_builder().if_(cond)
    else:
        return StaticIf(cond)


def switch(value: Any) -> Any:
    """Handle switch statements."""
    if is_ir_value(value):
        return get_current_builder().switch(value)
    else:
        # Fallback for host-side switch
        raise NotImplementedError("Host-side switch not yet supported")


def for_(iter_obj: Any, loop_var_name: Any) -> Any:
    """Handle for loops (returns iterable)."""
    if isinstance(iter_obj, (range, LuisaRange)):
        if isinstance(iter_obj, range):
            start_val, stop_val, step_val = iter_obj.start, iter_obj.stop, iter_obj.step
        else:
            args = iter_obj.args
            if len(args) == 1:
                start_val, stop_val, step_val = 0, args[0], 1
            elif len(args) == 2:
                start_val, stop_val, step_val = args[0], args[1], 1
            else:
                start_val, stop_val, step_val = args[0], args[1], args[2]

        start = to_ir_value(start_val)
        stop = to_ir_value(stop_val)
        step = to_ir_value(step_val)
        name = loop_var_name
        stmt = get_current_builder().for_range(start, stop, step, name)
        return [stmt]

    # Handle StaticRange without circular import
    if hasattr(iter_obj, 'rng') and iter_obj.__class__.__name__ == 'StaticRange':
        return iter_obj.rng

    return iter_obj


@contextmanager
def loop_scope(loop_item: Any):
    """Context manager for loop bodies."""
    if hasattr(loop_item, 'body_scope'):
        with loop_item.body_scope() as scope:
            yield scope
    else:
        yield loop_item


def while_(test_func: Callable[[], Any]) -> Any:
    """Handle while loops (returns generator)."""
    cond = test_func()
    if is_ir_value(cond):
        stmt = get_current_builder().while_(cond)
        yield stmt
    else:
        # Host side loop
        while cond:
            yield None
            cond = test_func()
            # If condition becomes an IR value during host execution, it's an error
            # or we need to switch to device execution (not supported mid-loop).
            if is_ir_value(cond):
                raise RuntimeError("Loop condition changed from host-side to device-side during execution. "
                                 "This usually happens when a loop variable is accidentally converted to a DSL variable.")


@contextmanager
def while_scope(loop_item: Any):
    """Context manager for while loop bodies."""
    if loop_item is not None and hasattr(loop_item, 'body_scope'):
        with loop_item.body_scope() as scope:
            yield scope
    else:
        yield None


# ============================================================================
# Static Constructs
# ============================================================================

class StaticIf:
    """Helper for host-side if statement (static evaluation)."""

    def __init__(self, cond: bool):
        self.cond = bool(cond)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @contextmanager
    def true_scope(self):
        if self.cond:
            yield self.cond
        else:
            pass

    @contextmanager
    def false_scope(self):
        if not self.cond:
            yield not self.cond
        else:
            pass


class StaticWhile:
    """Helper for host-side while loop (static evaluation)."""

    def __init__(self, test_func: Callable[[], bool]):
        self.test_func = test_func

    def __iter__(self):
        while self.test_func():
            yield None


class LuisaRange:
    """A range object that can contain IR Values."""

    def __init__(self, *args):
        self.args = args


# ============================================================================
# Accessors and Calls
# ============================================================================

def call(func: Any, *args, **kwargs) -> Any:
    """Handle function calls."""
    # Handle built-in range()
    if func is range:
        if any(is_ir_value(a) for a in args):
            return LuisaRange(*args)
        return range(*args)

    if isinstance(func, Type):
        # Type constructors can take multiple arguments (aggregate types)
        # or a single argument (casting or broadcasting)
        
        # If any argument is an IR value, emit a cast or call_builtin
        if any(is_ir_value(a) for a in args):
            # T(x) or T(x,y,...)
            # Delegate to the type's __call__ which knows how to emit IR
            return func(*args)
        
        # All constants - delegate to T.__call__ for host computation
        return func(*args)

    # Use duck typing or check class name to avoid circular import with multistage
    if func.__class__.__name__ == 'StagedFunction':
        # Check if we're inside a builder context (i.e., building IR)
        # If so, emit a CALL instruction
        # If not, call the staged function directly to get its compiled IR
        try:
            builder = get_current_builder()
            # We have a builder context - emit a CALL instruction
            return builder.call(func, *args)
        except RuntimeError:
            # No builder context - call the staged function directly
            # This returns the compiled Function, not an InstructionValue
            return func(*args, **kwargs)

    import builtins
    if builtins.callable(func):
        if any(is_ir_value(a) for a in args):
            new_args = []
            for a in args:
                if isinstance(a, str):
                    new_args.append(a)
                else:
                    new_args.append(try_to_ir_value(a))
            return func(*new_args, **kwargs)
        return func(*args, **kwargs)

    raise TypeError(f"Object {func} is not callable")


def subscript(value: Any, index: Any) -> Any:
    """Handle subscript access."""
    if is_ir_value(value) or is_ir_value(index):
        value = to_ir_value(value)
        index = to_ir_value(index)
        if isinstance(value.type, (Buffer, Array)):
            return get_current_builder().buffer_read(value, index, value.type.element)
        raise TypeError(f"Cannot subscript IR type {value.type}")
    return value[index]


def subscript_assign(value: Any, index: Any, rhs: Any) -> None:
    """Handle subscript assignment."""
    if is_ir_value(value) or is_ir_value(index) or is_ir_value(rhs):
        value = to_ir_value(value)
        index = to_ir_value(index)
        rhs = to_ir_value(rhs)
        if isinstance(value.type, (Buffer, Array)):
            get_current_builder().buffer_write(value, index, rhs)
        else:
            raise TypeError(f"Cannot assign to subscript of IR type {value.type}")
    else:
        value[index] = rhs


def attribute(value: Any, attr: str) -> Any:
    """Handle attribute access."""
    from ..transform.ir import ConstantValue
    from .types import Vector, Struct, _ConstValue
    
    # Handle _ConstValue (compile-time wrapper)
    if isinstance(value, _ConstValue):
        # Recursive call with the unwrapped value
        res = attribute(value.value, attr)
        # Re-wrap result in _ConstValue if it's not already a DSL value
        if not is_ir_value(res) and not isinstance(res, (ConstantValue, _ConstValue)):
            return _ConstValue(res)
        return res

    # Handle ConstantValue by extracting its value and type
    if isinstance(value, ConstantValue):
        typ = value.type
        val_obj = value.value
        
        if isinstance(typ, Vector):
            from .router import vector_swizzle
            res = vector_swizzle(val_obj, attr)
            return ConstantValue(typ=value_to_type(res), value=res)
        elif isinstance(typ, Struct):
            idx = typ.get_field_index(attr)
            if hasattr(val_obj, 'to_tuple'):
                res = getattr(val_obj, attr)
            else:
                res = val_obj[idx]
            return ConstantValue(typ=typ.fields[idx][1], value=res)
        
    if is_ir_value(value):
        # Allow accessing standard attributes of Value/InstructionValue
        if attr in ('type', 'typ', 'name', 'instruction'):
            return getattr(value, attr)

        if isinstance(value.type, Vector):
            return get_current_builder().swizzle(value, attr)
        raise AttributeError(f"IR type {value.type} has no attribute {attr}")
        
    # Host side
    # If it's a Struct object
    if hasattr(value, 'to_tuple') and hasattr(value, 'get_dsl_type'):
        return getattr(value, attr)

    # If it's a tuple representing a struct
    if isinstance(value, tuple) and hasattr(value, '_dsl_type'):
        idx = value._dsl_type.get_field_index(attr)
        return value[idx]
        
    # If it's a vector constant (tuple)
    if isinstance(value, tuple) and len(value) in (2,3,4):
        # Check if it looks like a swizzle
        from .router import vector_swizzle
        try:
            return vector_swizzle(value, attr)
        except ValueError:
            pass

    return getattr(value, attr)


# ============================================================================
# Core Language Operations
# ============================================================================

def return_(value: Any = None) -> None:
    """Handle return statement."""
    if value is not None:
        val = to_ir_value(value)
        get_current_builder().return_(val)
    else:
        get_current_builder().return_(None)


def local_assign(name: str, value: Any) -> Any:
    """Helper to store a value in the builder's local namespace."""
    return value


def local_var_assign(name: str, value: Any) -> Any:
    """
    Helper to create a DSL variable and store a value in it.
    
    This creates an alloca instruction and stores the value, returning
    the alloca'd location (which is a reference/pointer).
    
    If the value is not a DSL-compatible type (e.g., Builder, str, etc.),
    just return the value as-is (it's a Python variable).
    """
    from ..transform.ir import ConstantValue
    
    # Ensure it's a Value object for IR visibility
    value = try_to_ir_value(value)
    
    if isinstance(value, Value):
        builder = get_current_builder()
        var_ptr = builder.alloca(value.type, name=name)
        builder.store(var_ptr, value)
        return var_ptr
    
    # Otherwise, it's a Python variable (Builder, str, etc.) - just return as-is
    return value


def set_location(file: str, line: int) -> None:
    """Helper to set the current source location."""
    get_current_builder().set_location(file, line)


def load(ptr: Any) -> Any:
    """Helper to load from a reference."""
    return get_current_builder().load(ptr)


def maybe_load(ptr: Any) -> Any:
    """
    Helper to load from a reference, or convert to IR value if not a reference.
    
    This handles the case where a variable starts as a Python value but is
    later used in DSL context. If ptr is not a Value, it's converted to one.
    If ptr is a pointer (from alloca), it's loaded from.
    """
    if isinstance(ptr, Value) and ptr.is_pointer:
        return get_current_builder().load(ptr)
    else:
        # It's a Python value or a non-pointer IR value - convert to IR value
        return to_ir_value(ptr)


def store(ptr: Any, value: Any) -> None:
    """Helper to store to a reference."""
    get_current_builder().store(ptr, value)
