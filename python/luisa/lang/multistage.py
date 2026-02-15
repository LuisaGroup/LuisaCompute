"""
Multistage Programming Support for the LuisaCompute Python DSL v2.

This module combines the JIT compiler (StagedFunction) and the runtime support
needed for the rewritten AST to generate IR.
"""

from __future__ import annotations
import ast
import sys
import copy
import os
from typing import Callable, Optional, Any, TYPE_CHECKING
from contextlib import contextmanager

from .builder import IRBuilder
from .ir import Value, IROp, IRFunction
from .types import Type, value_to_type, bool_, int32, float32, Scalar, Vector, Buffer, Array
from .parser import parse_function, CapturedVar, ParsedFunction
from .rewriter import ASTRewriter
from .builtins.math import set_builder as set_math_builder


# ============================================================================
# Runtime Helper Functions (called by rewritten code)
# ============================================================================

def is_ir_value(val: Any) -> bool:
    """Check if a value is an IR Value."""
    return isinstance(val, Value)

def to_ir_value(builder: IRBuilder, val: Any) -> Value:
    """Ensure a value is an IR Value, converting literals if necessary."""
    if isinstance(val, Value):
        return val
    if isinstance(val, str):
        return val
    
    typ = value_to_type(val)
    if typ is None:
        raise TypeError(f"Cannot convert {type(val)} to Luisa type")
    return builder.constant(typ, val)

def _try_to_ir_value(builder: IRBuilder, val: Any) -> Any:
    """Try to convert to IR value, return original if not possible."""
    try:
        return to_ir_value(builder, val)
    except TypeError:
        return val

def l_binop(builder: IRBuilder, op: ast.operator, left: Any, right: Any) -> Any:
    """Handle binary operations."""
    if is_ir_value(left) or is_ir_value(right):
        left = to_ir_value(builder, left)
        right = to_ir_value(builder, right)
        if isinstance(op, ast.Add): return builder.add(left, right)
        if isinstance(op, ast.Sub): return builder.sub(left, right)
        if isinstance(op, ast.Mult): return builder.mul(left, right)
        if isinstance(op, ast.Div): return builder.div(left, right)
        if isinstance(op, ast.Mod): return builder.mod(left, right)
        if isinstance(op, ast.Pow): return builder.pow(left, right)
        if isinstance(op, ast.FloorDiv): 
            return builder.floor(builder.div(left, right))
        if isinstance(op, ast.BitAnd): return builder.bit_and(left, right)
        if isinstance(op, ast.BitOr): return builder.bit_or(left, right)
        if isinstance(op, ast.BitXor): return builder.bit_xor(left, right)
        if isinstance(op, ast.LShift): return builder.shl(left, right)
        if isinstance(op, ast.RShift): return builder.shr(left, right)
        raise NotImplementedError(f"Unsupported binary operator for IR: {type(op)}")
    else:
        # Host side
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right
        if isinstance(op, ast.Mod): return left % right
        if isinstance(op, ast.Pow): return left ** right
        if isinstance(op, ast.FloorDiv): return left // right
        if isinstance(op, ast.BitAnd): return left & right
        if isinstance(op, ast.BitOr): return left | right
        if isinstance(op, ast.BitXor): return left ^ right
        if isinstance(op, ast.LShift): return left << right
        if isinstance(op, ast.RShift): return left >> right
        raise NotImplementedError(f"Unsupported binary operator: {type(op)}")

def l_unaryop(builder: IRBuilder, op: ast.unaryop, operand: Any) -> Any:
    """Handle unary operations."""
    if is_ir_value(operand):
        operand = to_ir_value(builder, operand)
        if isinstance(op, ast.USub): return builder.neg(operand)
        if isinstance(op, ast.Not): return builder.logical_not(operand)
        if isinstance(op, ast.Invert): return builder.bit_not(operand)
        raise NotImplementedError(f"Unsupported unary operator for IR: {type(op)}")
    else:
        if isinstance(op, ast.USub): return -operand
        if isinstance(op, ast.Not): return not operand
        if isinstance(op, ast.Invert): return ~operand
        raise NotImplementedError(f"Unsupported unary operator: {type(op)}")

def l_compare(builder: IRBuilder, op: ast.cmpop, left: Any, right: Any) -> Any:
    """Handle comparison operations."""
    if is_ir_value(left) or is_ir_value(right):
        left = to_ir_value(builder, left)
        right = to_ir_value(builder, right)
        if isinstance(op, ast.Eq): return builder.eq(left, right)
        if isinstance(op, ast.NotEq): return builder.ne(left, right)
        if isinstance(op, ast.Lt): return builder.lt(left, right)
        if isinstance(op, ast.LtE): return builder.le(left, right)
        if isinstance(op, ast.Gt): return builder.gt(left, right)
        if isinstance(op, ast.GtE): return builder.ge(left, right)
        raise NotImplementedError(f"Unsupported comparison operator for IR: {type(op)}")
    else:
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.Is): return left is right
        if isinstance(op, ast.IsNot): return left is not right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
        raise NotImplementedError(f"Unsupported comparison operator: {type(op)}")

def l_boolop(builder: IRBuilder, op: ast.boolop, values: list[Any]) -> Any:
    """Handle boolean operations (and/or)."""
    is_ir = any(is_ir_value(v) for v in values)
    if is_ir:
        ir_values = [to_ir_value(builder, v) for v in values]
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

# ============================================================================
# Control Flow Helpers
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
            # We still need to yield but maybe something that indicates skip?
            # Actually if we are not in the scope, the body shouldn't run.
            # But 'with' always runs the body unless __enter__ raises or we use a generator.
            # StagedFunction uses @contextmanager which handles this.
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

HostIf = StaticIf # Legacy alias

def l_if(builder: IRBuilder, cond_func: Callable[[], Any]) -> Any:
    """Handle if statements."""
    cond = cond_func()
    if is_ir_value(cond):
        return builder.if_(cond)
    else:
        return StaticIf(cond)

def l_for(builder: IRBuilder, iter_obj: Any, loop_var_name: Any) -> Any:
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
        
        start = to_ir_value(builder, start_val)
        stop = to_ir_value(builder, stop_val)
        step = to_ir_value(builder, step_val)
        name = loop_var_name
        stmt = builder.for_range(start, stop, step, name)
        return [stmt]
    
    if isinstance(iter_obj, StaticRange):
        return iter_obj.rng
    
    return iter_obj

@contextmanager
def l_loop_scope(builder: IRBuilder, loop_item: Any):
    if hasattr(loop_item, 'body_scope'):
        with loop_item.body_scope() as scope:
            yield scope
    else:
        yield loop_item

def l_while(builder: IRBuilder, test_func: Callable[[], Any]) -> Any:
    """Handle while loops (returns generator)."""
    cond = test_func()
    if is_ir_value(cond):
        stmt = builder.while_(cond)
        yield stmt
    else:
        while cond:
            yield None
            cond = test_func()

@contextmanager
def l_while_scope(builder: IRBuilder, loop_item: Any):
    if loop_item is not None and hasattr(loop_item, 'body_scope'):
         with loop_item.body_scope() as scope:
            yield scope
    else:
        yield None

# ============================================================================
# Static Constructs
# ============================================================================

class StaticRange:
    def __init__(self, *args):
        self.rng = range(*args)
    def __iter__(self):
        return iter(self.rng)

def static_range(*args):
    """Static range for meta-programming loops."""
    return StaticRange(*args)

unrolled = static_range # Legacy alias

# ============================================================================
# Other Runtime Helpers
# ============================================================================

class LuisaRange:
    """A range object that can contain IR Values."""
    def __init__(self, *args):
        self.args = args

def l_call(builder: IRBuilder, func: Any, *args, **kwargs) -> Any:
    """Handle function calls."""
    from .types import Type
    
    # Handle built-in range()
    if func is range:
        if any(is_ir_value(a) for a in args):
            return LuisaRange(*args)
        return range(*args)

    if isinstance(func, Type):
        if len(args) != 1:
            raise ValueError(f"Type cast takes exactly one argument, got {len(args)}")
        val = to_ir_value(builder, args[0])
        return builder.cast(val, func)

    if isinstance(func, StagedFunction):
        arg_values = list(args)
        ir_args = [to_ir_value(builder, a) for a in arg_values]
        arg_types = tuple(a.type for a in ir_args)
        
        if arg_types not in func._cache:
            func._cache[arg_types] = func(*ir_args)
        
        return builder.call(func._cache[arg_types], ir_args)
    
    import builtins
    if builtins.callable(func):
        if any(is_ir_value(a) for a in args):
            new_args = []
            for a in args:
                if isinstance(a, str):
                    new_args.append(a)
                else:
                    new_args.append(_try_to_ir_value(builder, a))
            return func(*new_args, **kwargs)
        return func(*args, **kwargs)
    
    raise TypeError(f"Object {func} is not callable")

def l_subscript(builder: IRBuilder, value: Any, index: Any) -> Any:
    if is_ir_value(value) or is_ir_value(index):
        value = to_ir_value(builder, value)
        index = to_ir_value(builder, index)
        if isinstance(value.type, (Buffer, Array)):
            return builder.buffer_read(value, index, value.type.element)
        raise TypeError(f"Cannot subscript IR type {value.type}")
    return value[index]

def l_subscript_assign(builder: IRBuilder, value: Any, index: Any, rhs: Any) -> None:
    if is_ir_value(value) or is_ir_value(index) or is_ir_value(rhs):
        value = to_ir_value(builder, value)
        index = to_ir_value(builder, index)
        rhs = to_ir_value(builder, rhs)
        if isinstance(value.type, (Buffer, Array)):
            builder.buffer_write(value, index, rhs)
        else:
            raise TypeError(f"Cannot assign to subscript of IR type {value.type}")
    else:
        value[index] = rhs

def l_attribute(builder: IRBuilder, value: Any, attr: str) -> Any:
    if is_ir_value(value):
        # Allow accessing standard attributes of Value/InstructionValue
        if attr in ('type', 'typ', 'name', 'instruction'):
            return getattr(value, attr)
        
        from .types import Vector
        if isinstance(value.type, Vector):
            return builder.swizzle(value, attr)
        raise AttributeError(f"IR type {value.type} has no attribute {attr}")
    # Host side
    return getattr(value, attr)

def l_return(builder: IRBuilder, value: Any = None) -> None:
    if value is not None:
        val = to_ir_value(builder, value)
        builder.return_(val)
    else:
        builder.return_(None)

def l_local_assign(builder: IRBuilder, name: str, value: Any) -> Any:
    """Helper to store a value in the builder's local namespace."""
    # This ensures that even if we rewrite an assignment, 
    # we can still track the name if needed.
    # For now, it just returns the value so standard Python assignment works.
    return value

# ============================================================================
# Specialization
# ============================================================================

class Specialization:
    """Helper to manage specialized parameters."""
    def __init__(self, names: tuple[str, ...], values: tuple[Any, ...]):
        self.params = dict(zip(names, values))

    def __repr__(self) -> str:
        return f"Specialization({self.params})"


# ============================================================================
# Staged Function
# ============================================================================

class StagedFunction:
    """A staged function that generates IR when called."""
    
    def __init__(self, func: Callable, is_kernel: bool = False, 
                 parsed: Optional[ParsedFunction] = None,
                 template_params: Optional[tuple[str, ...]] = None,
                 ast_node: Optional[ast.FunctionDef] = None,
                 source: Optional[str] = None):
        self.pyfunc = func
        self.is_kernel = is_kernel
        
        if ast_node is not None:
            # If AST is provided, we use it (useful for nested functions)
            func._luisa_ast = ast_node
            
        if parsed is not None:
            self.parsed = parsed
        else:
            self.parsed = parse_function(func, source=source)
        
        self.template_params = template_params or ()
        # Cache for compiled versions (keyed by argument types AND specialization values)
        self._cache: dict[tuple[tuple[Type, ...], tuple[Any, ...]], IRFunction] = {}
        
        import inspect
        self.filename = inspect.getsourcefile(func) or "<unknown>"
        
        self.compiled_code = None
        
        # If it's NOT a template, we can rewrite now.
        if not self.template_params:
            self._do_compile()

    def _do_compile(self):
        """Perform AST rewrite and compilation."""
        if self.compiled_code is not None:
            return
            
        rewriter = ASTRewriter(file=self.filename, template_params=self.template_params)
        self.rewritten_ast = rewriter.rewrite(self.parsed.ast_node)
        ast.fix_missing_locations(self.rewritten_ast)
        
        if os.environ.get("LUISA_DUMP_REWRITTEN_AST") in ("1", "ON", "TRUE", "true", "yes"):
            print(f"DEBUG: Rewritten AST for {self.name}:\n{ast.unparse(self.rewritten_ast)}")
        
        self.compiled_code = compile(
            ast.Module(body=[self.rewritten_ast], type_ignores=[]), 
            filename=f"<luisa-built-{self.name}>", 
            mode="exec"
        )

    def builder_func(self, builder: IRBuilder, *args, specialization_values: tuple = ()):
        """The internal function that populates the IR builder."""
        self._do_compile()
        
        # Prepare namespace with specializations
        spec_dict = dict(zip(self.template_params, specialization_values))
        
        namespace = {
            "__luisa_rt": sys.modules[__name__],
            "ast": ast,
            "static_range": static_range,
            **{name: var.value for name, var in self.parsed.captured_vars.items()},
            **spec_dict # Inject template parameters
        }
        if self.pyfunc and hasattr(self.pyfunc, "__globals__"):
            for name, val in self.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val
        
        # Execute to define the built function
        exec(self.compiled_code, namespace)
        built_func = namespace[f"__luisa_built_{self.name}"]
        
        # Call it
        return built_func(builder, *args)
    
    @property
    def name(self) -> str:
        return self.parsed.name

    def __getitem__(self, items) -> SpecializedFunctionProxy:
        """Support func[int32, 2](x) syntax."""
        if not isinstance(items, tuple):
            items = (items,)
        return SpecializedFunctionProxy(self, items)
    
    def __call__(self, *args, specialization_values: tuple = (), **kwargs) -> IRFunction:
        arg_types = []
        arg_is_reference = []
        for i, arg in enumerate(args):
            if i < len(self.parsed.arg_annotations) and self.parsed.arg_annotations[i] is not None:
                ann = self.parsed.arg_annotations[i]
                is_ref = self.parsed.arg_is_reference[i]
                arg_types.append(ann)
                arg_is_reference.append(is_ref)
            else:
                arg_types.append(self._get_arg_type(arg))
                arg_is_reference.append(False)
        arg_types = tuple(arg_types)
        
        cache_key = (arg_types, specialization_values)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        builder = IRBuilder(
            name=self.parsed.name, 
            arg_types=arg_types, 
            ret_type=self.parsed.ret_annotation,
            arg_is_reference=arg_is_reference
        )
        set_math_builder(builder)
        try:
            # Set initial location
            builder.set_location(self.filename, self.parsed.ast_node.lineno)
            
            entry = builder.create_block("entry")
            builder.set_insert_point(entry)
            
            arg_values = [builder.get_argument(i) for i in range(len(arg_types))]
            self.builder_func(builder, *arg_values, specialization_values=specialization_values)
            
        finally:
            set_math_builder(None)
        
        ir_func = builder.build()
        ir_func.is_kernel = self.is_kernel
        self._cache[cache_key] = ir_func
        return ir_func

    def _get_arg_type(self, arg: Any) -> Type:
        inferred = value_to_type(arg)
        if inferred is not None:
            return inferred
        return arg.type if hasattr(arg, 'type') else arg.__class__.__name__


class SpecializedFunctionProxy:
    """Proxy for a staged function with applied specialization values."""
    def __init__(self, staged: StagedFunction, values: tuple):
        self.staged = staged
        self.values = values
    
    def __call__(self, *args, **kwargs) -> IRFunction:
        return self.staged(*args, specialization_values=self.values, **kwargs)


class StagedFunctionDecorator:
    """Wrapper for kernel/callable decorators to support indexing."""
    def __init__(self, is_kernel: bool):
        self.is_kernel = is_kernel
        self.params = None

    def __getitem__(self, params) -> StagedFunctionDecorator:
        if not isinstance(params, tuple):
            params = (params,)
        
        param_names = []
        for p in params:
            if isinstance(p, str):
                param_names.append(p)
            elif hasattr(p, '__name__'):
                param_names.append(p.__name__)
            else:
                param_names.append(str(p))
        
        self.params = tuple(param_names)
        return self

    def __call__(self, func: Callable, ast_node: Optional[ast.FunctionDef] = None, source: Optional[str] = None) -> StagedFunction:
        return StagedFunction(func, is_kernel=self.is_kernel, template_params=self.params, ast_node=ast_node, source=source)


kernel = StagedFunctionDecorator(is_kernel=True)
callable = StagedFunctionDecorator(is_kernel=False)
