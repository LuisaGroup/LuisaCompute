"""
JIT Compilation Support for the LuisaCompute Python DSL v2.

This module provides the @kernel and @callable decorators and JIT compilation logic.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Callable, Optional

from ..transform.builder import Builder, set_current_builder
from ..transform.inspect import ParsedFunction, parse_function
from ..transform.ir import Function
from ..transform.rewriter import ASTRewriter
from .types import Type, value_to_type

# ============================================================================
# Static Constructs (Meta-programming)
# ============================================================================

class UnrolledRange:
    """
    Marker class for unrolled loops.

    Usage:
        for i in unrolled(range(4)):
            ...  # This loop is unrolled at compile time

    The loop body will be replicated for each iteration.
    Use only for small iteration counts to avoid code bloat!
    """

    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1):
        if stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        """Python-side iteration (for reference)."""
        return iter(range(self.start, self.stop, self.step))

    def __len__(self) -> int:
        """Return the number of iterations."""
        return max(0, (self.stop - self.start + self.step - 1) // self.step)


def unrolled(r: range | int) -> UnrolledRange:
    """
    Mark a range for compile-time unrolling.

    Usage:
        for i in unrolled(range(4)):      # Unrolled: 0, 1, 2, 3
        for i in unrolled(4):             # Equivalent to above

    The loop body will be replicated for each iteration at compile time.
    """
    if isinstance(r, int):
        return UnrolledRange(r)
    return UnrolledRange(r.start, r.stop, r.step)


class StaticRange:
    """Range for host-side (static) iteration."""
    def __init__(self, *args):
        self.rng = range(*args)

    def __iter__(self):
        return iter(self.rng)


def static_range(*args):
    """Static range for meta-programming loops."""
    return StaticRange(*args)


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
        self._cache: dict[tuple[tuple[Type, ...], tuple[Any, ...]], Function] = {}

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

    def builder_func(self, *args, specialization_values: tuple = ()):
        """The internal function that populates the IR builder."""
        self._do_compile()

        # Prepare namespace with specializations
        spec_dict = dict(zip(self.template_params, specialization_values))

        from . import builtins
        from . import ops as rt
        namespace = {
            "__luisa_rt": rt,
            "ast": ast,
            "static_range": static_range,
            # Inject all builtins for easy access
            **{name: getattr(builtins, name) for name in builtins.__all__},
            **{name: var.value for name, var in self.parsed.captured_vars.items()},
            **spec_dict  # Inject template parameters
        }
        if self.pyfunc and hasattr(self.pyfunc, "__globals__"):
            for name, val in self.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val

        # Execute to define the built function
        exec(self.compiled_code, namespace)
        built_func = namespace[f"__luisa_built_{self.name}"]

        # Call it
        return built_func(*args)

    @property
    def name(self) -> str:
        return self.parsed.name

    def __getitem__(self, items) -> SpecializedFunctionProxy:
        """Support func[Int, 2](x) syntax."""
        if not isinstance(items, tuple):
            items = (items,)
        return SpecializedFunctionProxy(self, items)

    def compile(self, builder: Builder, *args, specialization_values: tuple = ()) -> Function:
        """Compile the function for given arguments and return the Function."""
        arg_values = list(args)
        # Use provided builder for compile entry point
        with set_current_builder(builder):
            from .ops import to_ir_value
            ir_args = [to_ir_value(a) for a in arg_values]
            arg_types = tuple(a.type for a in ir_args)

            cache_key = (arg_types, specialization_values)
            if cache_key not in self._cache:
                # This will populate self._cache[cache_key]
                self(*ir_args, specialization_values=specialization_values)

            return self._cache[cache_key]

    def __call__(self, *args, specialization_values: tuple = (), **kwargs) -> Function:
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

        builder = Builder(
            name=self.parsed.name,
            arg_types=arg_types,
            ret_type=self.parsed.ret_annotation,
            arg_is_reference=arg_is_reference
        )
        with set_current_builder(builder):
            # Set initial location
            builder.set_location(self.filename, self.parsed.ast_node.lineno)

            entry = builder.create_block("entry")
            builder.set_insert_point(entry)

            arg_values = [builder.get_argument(i) for i in range(len(arg_types))]
            self.builder_func(*arg_values, specialization_values=specialization_values)

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

    def __call__(self, *args, **kwargs) -> Function:
        return self.staged(*args, specialization_values=self.values, **kwargs)


class StagedFunctionDecorator:
    """Wrapper for kernel/callable decorators to support indexing."""

    def __init__(self, is_kernel: bool, params: Optional[tuple[str, ...]] = None):
        self.is_kernel = is_kernel
        self.params = params

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

        # Return a NEW decorator instance with the params set
        return StagedFunctionDecorator(self.is_kernel, params=tuple(param_names))

    def __call__(self, func: Callable, ast_node: Optional[ast.FunctionDef] = None,
                 source: Optional[str] = None) -> StagedFunction:
        return StagedFunction(func, is_kernel=self.is_kernel, template_params=self.params, ast_node=ast_node,
                              source=source)


kernel = StagedFunctionDecorator(is_kernel=True)
callable = StagedFunctionDecorator(is_kernel=False)
