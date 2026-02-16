"""
JIT Compilation Support for the LuisaCompute Python DSL v2.

This module provides the @kernel and @callable decorators and JIT compilation logic.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Callable, Optional, Union

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
    """Base class for staged functions, holding metadata and template specialization logic."""

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
        # Cache for specialized versions
        self._specialized_cache: dict[tuple, SpecializedFunction] = {}

        import inspect
        self.filename = inspect.getsourcefile(func) or "<unknown>"
        self.compiled_code = None

    @property
    def name(self) -> str:
        return self.parsed.name

    @property
    def ir(self) -> Function:
        """Templates cannot have IR until specialized."""
        raise TypeError(f"Template function {self.name} must be specialized with [] before accessing IR")

    def __call__(self, *args, **kwargs) -> Function:
        """Templates cannot be called until specialized."""
        raise TypeError(f"Template function {self.name} must be specialized with [] before calling")

    def _do_compile(self):
        """Perform AST rewrite and compilation (Stage 3)."""
        if self.compiled_code is not None:
            return self.compiled_code

        # Rewrite the AST
        rewriter = ASTRewriter(file=self.filename)
        rewritten_ast = rewriter.rewrite(self.parsed.ast_node)
        ast.fix_missing_locations(rewritten_ast)

        if os.environ.get("LUISA_DUMP_REWRITTEN_AST") in ("1", "ON", "TRUE", "true", "yes"):
            print(f"DEBUG: Rewritten AST for {self.name}:\n{ast.unparse(rewritten_ast)}")

        self.compiled_code = compile(
            ast.Module(body=[rewritten_ast], type_ignores=[]),
            filename=f"<luisa-built-{self.name}>",
            mode="exec"
        )
        self.rewritten_ast = rewritten_ast
        return self.compiled_code

    def builder_func(self, *args, specialization_values: tuple):
        """The internal function that populates the IR builder."""
        compiled_code = self._do_compile()

        # Prepare namespace with specializations
        spec_dict = dict(zip(self.template_params, specialization_values))

        from . import builtins
        from . import ops as rt
        
        # Inject all builtins for easy access, if available
        builtin_namespace = {}
        if hasattr(builtins, "__all__"):
            for name in builtins.__all__:
                if hasattr(builtins, name):
                    builtin_namespace[name] = getattr(builtins, name)

        namespace = {
            "__luisa_rt": rt,
            "ast": ast,
            "static_range": static_range,
            **builtin_namespace,
            **{name: var.value for name, var in self.parsed.captured_vars.items()},
            **spec_dict  # Inject template parameters
        }
        if self.pyfunc and hasattr(self.pyfunc, "__globals__"):
            for name, val in self.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val

        # Execute to define the built function
        exec(compiled_code, namespace)
        built_func = namespace[f"__luisa_built_{self.name}"]

        # Call it
        return built_func(*args)

    def __getitem__(self, items) -> SpecializedFunction:
        """Support func[Int, 2] specialization (Stage 2)."""
        if not isinstance(items, tuple):
            items = (items,)
        
        if items not in self._specialized_cache:
            self._specialized_cache[items] = SpecializedFunction(self, items)
        return self._specialized_cache[items]

    def resolve_annotation(self, ann: Any, specialization_values: tuple) -> Any:
        """Resolve a type annotation, potentially replacing template parameters."""
        if isinstance(ann, str) and self.template_params and specialization_values:
            # Create mapping of param names to values
            spec_map = dict(zip(self.template_params, specialization_values))
            if ann in spec_map:
                ann = spec_map[ann]
        
        # Convert to DSL type
        from .types import annotation_to_type
        typ, _ = annotation_to_type(ann)
        return typ


class SpecializedFunction:
    """A specialized staged function (normal function or specialized template)."""

    def __init__(self, staged: StagedFunction, specialization_values: tuple):
        self.staged = staged
        self.specialization_values = specialization_values
        # Cache for compiled versions (keyed by argument types)
        self._cache: dict[tuple[Type, ...], Function] = {}

        # Normal functions and specialized templates are compiled immediately (Stage 3)
        self.staged._do_compile()

        # If all argument types are known, pre-compile immediately (Stage 4)
        if all(ann is not None for ann in self.staged.parsed.arg_annotations):
            try:
                self._precompile()
            except (AttributeError, KeyError, NameError, TypeError):
                # Defer until first use if types/modules not ready
                pass

    @property
    def ir(self) -> Function:
        """Get the generated IR for this function."""
        # If already cached (e.g. from precompile or previous call), return it
        if self._cache:
            return next(iter(self._cache.values()))
        
        # If not cached, try to precompile if we have enough info
        return self._precompile()

    def _precompile(self) -> Function:
        """Internal helper to build IR using annotated types."""
        # Use None as placeholders for arguments - __call__ will handle type extraction from annotations
        placeholders = [None] * len(self.staged.parsed.arg_annotations)
        return self(*placeholders)

    @property
    def parsed(self) -> ParsedFunction:
        return self.staged.parsed

    @property
    def name(self) -> str:
        return self.staged.name

    @property
    def pyfunc(self) -> Optional[Callable]:
        return self.staged.pyfunc

    @property
    def is_kernel(self) -> bool:
        return self.staged.is_kernel

    def _do_compile(self):
        """Delegate compilation to the base staged function."""
        return self.staged._do_compile()

    def builder_func(self, *args):
        """Populate the IR builder using specialization values."""
        return self.staged.builder_func(*args, specialization_values=self.specialization_values)

    def compile(self, builder: Builder, *args) -> Function:
        """Compile the function for given arguments and return the Function."""
        arg_values = list(args)
        # Use provided builder for compile entry point
        with set_current_builder(builder):
            from .ops import to_ir_value
            ir_args = [to_ir_value(a) for a in arg_values]
            arg_types = tuple(a.type for a in ir_args)

            if arg_types not in self._cache:
                # This will populate self._cache[arg_types]
                self(*ir_args)

            return self._cache[arg_types]

    def __call__(self, *args, **kwargs) -> Function:
        arg_types = []
        arg_is_reference = []
        for i, arg in enumerate(args):
            is_ref = False
            if i < len(self.staged.parsed.arg_annotations):
                ann = self.staged.parsed.arg_annotations[i]
                is_ref = self.staged.parsed.arg_is_reference[i]
                
                # Resolve annotation if it's a template parameter
                if ann is not None:
                    resolved_ann = self.staged.resolve_annotation(ann, self.specialization_values)
                    
                    # If we have a resolved type and arg is None (placeholder), use the resolved type
                    if resolved_ann is not None and arg is None:
                        arg_types.append(resolved_ann)
                        arg_is_reference.append(is_ref)
                        continue
                    
                    # If arg is provided, we might still want to use its actual type if it's more specific,
                    # but usually we trust the annotation if it's present.
                    if resolved_ann is not None:
                        arg_types.append(resolved_ann)
                        arg_is_reference.append(is_ref)
                        continue

            # Fallback to inferred type from argument value
            inferred = self._get_arg_type(arg)
            arg_types.append(inferred)
            arg_is_reference.append(is_ref)
            
        arg_types = tuple(arg_types)

        if arg_types in self._cache:
            return self._cache[arg_types]

        # Resolve return type
        ret_type = self.staged.resolve_annotation(self.staged.parsed.ret_annotation, self.specialization_values)

        builder = Builder(
            name=self.staged.parsed.name,
            arg_types=arg_types,
            ret_type=ret_type,
            arg_is_reference=arg_is_reference
        )
        with set_current_builder(builder):
            # Set initial location
            builder.set_location(self.staged.filename, self.staged.parsed.ast_node.lineno)

            entry = builder.create_block("entry")
            builder.set_insert_point(entry)

            arg_values = [builder.get_argument(i) for i in range(len(arg_types))]
            self.staged.builder_func(*arg_values, specialization_values=self.specialization_values)

        ir_func = builder.build()
        ir_func.is_kernel = self.staged.is_kernel
        self._cache[arg_types] = ir_func
        return ir_func

    def _get_arg_type(self, arg: Any) -> Type:
        inferred = value_to_type(arg)
        if inferred is not None:
            return inferred
        return arg.type if hasattr(arg, 'type') else arg.__class__.__name__


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
                 source: Optional[str] = None) -> Union[StagedFunction, SpecializedFunction]:
        staged = StagedFunction(func, is_kernel=self.is_kernel, template_params=self.params,
                                ast_node=ast_node, source=source)
        if self.params:
            return staged
        else:
            return SpecializedFunction(staged, ())


kernel = StagedFunctionDecorator(is_kernel=True)
callable = StagedFunctionDecorator(is_kernel=False)
