"""
JIT Compilation Support for the LuisaCompute Python DSL v2.

This module provides the @kernel and @callable decorators and JIT compilation logic.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Callable, Optional, Union

from ..transform.builder import Builder, get_current_builder, set_current_builder
from ..transform.inspect import ParsedFunction, parse_function
from ..transform.ir import Function
from ..transform.rewriter import ASTRewriter
from .types import (
    Type, value_to_type, annotation_to_type,
    Buffer, Array, Vector, Matrix,
    Float, Int, Bool, UInt, Double
)

# ============================================================================
# Static Constructs (Meta-programming)
# ============================================================================

class UnrolledRange:
    """Marker class for unrolled loops."""
    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1):
        if stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return iter(range(self.start, self.stop, self.step))

    def __len__(self) -> int:
        return max(0, (self.stop - self.start + self.step - 1) // self.step)


def unrolled(r: range | int) -> UnrolledRange:
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
    return StaticRange(*args)


# ============================================================================
# Staged and Templated Functions
# ============================================================================

class KernelInvoke:
    """Records a kernel and its arguments for later dispatch."""
    def __init__(self, kernel: StagedFunction, args: tuple):
        self.kernel = kernel
        self.args = args

    def __repr__(self) -> str:
        return f"KernelInvoke(kernel={self.kernel.name}, args={self.args})"


class TemplatedFunction:
    """Base metadata and factory for specialized staged functions.
    
    Supports both explicit template params (from decorator) and implicit
    template params (arguments without type annotations).
    """

    def __init__(self, func: Callable, is_kernel: bool = False,
                 parsed: Optional[ParsedFunction] = None,
                 template_params: Optional[tuple[str, ...]] = None,
                 ast_node: Optional[ast.FunctionDef] = None,
                 source: Optional[str] = None,
                 specialization_values: tuple = (),
                 implicit_params: Optional[tuple[str, ...]] = None):
        self.pyfunc = func
        self.is_kernel = is_kernel
        # template_params are the EXPLICIT params from decorator like ['T', 'U']
        self.template_params = template_params or ()
        self.specialization_values = specialization_values
        self._implicit_params = implicit_params

        if ast_node is not None:
            func._luisa_ast = ast_node

        if parsed is not None:
            self.parsed = parsed
        else:
            self.parsed = parse_function(func, source=source)

        # Cache for specialized instances: (arg_types,) -> StagedFunction
        self._cache: dict[tuple[Type, ...], StagedFunction] = {}
        
        import inspect
        self.filename = inspect.getsourcefile(func) or "<unknown>"
        self.compiled_code = None
    
    @property
    def explicit_params(self) -> tuple[str, ...]:
        """Get explicit template params (from decorator)."""
        return self.template_params
    
    @property
    def implicit_params(self) -> tuple[str, ...]:
        """Get implicit template params (from unannotated arguments)."""
        if self._implicit_params is not None:
            return self._implicit_params
        return tuple(self.parsed.implicit_template_params)
    
    @property
    def all_template_params(self) -> tuple[str, ...]:
        """Get all template params (explicit + implicit)."""
        return self.explicit_params + self.implicit_params

    @property
    def name(self) -> str:
        return self.parsed.name

    def _do_compile(self):
        """Perform AST rewrite and compilation (Stage 3)."""
        if self.compiled_code is not None:
            return self.compiled_code

        rewriter = ASTRewriter(file=self.filename)
        rewritten_ast = rewriter.rewrite(self.parsed.ast_node)
        
        # Inject template param assignments at the start of the function body
        # This allows annotations and nested functions to naturally capture T, U, etc.
        # Include both explicit and implicit template params
        all_params = self.all_template_params
        if all_params:
            rewritten_ast = self._inject_template_params(rewritten_ast, all_params)
        
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

    def _inject_template_params(self, func_ast: ast.FunctionDef, 
                                 params: tuple[str, ...]) -> ast.FunctionDef:
        """Inject T = __luisa_spec.get('T') assignments at the start of function body.
        
        Uses .get() with default to handle partial specialization where not all
        template params are available yet.
        """
        # Create: T = __luisa_spec.get('T', T) for each template param
        # The default T preserves any existing binding (for nested scopes)
        inject_stmts = []
        for param in params:
            # T = __luisa_spec.get('T', T)  # second T will be NameError if not defined, which is fine
            assign = ast.Assign(
                targets=[ast.Name(id=param, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='__luisa_spec', ctx=ast.Load()),
                        attr='get',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=param)],
                    keywords=[]
                )
            )
            inject_stmts.append(assign)
        
        # Prepend to function body
        func_ast.body = inject_stmts + func_ast.body
        return func_ast

    def builder_func(self, *args, specialization_values: tuple):
        """Internal IR builder population."""
        compiled_code = self._do_compile()
        # Map all template params (explicit + implicit) to their values
        all_params = self.all_template_params
        spec_dict = dict(zip(all_params, specialization_values))

        from . import builtins
        from . import ops as rt
        
        builtin_namespace = {}
        if hasattr(builtins, "__all__"):
            for name in builtins.__all__:
                if hasattr(builtins, name):
                    builtin_namespace[name] = getattr(builtins, name)

        namespace = {
            "__luisa_rt": rt,
            "ast": ast,
            "static_range": static_range,
            "__luisa_spec": spec_dict,  # Template params injected via AST
            **builtin_namespace,
            **{name: var.value for name, var in self.parsed.captured_vars.items()},
        }
        if self.pyfunc and hasattr(self.pyfunc, "__globals__"):
            for name, val in self.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val

        exec(compiled_code, namespace)
        built_func = namespace[f"__luisa_built_{self.name}"]
        return built_func(*args)

    def __getitem__(self, items) -> Union[TemplatedFunction, StagedFunction]:
        """Support explicit template specialization.
        
        Only explicit template params (from decorator) can be specialized via [...].
        Implicit template params (unannotated args) are always deduced from call args.
        """
        explicit_params = self.explicit_params
        
        if not explicit_params:
            raise TypeError(f"Function '{self.name}' has no explicit template parameters to specialize")

        if not isinstance(items, tuple):
            items = (items,)
        
        new_explicit_values = self.specialization_values + items
        
        if len(new_explicit_values) > len(explicit_params):
            raise TypeError(f"Too many template arguments for '{self.name}': expected {len(explicit_params)}, got {len(new_explicit_values)}")

        # Check if we have all explicit params resolved and no implicit params
        if len(new_explicit_values) == len(explicit_params) and not self.implicit_params:
            # All params resolved and no implicit params - can create StagedFunction if annotations are complete
            if all(ann is not None for ann in self.parsed.arg_annotations):
                from .types import Type
                import inspect
                arg_types = tuple(self.resolve_annotation(ann, new_explicit_values) for ann in self.parsed.arg_annotations)
                if all(isinstance(t, Type) or (inspect.isclass(t) and issubclass(t, Type)) for t in arg_types):
                    return StagedFunction(self, new_explicit_values, arg_types)
        
        # Either still have unresolved explicit params, or have implicit params to deduce
        return TemplatedFunction(
            self.pyfunc, self.is_kernel, self.parsed, self.template_params, 
            specialization_values=new_explicit_values,
            implicit_params=self.implicit_params)

    def resolve_annotation(self, ann: Any, specialization_values: tuple) -> Any:
        """Resolve annotation, substituting template params with specialization values.
        
        Uses eval() with template params in namespace - this naturally handles
        complex nested generics like Buffer[Vector[T, 3]] via Python's evaluation.
        """
        if isinstance(ann, str):
            # Build namespace with template params for eval
            spec_map = dict(zip(self.template_params, specialization_values))
            
            # Add common types to namespace
            namespace = {
                **spec_map,
                'Buffer': Buffer, 'Array': Array,
                'Vector': Vector, 'Matrix': Matrix,
                'Float': Float, 'Int': Int, 'Bool': Bool,
                'UInt': UInt, 'Double': Double,
            }
            
            try:
                # Let Python eval resolve it naturally
                resolved = eval(ann, namespace)
                if isinstance(resolved, Type):
                    return resolved
            except (NameError, TypeError):
                pass
            
            # Fall back to standard parsing
            typ, _ = annotation_to_type(ann)
            return typ if typ is not None else ann

        # Already a Type object
        if isinstance(ann, Type):
            return ann

        typ, _ = annotation_to_type(ann)
        return typ

    def _get_or_create_staged(self, args: tuple) -> StagedFunction:
        """Infer types from arguments and get/create a StagedFunction.
        
        Also deduces implicit template params from unannotated arguments.
        """
        from .types import Type
        
        # Detect arg types and implicit template param values
        arg_types = []
        implicit_values = []  # Values for implicit template params
        
        for i, arg in enumerate(args):
            resolved_type = None
            # Try to resolve annotation if present
            if i < len(self.parsed.arg_annotations) and self.parsed.arg_annotations[i] is not None:
                resolved = self.resolve_annotation(self.parsed.arg_annotations[i], self.specialization_values)
                # If resolved to an actual Type (not a string/None), use it
                if isinstance(resolved, Type):
                    resolved_type = resolved
            
            # If not resolved from annotation, infer from value
            if resolved_type is None:
                resolved_type = value_to_type(arg) if not hasattr(arg, 'type') else arg.type
            
            arg_types.append(resolved_type)
            
            # Check if this arg corresponds to an implicit template param
            # Implicit params are stored as __impl_<arg_name>
            if i < len(self.parsed.arg_names):
                arg_name = self.parsed.arg_names[i]
                implicit_param_name = f"__impl_{arg_name}"
                if implicit_param_name in self.implicit_params:
                    # This is an implicit template param - use the inferred type as its value
                    implicit_values.append(resolved_type)
        
        arg_types = tuple(arg_types)
        
        # Combine explicit and implicit specialization values
        # explicit values + implicit values
        full_specialization_values = self.specialization_values + tuple(implicit_values)
        
        if arg_types not in self._cache:
            self._cache[arg_types] = StagedFunction(self, full_specialization_values, arg_types)
        return self._cache[arg_types]

    def __call__(self, *args, **kwargs) -> Any:
        """Handle DSL function call or Python-side IR retrieval."""
        staged = self._get_or_create_staged(args)
        return staged(*args, **kwargs)


class StagedFunction:
    """A fully specialized, ready-to-use DSL function instance."""

    def __init__(self, templated: TemplatedFunction, specialization_values: tuple, arg_types: tuple[Type, ...]):
        self.templated = templated
        self.specialization_values = specialization_values
        self.arg_types = arg_types
        
        # Resolve return type
        self.ret_type = templated.resolve_annotation(templated.parsed.ret_annotation, specialization_values)
        
        # Build IR immediately (Stage 4)
        self.ir = self._build_ir()

    def _build_ir(self) -> Function:
        builder = Builder(
            name=self.templated.parsed.name,
            arg_types=self.arg_types,
            ret_type=self.ret_type,
            arg_is_reference=self.templated.parsed.arg_is_reference
        )
        with set_current_builder(builder):
            builder.set_location(self.templated.filename, self.templated.parsed.ast_node.lineno)
            entry = builder.create_block("entry")
            builder.set_insert_point(entry)
            arg_values = [builder.get_argument(i) for i in range(len(self.arg_types))]
            self.templated.builder_func(*arg_values, specialization_values=self.specialization_values)

        ir_func = builder.build()
        ir_func.is_kernel = self.templated.is_kernel
        return ir_func

    @property
    def is_kernel(self) -> bool:
        return self.templated.is_kernel

    @property
    def name(self) -> str:
        return self.templated.name

    @property
    def parsed(self) -> ParsedFunction:
        return self.templated.parsed

    def _do_compile(self):
        return self.templated._do_compile()

    def builder_func(self, *args):
        return self.templated.builder_func(*args, specialization_values=self.specialization_values)

    def compile(self, builder: Builder, *args) -> Function:
        """Already specialized, just return IR."""
        return self.ir

    def __call__(self, *args, **kwargs) -> Any:
        """Handle DSL function call or Python-side IR retrieval."""
        try:
            builder = get_current_builder()
            if self.is_kernel:
                raise RuntimeError(f"Cannot call kernel '{self.name}' from within another kernel/callable.")
            return builder.call(self, *args)
        except RuntimeError as e:
            if "No active builder context" in str(e):
                if not self.is_kernel:
                    raise RuntimeError(f"Callable '{self.name}' can only be called from within a kernel or another callable.")
                return KernelInvoke(self, args)
            raise e


class StagedFunctionDecorator:
    """Wrapper for kernel/callable decorators."""

    def __init__(self, is_kernel: bool, params: Optional[tuple[str, ...]] = None):
        self.is_kernel = is_kernel
        self.params = params

    def __getitem__(self, params) -> StagedFunctionDecorator:
        if not isinstance(params, tuple):
            params = (params,)
        param_names = [p if isinstance(p, str) else (p.__name__ if hasattr(p, '__name__') else str(p)) for p in params]
        return StagedFunctionDecorator(self.is_kernel, params=tuple(param_names))

    def __call__(self, func: Callable, ast_node: Optional[ast.FunctionDef] = None,
                 source: Optional[str] = None) -> Union[TemplatedFunction, StagedFunction]:
        templated = TemplatedFunction(func, is_kernel=self.is_kernel, template_params=self.params,
                                      ast_node=ast_node, source=source)
        
        # Check if there are implicit template params (unannotated args)
        has_implicit_params = len(templated.implicit_params) > 0
        
        # If it's a normal function (no explicit params, no implicit params) and fully annotated, 
        # make it a StagedFunction immediately
        if not self.params and not has_implicit_params and all(ann is not None for ann in templated.parsed.arg_annotations):
            try:
                arg_types = tuple(templated.resolve_annotation(ann, ()) for ann in templated.parsed.arg_annotations)
                if all(t is not None for t in arg_types):
                    return StagedFunction(templated, (), arg_types)
            except Exception:
                pass # Defer if types/modules not ready
        
        return templated


kernel = StagedFunctionDecorator(is_kernel=True)
callable = StagedFunctionDecorator(is_kernel=False)
