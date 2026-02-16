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
from .types import Type, value_to_type

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
    """Base metadata and factory for specialized staged functions."""

    def __init__(self, func: Callable, is_kernel: bool = False,
                 parsed: Optional[ParsedFunction] = None,
                 template_params: Optional[tuple[str, ...]] = None,
                 ast_node: Optional[ast.FunctionDef] = None,
                 source: Optional[str] = None,
                 specialization_values: tuple = ()):
        self.pyfunc = func
        self.is_kernel = is_kernel
        self.template_params = template_params or ()
        self.specialization_values = specialization_values

        if ast_node is not None:
            func._luisa_ast = ast_node

        if parsed is not None:
            self.parsed = parsed
        else:
            self.parsed = parse_function(func, source=source)

        # Cache for specialized instances: (arg_types) -> StagedFunction
        self._cache: dict[tuple[Type, ...], StagedFunction] = {}
        
        import inspect
        self.filename = inspect.getsourcefile(func) or "<unknown>"
        self.compiled_code = None

    @property
    def name(self) -> str:
        return self.parsed.name

    def _do_compile(self):
        """Perform AST rewrite and compilation (Stage 3)."""
        if self.compiled_code is not None:
            return self.compiled_code

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
        """Internal IR builder population."""
        compiled_code = self._do_compile()
        spec_dict = dict(zip(self.template_params, specialization_values))

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
            **builtin_namespace,
            **{name: var.value for name, var in self.parsed.captured_vars.items()},
            **spec_dict
        }
        if self.pyfunc and hasattr(self.pyfunc, "__globals__"):
            for name, val in self.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val

        exec(compiled_code, namespace)
        built_func = namespace[f"__luisa_built_{self.name}"]
        return built_func(*args)

    def __getitem__(self, items) -> Union[TemplatedFunction, StagedFunction]:
        """Support template specialization."""
        if not self.template_params:
            raise TypeError(f"Function '{self.name}' is not templated (no template parameters declared in decorator)")

        if not isinstance(items, tuple):
            items = (items,)
        
        new_values = self.specialization_values + items
        
        if len(new_values) > len(self.template_params):
            raise TypeError(f"Too many template arguments for '{self.name}': expected {len(self.template_params)}, got {len(new_values)}")

        # Check if we have enough info to become a StagedFunction
        # (Must have all template params AND all args annotated)
        if len(new_values) == len(self.template_params) and \
           all(ann is not None for ann in self.parsed.arg_annotations):
            from .types import Type
            import inspect
            arg_types = tuple(self.resolve_annotation(ann, new_values) for ann in self.parsed.arg_annotations)
            if all(isinstance(t, Type) or (inspect.isclass(t) and issubclass(t, Type)) for t in arg_types):
                return StagedFunction(self, new_values, arg_types)
        
        # Still polymorphic (either template params missing or arg annotations missing)
        return TemplatedFunction(self.pyfunc, self.is_kernel, self.parsed, self.template_params, 
                                 specialization_values=new_values)

    def resolve_annotation(self, ann: Any, specialization_values: tuple) -> Any:
        """Resolve annotation, substituting template params with specialization values."""
        from .types import Type, Buffer, Array, annotation_to_type
        
        # Build spec map for template resolution
        spec_map = {}
        if self.template_params and specialization_values:
            spec_map = dict(zip(self.template_params, specialization_values))
        
        # Handle string annotations - may contain template params like 'Buffer[T]' or just 'T'
        if isinstance(ann, str):
            # Check if it's a simple template param like 'T'
            if ann in spec_map:
                return spec_map[ann]
            
            # Try to parse as a generic type like 'Buffer[T]'
            resolved = self._resolve_generic_string_annotation(ann, spec_map)
            if resolved is not None:
                return resolved
            
            # Fall back to standard annotation parsing
            typ, _ = annotation_to_type(ann)
            return typ if typ is not None else ann

        # Handle Type objects with template parameters, e.g., Buffer(element='T')
        if isinstance(ann, Type):
            if isinstance(ann, Buffer) and isinstance(ann.element, str):
                resolved_element = self.resolve_annotation(ann.element, specialization_values)
                if isinstance(resolved_element, Type) and resolved_element is not ann.element:
                    return Buffer(element=resolved_element)
            elif isinstance(ann, Array):
                if isinstance(ann.element, str):
                    resolved_element = self.resolve_annotation(ann.element, specialization_values)
                    if isinstance(resolved_element, Type) and resolved_element is not ann.element:
                        return Array(element=resolved_element, count=ann.count)
            # Add more generic types as needed (Texture2D, Texture3D, etc.)
            return ann

        typ, _ = annotation_to_type(ann)
        return typ

    def _resolve_generic_string_annotation(self, ann: str, spec_map: dict) -> Optional[Any]:
        """Parse and resolve generic type strings like 'Buffer[T]' or 'Array[T, 4]'."""
        from .types import Type, Buffer, Array, Vector, Matrix, ScalarType
        
        if not spec_map:
            return None
            
        try:
            tree = ast.parse(ann, mode='eval')
            body = tree.body
            
            # Handle subscript like Buffer[T] or Array[T, 4]
            if isinstance(body, ast.Subscript):
                base_name = body.value.id if isinstance(body.value, ast.Name) else None
                if base_name is None:
                    return None
                
                # Get the slice - handle both single element and tuple
                slice_node = body.slice
                
                if base_name == 'Buffer':
                    element_name = self._get_name_from_slice(slice_node)
                    if element_name and element_name in spec_map:
                        return Buffer(element=spec_map[element_name])
                    # Also check if element_name is a known type
                    if element_name:
                        from .types import annotation_to_type
                        typ, _ = annotation_to_type(element_name)
                        if isinstance(typ, Type):
                            return Buffer(element=typ)
                            
                elif base_name == 'Array':
                    # Array[T, 4] - tuple slice
                    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                        elem_node, count_node = slice_node.elts
                        element_name = elem_node.id if isinstance(elem_node, ast.Name) else None
                        count = count_node.n if isinstance(count_node, ast.Constant) else None
                        
                        if element_name and element_name in spec_map and count is not None:
                            return Array(element=spec_map[element_name], count=count)
                        if element_name and count is not None:
                            from .types import annotation_to_type
                            typ, _ = annotation_to_type(element_name)
                            if isinstance(typ, Type):
                                return Array(element=typ, count=count)
                
                # Add more generic types here (Vector, Matrix, Texture2D, etc.)
                
        except (SyntaxError, AttributeError):
            pass
        
        return None
    
    def _get_name_from_slice(self, slice_node) -> Optional[str]:
        """Extract a name from a slice node."""
        if isinstance(slice_node, ast.Name):
            return slice_node.id
        # Handle Python 3.9+ Index wrapper
        if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index):
            if isinstance(slice_node.value, ast.Name):
                return slice_node.value.id
        return None

    def _get_or_create_staged(self, args: tuple) -> StagedFunction:
        """Infer types from arguments and get/create a StagedFunction."""
        # Detect arg types
        arg_types = []
        for i, arg in enumerate(args):
            resolved_type = None
            # Try to resolve annotation if present
            if i < len(self.parsed.arg_annotations) and self.parsed.arg_annotations[i] is not None:
                resolved = self.resolve_annotation(self.parsed.arg_annotations[i], self.specialization_values)
                # If resolved to an actual Type (not a string/None), use it
                from .types import Type
                if isinstance(resolved, Type):
                    resolved_type = resolved
            
            # If not resolved from annotation, infer from value
            if resolved_type is None:
                resolved_type = value_to_type(arg) if not hasattr(arg, 'type') else arg.type
            
            arg_types.append(resolved_type)
        
        arg_types = tuple(arg_types)
        if arg_types not in self._cache:
            self._cache[arg_types] = StagedFunction(self, self.specialization_values, arg_types)
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
        
        # If it's a normal function (no params) and fully annotated, make it a StagedFunction immediately
        if not self.params and all(ann is not None for ann in templated.parsed.arg_annotations):
            try:
                arg_types = tuple(templated.resolve_annotation(ann, ()) for ann in templated.parsed.arg_annotations)
                if all(t is not None for t in arg_types):
                    return StagedFunction(templated, (), arg_types)
            except Exception:
                pass # Defer if types/modules not ready
        
        return templated


kernel = StagedFunctionDecorator(is_kernel=True)
callable = StagedFunctionDecorator(is_kernel=False)
