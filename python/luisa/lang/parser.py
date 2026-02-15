"""
AST Parser for the LuisaCompute Python DSL v2.

This module parses Python functions and creates a staged function
that can generate IR when called with actual arguments.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Callable, Optional, Any
from dataclasses import dataclass

# Runtime imports
from .types import Type
from .types import (
    Scalar, Vector, Buffer, Texture2D, Texture3D, Array, Void,
    bool_, int32, uint32, float32,
    python_type_to_dsl
)


# ============================================================================
# Parsed Function Representation
# ============================================================================

@dataclass
class CapturedVar:
    """Information about a captured variable."""
    name: str
    value: Any
    type: Optional[Type] = None

    def __post_init__(self):
        if self.type is None:
            self.type = value_to_type(self.value)


@dataclass
class ParsedFunction:
    """A Python function parsed into an AST with metadata."""
    name: str
    ast_node: ast.FunctionDef
    arg_names: list[str]
    arg_annotations: list[Optional[Type]]
    arg_is_reference: list[bool]
    ret_annotation: Optional[Type]
    captured_vars: dict[str, CapturedVar]
    source: str
    pyfunc: Optional[Callable] = None

    def get_arg_type(self, index: int) -> Optional[Type]:
        """Get the type annotation for an argument."""
        if index < len(self.arg_annotations):
            return self.arg_annotations[index]
        return None


# ============================================================================
# Type Conversion
# ============================================================================

def value_to_type(value: Any) -> Optional[Type]:
    """Infer DSL type from Python value."""
    if value is None:
        return Void()

    if isinstance(value, bool):
        return bool_
    if isinstance(value, int):
        return int32
    if isinstance(value, float):
        return float32

    # Could add more types here (tuples, lists, etc.)
    return None


def annotation_to_type(ann: Any) -> tuple[Optional[Type], bool]:
    """Convert a Python type annotation to a DSL type and a reference flag."""
    if ann is None or ann is inspect.Parameter.empty:
        return None, False

    # Handle direct type references
    if isinstance(ann, Type):
        return ann, False

    # Handle Python built-in types
    py_type = python_type_to_dsl(ann)
    if py_type is not None:
        return py_type, False

    # Handle generic types like Buffer[float]
    origin = getattr(ann, '__origin__', None)
    args = getattr(ann, '__args__', None)

    if origin is not None and args is not None:
        # Handle Ref[T]
        if origin.__name__ == 'Ref' or getattr(origin, '__name__', None) == 'Ref':
            elem_type, _ = annotation_to_type(args[0])
            return elem_type, True

        # Handle Buffer[T]
        if origin.__name__ == 'Buffer' or getattr(origin, '__name__', None) == 'buffer':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None:
                return Buffer(element=elem_type), False

        # Handle Texture2D[T]
        if origin.__name__ == 'Texture2D' or getattr(origin, '__name__', None) == 'Texture2D':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None and isinstance(elem_type, Scalar):
                return Texture2D(element=elem_type), False

        # Handle Texture3D[T]
        if origin.__name__ == 'Texture3D' or getattr(origin, '__name__', None) == 'Texture3D':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None and isinstance(elem_type, Scalar):
                return Texture3D(element=elem_type), False

    return None, False


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parse Python functions for DSL compilation."""

    def parse_function(self, func: Callable, source: Optional[str] = None) -> ParsedFunction:
        """
        Parse a Python function and return a ParsedFunction.
        
        This extracts:
        - The AST
        - Type annotations
        - Captured variables from the closure
        """
        # If we already have the AST
        if hasattr(func, '_luisa_ast'):
            func_def = func._luisa_ast
            source = ast.unparse(func_def)
        elif source is not None:
            # Parse AST from provided source
            try:
                tree = ast.parse(source)
                func_def = tree.body[0]
            except Exception as e:
                raise RuntimeError(f"Error parsing provided source for {func}: {e}") from e
        else:
            # Get source code
            try:
                lines, start_line = inspect.getsourcelines(func)
                source = "".join(lines)

                # Dedent source to handle nested function definitions
                source = textwrap.dedent(source)

                # Parse AST
                try:
                    tree = ast.parse(source)
                except SyntaxError as e:
                    raise RuntimeError(f"Syntax error in {func}: {e}") from e

                # Get function definition
                if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                    raise RuntimeError(f"Expected function definition, got {type(tree.body[0])}")

                func_def = tree.body[0]

                # Adjust line numbers to be global
                ast.increment_lineno(func_def, start_line - 1)
            except (OSError, TypeError) as e:
                raise RuntimeError(f"Cannot get source for {func}: {e}") from e

        # Get signature
        try:
            sig = inspect.signature(func)

            # Extract argument names and annotations
            arg_names = []
            arg_annotations = []
            arg_is_reference = []

            for name, param in sig.parameters.items():
                arg_names.append(name)
                ann, is_ref = annotation_to_type(param.annotation)
                arg_annotations.append(ann)
                arg_is_reference.append(is_ref)

            # Extract return annotation
            ret_annotation, _ = annotation_to_type(sig.return_annotation)
        except NameError:
            # Fallback for specialized functions where types are not yet defined
            # We'll extract names from AST and use None for annotations for now
            arg_names = [arg.arg for arg in func_def.args.args]
            arg_annotations = [None] * len(arg_names)
            arg_is_reference = [False] * len(arg_names)
            ret_annotation = None

        # Analyze captured variables
        captured_vars = self._analyze_captured_vars(func)

        return ParsedFunction(
            name=func.__name__,
            ast_node=func_def,
            arg_names=arg_names,
            arg_annotations=arg_annotations,
            arg_is_reference=arg_is_reference,
            ret_annotation=ret_annotation,
            captured_vars=captured_vars,
            source=source,
            pyfunc=func
        )

    def _analyze_captured_vars(self, func: Callable) -> dict[str, CapturedVar]:
        """Analyze captured (closure) variables."""
        captured = {}

        try:
            closure = inspect.getclosurevars(func)

            # Non-local variables
            for name, value in closure.nonlocals.items():
                captured[name] = CapturedVar(name=name, value=value)

            # Global variables
            for name, value in closure.globals.items():
                captured[name] = CapturedVar(name=name, value=value)

        except (TypeError, ValueError):
            # Function has no closure
            pass

        return captured


# ============================================================================
# Type Checker
# ============================================================================

class TypeChecker:
    """Type checker for DSL expressions."""

    def __init__(self):
        self.errors: list[str] = []

    def check_binary_op(self, op: ast.operator,
                        left_type: Type, right_type: Type) -> Optional[Type]:
        """Check and return the result type of a binary operation."""
        # Arithmetic operations
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            if left_type == right_type:
                return left_type

            # Check for scalar-vector broadcasting
            if isinstance(left_type, Scalar) and isinstance(right_type, Vector):
                if left_type == right_type.element:
                    return right_type
            if isinstance(left_type, Vector) and isinstance(right_type, Scalar):
                if left_type.element == right_type:
                    return left_type

            self.errors.append(
                f"Cannot apply {op.__class__.__name__} to {left_type} and {right_type}"
            )
            return None

        # Division
        if isinstance(op, ast.Div):
            # Division always returns float or float vector
            if isinstance(left_type, Scalar) and isinstance(right_type, Scalar):
                return float32
            if isinstance(left_type, Vector) and isinstance(right_type, Vector):
                if left_type.size == right_type.size:
                    return Vector(float32, left_type.size)

            self.errors.append(
                f"Cannot divide {left_type} by {right_type}"
            )
            return None

        # Comparison
        if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            # Comparisons return bool
            return bool_

        self.errors.append(f"Unsupported binary operator: {op.__class__.__name__}")
        return None

    def check_unary_op(self, op: ast.unaryop, operand_type: Type) -> Optional[Type]:
        """Check and return the result type of a unary operation."""
        if isinstance(op, ast.USub):
            # Negation preserves type
            return operand_type

        if isinstance(op, ast.Not):
            # Logical not returns bool
            return bool_

        self.errors.append(f"Unsupported unary operator: {op.__class__.__name__}")
        return None

    def check_subscript(self, value_type: Type, index_type: Type) -> Optional[Type]:
        """Check and return the result type of a subscript operation."""
        # Buffer subscript
        if isinstance(value_type, Buffer):
            if index_type == int32 or index_type == uint32:
                return value_type.element
            self.errors.append(f"Buffer index must be int or uint, got {index_type}")
            return None

        # Array subscript
        if isinstance(value_type, Array):
            if index_type == int32 or index_type == uint32:
                return value_type.element
            self.errors.append(f"Array index must be int or uint, got {index_type}")
            return None

        # Vector swizzle (handled separately, this is for element access)
        if isinstance(value_type, Vector):
            if index_type == int32 or index_type == uint32:
                return value_type.element
            self.errors.append(f"Vector index must be int or uint, got {index_type}")
            return None

        self.errors.append(f"Cannot subscript type {value_type}")
        return None

    def check_call(self, func_typ: Type, _arg_types: list[Type]) -> Optional[Type]:
        """Check a function call and return the result type."""
        if isinstance(func_typ, type(lambda: None)):
            # Python callable - need to handle this differently
            return None

        if not hasattr(func_typ, 'arg_types'):
            self.errors.append(f"Cannot call non-callable type {func_typ}")
            return None

        # For now, just return the return type if it exists
        return getattr(func_typ, 'ret_type', None)


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_function(func: Callable, source: Optional[str] = None) -> ParsedFunction:
    """Parse a function using the default parser."""
    parser = Parser()
    return parser.parse_function(func, source=source)


def check_types(_node: ast.AST,
                _type_env: Optional[dict[str, Type]] = None) -> Optional[Type]:
    """
    Check types for an AST node.

    This is a simplified entry point. Full type checking is done
    during the builder execution phase.
    """
    # Type checking happens during builder execution
    return None
