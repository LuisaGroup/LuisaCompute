"""
AST Parser for the LuisaCompute Python DSL v2.

This module parses Python functions and creates a staged function
that can generate IR when called with actual arguments.
"""

from __future__ import annotations
import ast
import inspect
from typing import Callable, Optional, Any, TYPE_CHECKING, Union
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .dsl_types import Type
    from .ir import Value

from .dsl_types import (
    Type, Scalar, Vector, Matrix, Array, Struct, Buffer, Void,
    bool_, int32, uint32, float32, float64, float3,
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
    ret_annotation: Optional[Type]
    captured_vars: dict[str, CapturedVar]
    source: str
    
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


def annotation_to_type(ann: Any) -> Optional[Type]:
    """Convert a Python type annotation to a DSL type."""
    if ann is None or ann is inspect.Parameter.empty:
        return None
    
    # Handle direct type references
    if isinstance(ann, Type):
        return ann
    
    # Handle Python built-in types
    py_type = python_type_to_dsl(ann)
    if py_type is not None:
        return py_type
    
    # Handle generic types like Buffer[float]
    # This requires more complex handling with typing.get_origin/args
    origin = getattr(ann, '__origin__', None)
    args = getattr(ann, '__args__', None)
    
    if origin is not None and args is not None:
        # Handle Buffer[T]
        if origin.__name__ == 'Buffer' or getattr(origin, '__name__', None) == 'buffer':
            elem_type = annotation_to_type(args[0])
            if elem_type is not None:
                return Buffer(element=elem_type)
        
        # Handle list[T] for arrays
        if origin is list and len(args) == 1:
            elem_type = annotation_to_type(args[0])
            if elem_type is not None:
                # Array size is unknown from type annotation alone
                return None
    
    return None


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parse Python functions for DSL compilation."""
    
    def parse_function(self, func: Callable) -> ParsedFunction:
        """
        Parse a Python function and return a ParsedFunction.
        
        This extracts:
        - The AST
        - Type annotations
        - Captured variables from the closure
        """
        # Get source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as e:
            raise RuntimeError(f"Cannot get source for {func}: {e}")
        
        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise RuntimeError(f"Syntax error in {func}: {e}")
        
        # Get function definition
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            raise RuntimeError(f"Expected function definition, got {type(tree.body[0])}")
        
        func_def = tree.body[0]
        
        # Get signature
        sig = inspect.signature(func)
        
        # Extract argument names and annotations
        arg_names = []
        arg_annotations = []
        
        for name, param in sig.parameters.items():
            arg_names.append(name)
            ann = annotation_to_type(param.annotation)
            arg_annotations.append(ann)
        
        # Extract return annotation
        ret_annotation = annotation_to_type(sig.return_annotation)
        
        # Analyze captured variables
        captured_vars = self._analyze_captured_vars(func)
        
        return ParsedFunction(
            name=func.__name__,
            ast_node=func_def,
            arg_names=arg_names,
            arg_annotations=arg_annotations,
            ret_annotation=ret_annotation,
            captured_vars=captured_vars,
            source=source
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
    
    def check_call(self, func_type: Type, arg_types: list[Type]) -> Optional[Type]:
        """Check a function call and return the result type."""
        if isinstance(func_type, type(lambda: None)):
            # Python callable - need to handle this differently
            return None
        
        if not hasattr(func_type, 'arg_types'):
            self.errors.append(f"Cannot call non-callable type {func_type}")
            return None
        
        # For now, just return the return type if it exists
        return getattr(func_type, 'ret_type', None)


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_function(func: Callable) -> ParsedFunction:
    """Parse a function using the default parser."""
    parser = Parser()
    return parser.parse_function(func)


def check_types(node: ast.AST, 
               type_env: Optional[dict[str, Type]] = None) -> Optional[Type]:
    """
    Check types for an AST node.
    
    This is a simplified entry point. Full type checking is done
    during the builder execution phase.
    """
    checker = TypeChecker()
    # Type checking happens during builder execution
    return None
