"""
Introspection utilities for the LuisaCompute Python DSL v2.

This module provides utilities for inspecting staged functions,
IR, and types.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass

from .ir import Function, Instruction
from .op import Op

if TYPE_CHECKING:
    from ..lang.types import Type


# ============================================================================
# Compilation Metadata
# ============================================================================

@dataclass
class CapturedVar:
    """Information about a captured variable."""
    name: str
    value: Any
    type: Optional[Type] = None

    def __post_init__(self):
        if self.type is None:
            from ..lang.types import value_to_type
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
# Function Parsing
# ============================================================================

def parse_function(func: Callable, source: Optional[str] = None) -> ParsedFunction:
    """Parse a Python function and return its metadata."""
    from ..lang.types import annotation_to_type
    
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
    except (NameError, TypeError):
        # Fallback for specialized functions where types are not yet defined
        arg_names = [arg.arg for arg in func_def.args.args]
        arg_annotations = [None] * len(arg_names)
        arg_is_reference = [False] * len(arg_names)
        ret_annotation = None

    # Analyze captured variables
    captured_vars = _analyze_captured_vars(func)

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


def _analyze_captured_vars(func: Callable) -> dict[str, CapturedVar]:
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
# IR Inspection
# ============================================================================


def get_ir_source(func: Callable) -> Optional[str]:
    """
    Get the IR source code for a staged function.
    
    Args:
        func: A staged function
        
    Returns:
        The IR source code or None if not a staged function
    """
    # Check if it has the parsed attribute (StagedFunction)
    if hasattr(func, 'parsed'):
        return func.parsed.source
    return None


def get_ir_ast(func: Callable) -> Optional[ast.AST]:
    """
    Get the AST for a staged function.
    
    Args:
        func: A staged function
        
    Returns:
        The function AST or None if not a staged function
    """
    if hasattr(func, 'parsed'):
        return func.parsed.ast_node
    return None


def get_ir_types(func: Callable) -> Optional[dict[str, Any]]:
    """
    Get type information for a staged function.
    
    Args:
        func: A staged function
        
    Returns:
        Dictionary with type information
    """
    if not hasattr(func, 'parsed'):
        return None

    return {
        'arg_types': func.parsed.arg_annotations,
        'ret_type': func.parsed.ret_annotation,
        'captured_vars': list(func.parsed.captured_vars.keys())
    }


def count_instructions(ir: Function) -> dict[str, int]:
    """
    Count instructions in an IR function by type.
    
    Args:
        ir: The IR function
        
    Returns:
        Dictionary mapping operation names to counts
    """
    counts: dict[str, int] = {}

    def scan_block(block):
        if not hasattr(block, 'instructions'): return
        for inst in block.instructions:
            op_name = inst.op.name
            counts[op_name] = counts.get(op_name, 0) + 1
            # Scan nested blocks
            for arg in inst.args:
                if hasattr(arg, 'instructions'):
                    scan_block(arg)
                elif isinstance(arg, list):
                    for item in arg:
                        if hasattr(item, 'instructions'):
                            scan_block(item)
                        elif isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], 'instructions'):
                            # Switch cases
                            scan_block(item[1])

    for block in ir.blocks:
        scan_block(block)

    return counts


def get_basic_block_count(ir: Function) -> int:
    """Get the number of basic blocks in a function."""
    return len(ir.blocks)


def get_instruction_count(ir: Function) -> int:
    """Get the total number of instructions in a function."""
    counts = count_instructions(ir)
    return sum(counts.values())


def find_operations(ir: Function, op: Op) -> list[Instruction]:
    """
    Find all instructions of a specific type.
    
    Args:
        ir: The IR function
        op: The operation type to find
        
    Returns:
        List of matching instructions
    """
    results = []

    def scan_block(block):
        if not hasattr(block, 'instructions'): return
        for inst in block.instructions:
            if inst.op == op:
                results.append(inst)
            # Scan nested blocks
            for arg in inst.args:
                if hasattr(arg, 'instructions'):
                    scan_block(arg)
                elif isinstance(arg, list):
                    for item in arg:
                        if hasattr(item, 'instructions'):
                            scan_block(item)
                        elif isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], 'instructions'):
                            # Switch cases (values, block)
                            scan_block(item[1])

    for block in ir.blocks:
        scan_block(block)
    return results


def analyze_control_flow(ir: Function) -> dict[str, Any]:
    """
    Analyze control flow in an IR function.
    
    Returns:
        Dictionary with control flow analysis
    """
    ifs = len(find_operations(ir, Op.IF))
    loops = len(find_operations(ir, Op.LOOP))
    switches = len(find_operations(ir, Op.SWITCH))
    returns = len(find_operations(ir, Op.RETURN))

    return {
        'blocks': len(ir.blocks),
        'branches': 0,  # Structured IR has no explicit branches
        'ifs': ifs,
        'loops': loops,
        'switches': switches,
        'conditional_branches': ifs + switches,
        'returns': returns,
        'has_loops': loops > 0
    }


def is_kernel(ir: Function) -> bool:
    """Check if an IR function is a kernel."""
    return ir.is_kernel


def get_type_size(t: Any) -> int:
    """
    Get the size in bytes of a type.
    
    This is an approximation for frontend use.
    """
    from ..lang.types import Scalar, Vector, Matrix
    
    if isinstance(t, Scalar):
        type_sizes = {
            'BOOL': 1,
            'INT8': 1, 'UINT8': 1,
            'INT16': 2, 'UINT16': 2,
            'INT32': 4, 'UINT32': 4,
            'INT64': 8, 'UINT64': 8,
            'FLOAT16': 2,
            'FLOAT32': 4,
            'FLOAT64': 8,
        }
        return type_sizes.get(t.dtype.name, 4)

    if isinstance(t, Vector):
        return get_type_size(t.element) * t.size

    if isinstance(t, Matrix):
        return get_type_size(t.element) * t.size * t.size

    return 4  # Default


def format_ir_summary(ir: Function) -> str:
    """
    Create a human-readable summary of an IR function.
    
    Args:
        ir: The IR function
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append(f"Function: {ir.name}")
    lines.append(f"  Type: {'Kernel' if ir.is_kernel else 'Callable'}")
    lines.append(f"  Arguments: {len(ir.arg_types)}")
    lines.append(f"  Return: {ir.ret_type or 'void'}")
    lines.append(f"  Blocks: {len(ir.blocks)}")
    lines.append(f"  Instructions: {get_instruction_count(ir)}")

    # Control flow
    cf = analyze_control_flow(ir)
    if cf['conditional_branches'] > 0:
        lines.append(f"  Control Flow: {cf['conditional_branches']} condition(s)")

    # Instructions by type
    counts = count_instructions(ir)
    if counts:
        lines.append("  Operations:")
        for op_name, count in sorted(counts.items()):
            lines.append(f"    {op_name}: {count}")

    return '\n'.join(lines)
