"""
Introspection utilities for the LuisaCompute Python DSL v2.

This module provides utilities for inspecting staged functions,
IR, and types.
"""

from __future__ import annotations
from typing import Optional, Any, Callable
import ast

from .ir import IRFunction, IRInstruction, IROp
from .types import Type, Scalar, Vector, Matrix


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


def count_instructions(ir: IRFunction) -> dict[str, int]:
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


def get_basic_block_count(ir: IRFunction) -> int:
    """Get the number of basic blocks in a function."""
    return len(ir.blocks)


def get_instruction_count(ir: IRFunction) -> int:
    """Get the total number of instructions in a function."""
    counts = count_instructions(ir)
    return sum(counts.values())


def find_operations(ir: IRFunction, op: IROp) -> list[IRInstruction]:
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


def analyze_control_flow(ir: IRFunction) -> dict[str, Any]:
    """
    Analyze control flow in an IR function.
    
    Returns:
        Dictionary with control flow analysis
    """
    ifs = len(find_operations(ir, IROp.IF))
    loops = len(find_operations(ir, IROp.LOOP))
    switches = len(find_operations(ir, IROp.SWITCH))
    returns = len(find_operations(ir, IROp.RETURN))

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


def is_kernel(ir: IRFunction) -> bool:
    """Check if an IR function is a kernel."""
    return ir.is_kernel


def get_type_size(t: Type) -> int:
    """
    Get the size in bytes of a type.
    
    This is an approximation for frontend use.
    """
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


def format_ir_summary(ir: IRFunction) -> str:
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
