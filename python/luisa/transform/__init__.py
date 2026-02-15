"""
Transformation machinery for the LuisaCompute Python DSL v2.

This package contains the internal engine that converts Python AST
into Luisa IR.
"""

from __future__ import annotations

# Export transformation components
from .op import Op, get_op_name, is_arithmetic_op, is_comparison_op, is_logical_op, is_terminator_op, is_memory_op, is_resource_op
from .ir import (
    SourceLocation,
    Value, ConstantValue, ArgumentValue, InstructionValue,
    Instruction, BasicBlock, Function, Module
)
from .builder import Builder, get_current_builder, set_current_builder
from .rewriter import ASTRewriter
from .inspect import (
    parse_function, ParsedFunction, CapturedVar,
    get_ir_source, get_ir_ast, get_ir_types,
    count_instructions, get_basic_block_count, get_instruction_count,
    find_operations, analyze_control_flow, is_kernel, get_type_size, format_ir_summary
)

__all__ = [
    # Op
    'Op', 'get_op_name', 'is_arithmetic_op', 'is_comparison_op', 'is_logical_op',
    'is_terminator_op', 'is_memory_op', 'is_resource_op',
    
    # IR
    'SourceLocation',
    'Value', 'ConstantValue', 'ArgumentValue', 'InstructionValue',
    'Instruction', 'BasicBlock', 'Function', 'Module',
    
    # Builder
    'Builder', 'get_current_builder', 'set_current_builder',
    
    # Rewriter
    'ASTRewriter', 'parse_function', 'ParsedFunction', 'CapturedVar',
    
    # Inspect
    'get_ir_source', 'get_ir_ast', 'get_ir_types',
    'count_instructions', 'get_basic_block_count', 'get_instruction_count',
    'find_operations', 'analyze_control_flow', 'is_kernel', 'get_type_size', 'format_ir_summary',
]
