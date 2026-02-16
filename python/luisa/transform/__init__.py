"""
Transformation machinery for the LuisaCompute Python DSL v2.

This package contains the internal engine that converts Python AST
into Luisa IR.
"""

from __future__ import annotations

from .builder import Builder, get_current_builder, set_current_builder
from .inspect import (CapturedVar, ParsedFunction, analyze_control_flow,
                      count_instructions, find_operations, format_ir_summary,
                      get_basic_block_count, get_instruction_count, get_ir_ast,
                      get_ir_source, get_ir_types, get_type_size, is_kernel,
                      parse_function)
from .ir import (ArgumentValue, BasicBlock, ConstantValue, Function,
                 Instruction, InstructionValue, Module, SourceLocation, Value)
# Export transformation components
from .op import (Op, get_op_name, is_arithmetic_op, is_comparison_op,
                 is_logical_op, is_memory_op, is_resource_op, is_terminator_op)
from .rewriter import ASTRewriter

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
