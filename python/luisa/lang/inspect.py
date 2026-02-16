"""
Introspection utilities for the LuisaCompute Python DSL v2.

This module provides utilities for inspecting staged functions,
IR, and types.

Note: This module is deprecated. Use luisa.transform.inspect instead.
"""

from __future__ import annotations

# Re-export from transform.inspect for backward compatibility
from ..transform.inspect import (analyze_control_flow, count_instructions,
                                 find_operations, format_ir_summary,
                                 get_basic_block_count, get_instruction_count,
                                 get_ir_ast, get_ir_source, get_ir_types,
                                 get_type_size, is_kernel)

__all__ = [
    'get_ir_source', 'get_ir_ast', 'get_ir_types',
    'count_instructions', 'get_basic_block_count', 'get_instruction_count',
    'find_operations', 'analyze_control_flow', 'is_kernel', 'get_type_size', 'format_ir_summary'
]
