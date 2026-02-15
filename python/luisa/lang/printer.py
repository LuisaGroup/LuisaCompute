"""
IR printing utilities for the LuisaCompute Python DSL v2.

This module provides utilities for printing IR in various formats.
It's a simpler alternative to the full pretty printer.
"""

from __future__ import annotations

from .ir import Function, Module, BasicBlock, Instruction, Value
from .types import Type


class SimplePrinter:
    """Simple printer for IR debugging."""

    def __init__(self, indent: int = 2):
        self.indent = indent

    def print_function(self, func: Function) -> str:
        """Print a function."""
        lines = []
        kind = "kernel" if func.is_kernel else "func"
        ret = str(func.ret_type) if func.ret_type else "void"
        args = ", ".join(str(t) for t in func.arg_types)
        lines.append(f"{kind} @{func.name}({args}) -> {ret} {{")

        for block in func.blocks:
            lines.append(self._indent_block(block))

        lines.append("}")
        return "\n".join(lines)

    def _indent_block(self, block: BasicBlock) -> str:
        """Print a basic block with indentation."""
        lines = [f"  {block.name}:"]
        for inst in block.instructions:
            lines.append(f"    {inst}")
        return "\n".join(lines)

    def print_module(self, module: Module) -> str:
        """Print a module."""
        lines = ["module {"]
        for func in module.functions:
            func_str = self.print_function(func)
            # Indent function
            for line in func_str.split("\n"):
                lines.append(f"  {line}")
        lines.append("}")
        return "\n".join(lines)


def print_function(func: Function) -> str:
    """Print a function to string."""
    printer = SimplePrinter()
    return printer.print_function(func)


def print_module(module: Module) -> str:
    """Print a module to string."""
    printer = SimplePrinter()
    return printer.print_module(module)


def print_instruction(inst: Instruction) -> str:
    """Print a single instruction."""
    return str(inst)


def print_value(val: Value) -> str:
    """Print a value."""
    return str(val)


def print_type(t: Type) -> str:
    """Print a type."""
    return str(t)
