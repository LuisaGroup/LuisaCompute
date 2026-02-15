"""
Pretty Printer CodeGen for the LuisaCompute Python DSL v2.

Provides human-readable output of IR for debugging purposes.
"""

from __future__ import annotations
from typing import Any, Optional
from io import StringIO

# Runtime imports
from ..lang.ir import Function, Module, BasicBlock, Instruction, Op, SourceLocation
from ..lang.type import Type


class PrettyPrinter:
    """Pretty printer for IR."""

    def __init__(self, indent_size: int = 2):
        self.indent_size = indent_size
        self._output = StringIO()
        self._indent_level = 0

    def _indent(self) -> str:
        """Get current indentation string."""
        return ' ' * (self.indent_size * self._indent_level)

    def _write(self, s: str) -> None:
        """Write a string to output."""
        self._output.write(s)

    def _write_line(self, s: str = '') -> None:
        """Write a line with proper indentation."""
        if s:
            self._write(self._indent() + s + '\n')
        else:
            self._write('\n')

    def _increase_indent(self) -> None:
        """Increase indentation level."""
        self._indent_level += 1

    def _decrease_indent(self) -> None:
        """Decrease indentation level."""
        self._indent_level -= 1

    def _format_loc(self, loc: Optional[SourceLocation]) -> str:
        """Format a source location as a comment string."""
        return f" ![{loc}]" if loc else ""

    def print(self, obj: Function | Module) -> str:
        """Print an IR object and return the result."""
        self._output = StringIO()
        self._indent_level = 0

        if isinstance(obj, Function):
            self._print_function(obj)
        elif isinstance(obj, Module):
            self._print_module(obj)
        else:
            raise TypeError(f"Cannot print object of type {type(obj)}")

        return self._output.getvalue()

    def _print_module(self, module: Module) -> None:
        """Print an IR module."""
        self._write_line(f"Module with {len(module.functions)} function(s):")
        self._increase_indent()
        for i, func in enumerate(module.functions):
            if i > 0:
                self._write_line()
            self._print_function(func)
        self._decrease_indent()

    def _print_function(self, func: Function) -> None:
        """Print an IR function."""
        # Function signature
        ret_type = str(func.ret_type) if func.ret_type else 'void'

        # Format arguments: type arg_name
        args_formatted = []
        for i, (t, is_ref) in enumerate(zip(func.arg_types, func.arg_is_reference)):
            type_str = str(t)
            if is_ref:
                type_str = f"ref<{type_str}>"
            args_formatted.append(f"{type_str} arg{i}")

        kernel_marker = 'kernel ' if func.is_kernel else ''
        block_size = f" /* block_size={func.block_size} */" if func.block_size else ''
        loc_str = self._format_loc(func.loc)

        self._write_line(f"{kernel_marker}{ret_type} {func.name}({', '.join(args_formatted)}){block_size}{loc_str} {{")

        self._increase_indent()

        # In structured IR, we only need to print the entry block
        # Nested blocks are printed as part of structured instructions
        if func.blocks:
            self._print_block(func.blocks[0])

        self._decrease_indent()
        self._write_line('}')

    def _print_block(self, block: BasicBlock) -> None:
        """Print a basic block."""
        self._write_line(f"{block.name}:")
        self._increase_indent()

        for inst in block.instructions:
            self._print_instruction(inst)

        if not block.instructions:
            self._write_line('(empty)')

        self._decrease_indent()

    def _print_instruction(self, inst: Instruction) -> None:
        """Print an instruction."""
        op_str = self._op_to_str(inst.op)
        type_str = str(inst.type)
        loc_str = self._format_loc(inst.loc)

        # Handle structured instructions specially
        if inst.op == Op.IF:
            cond_str = self._arg_to_str(inst.args[0])
            self._write_line(f"if ({cond_str}) {{ {loc_str}")
            self._increase_indent()
            self._print_block_inline(inst.args[1])
            self._decrease_indent()
            self._write_line("} else {")
            self._increase_indent()
            self._print_block_inline(inst.args[2])
            self._decrease_indent()
            self._write_line("}")
        elif inst.op == Op.LOOP:
            self._write_line(f"while (true) {{ {loc_str}")
            self._increase_indent()
            self._print_block_inline(inst.args[0])
            self._decrease_indent()
            self._write_line("}")
        elif inst.op == Op.SWITCH:
            val_str = self._arg_to_str(inst.args[0])
            self._write_line(f"switch ({val_str}) {{ {loc_str}")
            self._increase_indent()
            cases = inst.args[1]
            for vals, block in cases:
                self._write_line(f"case {', '.join(map(str, vals))}: {{")
                self._increase_indent()
                self._print_block_inline(block)
                self._decrease_indent()
                self._write_line("}")
            if inst.args[2]:
                self._write_line("default: {")
                self._increase_indent()
                self._print_block_inline(inst.args[2])
                self._decrease_indent()
                self._write_line("}")
            self._decrease_indent()
            self._write_line("}")
        elif inst.op == Op.RETURN:
            if inst.args:
                self._write_line(f"return {self._arg_to_str(inst.args[0])};")
            else:
                self._write_line("return;")
        elif inst.op == Op.BREAK:
            self._write_line("break;")
        elif inst.op == Op.CONTINUE:
            self._write_line("continue;")
        else:
            args_str = self._args_to_str(inst.args)
            if inst.result and inst.type is not None:
                self._write_line(f"{type_str} {inst.result} = {op_str}({args_str});{loc_str}")
            else:
                self._write_line(f"{op_str}({args_str});{loc_str}")

    def _print_block_inline(self, block: Any) -> None:
        """Print instructions of a block without the label and additional indent."""
        if not hasattr(block, 'instructions'):
            self._write_line(repr(block))
            return
        for inst in block.instructions:
            self._print_instruction(inst)
        if not block.instructions:
            self._write_line('(empty)')

    def _op_to_str(self, op: Op) -> str:
        """Convert an Op to a string."""
        return op.name.lower()

    def _arg_to_str(self, arg: Any) -> str:
        """Convert a single argument to string."""
        if hasattr(arg, 'name') and hasattr(arg, 'instructions'):
            # BasicBlock
            return f"{arg.name}"
        elif hasattr(arg, 'name') and hasattr(arg, 'type'):
            # Value (InstructionValue, ArgumentValue, ConstantValue)
            if hasattr(arg, 'value'):
                # ConstantValue
                return f"{arg.value}"
            elif hasattr(arg, 'index'):
                # ArgumentValue
                return f"arg{arg.index}"
            elif arg.name:
                return f"{arg.name}"
            else:
                return str(arg)
        elif hasattr(arg, 'name'):
            # BasicBlock
            return f"{arg.name}"
        else:
            # String or other literal
            return repr(arg)

    def _args_to_str(self, args: list[Any]) -> str:
        """Convert instruction arguments to a string."""
        return ', '.join(self._arg_to_str(a) for a in args)


# Convenience functions
def pprint(obj: Function | Module, indent_size: int = 2) -> str:
    """
    Pretty print an IR object.
    
    Args:
        obj: The IR function or module to print
        indent_size: Number of spaces per indentation level
    
    Returns:
        Pretty-printed string representation
    
    Example:
        >>> func = my_kernel.compile()
        >>> print(pprint(func))
        [kernel] func my_kernel(Buffer<float>) -> void {
          entry:
            v0: UInt3 = dispatch_id 
            ...
        }
    """
    printer = PrettyPrinter(indent_size)
    return printer.print(obj)


def pprint_to_file(obj, path: str, indent_size: int = 2) -> None:  # type: ignore
    """Pretty print an IR object to a file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(pprint(obj, indent_size))
