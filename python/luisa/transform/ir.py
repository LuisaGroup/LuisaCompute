"""
Intermediate Representation (IR) for the LuisaCompute Python DSL v2.

This module defines the IR data structures that are used to represent
the compiled kernel code. The IR is designed to be JSON-serializable
for easy exchange with the C++ backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..lang.types import Type

from .op import Op

# ============================================================================
# Source Location
# ============================================================================

@dataclass(frozen=True)
class SourceLocation:
    """Source code location for debugging."""
    file: str
    line: int

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}"


# ============================================================================
# IR Values
# ============================================================================

@dataclass
class Value:
    """Base class for IR values."""
    typ: Any  # Type, use Any to avoid circular import
    name: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"%{self.name}"
        return "%<unnamed>"

    @property
    def type(self) -> Any:
        """Backward compatibility accessor for type."""
        return self.typ

    @property
    def is_pointer(self) -> bool:
        """Check if this value is a pointer/reference."""
        return False


@dataclass
class ConstantValue(Value):
    """Constant value in IR."""
    value: Any = None

    def __post_init__(self):
        if self.name is None:
            self.name = str(self.value)

    def __repr__(self) -> str:
        return f"const {self.value}"

    # ============================================================================
    # Arithmetic operations for constant folding
    # ============================================================================

    def _binary_op(self, other, op):
        """Helper for binary operations."""
        # Extract Python value from other if it's a ConstantValue
        if isinstance(other, ConstantValue):
            other_val = other.value
        else:
            other_val = other

        # Perform the operation on raw Python values
        result = op(self.value, other_val)

        # Return a new ConstantValue with the result
        from ..lang.types import value_to_type
        result_type = value_to_type(result)
        if result_type is None:
            # For operations that return Python types not directly supported,
            # just return the raw Python value
            return result
        return ConstantValue(typ=result_type, value=result)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: b / a)

    def __floordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b)

    def __rfloordiv__(self, other):
        return self._binary_op(other, lambda a, b: b // a)

    def __mod__(self, other):
        return self._binary_op(other, lambda a, b: a % b)

    def __rmod__(self, other):
        return self._binary_op(other, lambda a, b: b % a)

    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._binary_op(other, lambda a, b: b ** a)

    def __neg__(self):
        from ..lang.types import value_to_type
        result_type = value_to_type(-self.value)
        if result_type is None:
            return -self.value
        return ConstantValue(typ=result_type, value=-self.value)

    def __pos__(self):
        return self

    def __abs__(self):
        from ..lang.types import value_to_type
        result_type = value_to_type(abs(self.value))
        if result_type is None:
            return abs(self.value)
        return ConstantValue(typ=result_type, value=abs(self.value))

    # Comparison operators
    def __eq__(self, other):
        if isinstance(other, ConstantValue):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ConstantValue):
            return self.value != other.value
        return self.value != other

    def __lt__(self, other):
        if isinstance(other, ConstantValue):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, ConstantValue):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, ConstantValue):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, ConstantValue):
            return self.value >= other.value
        return self.value >= other

    def __bool__(self):
        return bool(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


@dataclass
class ArgumentValue(Value):
    """Function argument value."""
    index: int = 0
    is_reference: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = f"arg{self.index}"

    @property
    def is_pointer(self) -> bool:
        return self.is_reference


# Forward reference for Instruction
@dataclass
class Instruction:
    """IR instruction."""
    op: Op
    typ: Any  # Type, use Any to avoid circular import
    args: list[Union[int, str, Value, 'BasicBlock']] = field(default_factory=list)
    result: Optional[str] = None
    loc: Optional[SourceLocation] = None

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        if self.result:
            return f"%{self.result} = {self.op.name}({args_str})"
        return f"{self.op.name}({args_str})"

    @property
    def type(self) -> Any:
        """Backward compatibility accessor for type."""
        return self.typ


@dataclass
class InstructionValue(Value):
    """Value produced by an instruction."""
    instruction: Optional[Instruction] = None

    def __repr__(self) -> str:
        if self.name:
            return f"%{self.name}"
        return f"%<inst>"

    @property
    def is_pointer(self) -> bool:
        if self.instruction:
            from .op import Op
            return self.instruction.op in (Op.ALLOCA, Op.GEP)
        return False


# Update the forward reference
InstructionValue.__annotations__['instruction'] = Optional[Instruction]


# ============================================================================
# Basic Blocks and Functions
# ============================================================================

@dataclass
class BasicBlock:
    """Basic block in IR."""
    name: str
    instructions: list[Instruction] = field(default_factory=list)
    loc: Optional[SourceLocation] = None

    def __repr__(self) -> str:
        lines = [f"{self.name}:"]
        for inst in self.instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)

    def is_terminated(self) -> bool:
        """Check if the block has a terminator instruction."""
        if not self.instructions:
            return False
        last = self.instructions[-1]
        return last.op in (
            Op.RETURN, Op.BREAK, Op.CONTINUE
        )

    def add_instruction(self, inst: Instruction) -> None:
        """Add an instruction to the block."""
        if self.is_terminated():
            raise RuntimeError(f"Cannot add instruction to terminated block {self.name}")
        self.instructions.append(inst)


@dataclass
class Function:
    """IR function."""
    name: str
    arg_types: list[Any]  # Type
    ret_type: Optional[Any]  # Type
    blocks: list[BasicBlock] = field(default_factory=list)
    is_kernel: bool = False
    arg_is_reference: list[bool] = field(default_factory=list)
    block_size: Optional[tuple[int, int, int]] = None
    loc: Optional[SourceLocation] = None

    def __post_init__(self):
        if not self.arg_is_reference:
            self.arg_is_reference = [False] * len(self.arg_types)

    def __repr__(self) -> str:
        args_strs = []
        for t, is_ref in zip(self.arg_types, self.arg_is_reference):
            s = str(t)
            if is_ref:
                s = f"ref<{s}>"
            args_strs.append(s)
        args_str = ", ".join(args_strs)
        ret_str = str(self.ret_type) if self.ret_type else "void"
        kind = "kernel" if self.is_kernel else "callable"
        lines = [f"{kind} @{self.name}({args_str}) -> {ret_str} {{"]
        for block in self.blocks:
            lines.append(str(block))
        lines.append("}")
        return "\n".join(lines)

    def get_block(self, name: str) -> Optional[BasicBlock]:
        """Get a block by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None


@dataclass
class Module:
    """IR module containing functions."""
    functions: list[Function] = field(default_factory=list)
    constants: list[ConstantValue] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["module {"]
        for func in self.functions:
            lines.append(str(func))
        lines.append("}")
        return "\n".join(lines)

    def add_function(self, func: Function) -> None:
        """Add a function to the module."""
        self.functions.append(func)

    def get_function(self, name: str) -> Optional[Function]:
        """Get a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None
