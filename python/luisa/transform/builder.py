"""
IR Builder for the LuisaCompute Python DSL v2.

This module provides the Builder class that constructs IR using
a fluent API with context managers for control flow.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

from .ir import (ArgumentValue, BasicBlock, ConstantValue, Function,
                 Instruction, InstructionValue, SourceLocation, Value)
# Runtime imports
from .op import Op

# ============================================================================
# Global Builder Context
# ============================================================================

_current_builder: Builder | None = None


def get_current_builder() -> Builder:
    """Get the current builder."""
    if _current_builder is None:
        raise RuntimeError("No active builder context")
    return _current_builder


@contextmanager
def set_current_builder(builder: Builder | None):
    """Set the current builder (as a context manager)."""
    global _current_builder
    old_builder = _current_builder
    _current_builder = builder
    try:
        yield
    finally:
        _current_builder = old_builder


# ============================================================================
# IR Builder
# ============================================================================

class Builder:
    """
    Builder for constructing IR.

    This class provides methods that directly correspond to IR operations.
    When executed, it constructs the IR data structures.
    """

    def __init__(self, name: str, arg_types: tuple[Any, ...],
                 ret_type: Optional[Any],
                 arg_is_reference: Optional[list[bool]] = None):
        self.name = name
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self.blocks: list[BasicBlock] = []
        self.current_block: Optional[BasicBlock] = None
        self.instruction_counter = 0
        self.local_vars: dict[str, Value] = {}  # name -> Value
        self.arg_values: list[ArgumentValue] = []
        self.arg_is_reference = arg_is_reference if arg_is_reference else [False] * len(arg_types)
        self.current_loc: Optional[SourceLocation] = None

        # Initialize argument values
        for i, (arg_typ, is_ref) in enumerate(zip(self.arg_types, self.arg_is_reference)):
            arg_val = ArgumentValue(typ=arg_typ, index=i, is_reference=is_ref)
            self.arg_values.append(arg_val)

    # ========================================================================
    # Location Management
    # ========================================================================

    def set_location(self, file: str, line: int) -> None:
        """Set the current source location for debugging."""
        self.current_loc = SourceLocation(file=file, line=line)

    def clear_location(self) -> None:
        """Clear the current source location."""
        self.current_loc = None

    # ========================================================================
    # Value Creation
    # ========================================================================

    def constant(self, typ: Any, value: Any) -> ConstantValue:
        """Create a constant value."""
        return ConstantValue(typ=typ, value=value)

    def get_argument(self, index: int) -> ArgumentValue:
        """Get an argument value by index."""
        return self.arg_values[index]

    def bind_local(self, name: str, value: Value) -> None:
        """Bind a local variable name to a value."""
        self.local_vars[name] = value

    def lookup_local(self, name: str) -> Optional[Value]:
        """Lookup a local variable by name."""
        return self.local_vars.get(name)

    # ========================================================================
    # Block Management
    # ========================================================================

    def create_block(self, name: str, loc: Optional[SourceLocation] = None) -> BasicBlock:
        """Create a new basic block."""
        # Ensure unique name
        base_name = name
        counter = 0
        while any(b.name == name for b in self.blocks):
            name = f"{base_name}_{counter}"
            counter += 1

        if loc is None:
            loc = self.current_loc

        block = BasicBlock(name=name, loc=loc)
        self.blocks.append(block)
        return block

    def set_insert_point(self, block: BasicBlock) -> None:
        """Set the current insertion point to a block."""
        self.current_block = block

    def get_insert_point(self) -> Optional[BasicBlock]:
        """Get the current insertion point."""
        return self.current_block

    @contextmanager
    def scope(self, block: BasicBlock):
        """Context manager to set insertion point to a block."""
        old_block = self.current_block
        self.set_insert_point(block)
        try:
            yield block
        finally:
            self.set_insert_point(old_block)

    # ========================================================================
    # Instruction Emission
    # ========================================================================

    def _resolve_type(self, typ: Any) -> Any:
        """Resolve a DSL type (handles @struct decorated classes)."""
        if hasattr(typ, '_dsl_type'):
            if typ._dsl_type is not None:
                return typ._dsl_type
            if hasattr(typ, 'get_dsl_type'):
                return typ.get_dsl_type()
        return typ

    def _emit(self, op: Op, typ: Any, args: list,
              name: Optional[str] = None,
              loc: Optional[SourceLocation] = None) -> InstructionValue:
        """Emit an instruction and return its result value."""
        if self.current_block is None:
            raise RuntimeError("No current block set")

        if self.current_block.is_terminated():
            raise RuntimeError(f"Cannot emit instruction to terminated block {self.current_block.name}")

        # Resolve type (handles @struct classes)
        typ = self._resolve_type(typ)

        # Generate result name if not provided
        if name is None:
            name = f"v{self.instruction_counter}"
        elif not name.startswith('v'):
            name = f"v{name}"
        self.instruction_counter += 1

        if loc is None:
            loc = self.current_loc

        # Create instruction
        inst = Instruction(op=op, typ=typ, args=args, result=name, loc=loc)
        self.current_block.add_instruction(inst)

        # Create and return result value
        result = InstructionValue(typ=typ, name=name, instruction=inst)
        return result

    # ========================================================================
    # Arithmetic Operations
    # ========================================================================

    def add(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit an add instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.ADD, result_type, [left, right])

    def sub(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit a sub instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.SUB, result_type, [left, right])

    def mul(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit a mul instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.MUL, result_type, [left, right])

    def div(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit a div instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.DIV, result_type, [left, right])

    def neg(self, operand: Value) -> InstructionValue:
        """Emit a neg instruction."""
        return self._emit(Op.NEG, operand.type, [operand])

    def mod(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit a modulo instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.MOD, result_type, [left, right])

    def pow(self, left: Value, right: Value,
            result_type: Optional[Any] = None) -> InstructionValue:
        """Emit a power instruction."""
        if result_type is None:
            from ..lang.types import promote_types
            result_type = promote_types(left.type, right.type)
        return self._emit(Op.POW, result_type, [left, right])

    def floor(self, operand: Value) -> InstructionValue:
        """Emit a floor instruction."""
        return self._emit(Op.FLOOR, operand.type, [operand])

    def bit_and(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise AND instruction."""
        return self._emit(Op.BIT_AND, left.type, [left, right])

    def bit_or(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise OR instruction."""
        return self._emit(Op.BIT_OR, left.type, [left, right])

    def bit_xor(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise XOR instruction."""
        return self._emit(Op.BIT_XOR, left.type, [left, right])

    def bit_not(self, operand: Value) -> InstructionValue:
        """Emit a bitwise NOT instruction."""
        return self._emit(Op.BIT_NOT, operand.type, [operand])

    def shl(self, left: Value, right: Value) -> InstructionValue:
        """Emit a left shift instruction."""
        return self._emit(Op.SHL, left.type, [left, right])

    def shr(self, left: Value, right: Value) -> InstructionValue:
        """Emit a right shift instruction."""
        return self._emit(Op.SHR, left.type, [left, right])

    def swizzle(self, vector: Value, pattern: str) -> InstructionValue:
        """
        Emit a vector swizzle operation.

        Args:
            vector: The vector to swizzle
            pattern: Swizzle pattern like 'x', 'xy', 'xyz', 'xyzw', 'rgba', etc.
        """
        from ..lang.types import Vector

        if not isinstance(vector.type, Vector):
            raise TypeError(f"Can only swizzle vectors, got {vector.type}")

        # Determine result type based on pattern length
        result_size = len(pattern)
        if result_size < 1 or result_size > 4:
            raise ValueError(f"Invalid swizzle pattern length: {len(pattern)}")

        # Single component swizzle returns scalar, not Vector(1)
        if result_size == 1:
            result_type = vector.type.element
        else:
            result_type = Vector(vector.type.element, result_size)

        # For now, emit as a generic swizzle - the backend handles the pattern
        # In a full implementation, we'd parse the pattern into component indices
        return self._emit(Op.SWIZZLE, result_type, [vector, pattern])

    def member(self, struct_val: Value, member_name: str) -> InstructionValue:
        """
        Emit a struct member access operation.

        Args:
            struct_val: The struct value
            member_name: Name of the field to access
        """
        from ..lang.types import Struct

        # Resolve type (handles @struct classes)
        typ = self._resolve_type(struct_val.type)

        if not isinstance(typ, Struct):
            raise TypeError(f"Can only access members of structs, got {typ}")

        # Determine result type from struct field
        result_type = typ.get_field_type(member_name)
        if result_type is None:
            raise AttributeError(f"Struct {typ.name} has no member '{member_name}'")

        return self._emit(Op.MEMBER_ACCESS, result_type, [struct_val, member_name])

    # ========================================================================
    # Comparison Operations
    # ========================================================================

    def eq(self, left: Value, right: Value) -> InstructionValue:
        """Emit an equality comparison."""
        from ..lang.types import Bool
        return self._emit(Op.EQ, Bool, [left, right])

    def ne(self, left: Value, right: Value) -> InstructionValue:
        """Emit a not-equal comparison."""
        from ..lang.types import Bool
        return self._emit(Op.NE, Bool, [left, right])

    def lt(self, left: Value, right: Value) -> InstructionValue:
        """Emit a less-than comparison."""
        from ..lang.types import Bool
        return self._emit(Op.LT, Bool, [left, right])

    def le(self, left: Value, right: Value) -> InstructionValue:
        """Emit a less-than-or-equal comparison."""
        from ..lang.types import Bool
        return self._emit(Op.LE, Bool, [left, right])

    def gt(self, left: Value, right: Value) -> InstructionValue:
        """Emit a greater-than comparison."""
        from ..lang.types import Bool
        return self._emit(Op.GT, Bool, [left, right])

    def ge(self, left: Value, right: Value) -> InstructionValue:
        """Emit a greater-than-or-equal comparison."""
        from ..lang.types import Bool
        return self._emit(Op.GE, Bool, [left, right])

    # ========================================================================
    # Logical Operations
    # ========================================================================

    def logical_and(self, left: Value, right: Value) -> InstructionValue:
        """Emit a logical AND."""
        from ..lang.types import Bool
        return self._emit(Op.LOGICAL_AND, Bool, [left, right])

    def logical_or(self, left: Value, right: Value) -> InstructionValue:
        """Emit a logical OR."""
        from ..lang.types import Bool
        return self._emit(Op.LOGICAL_OR, Bool, [left, right])

    def logical_not(self, operand: Value) -> InstructionValue:
        """Emit a logical NOT."""
        from ..lang.types import Bool
        return self._emit(Op.LOGICAL_NOT, Bool, [operand])

    # ========================================================================
    # Memory Operations
    # ========================================================================

    def alloca(self, typ: Any, name: Optional[str] = None) -> InstructionValue:
        """Emit an alloca instruction (allocate local variable)."""
        if name is not None and not name.startswith('v'):
            name = f"v{name}"
        return self._emit(Op.ALLOCA, typ, [], name)

    def load(self, ptr: Value, typ: Optional[Any] = None) -> InstructionValue:
        """Emit a load instruction."""
        if typ is None:
            typ = ptr.type
        return self._emit(Op.LOAD, typ, [ptr])

    def store(self, ptr: Value, value: Value) -> InstructionValue:
        """Emit a store instruction."""
        return self._emit(Op.STORE, None, [ptr, value])

    def buffer_read(self, buffer: Value, index: Value, elem_type: Any) -> InstructionValue:
        """Emit a buffer read instruction."""
        return self._emit(Op.BUFFER_READ, elem_type, [buffer, index])

    def buffer_write(self, buffer: Value, index: Value, value: Value) -> InstructionValue:
        """Emit a buffer write instruction."""
        return self._emit(Op.BUFFER_WRITE, None, [buffer, index, value])

    # ========================================================================
    # Control Flow
    # ========================================================================

    def return_(self, value: Optional[Value] = None) -> InstructionValue:
        """Emit a return instruction."""
        if value is None:
            return self._emit(Op.RETURN, None, [])
        return self._emit(Op.RETURN, value.type, [value])

    def break_(self) -> InstructionValue:
        """Emit a break instruction."""
        return self._emit(Op.BREAK, None, [])

    def continue_(self) -> InstructionValue:
        """Emit a continue instruction."""
        return self._emit(Op.CONTINUE, None, [])

    def call(self, func: Any, *args: Value) -> InstructionValue:
        """Emit a function call instruction."""
        if hasattr(func, 'compile'):
            # It's likely a StagedFunction, compile it for these arguments
            func = func.compile(self, *args)

        # Determine return type
        ret_type = func.ret_type if func.ret_type is not None else None
        # Include function object as first argument, then actual args
        call_args = [func] + list(args)
        return self._emit(Op.CALL, ret_type, call_args)

    def cast(self, value: Value, target_typ: Any) -> InstructionValue:
        """Emit a type cast instruction."""
        return self._emit(Op.CAST, target_typ, [value])

    def bitcast(self, value: Value, target_typ: Any) -> InstructionValue:
        """Emit a bitcast instruction (preserves bit pattern)."""
        return self._emit(Op.BITCAST, target_typ, [value])

    # ========================================================================
    # Control Flow - Structured API (New Style)
    # ========================================================================

    def if_(self, cond: Value) -> 'IfStmt':
        """
        Create an if statement.

        Usage:
            if_ = builder.if_(cond)
            with if_.true_scope():
                ...  # true branch
            with if_.false_scope():  # optional
                ...  # false branch
        """
        from ..lang.control_flow import IfStmt
        return IfStmt(self, cond)

    def while_(self, cond: Value) -> 'WhileStmt':
        """
        Create a while loop statement.

        Usage:
            while_ = builder.while_(cond)
            with while_.body_scope():
                ...  # loop body
        """
        from ..lang.control_flow import WhileStmt
        return WhileStmt(self, cond)

    def for_range(self, start: Value, stop: Value,
                  step: Value, loop_var: str) -> 'ForRangeStmt':
        """
        Create a for-range loop statement (dynamic device-side).

        Usage:
            for_ = builder.for_range(start, stop, step, 'i')
            with for_.body_scope():
                ...  # loop body, 'i' is bound to loop variable
        """
        from ..lang.control_flow import ForRangeStmt
        return ForRangeStmt(self, start, stop, step, loop_var)

    def for_unrolled(self, start: int, stop: int,
                     step: int, loop_var: str) -> 'UnrolledForStmt':
        """
        Create an unrolled for loop statement (compile-time unrolling).

        Usage:
            for_ = builder.for_unrolled(0, 4, 1, 'i')
            for i in for_.body_scope():
                ...  # loop body, fully unrolled

        Use only for small iteration counts!
        """
        from ..lang.control_flow import UnrolledForStmt
        return UnrolledForStmt(self, start, stop, step, loop_var)

    def switch(self, value: Value) -> 'SwitchStmt':
        """
        Create a switch statement.

        Usage:
            switch = builder.switch(value)
            with switch.case_scope(1):
                ...  # case 1
            with switch.case_scope(2, 3):
                ...  # case 2 or 3
            with switch.default_scope():
                ...  # default case
        """
        from ..lang.control_flow import SwitchStmt
        return SwitchStmt(self, value)

    # ========================================================================
    # Build
    # ========================================================================

    def build(self) -> Function:
        """Finalize and return the IR function."""
        return Function(
            name=self.name,
            arg_types=self.arg_types,
            ret_type=self.ret_type,
            blocks=self.blocks,
            is_kernel=False,  # Set by caller if needed
            arg_is_reference=self.arg_is_reference,
            loc=self.current_loc
        )
