"""
IR Builder for the LuisaCompute Python DSL v2.

This module provides the IRBuilder class that constructs IR using
a fluent API with context managers for control flow.
"""

from __future__ import annotations
from typing import Optional, Any
from contextlib import contextmanager

from dataclasses import dataclass

# Runtime imports
from .types import Type
from .types import (
    Void, bool_, promote_types
)
from .ir import (
    IROp, Value, ConstantValue, InstructionValue, ArgumentValue,
    IRInstruction, IRBasicBlock, IRFunction, IRModule, SourceLocation
)

# ============================================================================
# Global Builder Context
# ============================================================================

_current_builder: IRBuilder | None = None


def get_current_builder() -> IRBuilder:
    """Get the current builder."""
    if _current_builder is None:
        raise RuntimeError("No active builder context")
    return _current_builder


def set_current_builder(builder: IRBuilder | None) -> None:
    """Set the current builder (called by executor)."""
    global _current_builder
    _current_builder = builder


# ============================================================================
# IR Builder
# ============================================================================

class IRBuilder:
    """
    Builder for constructing IR.
    
    This class provides methods that directly correspond to IR operations.
    When executed, it constructs the IR data structures.
    """
    
    def __init__(self, name: str, arg_types: tuple[Type, ...], 
                 ret_type: Optional[Type],
                 arg_is_reference: Optional[list[bool]] = None):
        self.name = name
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self.blocks: list[IRBasicBlock] = []
        self.current_block: Optional[IRBasicBlock] = None
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
    
    def constant(self, typ: Type, value: Any) -> ConstantValue:
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
    
    def create_block(self, name: str, loc: Optional[SourceLocation] = None) -> IRBasicBlock:
        """Create a new basic block."""
        # Ensure unique name
        base_name = name
        counter = 0
        while any(b.name == name for b in self.blocks):
            name = f"{base_name}_{counter}"
            counter += 1
        
        if loc is None:
            loc = self.current_loc
            
        block = IRBasicBlock(name=name, loc=loc)
        self.blocks.append(block)
        return block
    
    def set_insert_point(self, block: IRBasicBlock) -> None:
        """Set the current insertion point to a block."""
        self.current_block = block
    
    def get_insert_point(self) -> Optional[IRBasicBlock]:
        """Get the current insertion point."""
        return self.current_block
    
    @contextmanager
    def scope(self, block: IRBasicBlock):
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
    
    def _emit(self, op: IROp, typ: Type, args: list,
              name: Optional[str] = None,
              loc: Optional[SourceLocation] = None) -> InstructionValue:
        """Emit an instruction and return its result value."""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        
        if self.current_block.is_terminated():
            raise RuntimeError(f"Cannot emit instruction to terminated block {self.current_block.name}")
        
        # Generate result name if not provided
        if name is None:
            name = f"t{self.instruction_counter}"
        self.instruction_counter += 1
        
        if loc is None:
            loc = self.current_loc
            
        # Create instruction
        inst = IRInstruction(op=op, typ=typ, args=args, result=name, loc=loc)
        self.current_block.add_instruction(inst)

        # Create and return result value
        result = InstructionValue(typ=typ, name=name, instruction=inst)
        return result
    
    # ========================================================================
    # Arithmetic Operations
    # ========================================================================
    
    def add(self, left: Value, right: Value, 
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit an add instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.ADD, result_type, [left, right])
    
    def sub(self, left: Value, right: Value,
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit a sub instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.SUB, result_type, [left, right])
    
    def mul(self, left: Value, right: Value,
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit a mul instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.MUL, result_type, [left, right])
    
    def div(self, left: Value, right: Value,
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit a div instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.DIV, result_type, [left, right])
    
    def neg(self, operand: Value) -> InstructionValue:
        """Emit a neg instruction."""
        return self._emit(IROp.NEG, operand.type, [operand])
    
    def mod(self, left: Value, right: Value,
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit a modulo instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.MOD, result_type, [left, right])
    
    def pow(self, left: Value, right: Value,
            result_type: Optional[Type] = None) -> InstructionValue:
        """Emit a power instruction."""
        if result_type is None:
            result_type = promote_types(left.type, right.type)
        return self._emit(IROp.POW, result_type, [left, right])
    
    def floor(self, operand: Value) -> InstructionValue:
        """Emit a floor instruction."""
        return self._emit(IROp.FLOOR, operand.type, [operand])
    
    def bit_and(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise AND instruction."""
        return self._emit(IROp.BIT_AND, left.type, [left, right])
    
    def bit_or(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise OR instruction."""
        return self._emit(IROp.BIT_OR, left.type, [left, right])
    
    def bit_xor(self, left: Value, right: Value) -> InstructionValue:
        """Emit a bitwise XOR instruction."""
        return self._emit(IROp.BIT_XOR, left.type, [left, right])
    
    def bit_not(self, operand: Value) -> InstructionValue:
        """Emit a bitwise NOT instruction."""
        return self._emit(IROp.BIT_NOT, operand.type, [operand])
    
    def shl(self, left: Value, right: Value) -> InstructionValue:
        """Emit a left shift instruction."""
        return self._emit(IROp.SHL, left.type, [left, right])
    
    def shr(self, left: Value, right: Value) -> InstructionValue:
        """Emit a right shift instruction."""
        return self._emit(IROp.SHR, left.type, [left, right])
    
    def swizzle(self, vector: Value, pattern: str) -> InstructionValue:
        """
        Emit a vector swizzle operation.

        Args:
            vector: The vector to swizzle
            pattern: Swizzle pattern like 'x', 'xy', 'xyz', 'xyzw', 'rgba', etc.
        """
        from .types import Vector

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
        return self._emit(IROp.SWIZZLE, result_type, [vector, pattern])
    
    # ========================================================================
    # Comparison Operations
    # ========================================================================
    
    def eq(self, left: Value, right: Value) -> InstructionValue:
        """Emit an equality comparison."""
        return self._emit(IROp.EQ, bool_, [left, right])
    
    def ne(self, left: Value, right: Value) -> InstructionValue:
        """Emit a not-equal comparison."""
        return self._emit(IROp.NE, bool_, [left, right])
    
    def lt(self, left: Value, right: Value) -> InstructionValue:
        """Emit a less-than comparison."""
        return self._emit(IROp.LT, bool_, [left, right])
    
    def le(self, left: Value, right: Value) -> InstructionValue:
        """Emit a less-than-or-equal comparison."""
        return self._emit(IROp.LE, bool_, [left, right])
    
    def gt(self, left: Value, right: Value) -> InstructionValue:
        """Emit a greater-than comparison."""
        return self._emit(IROp.GT, bool_, [left, right])
    
    def ge(self, left: Value, right: Value) -> InstructionValue:
        """Emit a greater-than-or-equal comparison."""
        return self._emit(IROp.GE, bool_, [left, right])
    
    # ========================================================================
    # Logical Operations
    # ========================================================================
    
    def logical_and(self, left: Value, right: Value) -> InstructionValue:
        """Emit a logical AND."""
        return self._emit(IROp.LOGICAL_AND, bool_, [left, right])
    
    def logical_or(self, left: Value, right: Value) -> InstructionValue:
        """Emit a logical OR."""
        return self._emit(IROp.LOGICAL_OR, bool_, [left, right])
    
    def logical_not(self, operand: Value) -> InstructionValue:
        """Emit a logical NOT."""
        return self._emit(IROp.LOGICAL_NOT, bool_, [operand])
    
    # ========================================================================
    # Memory Operations
    # ========================================================================
    
    def alloca(self, typ: Type, name: Optional[str] = None) -> InstructionValue:
        """Emit an alloca instruction (allocate local variable)."""
        return self._emit(IROp.ALLOCA, typ, [], name)
    
    def load(self, ptr: Value, typ: Optional[Type] = None) -> InstructionValue:
        """Emit a load instruction."""
        if typ is None:
            typ = ptr.type
        return self._emit(IROp.LOAD, typ, [ptr])
    
    def store(self, ptr: Value, value: Value) -> InstructionValue:
        """Emit a store instruction."""
        return self._emit(IROp.STORE, Void(), [ptr, value])
    
    def buffer_read(self, buffer: Value, index: Value, elem_type: Type) -> InstructionValue:
        """Emit a buffer read instruction."""
        return self._emit(IROp.BUFFER_READ, elem_type, [buffer, index])
    
    def buffer_write(self, buffer: Value, index: Value, value: Value) -> InstructionValue:
        """Emit a buffer write instruction."""
        return self._emit(IROp.BUFFER_WRITE, Void(), [buffer, index, value])
    
    # ========================================================================
    # Control Flow
    # ========================================================================
    
    def return_(self, value: Optional[Value] = None) -> InstructionValue:
        """Emit a return instruction."""
        if value is None:
            return self._emit(IROp.RETURN, Void(), [])
        return self._emit(IROp.RETURN, value.type, [value])
    
    def break_(self) -> InstructionValue:
        """Emit a break instruction."""
        return self._emit(IROp.BREAK, Void(), [])
    
    def continue_(self) -> InstructionValue:
        """Emit a continue instruction."""
        return self._emit(IROp.CONTINUE, Void(), [])
    
    def call(self, func: 'IRFunction', args: list[Value]) -> InstructionValue:
        """Emit a function call instruction."""
        # Determine return type
        ret_type = func.ret_type if func.ret_type else Void()
        # Include function name as first argument, then actual args
        call_args = [func.name] + args
        return self._emit(IROp.CALL, ret_type, call_args)
    
    def cast(self, value: Value, target_typ: Type) -> InstructionValue:
        """Emit a type cast instruction."""
        return self._emit(IROp.CAST, target_typ, [value])

    def bitcast(self, value: Value, target_typ: Type) -> InstructionValue:
        """Emit a bitcast instruction (preserves bit pattern)."""
        return self._emit(IROp.BITCAST, target_typ, [value])
    
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
        from . import control_flow
        return control_flow.IfStmt(self, cond)
    
    def while_(self, cond: Value) -> 'WhileStmt':
        """
        Create a while loop statement.
        
        Usage:
            while_ = builder.while_(cond)
            with while_.body_scope():
                ...  # loop body
        """
        from . import control_flow
        return control_flow.WhileStmt(self, cond)
    
    def for_range(self, start: Value, stop: Value,
                  step: Value, loop_var: str) -> 'ForRangeStmt':
        """
        Create a for-range loop statement (dynamic device-side).
        
        Usage:
            for_ = builder.for_range(start, stop, step, 'i')
            with for_.body_scope():
                ...  # loop body, 'i' is bound to loop variable
        """
        from . import control_flow
        return control_flow.ForRangeStmt(self, start, stop, step, loop_var)
    
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
        from . import control_flow
        return control_flow.UnrolledForStmt(self, start, stop, step, loop_var)
    
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
        from . import control_flow
        return control_flow.SwitchStmt(self, value)
    
    # ========================================================================
    # Build
    # ========================================================================
    
    def build(self) -> IRFunction:
        """Finalize and return the IR function."""
        return IRFunction(
            name=self.name,
            arg_types=self.arg_types,
            ret_type=self.ret_type,
            blocks=self.blocks,
            is_kernel=False,  # Set by caller if needed
            arg_is_reference=self.arg_is_reference,
            loc=self.current_loc
        )
