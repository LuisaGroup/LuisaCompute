"""
IR Builder for the LuisaCompute Python DSL v2.

This module provides the IRBuilder class that constructs IR using
a fluent API with context managers for control flow.
"""

from __future__ import annotations
from typing import Optional, Union, Any, TYPE_CHECKING, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .dsl_types import Type
    from .ir import Value, ConstantValue, IRInstruction, IRBasicBlock, IRFunction, IROp

from .dsl_types import (
    Scalar, Vector, Matrix, Void, bool_, int32, uint32, float32,
    is_integer_type, is_float_type, is_bool_type, promote_types
)
from .ir import (
    IROp, Value, ConstantValue, InstructionValue, ArgumentValue,
    IRInstruction, IRBasicBlock, IRFunction, IRModule
)


# ============================================================================
# Loop Context
# ============================================================================

@dataclass
class LoopContext:
    """Context for a loop, used by break and continue."""
    header_block: IRBasicBlock
    exit_block: IRBasicBlock


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
                 ret_type: Optional[Type]):
        self.name = name
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self.blocks: list[IRBasicBlock] = []
        self.current_block: Optional[IRBasicBlock] = None
        self.instruction_counter = 0
        self.local_vars: dict[str, Value] = {}  # name -> Value
        self.loop_stack: list[LoopContext] = []  # Stack for break/continue
        self.arg_values: list[ArgumentValue] = []
        
        # Initialize argument values
        for i, arg_type in enumerate(arg_types):
            arg_val = ArgumentValue(type=arg_type, index=i)
            self.arg_values.append(arg_val)
    
    # ========================================================================
    # Value Creation
    # ========================================================================
    
    def constant(self, type: Type, value: Any) -> ConstantValue:
        """Create a constant value."""
        return ConstantValue(type=type, value=value)
    
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
    
    def create_block(self, name: str) -> IRBasicBlock:
        """Create a new basic block."""
        # Ensure unique name
        base_name = name
        counter = 0
        while any(b.name == name for b in self.blocks):
            name = f"{base_name}_{counter}"
            counter += 1
        
        block = IRBasicBlock(name=name)
        self.blocks.append(block)
        return block
    
    def set_insert_point(self, block: IRBasicBlock) -> None:
        """Set the current insertion point to a block."""
        self.current_block = block
    
    def get_insert_point(self) -> Optional[IRBasicBlock]:
        """Get the current insertion point."""
        return self.current_block
    
    # ========================================================================
    # Instruction Emission
    # ========================================================================
    
    def _emit(self, op: IROp, type: Type, args: list, 
              name: Optional[str] = None) -> InstructionValue:
        """Emit an instruction and return its result value."""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        
        if self.current_block.is_terminated():
            raise RuntimeError(f"Cannot emit instruction to terminated block {self.current_block.name}")
        
        # Generate result name if not provided
        if name is None:
            name = f"t{self.instruction_counter}"
        self.instruction_counter += 1
        
        # Create instruction
        inst = IRInstruction(op=op, type=type, args=args, result=name)
        self.current_block.add_instruction(inst)
        
        # Create and return result value
        result = InstructionValue(type=type, name=name, instruction=inst)
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
    
    def alloca(self, type: Type, name: Optional[str] = None) -> InstructionValue:
        """Emit an alloca instruction (allocate local variable)."""
        return self._emit(IROp.ALLOCA, type, [], name)
    
    def load(self, ptr: Value, type: Optional[Type] = None) -> InstructionValue:
        """Emit a load instruction."""
        if type is None:
            type = ptr.type
        return self._emit(IROp.LOAD, type, [ptr])
    
    def store(self, ptr: Value, value: Value) -> InstructionValue:
        """Emit a store instruction."""
        return self._emit(IROp.STORE, Void(), [ptr, value])
    
    # ========================================================================
    # Control Flow - Branches
    # ========================================================================
    
    def branch(self, target: IRBasicBlock) -> IRInstruction:
        """Emit an unconditional branch."""
        return self._emit(IROp.BR, Void(), [target.name])
    
    def branch_conditional(self, cond: Value, 
                          true_target: IRBasicBlock,
                          false_target: IRBasicBlock) -> IRInstruction:
        """Emit a conditional branch."""
        return self._emit(IROp.COND_BR, Void(), 
                         [cond, true_target.name, false_target.name])
    
    def return_(self, value: Optional[Value] = None) -> IRInstruction:
        """Emit a return instruction."""
        if value is None:
            return self._emit(IROp.RETURN, Void(), [])
        return self._emit(IROp.RETURN, value.type, [value])
    
    # ========================================================================
    # Control Flow - High Level (Context Managers)
    # ========================================================================
    
    def if_(self, cond: Value) -> IfScope:
        """Create an if scope for conditional branching."""
        return IfScope(self, cond)
    
    def while_(self, cond: Value) -> WhileScope:
        """Create a while loop scope."""
        return WhileScope(self, cond)
    
    def for_range(self, start: Value, stop: Value, 
                  step: Value, loop_var: str) -> ForRangeScope:
        """Create a for-range loop scope (dynamic)."""
        return ForRangeScope(self, start, stop, step, loop_var)
    
    def for_unrolled(self, start: int, stop: int, 
                     step: int, loop_var: str) -> UnrolledForScope:
        """Create an unrolled for-range loop scope."""
        return UnrolledForScope(self, start, stop, step, loop_var)
    
    def switch(self, value: Value) -> SwitchScope:
        """Create a switch scope."""
        return SwitchScope(self, value)
    
    def break_(self) -> None:
        """Emit a break instruction."""
        if not self.loop_stack:
            raise RuntimeError("break outside of loop")
        loop_ctx = self.loop_stack[-1]
        self.branch(loop_ctx.exit_block)
    
    def continue_(self) -> None:
        """Emit a continue instruction."""
        if not self.loop_stack:
            raise RuntimeError("continue outside of loop")
        loop_ctx = self.loop_stack[-1]
        self.branch(loop_ctx.header_block)
    
    def push_loop(self, header: IRBasicBlock, exit_block: IRBasicBlock) -> None:
        """Push a loop context onto the stack."""
        self.loop_stack.append(LoopContext(header, exit_block))
    
    def pop_loop(self) -> None:
        """Pop a loop context from the stack."""
        if self.loop_stack:
            self.loop_stack.pop()
    
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
            is_kernel=False  # Set by caller if needed
        )


# ============================================================================
# Control Flow Scopes
# ============================================================================

class IfScope:
    """
    Context manager for if-then-else control flow.
    
    Automatically folds constant conditions:
    - If cond is known True: only executes true branch
    - If cond is known False: only executes false branch (if exists)
    - Otherwise: generates conditional branch IR
    """
    
    def __init__(self, builder: IRBuilder, condition: Value):
        self.builder = builder
        self.condition = condition
        self.true_block: Optional[IRBasicBlock] = None
        self.false_block: Optional[IRBasicBlock] = None
        self.merge_block: Optional[IRBasicBlock] = None
        self.has_false_branch = False
        self._folded = False
        self._fold_result = None
        
    def __enter__(self):
        # Check for constant folding
        if isinstance(self.condition, ConstantValue):
            if self.condition.value == True:
                # Constant True - no branching needed
                self._folded = True
                self._fold_result = True
                return self
            elif self.condition.value == False:
                # Constant False - skip true branch
                self._folded = True
                self._fold_result = False
                # Return a no-op context
                return NoOpScope(self.builder)
        
        # Not folded - create blocks
        self.true_block = self.builder.create_block("if_true")
        self.merge_block = self.builder.create_block("if_merge")
        
        # Emit conditional branch (initially without false block)
        self.builder.branch_conditional(
            self.condition,
            self.true_block,
            self.merge_block
        )
        
        # Start true block
        self.builder.set_insert_point(self.true_block)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        if self._folded:
            return True
            
        # Branch to merge if not terminated
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.merge_block)
        
        # If no else branch, set insert point to merge
        if not self.has_false_branch:
            self.builder.set_insert_point(self.merge_block)
        
        return True
    
    def otherwise(self):
        """Start the else branch."""
        if self._folded:
            # Return appropriate scope based on constant folding
            if self._fold_result:
                # True was folded, so else branch never executes
                return NoOpScope(self.builder)
            else:
                # False was folded, so else branch always executes
                return DirectScope(self.builder)
        
        self.has_false_branch = True
        
        # Create false block
        self.false_block = self.builder.create_block("if_false")
        
        # We need to update the conditional branch, but since we can't modify
        # existing instructions, we'll emit a new one in a different way
        # For now, we just set up the blocks correctly
        
        # Start false block
        self.builder.set_insert_point(self.false_block)
        
        return ElseScope(self.builder, self.merge_block)


class ElseScope:
    """Context manager for else branch."""
    
    def __init__(self, builder: IRBuilder, merge_block: IRBasicBlock):
        self.builder = builder
        self.merge_block = merge_block
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Branch to merge
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.merge_block)
        
        # Set insert point to merge block
        self.builder.set_insert_point(self.merge_block)
        return True


class NoOpScope:
    """No-op scope for constant-folded branches that don't execute."""
    
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    
    def otherwise(self):
        # Return a scope that will execute
        return DirectScope(self.builder)


class DirectScope:
    """Direct execution scope for constant-folded branches that always execute."""
    
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    
    def otherwise(self):
        # Return no-op for else branch
        return NoOpScope(self.builder)


class WhileScope:
    """
    Context manager for while loops.
    
    Structure:
        header_block:
            if condition:
                goto body_block
            else:
                goto exit_block
        body_block:
            ... body ...
            goto header_block
        exit_block:
    """
    
    def __init__(self, builder: IRBuilder, condition: Value):
        self.builder = builder
        self.condition = condition
        self.header_block: Optional[IRBasicBlock] = None
        self.body_block: Optional[IRBasicBlock] = None
        self.exit_block: Optional[IRBasicBlock] = None
        
    def __enter__(self):
        # Check for constant folding - if False, skip entire loop
        if isinstance(self.condition, ConstantValue) and self.condition.value == False:
            return NoOpScope(self.builder)
        
        # Create blocks
        self.header_block = self.builder.create_block("while_header")
        self.body_block = self.builder.create_block("while_body")
        self.exit_block = self.builder.create_block("while_exit")
        
        # Branch to header
        self.builder.branch(self.header_block)
        
        # Emit condition check in header
        self.builder.set_insert_point(self.header_block)
        self.builder.branch_conditional(
            self.condition,
            self.body_block,
            self.exit_block
        )
        
        # Push loop context for break/continue
        self.builder.push_loop(self.header_block, self.exit_block)
        
        # Start body
        self.builder.set_insert_point(self.body_block)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Loop back to header
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context
        self.builder.pop_loop()
        
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True


class ForRangeScope:
    """
    Context manager for for-loops over range().
    
    This generates a dynamic loop on the device-side.
    """
    
    def __init__(self, builder: IRBuilder, 
                 start: Value, stop: Value, step: Value,
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.loop_var: Optional[InstructionValue] = None
        self.header_block: Optional[IRBasicBlock] = None
        self.body_block: Optional[IRBasicBlock] = None
        self.exit_block: Optional[IRBasicBlock] = None
        
    def __enter__(self):
        # Create blocks
        self.header_block = self.builder.create_block("for_header")
        self.body_block = self.builder.create_block("for_body")
        self.exit_block = self.builder.create_block("for_exit")
        
        # Allocate loop variable
        self.loop_var = self.builder.alloca(self.start.type, name=self.loop_var_name)
        self.builder.store(self.loop_var, self.start)
        
        # Branch to header
        self.builder.branch(self.header_block)
        
        # Header: check condition
        self.builder.set_insert_point(self.header_block)
        current_val = self.builder.load(self.loop_var)
        cond = self.builder.lt(current_val, self.stop)  # i < stop
        self.builder.branch_conditional(cond, self.body_block, self.exit_block)
        
        # Push loop context
        self.builder.push_loop(self.header_block, self.exit_block)
        
        # Start body
        self.builder.set_insert_point(self.body_block)
        
        # Bind loop variable name to current value
        current_val = self.builder.load(self.loop_var)
        self.builder.bind_local(self.loop_var_name, current_val)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Increment loop variable
        current_val = self.builder.load(self.loop_var)
        next_val = self.builder.add(current_val, self.step)
        self.builder.store(self.loop_var, next_val)
        
        # Loop back
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context
        self.builder.pop_loop()
        
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True


class UnrolledForScope:
    """
    Context manager for explicitly unrolled for-loops.
    
    This fully unrolls the loop at compile time.
    Use only for small iteration counts!
    """
    
    def __init__(self, builder: IRBuilder, 
                 start: int, stop: int, step: int,
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.iterations = list(range(start, stop, step))
        self.current_iteration = 0
        
    def __enter__(self):
        # Start first iteration
        self._bind_iteration(0)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        return True
    
    def _bind_iteration(self, idx: int) -> bool:
        """Bind the current iteration value."""
        if idx >= len(self.iterations):
            return False
        
        val = self.iterations[idx]
        const_val = self.builder.constant(int32, val)
        self.builder.bind_local(self.loop_var_name, const_val)
        self.current_iteration = idx
        return True
    
    def next_iteration(self) -> bool:
        """Move to the next iteration. Called by the executor."""
        return self._bind_iteration(self.current_iteration + 1)


class SwitchScope:
    """
    Context manager for switch statements.
    
    Supports constant folding when the switch value is known.
    """
    
    def __init__(self, builder: IRBuilder, value: Value):
        self.builder = builder
        self.value = value
        self.cases: list[tuple[list[int], IRBasicBlock]] = []
        self.default_block: Optional[IRBasicBlock] = None
        self.exit_block: Optional[IRBasicBlock] = None
        self._folded = False
        self._constant_value = None
        
    def __enter__(self):
        # Check for constant folding
        if isinstance(self.value, ConstantValue):
            self._constant_value = self.value.value
            self._folded = True
        else:
            self._folded = False
            self.exit_block = self.builder.create_block("switch_exit")
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        if self._folded:
            return True
            
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True
    
    def case(self, *values: int):
        """Start a case block."""
        if self._folded:
            # Check if this case matches the constant value
            if self._constant_value in values:
                # This case executes
                return DirectScope(self.builder)
            else:
                # This case is skipped
                return NoOpScope(self.builder)
        
        # Create case block
        case_block = self.builder.create_block(f"case_{values[0]}")
        self.cases.append((list(values), case_block))
        
        return CaseScope(self.builder, case_block, self.exit_block)
    
    def default(self):
        """Start the default block."""
        if self._folded:
            # Check if default should execute
            all_case_values = []
            for vals, _ in self.cases:
                all_case_values.extend(vals)
            if self._constant_value not in all_case_values:
                return DirectScope(self.builder)
            else:
                return NoOpScope(self.builder)
        
        self.default_block = self.builder.create_block("case_default")
        return CaseScope(self.builder, self.default_block, self.exit_block)


class CaseScope:
    """Context manager for a case block."""
    
    def __init__(self, builder: IRBuilder, 
                 case_block: IRBasicBlock, exit_block: IRBasicBlock):
        self.builder = builder
        self.case_block = case_block
        self.exit_block = exit_block
        
    def __enter__(self):
        self.builder.set_insert_point(self.case_block)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Break to exit
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.exit_block)
            
        return True
