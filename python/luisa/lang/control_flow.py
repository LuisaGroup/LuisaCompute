"""
Structured Control Flow for the LuisaCompute Python DSL v2.

This module provides a structured API for control flow:
    if_ = IfStmt(builder, cond)
    with if_.true_scope():
        ...  # true branch
    with if_.false_scope():
        ...  # false branch (optional)

Similar patterns for loops and switches.
"""

from __future__ import annotations
from typing import Optional


# Runtime imports
from .ast import Value, IRBasicBlock, ConstantValue
from .builder import IRBuilder


# ============================================================================
# If Statement
# ============================================================================

class IfStmt:
    """
    Structured if statement.
    
    Usage:
        if_ = IfStmt(builder, condition)
        with if_.true_scope():
            ...  # true branch
        with if_.false_scope():  # optional
            ...  # false branch
    
    Supports constant folding when condition is known at compile time.
    """
    
    def __init__(self, builder: 'IRBuilder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self._true_block: Optional['IRBasicBlock'] = None
        self._false_block: Optional['IRBasicBlock'] = None
        self._merge_block: Optional['IRBasicBlock'] = None
        self._has_false_branch = False
        # Check for constant folding at runtime
        self._constant_fold = isinstance(condition, ConstantValue)
        self._fold_true = condition.value if self._constant_fold else None
        
    def true_scope(self):
        """Get context manager for the true branch."""
        # Check for constant folding
        if isinstance(self.condition, ConstantValue):
            if self.condition.value == True:
                # Always true - use direct scope
                return _DirectScope(self.builder)
            elif self.condition.value == False:
                # Always false - skip true branch
                return _NoOpScope()
        
        # Create blocks if not already created
        if self._true_block is None:
            self._true_block = self.builder.create_block("if_true")
            self._merge_block = self.builder.create_block("if_merge")
            
            # Emit conditional branch
            self.builder.branch_conditional(
                self.condition,
                self._true_block,
                self._merge_block
            )
        
        return _BranchScope(
            self.builder,
            self._true_block,
            self._merge_block,
            is_false_branch=False,
            parent=self
        )
    
    def false_scope(self):
        """Get context manager for the false branch."""
        # Check for constant folding
        if isinstance(self.condition, ConstantValue):
            if self.condition.value == True:
                # Always true - skip false branch
                return _NoOpScope()
            elif self.condition.value == False:
                # Always false - use direct scope
                return _DirectScope(self.builder)
        
        # Create false block if not already created
        if self._false_block is None:
            self._false_block = self.builder.create_block("if_false")
            self._has_false_branch = True
            
            # Update the conditional branch to point to false block
            # We need to find and update the last instruction
            if self.builder.current_block and self.builder.current_block.instructions:
                last_inst = self.builder.current_block.instructions[-1]
                if last_inst.op.name == 'COND_BR':
                    # Update the false target
                    last_inst.args = [last_inst.args[0], last_inst.args[1], self._false_block.name]
        
        return _BranchScope(
            self.builder,
            self._false_block,
            self._merge_block,
            is_false_branch=True,
            parent=self
        )


# ============================================================================
# While Loop
# ============================================================================

class WhileStmt:
    """
    Structured while loop.
    
    Usage:
        while_ = WhileStmt(builder, condition)
        with while_.body_scope():
            ...  # loop body
            # break/continue work within this scope
    """
    
    def __init__(self, builder: 'IRBuilder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self._header_block: Optional[IRBasicBlock] = None
        self._body_block: Optional[IRBasicBlock] = None
        self._exit_block: Optional[IRBasicBlock] = None
        
    def body_scope(self):
        """Get context manager for the loop body."""
        # Check for constant folding - if False, skip entire loop
        if isinstance(self.condition, ConstantValue) and self.condition.value == False:
            return _NoOpScope()
        
        # Create blocks
        self._header_block = self.builder.create_block("while_header")
        self._body_block = self.builder.create_block("while_body")
        self._exit_block = self.builder.create_block("while_exit")
        
        # Branch to header
        self.builder.branch(self._header_block)
        
        # Emit condition check in header
        self.builder.set_insert_point(self._header_block)
        self.builder.branch_conditional(
            self.condition,
            self._body_block,
            self._exit_block
        )
        
        # Push loop context for break/continue
        self.builder.push_loop(self._header_block, self._exit_block)
        
        return _LoopScope(
            self.builder,
            self._body_block,
            self._header_block,
            self._exit_block
        )


# ============================================================================
# For Range Loop
# ============================================================================

class ForRangeStmt:
    """
    Structured for-range loop (dynamic device-side loop).
    
    Usage:
        for_ = ForRangeStmt(builder, start, stop, step, loop_var_name)
        with for_.body_scope():
            ...  # loop body, loop var is bound to name
    """
    
    def __init__(self, builder: 'IRBuilder',
                 start: 'Value', stop: 'Value', step: 'Value',
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self._header_block: Optional[IRBasicBlock] = None
        self._body_block: Optional[IRBasicBlock] = None
        self._exit_block: Optional[IRBasicBlock] = None
        self._loop_var: Optional[Value] = None
        
    def body_scope(self):
        """Get context manager for the loop body."""
        # Create blocks
        self._header_block = self.builder.create_block("for_header")
        self._body_block = self.builder.create_block("for_body")
        self._exit_block = self.builder.create_block("for_exit")
        
        # Allocate loop variable
        self._loop_var = self.builder.alloca(self.start.type, name=self.loop_var_name)
        self.builder.store(self._loop_var, self.start)
        
        # Branch to header
        self.builder.branch(self._header_block)
        
        # Header: check condition
        self.builder.set_insert_point(self._header_block)
        current_val = self.builder.load(self._loop_var)
        cond = self.builder.lt(current_val, self.stop)  # i < stop
        self.builder.branch_conditional(cond, self._body_block, self._exit_block)
        
        # Push loop context
        self.builder.push_loop(self._header_block, self._exit_block)
        
        return _ForLoopScope(
            self.builder,
            self._body_block,
            self._header_block,
            self._exit_block,
            self._loop_var,
            self.step,
            self.loop_var_name
        )


# ============================================================================
# Unrolled For Loop
# ============================================================================

class UnrolledForStmt:
    """
    Structured unrolled for loop (compile-time unrolling).
    
    Usage:
        for_ = UnrolledForStmt(builder, 0, 4, 1, loop_var_name)
        for _.body_scope() as scope:
            ...  # loop body, fully unrolled
    
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
        self._current_idx = 0
        
    def body_scope(self):
        """Get context manager for unrolled iterations."""
        return _UnrolledScope(self.builder, self.iterations, self.loop_var_name)


# ============================================================================
# Switch Statement
# ============================================================================

class SwitchStmt:
    """
    Structured switch statement.
    
    Usage:
        switch = SwitchStmt(builder, value)
        with switch.case_scope(1):
            ...  # case 1
        with switch.case_scope(2, 3):
            ...  # case 2 or 3
        with switch.default_scope():
            ...  # default case
    """
    
    def __init__(self, builder: 'IRBuilder', value: 'Value'):
        self.builder = builder
        self.value = value
        self._cases: list[tuple[list[int], IRBasicBlock]] = []
        self._default_block: Optional[IRBasicBlock] = None
        self._exit_block: Optional[IRBasicBlock] = None
        
        self._folded = isinstance(value, ConstantValue)
        self._constant_value = value.value if self._folded else None
        
        if not self._folded:
            self._exit_block = self.builder.create_block("switch_exit")
    
    def case_scope(self, *values: int):
        """Get context manager for a case."""
        if self._folded:
            # Check if this case matches
            if self._constant_value in values:
                return _DirectScope(self.builder)
            else:
                return _NoOpScope()
        
        # Create case block
        case_block = self.builder.create_block(f"case_{values[0]}")
        self._cases.append((list(values), case_block))
        
        return _CaseScope(self.builder, case_block, self._exit_block)
    
    def default_scope(self):
        """Get context manager for the default case."""
        if self._folded:
            # Check if default should execute
            all_values = []
            for vals, _ in self._cases:
                all_values.extend(vals)
            if self._constant_value not in all_values:
                return _DirectScope(self.builder)
            else:
                return _NoOpScope()
        
        self._default_block = self.builder.create_block("case_default")
        return _CaseScope(self.builder, self._default_block, self._exit_block)


# ============================================================================
# Internal Scope Classes
# ============================================================================

class _BranchScope:
    """Context manager for a branch (true or false)."""
    
    def __init__(self, builder: 'IRBuilder',
                 block: 'IRBasicBlock',
                 merge_block: 'IRBasicBlock',
                 is_false_branch: bool,
                 parent: 'IfStmt'):
        self.builder = builder
        self.block = block
        self.merge_block = merge_block
        self.is_false_branch = is_false_branch
        self.parent = parent
    
    def __enter__(self):
        self.builder.set_insert_point(self.block)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        
        # Branch to merge if not terminated
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.merge_block)
        
        # Set insert point to merge after false branch
        if self.is_false_branch:
            self.builder.set_insert_point(self.merge_block)
        
        return True


class _LoopScope:
    """Context manager for a loop body."""
    
    def __init__(self, builder: 'IRBuilder',
                 body_block: 'IRBasicBlock',
                 header_block: 'IRBasicBlock',
                 exit_block: 'IRBasicBlock'):
        self.builder = builder
        self.body_block = body_block
        self.header_block = header_block
        self.exit_block = exit_block
    
    def __enter__(self):
        self.builder.set_insert_point(self.body_block)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        
        # Loop back to header
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context and continue in exit block
        self.builder.pop_loop()
        self.builder.set_insert_point(self.exit_block)
        return True


class _ForLoopScope:
    """Context manager for a for loop body with iteration variable."""
    
    def __init__(self, builder: 'IRBuilder',
                 body_block: 'IRBasicBlock',
                 header_block: 'IRBasicBlock',
                 exit_block: 'IRBasicBlock',
                 loop_var: 'Value',
                 step: 'Value',
                 loop_var_name: str):
        self.builder = builder
        self.body_block = body_block
        self.header_block = header_block
        self.exit_block = exit_block
        self.loop_var = loop_var
        self.step = step
        self.loop_var_name = loop_var_name
    
    def __enter__(self):
        self.builder.set_insert_point(self.body_block)
        # Bind loop variable to current value
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
        
        # Loop back to header
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context and continue in exit block
        self.builder.pop_loop()
        self.builder.set_insert_point(self.exit_block)
        return True


class _UnrolledScope:
    """Context manager for unrolled loop iterations."""
    
    def __init__(self, builder: IRBuilder, 
                 iterations: list[int], 
                 loop_var_name: str):
        self.builder = builder
        self.iterations = iterations
        self.loop_var_name = loop_var_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    
    def __iter__(self):
        """Iterate over unrolled iterations."""
        from .types import int32
        for val in self.iterations:
            const_val = self.builder.constant(int32, val)
            self.builder.bind_local(self.loop_var_name, const_val)
            yield const_val


class _CaseScope:
    """Context manager for a switch case."""
    
    def __init__(self, builder: 'IRBuilder',
                 case_block: 'IRBasicBlock',
                 exit_block: 'IRBasicBlock'):
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


class _NoOpScope:
    """No-op scope for folded-away branches."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class _DirectScope:
    """Direct scope for branches that always execute (folded)."""
    
    def __init__(self, builder: IRBuilder):
        self.builder = builder
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
