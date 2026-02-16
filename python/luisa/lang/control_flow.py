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

from contextlib import contextmanager
from typing import Optional

from ..transform.builder import Builder
# Runtime imports
from ..transform.ir import BasicBlock, ConstantValue, Value
from ..transform.op import Op

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
    """

    def __init__(self, builder: 'Builder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self.true_block = self.builder.create_block("if_true")
        self.false_block = self.builder.create_block("if_false")

        # Check for constant folding at runtime
        self._constant_fold = isinstance(condition, ConstantValue)
        self._fold_true = condition.value if self._constant_fold else None

        if not self._constant_fold:
            from .types import Void
            self.builder._emit(Op.IF, Void, [self.condition, self.true_block, self.false_block])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def true_scope(self):
        """Get context manager for the true branch."""
        if self._constant_fold:
            return _DirectScope(self.builder) if self._fold_true else _NoOpScope()

        return self.builder.scope(self.true_block)

    def false_scope(self):
        """Get context manager for the false branch."""
        if self._constant_fold:
            return _NoOpScope() if self._fold_true else _DirectScope(self.builder)

        return self.builder.scope(self.false_block)


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
    """

    def __init__(self, builder: 'Builder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self.body_block = self.builder.create_block("while_body")

        # Check for constant folding - if False, skip entire loop
        self._constant_fold = isinstance(condition, ConstantValue) and condition.value == False

        if not self._constant_fold:
            from .types import Void

            # We emit a LOOP instruction that contains the body.
            # In Luisa IR, structured loops usually have the condition at the beginning of the body.
            # Our WhileStmt will handle this by injecting an IF BREAK at the start of body_block.
            self.builder._emit(Op.LOOP, Void, [self.body_block])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def body_scope(self):
        """Get context manager for the loop body."""
        if self._constant_fold:
            return _NoOpScope()

        @contextmanager
        def loop_body_wrapper():
            with self.builder.scope(self.body_block):
                # Inject condition check: if !condition: break
                from .types import Bool, Void
                not_cond = self.builder._emit(Op.LOGICAL_NOT, Bool, [self.condition])
                break_block = self.builder.create_block("while_break")
                with self.builder.scope(break_block):
                    self.builder.break_()
                self.builder._emit(Op.IF, Void,
                                   [not_cond, break_block, self.builder.create_block("while_continue")])

                yield True

        return loop_body_wrapper()


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

    def __init__(self, builder: 'Builder',
                 start: 'Value', stop: 'Value', step: 'Value',
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.body_block = self.builder.create_block("for_body")

        # Allocate and initialize loop variable
        self.loop_var_ptr = self.builder.alloca(self.start.type, name=self.loop_var_name)
        self.builder.store(self.loop_var_ptr, self.start)

        from .types import Void
        self.builder._emit(Op.LOOP, Void, [self.body_block])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def body_scope(self):
        """Get context manager for the loop body."""

        @contextmanager
        def loop_body_wrapper():
            with self.builder.scope(self.body_block):
                # 1. Load current value
                current_val = self.builder.load(self.loop_var_ptr)
                self.builder.bind_local(self.loop_var_name, current_val)

                # 2. Check condition: if i >= stop: break
                from .types import Bool, Void
                cond = self.builder.lt(current_val, self.stop)
                not_cond = self.builder._emit(Op.LOGICAL_NOT, Bool, [cond])

                break_block = self.builder.create_block("for_break")
                with self.builder.scope(break_block):
                    self.builder.break_()

                self.builder._emit(Op.IF, Void, [not_cond, break_block, self.builder.create_block("for_continue")])

                # 3. Yield to user body
                yield current_val

                # 4. Increment
                next_val = self.builder.add(current_val, self.step)
                self.builder.store(self.loop_var_ptr, next_val)

        return loop_body_wrapper()


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

    def __init__(self, builder: Builder,
                 start: int, stop: int, step: int,
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.iterations = list(range(start, stop, step))

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

    def __init__(self, builder: 'Builder', value: 'Value'):
        self.builder = builder
        self.value = value
        self._folded = isinstance(value, ConstantValue)
        self._constant_value = value.value if self._folded else None

        if not self._folded:
            from .types import Void
            self.cases: list[tuple[list[int], BasicBlock]] = []
            self.default_block: Optional[BasicBlock] = None
            self.inst = self.builder._emit(Op.SWITCH, Void, [self.value, self.cases, None])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._folded:
            self.inst.instruction.args[2] = self.default_block

    def case_scope(self, *values: int):
        """Get context manager for a case."""
        if self._folded:
            if self._constant_value in values:
                return _DirectScope(self.builder)
            else:
                return _NoOpScope()

        case_block = self.builder.create_block(f"case_{values[0]}")
        self.cases.append((list(values), case_block))
        return self.builder.scope(case_block)

    def default_scope(self):
        """Get context manager for the default case."""
        if self._folded:
            return _DirectScope(self.builder)

        self.default_block = self.builder.create_block("case_default")
        return self.builder.scope(self.default_block)


# ============================================================================
# Internal Scope Classes
# ============================================================================

class _NoOpScope:
    """No-op scope for folded-away branches."""

    def __enter__(self):
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class _DirectScope:
    """Direct scope for branches that always execute (folded)."""

    def __init__(self, builder: Builder):
        self.builder = builder

    def __enter__(self):
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class _UnrolledScope:
    """Scope for unrolled for loops."""

    def __init__(self, builder: Builder, iterations: list[int], loop_var_name: str):
        self.builder = builder
        self.iterations = iterations
        self.loop_var_name = loop_var_name

    def __iter__(self):
        for val in self.iterations:
            # Bind loop variable to constant for each iteration
            # We need to find the correct scalar type for the iteration value
            from .types import Int
            const_val = self.builder.constant(Int, val)
            self.builder.bind_local(self.loop_var_name, const_val)
            yield const_val
