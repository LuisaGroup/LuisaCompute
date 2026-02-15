"""
Core builtin functions for the LuisaCompute Python DSL v2.

These include special registers, synchronization, and other non-math operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue
    from ..types import Type

from ..ir import Op
from ..types import UInt3, UInt, Bool, Float3
from ..builder import get_current_builder


# ============================================================================
# Special Registers (Kernel Execution Context)
# ============================================================================

def dispatch_id() -> InstructionValue:
    """Get the global dispatch ID (3D)."""
    return get_current_builder()._emit(Op.DISPATCH_ID, UInt3, [])


def thread_id() -> InstructionValue:
    """Get the local thread ID within a block (3D)."""
    return get_current_builder()._emit(Op.THREAD_ID, UInt3, [])


def block_id() -> InstructionValue:
    """Get the block ID (3D)."""
    return get_current_builder()._emit(Op.BLOCK_ID, UInt3, [])


def dispatch_size() -> InstructionValue:
    """Get the total dispatch size (3D)."""
    return get_current_builder()._emit(Op.DISPATCH_SIZE, UInt3, [])


def kernel_id() -> InstructionValue:
    """Get the kernel ID."""
    return get_current_builder()._emit(Op.KERNEL_ID, UInt, [])


def object_id() -> InstructionValue:
    """Get the object ID (for rasterization)."""
    return get_current_builder()._emit(Op.OBJECT_ID, UInt, [])


# ============================================================================
# Synchronization
# ============================================================================

def sync_block() -> InstructionValue:
    """Synchronize all threads in a block."""
    from ..types import Void
    return get_current_builder()._emit(Op.SYNC_BLOCK, Void(), [])


# ============================================================================
# Type Casting
# ============================================================================

def cast(value: Value, target_type: Type) -> InstructionValue:
    """Static cast to target type."""
    return get_current_builder()._emit(Op.CAST, target_type, [value])


def bitcast(value: Value, target_type: Type) -> InstructionValue:
    """Bitwise cast to target type (preserves bit pattern)."""
    return get_current_builder()._emit(Op.BITCAST, target_type, [value])


# ============================================================================
# Print
# ============================================================================

def device_print(fmt: str, *values: Value) -> InstructionValue:
    """
    Print a message from the kernel.
    
    Args:
        fmt: Format string with {} placeholders
        *values: Values to print
    """
    from ..types import Void
    args = [fmt] + list(values)
    return get_current_builder()._emit(Op.PRINT, Void(), args)


# ============================================================================
# Assumptions and Assertions
# ============================================================================

def assume(condition: Value, message: str = "") -> InstructionValue:
    """Provide a compiler assumption for optimization."""
    from ..types import Void
    return get_current_builder()._emit(Op.ASSUME, Void(), [condition, message])


def device_assert(condition: Value, message: str = "") -> InstructionValue:
    """Runtime assertion (may be disabled in release)."""
    from ..types import Void
    return get_current_builder()._emit(Op.ASSERT, Void(), [condition, message])


def unreachable(message: str = "") -> InstructionValue:
    """Mark a code path as unreachable."""
    from ..types import Void
    return get_current_builder()._emit(Op.UNREACHABLE, Void(), [message])


# ============================================================================
# Clock (for profiling)
# ============================================================================

def clock() -> InstructionValue:
    """Get current clock value (for timing)."""
    from ..types import ULong
    return get_current_builder()._emit(Op.CLOCK, ULong, [])
