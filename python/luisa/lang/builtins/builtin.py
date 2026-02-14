"""
Core builtin functions for the LuisaCompute Python DSL v2.

These include special registers, synchronization, and other non-math operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ast import Value, InstructionValue
    from ..types import Type

from ..ast import IROp
from ..types import uint3, uint, bool_, float3
from .math import _get_builder


# ============================================================================
# Special Registers (Kernel Execution Context)
# ============================================================================

def dispatch_id() -> InstructionValue:
    """Get the global dispatch ID (3D)."""
    return _get_builder()._emit(IROp.DISPATCH_ID, uint3, [])


def dispatch_idx() -> InstructionValue:
    """Get the linearized 1D dispatch index."""
    # This would typically be computed from dispatch_id() in a real implementation
    return _get_builder()._emit(IROp.DISPATCH_ID, uint, [])


def thread_id() -> InstructionValue:
    """Get the local thread ID within a block (3D)."""
    return _get_builder()._emit(IROp.THREAD_ID, uint3, [])


def block_id() -> InstructionValue:
    """Get the block ID (3D)."""
    return _get_builder()._emit(IROp.BLOCK_ID, uint3, [])


def dispatch_size() -> InstructionValue:
    """Get the total dispatch size (3D)."""
    return _get_builder()._emit(IROp.DISPATCH_SIZE, uint3, [])


def kernel_id() -> InstructionValue:
    """Get the kernel ID."""
    return _get_builder()._emit(IROp.KERNEL_ID, uint, [])


def object_id() -> InstructionValue:
    """Get the object ID (for rasterization)."""
    return _get_builder()._emit(IROp.OBJECT_ID, uint, [])


# ============================================================================
# Synchronization
# ============================================================================

def sync_block() -> InstructionValue:
    """Synchronize all threads in a block."""
    from ..types import Void
    return _get_builder()._emit(IROp.SYNC_BLOCK, Void(), [])


# ============================================================================
# Type Casting
# ============================================================================

def cast(value: Value, target_type: Type) -> InstructionValue:
    """Static cast to target type."""
    return _get_builder()._emit(IROp.CAST, target_type, [value])


def bitcast(value: Value, target_type: Type) -> InstructionValue:
    """Bitwise cast to target type (preserves bit pattern)."""
    return _get_builder()._emit(IROp.BITCAST, target_type, [value])


# ============================================================================
# Print
# ============================================================================

def print_msg(fmt: str, *values: Value) -> InstructionValue:
    """
    Print a message from the kernel.
    
    Args:
        fmt: Format string with {} placeholders
        *values: Values to print
    """
    from ..types import Void
    args = [fmt] + list(values)
    return _get_builder()._emit(IROp.PRINT, Void(), args)


# ============================================================================
# Assumptions and Assertions
# ============================================================================

def assume(condition: Value, message: str = "") -> InstructionValue:
    """Provide a compiler assumption for optimization."""
    from ..types import Void
    return _get_builder()._emit(IROp.ASSUME, Void(), [condition, message])


def assert_(condition: Value, message: str = "") -> InstructionValue:
    """Runtime assertion (may be disabled in release)."""
    from ..types import Void
    return _get_builder()._emit(IROp.ASSERT, Void(), [condition, message])


# ============================================================================
# Clock (for profiling)
# ============================================================================

def clock() -> InstructionValue:
    """Get current clock value (for timing)."""
    from ..types import uint64
    return _get_builder()._emit(IROp.CLOCK, uint64, [])
