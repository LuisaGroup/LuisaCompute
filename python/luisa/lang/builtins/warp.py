"""
Warp builtin functions for the LuisaCompute Python DSL v2.

Warp-level primitives for GPU programming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...transform.ir import Value, InstructionValue

from ...transform.builder import get_current_builder
from ...transform.op import Op
from ..types import Bool, UInt

# ============================================================================
# Warp Query
# ============================================================================

def warp_is_first_active_lane() -> InstructionValue:
    """Check if this is the first active lane in the warp."""
    return get_current_builder()._emit(Op.WARP_IS_FIRST_ACTIVE_LANE, Bool, [])


def warp_first_active_lane() -> InstructionValue:
    """Get the index of the first active lane in the warp."""
    return get_current_builder()._emit(Op.WARP_FIRST_ACTIVE_LANE, UInt, [])


def warp_active_count_bits(value: Value) -> InstructionValue:
    """Count the number of active (True) lanes in the warp."""
    return get_current_builder()._emit(Op.WARP_ACTIVE_COUNT_BITS, UInt, [value])


# ============================================================================
# Warp Reduction
# ============================================================================

def warp_sum(value: Value) -> InstructionValue:
    """Sum values across all active lanes in the warp."""
    return get_current_builder()._emit(Op.WARP_SUM, value.type, [value])


def warp_product(value: Value) -> InstructionValue:
    """Multiply values across all active lanes in the warp."""
    return get_current_builder()._emit(Op.WARP_PRODUCT, value.type, [value])


def warp_min(value: Value) -> InstructionValue:
    """Find minimum value across all active lanes in the warp."""
    return get_current_builder()._emit(Op.WARP_MIN, value.type, [value])


def warp_max(value: Value) -> InstructionValue:
    """Find maximum value across all active lanes in the warp."""
    return get_current_builder()._emit(Op.WARP_MAX, value.type, [value])


def warp_all(value: Value) -> InstructionValue:
    """Check if all active lanes have True."""
    return get_current_builder()._emit(Op.WARP_ALL, Bool, [value])


def warp_any(value: Value) -> InstructionValue:
    """Check if any active lane has True."""
    return get_current_builder()._emit(Op.WARP_ANY, Bool, [value])


def warp_all_equal(value: Value) -> InstructionValue:
    """Check if all active lanes have the same value."""
    return get_current_builder()._emit(Op.WARP_ACTIVE_ALL_EQUAL, Bool, [value])


# ============================================================================
# Warp Prefix Operations
# ============================================================================

def warp_prefix_sum(value: Value) -> InstructionValue:
    """
    Compute prefix sum (exclusive scan) across active lanes.

    Returns the sum of values from lanes with lower indices.
    """
    return get_current_builder()._emit(Op.WARP_PREFIX_SUM, value.type, [value])


def warp_prefix_product(value: Value) -> InstructionValue:
    """
    Compute prefix product (exclusive scan) across active lanes.

    Returns the product of values from lanes with lower indices.
    """
    return get_current_builder()._emit(Op.WARP_PREFIX_PRODUCT, value.type, [value])


def warp_prefix_count_bits(value: Value) -> InstructionValue:
    """
    Count the number of True values in active lanes with lower indices.
    """
    return get_current_builder()._emit(Op.WARP_PREFIX_COUNT_BITS, UInt, [value])


# ============================================================================
# Warp Broadcast
# ============================================================================

def warp_read_lane(value: Value, lane: Value) -> InstructionValue:
    """
    Read value from a specific lane.

    Args:
        value: The value to broadcast
        lane: The source lane index

    Returns:
        The value from the specified lane
    """
    return get_current_builder()._emit(Op.WARP_READ_LANE, value.type, [value, lane])


def warp_read_first_lane(value: Value) -> InstructionValue:
    """
    Read value from the first active lane.

    This is a broadcast operation.
    """
    return get_current_builder()._emit(Op.WARP_READ_FIRST_ACTIVE_LANE, value.type, [value])


# ============================================================================
# Warp Bitwise Operations
# ============================================================================

def warp_bit_and(value: Value) -> InstructionValue:
    """Bitwise AND across all active lanes."""
    return get_current_builder()._emit(Op.WARP_ACTIVE_BIT_AND, value.type, [value])


def warp_bit_or(value: Value) -> InstructionValue:
    """Bitwise OR across all active lanes."""
    return get_current_builder()._emit(Op.WARP_ACTIVE_BIT_OR, value.type, [value])


def warp_bit_xor(value: Value) -> InstructionValue:
    """Bitwise XOR across all active lanes."""
    return get_current_builder()._emit(Op.WARP_ACTIVE_BIT_XOR, value.type, [value])


def warp_bit_mask(value: Value) -> InstructionValue:
    """
    Get a bitmask of active lanes where value is True.

    Returns a 128-bit mask (UInt4).
    """
    from ..types import UInt4
    return get_current_builder()._emit(Op.WARP_ACTIVE_BIT_MASK, UInt4, [value])
