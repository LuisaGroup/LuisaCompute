"""
Atomic builtin functions for the LuisaCompute Python DSL v2.

Atomic operations on buffers and shared memory.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue

from ..ir import IROp
from .math import _get_builder


# ============================================================================
# Atomic Operations
# ============================================================================

def atomic_exchange(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically exchange value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: New value to store
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_EXCHANGE, elem_type, [buffer, index, value])


def atomic_compare_exchange(buffer: Value, index: Value, expected: Value, desired: Value) -> InstructionValue:
    """
    Atomically compare and exchange.
    
    Stores 'desired' if current value equals 'expected'.
    Returns the old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        expected: Expected current value
        desired: New value to store if comparison succeeds
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_CMP_EXCH, elem_type, 
                                 [buffer, index, expected, desired])


def atomic_add(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically add value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to add
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_ADD, elem_type, [buffer, index, value])


def atomic_sub(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically subtract value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to subtract
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_SUB, elem_type, [buffer, index, value])


def atomic_and(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically bitwise AND value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to AND
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_AND, elem_type, [buffer, index, value])


def atomic_or(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically bitwise OR value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to OR
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_OR, elem_type, [buffer, index, value])


def atomic_xor(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically bitwise XOR value at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to XOR
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_XOR, elem_type, [buffer, index, value])


def atomic_min(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically compute minimum at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to compare with
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_MIN, elem_type, [buffer, index, value])


def atomic_max(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Atomically compute maximum at index and return old value.
    
    Args:
        buffer: Buffer handle
        index: Element index
        value: Value to compare with
    
    Returns:
        The old value at the index
    """
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.ATOMIC_MAX, elem_type, [buffer, index, value])
