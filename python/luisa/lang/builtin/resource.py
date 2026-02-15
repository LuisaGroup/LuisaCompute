"""
Resource builtin functions for the LuisaCompute Python DSL v2.

Buffer, texture, and other memory operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue
    from ..type import Type

from ..ir import Op
from ..type import UInt, UInt2, UInt3, ULong, Void, Float4
from ..builder import get_current_builder


# ============================================================================
# Buffer Operations
# ============================================================================

def buffer_read(buffer: Value, index: Value) -> InstructionValue:
    """
    Read from a buffer.
    
    Args:
        buffer: Buffer handle
        index: Element index (UInt)
    
    Returns:
        The element value at the specified index
    """
    # Get element type from buffer type
    elem_type = buffer.type.element
    return get_current_builder()._emit(Op.BUFFER_READ, elem_type, [buffer, index])


def buffer_write(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Write to a buffer.
    
    Args:
        buffer: Buffer handle
        index: Element index (UInt)
        value: Value to write
    """
    return get_current_builder()._emit(Op.BUFFER_WRITE, Void, [buffer, index, value])


def buffer_size(buffer: Value) -> InstructionValue:
    """
    Get the size (number of elements) of a buffer.
    
    Args:
        buffer: Buffer handle
    
    Returns:
        Number of elements (UInt)
    """
    return get_current_builder()._emit(Op.BUFFER_SIZE, UInt, [buffer])


def buffer_device_address(buffer: Value) -> InstructionValue:
    """
    Get the device address of a buffer.
    
    Args:
        buffer: Buffer handle
    
    Returns:
        64-bit device address
    """
    return get_current_builder()._emit(Op.BUFFER_DEVICE_ADDRESS, ULong, [buffer])


# ============================================================================
# Texture2D Operations
# ============================================================================

def texture2d_read(texture: Value, coord: Value) -> InstructionValue:
    """
    Read from a 2D texture.
    
    Args:
        texture: Texture2D handle
        coord: Integer coordinates (UInt2)
    
    Returns:
        The texture value at the specified coordinates
    """
    return get_current_builder()._emit(Op.TEXTURE2D_READ, Float4, [texture, coord])


def texture2d_write(texture: Value, coord: Value, value: Value) -> InstructionValue:
    """
    Write to a 2D texture.
    
    Args:
        texture: Texture2D handle
        coord: Integer coordinates (UInt2)
        value: Value to write
    """
    return get_current_builder()._emit(Op.TEXTURE2D_WRITE, Void, [texture, coord, value])


def texture2d_sample(texture: Value, uv: Value) -> InstructionValue:
    """
    Sample from a 2D texture with filtering.
    
    Args:
        texture: Texture2D handle
        uv: Floating-point UV coordinates (Float2)
    
    Returns:
        The sampled value (Float4)
    """
    return get_current_builder()._emit(Op.TEXTURE2D_SAMPLE, Float4, [texture, uv])


def texture2d_sample_level(texture: Value, uv: Value, level: Value) -> InstructionValue:
    """
    Sample from a specific mipmap level of a 2D texture.
    
    Args:
        texture: Texture2D handle
        uv: Floating-point UV coordinates (Float2)
        level: Mipmap level (float)
    
    Returns:
        The sampled value (Float4)
    """
    return get_current_builder()._emit(Op.TEXTURE2D_SAMPLE_LEVEL, Float4, [texture, uv, level])


def texture2d_size(texture: Value) -> InstructionValue:
    """
    Get the size of a 2D texture.
    
    Args:
        texture: Texture2D handle
    
    Returns:
        Texture dimensions (UInt2)
    """
    return get_current_builder()._emit(Op.TEXTURE2D_SIZE, UInt2, [texture])


# ============================================================================
# Texture3D Operations
# ============================================================================

def texture3d_read(texture: Value, coord: Value) -> InstructionValue:
    """
    Read from a 3D texture.
    
    Args:
        texture: Texture3D handle
        coord: Integer coordinates (UInt3)
    
    Returns:
        The texture value at the specified coordinates
    """
    return get_current_builder()._emit(Op.TEXTURE3D_READ, Float4, [texture, coord])


def texture3d_write(texture: Value, coord: Value, value: Value) -> InstructionValue:
    """
    Write to a 3D texture.
    
    Args:
        texture: Texture3D handle
        coord: Integer coordinates (UInt3)
        value: Value to write
    """
    return get_current_builder()._emit(Op.TEXTURE3D_WRITE, Void, [texture, coord, value])


def texture3d_sample(texture: Value, uvw: Value) -> InstructionValue:
    """
    Sample from a 3D texture.
    
    Args:
        texture: Texture3D handle
        uvw: Floating-point UVW coordinates (Float3)
    
    Returns:
        The sampled value (Float4)
    """
    return get_current_builder()._emit(Op.TEXTURE3D_SAMPLE, Float4, [texture, uvw])


def texture3d_size(texture: Value) -> InstructionValue:
    """
    Get the size of a 3D texture.
    
    Args:
        texture: Texture3D handle
    
    Returns:
        Texture dimensions (UInt3)
    """
    return get_current_builder()._emit(Op.TEXTURE3D_SIZE, UInt3, [texture])


# ============================================================================
# Device Address Operations
# ============================================================================

def device_address_load(address: Value, elem_type: Type) -> InstructionValue:
    """
    Load from a device address.
    
    Args:
        address: Device address (ULong)
        elem_type: Type of element to load
    
    Returns:
        The loaded value
    """
    return get_current_builder()._emit(Op.DEVICE_ADDRESS_READ, elem_type, [address])


def device_address_store(address: Value, value: Value) -> InstructionValue:
    """
    Store to a device address.
    
    Args:
        address: Device address (ULong)
        value: Value to store
    """
    return get_current_builder()._emit(Op.DEVICE_ADDRESS_WRITE, Void, [address, value])
