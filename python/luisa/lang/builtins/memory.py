"""
Memory builtin functions for the LuisaCompute Python DSL v2.

Buffer, texture, and other memory operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue
    from ..types import Type

from ..ir import IROp
from ..types import uint, uint2, uint3, uint64
from .math import _get_builder


# ============================================================================
# Buffer Operations
# ============================================================================

def buffer_read(buffer: Value, index: Value) -> InstructionValue:
    """
    Read from a buffer.
    
    Args:
        buffer: Buffer handle
        index: Element index (uint)
    
    Returns:
        The element value at the specified index
    """
    # Get element type from buffer type
    elem_type = buffer.type.element
    return _get_builder()._emit(IROp.BUFFER_READ, elem_type, [buffer, index])


def buffer_write(buffer: Value, index: Value, value: Value) -> InstructionValue:
    """
    Write to a buffer.
    
    Args:
        buffer: Buffer handle
        index: Element index (uint)
        value: Value to write
    """
    from ..types import Void
    return _get_builder()._emit(IROp.BUFFER_WRITE, Void(), [buffer, index, value])


def buffer_size(buffer: Value) -> InstructionValue:
    """
    Get the size (number of elements) of a buffer.
    
    Args:
        buffer: Buffer handle
    
    Returns:
        Number of elements (uint)
    """
    return _get_builder()._emit(IROp.BUFFER_SIZE, uint, [buffer])


def buffer_device_address(buffer: Value) -> InstructionValue:
    """
    Get the device address of a buffer.
    
    Args:
        buffer: Buffer handle
    
    Returns:
        64-bit device address
    """
    return _get_builder()._emit(IROp.BUFFER_DEVICE_ADDRESS, uint64, [buffer])


# ============================================================================
# Texture2D Operations
# ============================================================================

def texture2d_read(texture: Value, coord: Value) -> InstructionValue:
    """
    Read from a 2D texture.
    
    Args:
        texture: Texture2D handle
        coord: Integer coordinates (uint2)
    
    Returns:
        The texture value at the specified coordinates
    """
    from ..types import float4
    return _get_builder()._emit(IROp.TEXTURE2D_READ, float4, [texture, coord])


def texture2d_write(texture: Value, coord: Value, value: Value) -> InstructionValue:
    """
    Write to a 2D texture.
    
    Args:
        texture: Texture2D handle
        coord: Integer coordinates (uint2)
        value: Value to write
    """
    from ..types import Void
    return _get_builder()._emit(IROp.TEXTURE2D_WRITE, Void(), [texture, coord, value])


def texture2d_sample(texture: Value, uv: Value) -> InstructionValue:
    """
    Sample from a 2D texture with filtering.
    
    Args:
        texture: Texture2D handle
        uv: Floating-point UV coordinates (float2)
    
    Returns:
        The sampled value (float4)
    """
    from ..types import float4
    return _get_builder()._emit(IROp.TEXTURE2D_SAMPLE, float4, [texture, uv])


def texture2d_sample_level(texture: Value, uv: Value, level: Value) -> InstructionValue:
    """
    Sample from a specific mipmap level of a 2D texture.
    
    Args:
        texture: Texture2D handle
        uv: Floating-point UV coordinates (float2)
        level: Mipmap level (float)
    
    Returns:
        The sampled value (float4)
    """
    from ..types import float4
    return _get_builder()._emit(IROp.TEXTURE2D_SAMPLE_LEVEL, float4, [texture, uv, level])


def texture2d_size(texture: Value) -> InstructionValue:
    """
    Get the size of a 2D texture.
    
    Args:
        texture: Texture2D handle
    
    Returns:
        Texture dimensions (uint2)
    """
    return _get_builder()._emit(IROp.TEXTURE2D_SIZE, uint2, [texture])


# ============================================================================
# Texture3D Operations
# ============================================================================

def texture3d_read(texture: Value, coord: Value) -> InstructionValue:
    """
    Read from a 3D texture.
    
    Args:
        texture: Texture3D handle
        coord: Integer coordinates (uint3)
    
    Returns:
        The texture value at the specified coordinates
    """
    from ..types import float4
    return _get_builder()._emit(IROp.TEXTURE3D_READ, float4, [texture, coord])


def texture3d_write(texture: Value, coord: Value, value: Value) -> InstructionValue:
    """
    Write to a 3D texture.
    
    Args:
        texture: Texture3D handle
        coord: Integer coordinates (uint3)
        value: Value to write
    """
    from ..types import Void
    return _get_builder()._emit(IROp.TEXTURE3D_WRITE, Void(), [texture, coord, value])


def texture3d_sample(texture: Value, uvw: Value) -> InstructionValue:
    """
    Sample from a 3D texture.
    
    Args:
        texture: Texture3D handle
        uvw: Floating-point UVW coordinates (float3)
    
    Returns:
        The sampled value (float4)
    """
    from ..types import float4
    return _get_builder()._emit(IROp.TEXTURE3D_SAMPLE, float4, [texture, uvw])


def texture3d_size(texture: Value) -> InstructionValue:
    """
    Get the size of a 3D texture.
    
    Args:
        texture: Texture3D handle
    
    Returns:
        Texture dimensions (uint3)
    """
    return _get_builder()._emit(IROp.TEXTURE3D_SIZE, uint3, [texture])


# ============================================================================
# Device Address Operations
# ============================================================================

def device_address_load(address: Value, elem_type: Type) -> InstructionValue:
    """
    Load from a device address.
    
    Args:
        address: Device address (uint64)
        elem_type: Type of element to load
    
    Returns:
        The loaded value
    """
    return _get_builder()._emit(IROp.DEVICE_ADDRESS_READ, elem_type, [address])


def device_address_store(address: Value, value: Value) -> InstructionValue:
    """
    Store to a device address.
    
    Args:
        address: Device address (uint64)
        value: Value to store
    """
    from ..types import Void
    return _get_builder()._emit(IROp.DEVICE_ADDRESS_WRITE, Void(), [address, value])
