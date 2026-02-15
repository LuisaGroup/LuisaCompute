"""
Builtin functions for the LuisaCompute Python DSL v2.
"""

# Math functions
from .math import (
    # Unary
    sqrt, abs, sin, cos, tan, asin, acos, atan, atan2,
    exp, exp2, log, log2, log10,
    floor, ceil, round, trunc, fract, saturate,
    normalize, length, length_squared,
    # Binary
    min, max, clamp, lerp, step, smoothstep, pow,
    dot, cross, distance, reflect, refract, faceforward,
    # Matrix
    transpose, inverse, determinant,
)

# Core builtins (special registers, sync, etc.)
from .builtin import (
    # Special registers
    dispatch_id, thread_id, block_id, dispatch_size,
    kernel_id, object_id,
    # Synchronization
    sync_block,
    # Type casting
    cast, bitcast,
    # Print
    device_print,
    # Assertions
    assume, device_assert, unreachable,
    # Profiling
    clock,
)

# Memory operations
from .memory import (
    # Buffer
    buffer_read, buffer_write, buffer_size, buffer_device_address,
    # Texture2D
    texture2d_read, texture2d_write, texture2d_sample, texture2d_sample_level, texture2d_size,
    # Texture3D
    texture3d_read, texture3d_write, texture3d_sample, texture3d_size,
    # Device address
    device_address_load, device_address_store,
)

# Atomic operations
from .atomic import (
    atomic_exchange, atomic_compare_exchange,
    atomic_add, atomic_sub,
    atomic_and, atomic_or, atomic_xor,
    atomic_min, atomic_max,
)

# Warp operations
from .warp import (
    # Query
    warp_is_first_active_lane, warp_first_active_lane, warp_active_count_bits,
    # Reduction
    warp_sum, warp_product, warp_min, warp_max,
    warp_all, warp_any, warp_all_equal,
    # Prefix
    warp_prefix_sum, warp_prefix_product, warp_prefix_count_bits,
    # Broadcast
    warp_read_lane, warp_read_first_lane,
    # Bitwise
    warp_bit_and, warp_bit_or, warp_bit_xor, warp_bit_mask,
)

# Ray tracing
from .rtx import (
    # Types
    Ray, TriangleHit, ProceduralHit, CommittedHit,
    # Tracing
    trace_closest, trace_any, ray_query_all, ray_query_any,
    # Ray query operations
    ray_query_world_space_ray, ray_query_proceed,
    ray_query_committed_hit, ray_query_candidate_triangle_hit, ray_query_candidate_procedural_hit,
    ray_query_commit_triangle, ray_query_commit_procedural, ray_query_terminate,
    # Accel operations
    accel_instance_transform, accel_instance_user_id, accel_instance_visibility_mask,
    make_ray,
)

__all__ = [
    # Math - Unary
    'sqrt', 'abs', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'exp', 'exp2', 'log', 'log2', 'log10',
    'floor', 'ceil', 'round', 'trunc', 'fract', 'saturate',
    'normalize', 'length', 'length_squared',
    # Math - Binary
    'min', 'max', 'clamp', 'lerp', 'step', 'smoothstep', 'pow',
    'dot', 'cross', 'distance', 'reflect', 'refract', 'faceforward',
    # Math - Matrix
    'transpose', 'inverse', 'determinant',

    # Core - Special registers
    'dispatch_id', 'thread_id', 'block_id', 'dispatch_size',
    'kernel_id', 'object_id',
    # Core - Synchronization
    'sync_block',
    # Core - Type casting
    'cast', 'bitcast',
    # Core - Print
    'device_print',
    # Core - Assertions
    'assume', 'device_assert', 'unreachable',
    # Core - Profiling
    'clock',

    # Memory - Buffer
    'buffer_read', 'buffer_write', 'buffer_size', 'buffer_device_address',
    # Memory - Texture2D
    'texture2d_read', 'texture2d_write', 'texture2d_sample', 'texture2d_sample_level', 'texture2d_size',
    # Memory - Texture3D
    'texture3d_read', 'texture3d_write', 'texture3d_sample', 'texture3d_size',
    # Memory - Device address
    'device_address_load', 'device_address_store',

    # Atomic
    'atomic_exchange', 'atomic_compare_exchange',
    'atomic_add', 'atomic_sub',
    'atomic_and', 'atomic_or', 'atomic_xor',
    'atomic_min', 'atomic_max',

    # Warp - Query
    'warp_is_first_active_lane', 'warp_first_active_lane', 'warp_active_count_bits',
    # Warp - Reduction
    'warp_sum', 'warp_product', 'warp_min', 'warp_max',
    'warp_all', 'warp_any', 'warp_all_equal',
    # Warp - Prefix
    'warp_prefix_sum', 'warp_prefix_product', 'warp_prefix_count_bits',
    # Warp - Broadcast
    'warp_read_lane', 'warp_read_first_lane',
    # Warp - Bitwise
    'warp_bit_and', 'warp_bit_or', 'warp_bit_xor', 'warp_bit_mask',

    # Ray Tracing - Types
    'Ray', 'TriangleHit', 'ProceduralHit', 'CommittedHit',
    # Ray Tracing - Queries
    'trace_closest', 'trace_any', 'ray_query_all', 'ray_query_any',
    # Ray Tracing - Ray query operations
    'ray_query_world_space_ray', 'ray_query_proceed',
    'ray_query_committed_hit', 'ray_query_candidate_triangle_hit', 'ray_query_candidate_procedural_hit',
    'ray_query_commit_triangle', 'ray_query_commit_procedural', 'ray_query_terminate',
    # Ray Tracing - Accel operations
    'accel_instance_transform', 'accel_instance_user_id', 'accel_instance_visibility_mask',
    'make_ray',
]
