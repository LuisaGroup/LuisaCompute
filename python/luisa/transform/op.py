"""
IR Operations Enum for the LuisaCompute Python DSL v2.

This module defines the Op enum used across the DSL to avoid
circular dependencies between types and IR.
"""

from __future__ import annotations
from enum import Enum, auto


class Op(Enum):
    """IR operation types."""

    # Literals and constants
    CONST = auto()

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()

    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    SHL = auto()  # Shift left
    SHR = auto()  # Shift right

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Logical
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()

    # Math functions
    SQRT = auto()
    RSQRT = auto()  # Reciprocal sqrt
    POW = auto()
    EXP = auto()
    EXP2 = auto()
    EXP10 = auto()  # Base-10 exponential
    LOG = auto()
    LOG2 = auto()
    LOG10 = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()
    ASINH = auto()  # Inverse hyperbolic sine
    ACOSH = auto()  # Inverse hyperbolic cosine
    ATANH = auto()  # Inverse hyperbolic tangent
    ABS = auto()
    FLOOR = auto()
    CEIL = auto()
    ROUND = auto()
    TRUNC = auto()
    FRACT = auto()
    MIN = auto()
    MAX = auto()
    CLAMP = auto()
    LERP = auto()
    SATURATE = auto()
    STEP = auto()
    SMOOTHSTEP = auto()
    FMA = auto()  # Fused multiply-add: a*b + c
    COPYSIGN = auto()  # Copy sign from one value to another
    ISINF = auto()  # Check for infinity
    ISNAN = auto()  # Check for NaN
    
    # Integer bit operations
    CLZ = auto()  # Count leading zeros
    CTZ = auto()  # Count trailing zeros
    POPCOUNT = auto()  # Count set bits
    REVERSE = auto()  # Bit reversal
    
    # Vector operations
    DOT = auto()
    CROSS = auto()
    NORMALIZE = auto()
    LENGTH = auto()
    LENGTH_SQUARED = auto()
    DISTANCE = auto()
    REFLECT = auto()
    REFRACT = auto()
    FACEFORWARD = auto()

    # Matrix operations
    MATRIX_DETERMINANT = auto()
    MATRIX_TRANSPOSE = auto()
    MATRIX_INVERSE = auto()

    # Memory
    ALLOCA = auto()
    LOAD = auto()
    STORE = auto()
    GEP = auto()  # Get element pointer
    MEMBER_ACCESS = auto()  # Struct member access

    # Resources
    BUFFER_READ = auto()
    BUFFER_WRITE = auto()
    BUFFER_SIZE = auto()
    TEXTURE2D_READ = auto()
    TEXTURE2D_WRITE = auto()
    TEXTURE2D_SAMPLE = auto()
    TEXTURE2D_SAMPLE_LEVEL = auto()
    TEXTURE3D_READ = auto()
    TEXTURE3D_WRITE = auto()
    TEXTURE3D_SAMPLE = auto()

    # Control flow
    PHI = auto()  # Phi node for SSA
    RETURN = auto()

    # Structured Control Flow
    IF = auto()
    LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    SWITCH = auto()

    # Function calls
    CALL = auto()
    CALL_BUILTIN = auto()

    # Cast
    CAST = auto()  # Static cast
    BITCAST = auto()  # Bitwise cast

    # Special registers
    THREAD_ID = auto()
    BLOCK_ID = auto()
    DISPATCH_ID = auto()
    DISPATCH_SIZE = auto()
    KERNEL_ID = auto()
    OBJECT_ID = auto()

    # Ray tracing
    TRACE_CLOSEST = auto()
    TRACE_ANY = auto()
    RAY_QUERY_ALL = auto()
    RAY_QUERY_ANY = auto()

    # Atomic operations
    ATOMIC_EXCHANGE = auto()
    ATOMIC_ADD = auto()
    ATOMIC_SUB = auto()
    ATOMIC_AND = auto()
    ATOMIC_OR = auto()
    ATOMIC_XOR = auto()
    ATOMIC_MIN = auto()
    ATOMIC_MAX = auto()
    ATOMIC_CMP_EXCH = auto()

    # Warp operations
    WARP_IS_FIRST_ACTIVE_LANE = auto()
    WARP_FIRST_ACTIVE_LANE = auto()
    WARP_ACTIVE_COUNT_BITS = auto()
    WARP_SUM = auto()
    WARP_PRODUCT = auto()
    WARP_MIN = auto()
    WARP_MAX = auto()
    WARP_ALL = auto()
    WARP_ANY = auto()
    WARP_ACTIVE_ALL_EQUAL = auto()
    WARP_ACTIVE_BIT_AND = auto()
    WARP_ACTIVE_BIT_OR = auto()
    WARP_ACTIVE_BIT_XOR = auto()
    WARP_ACTIVE_BIT_MASK = auto()
    WARP_BROADCAST = auto()
    WARP_PREFIX_SUM = auto()
    WARP_PREFIX_PRODUCT = auto()
    WARP_PREFIX_COUNT_BITS = auto()
    WARP_READ_LANE = auto()
    WARP_READ_FIRST_ACTIVE_LANE = auto()

    # Synchronization
    SYNC_BLOCK = auto()

    # Print
    PRINT = auto()

    # Swizzle
    SWIZZLE = auto()

    # Additional resource operations
    TEXTURE2D_SIZE = auto()
    TEXTURE3D_SIZE = auto()
    BUFFER_DEVICE_ADDRESS = auto()
    DEVICE_ADDRESS_READ = auto()
    DEVICE_ADDRESS_WRITE = auto()

    # Additional ray tracing
    RAY_QUERY_WORLD_RAY = auto()
    RAY_QUERY_PROCEED = auto()
    RAY_QUERY_COMMITTED_HIT = auto()
    RAY_QUERY_CANDIDATE_TRIANGLE_HIT = auto()
    RAY_QUERY_CANDIDATE_PROCEDURAL_HIT = auto()
    RAY_QUERY_COMMIT_TRIANGLE = auto()
    RAY_QUERY_COMMIT_PROCEDURAL = auto()
    RAY_QUERY_TERMINATE = auto()
    ACCEL_INSTANCE_TRANSFORM = auto()
    ACCEL_INSTANCE_USER_ID = auto()
    ACCEL_INSTANCE_VISIBILITY_MASK = auto()

    # Additional operations
    ASSERT = auto()
    ASSUME = auto()
    UNREACHABLE = auto()
    CLOCK = auto()


# Utility functions for Op
def get_op_name(op: Op) -> str:
    """Get the name of an IR operation."""
    return op.name


def is_arithmetic_op(op: Op) -> bool:
    """Check if an operation is arithmetic."""
    return op in (
        Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.NEG
    )


def is_comparison_op(op: Op) -> bool:
    """Check if an operation is a comparison."""
    return op in (
        Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE
    )


def is_logical_op(op: Op) -> bool:
    """Check if an operation is logical."""
    return op in (
        Op.LOGICAL_AND, Op.LOGICAL_OR, Op.LOGICAL_NOT
    )


def is_terminator_op(op: Op) -> bool:
    """Check if an operation is a terminator."""
    return op in (Op.RETURN, Op.BREAK, Op.CONTINUE)


def is_memory_op(op: Op) -> bool:
    """Check if an operation is a memory operation."""
    return op in (
        Op.ALLOCA, Op.LOAD, Op.STORE, Op.GEP, Op.MEMBER_ACCESS
    )


def is_resource_op(op: Op) -> bool:
    """Check if an operation is a resource operation."""
    return op in (
        Op.BUFFER_READ, Op.BUFFER_WRITE, Op.BUFFER_SIZE,
        Op.TEXTURE2D_READ, Op.TEXTURE2D_WRITE,
        Op.TEXTURE2D_SAMPLE, Op.TEXTURE2D_SAMPLE_LEVEL,
        Op.TEXTURE3D_READ, Op.TEXTURE3D_WRITE, Op.TEXTURE3D_SAMPLE,
    )
