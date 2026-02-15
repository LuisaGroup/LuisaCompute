"""
Math builtin functions for the LuisaCompute Python DSL v2.

These functions operate on DSL values and generate appropriate IR instructions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue
    from ..types import Type

from ..ir import Op
from ..builder import Builder, get_current_builder, set_current_builder


# ============================================================================
# Unary Math Functions
# ============================================================================

def sqrt(x: Value) -> InstructionValue:
    """Compute square root."""
    return get_current_builder()._emit(Op.SQRT, x.type, [x])


def abs(x: Value) -> InstructionValue:
    """Compute absolute value."""
    return get_current_builder()._emit(Op.ABS, x.type, [x])


def sin(x: Value) -> InstructionValue:
    """Compute sine."""
    return get_current_builder()._emit(Op.SIN, x.type, [x])


def cos(x: Value) -> InstructionValue:
    """Compute cosine."""
    return get_current_builder()._emit(Op.COS, x.type, [x])


def tan(x: Value) -> InstructionValue:
    """Compute tangent."""
    return get_current_builder()._emit(Op.TAN, x.type, [x])


def asin(x: Value) -> InstructionValue:
    """Compute arc sine."""
    return get_current_builder()._emit(Op.ASIN, x.type, [x])


def acos(x: Value) -> InstructionValue:
    """Compute arc cosine."""
    return get_current_builder()._emit(Op.ACOS, x.type, [x])


def atan(x: Value) -> InstructionValue:
    """Compute arc tangent."""
    return get_current_builder()._emit(Op.ATAN, x.type, [x])


def atan2(y: Value, x: Value) -> InstructionValue:
    """Compute arc tangent of y/x."""
    return get_current_builder()._emit(Op.ATAN2, y.type, [y, x])


def exp(x: Value) -> InstructionValue:
    """Compute exponential."""
    return get_current_builder()._emit(Op.EXP, x.type, [x])


def exp2(x: Value) -> InstructionValue:
    """Compute base-2 exponential."""
    return get_current_builder()._emit(Op.EXP2, x.type, [x])


def log(x: Value) -> InstructionValue:
    """Compute natural logarithm."""
    return get_current_builder()._emit(Op.LOG, x.type, [x])


def log2(x: Value) -> InstructionValue:
    """Compute base-2 logarithm."""
    return get_current_builder()._emit(Op.LOG2, x.type, [x])


def log10(x: Value) -> InstructionValue:
    """Compute base-10 logarithm."""
    return get_current_builder()._emit(Op.LOG10, x.type, [x])


def floor(x: Value) -> InstructionValue:
    """Compute floor."""
    return get_current_builder()._emit(Op.FLOOR, x.type, [x])


def ceil(x: Value) -> InstructionValue:
    """Compute ceiling."""
    return get_current_builder()._emit(Op.CEIL, x.type, [x])


def round(x: Value) -> InstructionValue:
    """Round to nearest integer."""
    return get_current_builder()._emit(Op.ROUND, x.type, [x])


def trunc(x: Value) -> InstructionValue:
    """Truncate to integer."""
    return get_current_builder()._emit(Op.TRUNC, x.type, [x])


def fract(x: Value) -> InstructionValue:
    """Compute fractional part."""
    return get_current_builder()._emit(Op.FRACT, x.type, [x])


def saturate(x: Value) -> InstructionValue:
    """Clamp to [0, 1]."""
    return get_current_builder()._emit(Op.SATURATE, x.type, [x])


def normalize(x: Value) -> InstructionValue:
    """Normalize vector."""
    return get_current_builder()._emit(Op.NORMALIZE, x.type, [x])


def length(x: Value) -> InstructionValue:
    """Compute vector length."""
    # Length returns scalar
    from ..types import Float
    return get_current_builder()._emit(Op.LENGTH, Float, [x])


def length_squared(x: Value) -> InstructionValue:
    """Compute squared vector length."""
    from ..types import Float
    return get_current_builder()._emit(Op.LENGTH_SQUARED, Float, [x])


# ============================================================================
# Binary Math Functions
# ============================================================================

def min(a: Value, b: Value) -> InstructionValue:
    """Compute minimum."""
    return get_current_builder()._emit(Op.MIN, a.type, [a, b])


def max(a: Value, b: Value) -> InstructionValue:
    """Compute maximum."""
    return get_current_builder()._emit(Op.MAX, a.type, [a, b])


def clamp(x: Value, min_val: Value, max_val: Value) -> InstructionValue:
    """Clamp to range [min_val, max_val]."""
    return get_current_builder()._emit(Op.CLAMP, x.type, [x, min_val, max_val])


def lerp(a: Value, b: Value, t: Value) -> InstructionValue:
    """Linear interpolation: a + (b - a) * t"""
    return get_current_builder()._emit(Op.LERP, a.type, [a, b, t])


def step(edge: Value, x: Value) -> InstructionValue:
    """Step function: (x >= edge) ? 1 : 0"""
    return get_current_builder()._emit(Op.STEP, x.type, [edge, x])


def smoothstep(edge0: Value, edge1: Value, x: Value) -> InstructionValue:
    """Smooth Hermite interpolation."""
    return get_current_builder()._emit(Op.SMOOTHSTEP, x.type, [edge0, edge1, x])


def pow(base: Value, exp: Value) -> InstructionValue:
    """Compute power."""
    return get_current_builder()._emit(Op.POW, base.type, [base, exp])


def dot(a: Value, b: Value) -> InstructionValue:
    """Compute dot product."""
    from ..types import Float
    return get_current_builder()._emit(Op.DOT, Float, [a, b])


def cross(a: Value, b: Value) -> InstructionValue:
    """Compute cross product (3D vectors only)."""
    return get_current_builder()._emit(Op.CROSS, a.type, [a, b])


def distance(a: Value, b: Value) -> InstructionValue:
    """Compute distance between two points."""
    from ..types import Float
    return get_current_builder()._emit(Op.DISTANCE, Float, [a, b])


def reflect(i: Value, n: Value) -> InstructionValue:
    """Reflect vector."""
    return get_current_builder()._emit(Op.REFLECT, i.type, [i, n])


def refract(i: Value, n: Value, eta: Value) -> InstructionValue:
    """Refract vector."""
    return get_current_builder()._emit(Op.REFRACT, i.type, [i, n, eta])


def faceforward(n: Value, i: Value, ng: Value) -> InstructionValue:
    """Face forward."""
    return get_current_builder()._emit(Op.FACEFORWARD, n.type, [n, i, ng])


# ============================================================================
# Matrix Functions
# ============================================================================

def transpose(m: Value) -> InstructionValue:
    """Transpose matrix."""
    return get_current_builder()._emit(Op.MATRIX_TRANSPOSE, m.type, [m])


def inverse(m: Value) -> InstructionValue:
    """Compute matrix inverse."""
    return get_current_builder()._emit(Op.MATRIX_INVERSE, m.type, [m])


def determinant(m: Value) -> InstructionValue:
    """Compute matrix determinant."""
    from ..types import Float
    return get_current_builder()._emit(Op.MATRIX_DETERMINANT, Float, [m])
