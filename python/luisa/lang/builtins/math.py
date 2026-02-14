"""
Math builtin functions for the LuisaCompute Python DSL v2.

These functions operate on DSL values and generate appropriate IR instructions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import Value, InstructionValue
    from ..types import Type

from ..ir import IROp
from ..builder import IRBuilder

# Global builder reference (set during execution)
_current_builder: IRBuilder | None = None


def _get_builder() -> IRBuilder:
    """Get the current builder."""
    if _current_builder is None:
        raise RuntimeError("No active builder context")
    return _current_builder


def set_builder(builder: IRBuilder | None) -> None:
    """Set the current builder (called by executor)."""
    global _current_builder
    _current_builder = builder


# ============================================================================
# Unary Math Functions
# ============================================================================

def sqrt(x: Value) -> InstructionValue:
    """Compute square root."""
    return _get_builder()._emit(IROp.SQRT, x.type, [x])


def abs(x: Value) -> InstructionValue:
    """Compute absolute value."""
    return _get_builder()._emit(IROp.ABS, x.type, [x])


def sin(x: Value) -> InstructionValue:
    """Compute sine."""
    return _get_builder()._emit(IROp.SIN, x.type, [x])


def cos(x: Value) -> InstructionValue:
    """Compute cosine."""
    return _get_builder()._emit(IROp.COS, x.type, [x])


def tan(x: Value) -> InstructionValue:
    """Compute tangent."""
    return _get_builder()._emit(IROp.TAN, x.type, [x])


def asin(x: Value) -> InstructionValue:
    """Compute arc sine."""
    return _get_builder()._emit(IROp.ASIN, x.type, [x])


def acos(x: Value) -> InstructionValue:
    """Compute arc cosine."""
    return _get_builder()._emit(IROp.ACOS, x.type, [x])


def atan(x: Value) -> InstructionValue:
    """Compute arc tangent."""
    return _get_builder()._emit(IROp.ATAN, x.type, [x])


def atan2(y: Value, x: Value) -> InstructionValue:
    """Compute arc tangent of y/x."""
    return _get_builder()._emit(IROp.ATAN2, y.type, [y, x])


def exp(x: Value) -> InstructionValue:
    """Compute exponential."""
    return _get_builder()._emit(IROp.EXP, x.type, [x])


def exp2(x: Value) -> InstructionValue:
    """Compute base-2 exponential."""
    return _get_builder()._emit(IROp.EXP2, x.type, [x])


def log(x: Value) -> InstructionValue:
    """Compute natural logarithm."""
    return _get_builder()._emit(IROp.LOG, x.type, [x])


def log2(x: Value) -> InstructionValue:
    """Compute base-2 logarithm."""
    return _get_builder()._emit(IROp.LOG2, x.type, [x])


def log10(x: Value) -> InstructionValue:
    """Compute base-10 logarithm."""
    return _get_builder()._emit(IROp.LOG10, x.type, [x])


def floor(x: Value) -> InstructionValue:
    """Compute floor."""
    return _get_builder()._emit(IROp.FLOOR, x.type, [x])


def ceil(x: Value) -> InstructionValue:
    """Compute ceiling."""
    return _get_builder()._emit(IROp.CEIL, x.type, [x])


def round(x: Value) -> InstructionValue:
    """Round to nearest integer."""
    return _get_builder()._emit(IROp.ROUND, x.type, [x])


def trunc(x: Value) -> InstructionValue:
    """Truncate to integer."""
    return _get_builder()._emit(IROp.TRUNC, x.type, [x])


def fract(x: Value) -> InstructionValue:
    """Compute fractional part."""
    return _get_builder()._emit(IROp.FRACT, x.type, [x])


def saturate(x: Value) -> InstructionValue:
    """Clamp to [0, 1]."""
    return _get_builder()._emit(IROp.SATURATE, x.type, [x])


def normalize(x: Value) -> InstructionValue:
    """Normalize vector."""
    return _get_builder()._emit(IROp.NORMALIZE, x.type, [x])


def length(x: Value) -> InstructionValue:
    """Compute vector length."""
    # Length returns scalar
    from ..types import Scalar, ScalarType
    return _get_builder()._emit(IROp.LENGTH, Scalar(ScalarType.FLOAT32), [x])


def length_squared(x: Value) -> InstructionValue:
    """Compute squared vector length."""
    from ..types import Scalar, ScalarType
    return _get_builder()._emit(IROp.LENGTH_SQUARED, Scalar(ScalarType.FLOAT32), [x])


# ============================================================================
# Binary Math Functions
# ============================================================================

def min(a: Value, b: Value) -> InstructionValue:
    """Compute minimum."""
    return _get_builder()._emit(IROp.MIN, a.type, [a, b])


def max(a: Value, b: Value) -> InstructionValue:
    """Compute maximum."""
    return _get_builder()._emit(IROp.MAX, a.type, [a, b])


def clamp(x: Value, min_val: Value, max_val: Value) -> InstructionValue:
    """Clamp to range [min_val, max_val]."""
    return _get_builder()._emit(IROp.CLAMP, x.type, [x, min_val, max_val])


def lerp(a: Value, b: Value, t: Value) -> InstructionValue:
    """Linear interpolation: a + (b - a) * t"""
    return _get_builder()._emit(IROp.LERP, a.type, [a, b, t])


def step(edge: Value, x: Value) -> InstructionValue:
    """Step function: (x >= edge) ? 1 : 0"""
    return _get_builder()._emit(IROp.STEP, x.type, [edge, x])


def smoothstep(edge0: Value, edge1: Value, x: Value) -> InstructionValue:
    """Smooth Hermite interpolation."""
    return _get_builder()._emit(IROp.SMOOTHSTEP, x.type, [edge0, edge1, x])


def pow(base: Value, exp: Value) -> InstructionValue:
    """Compute power."""
    return _get_builder()._emit(IROp.POW, base.type, [base, exp])


def dot(a: Value, b: Value) -> InstructionValue:
    """Compute dot product."""
    from ..types import float32
    return _get_builder()._emit(IROp.DOT, float32, [a, b])


def cross(a: Value, b: Value) -> InstructionValue:
    """Compute cross product (3D vectors only)."""
    return _get_builder()._emit(IROp.CROSS, a.type, [a, b])


def distance(a: Value, b: Value) -> InstructionValue:
    """Compute distance between two points."""
    from ..types import float32
    return _get_builder()._emit(IROp.DISTANCE, float32, [a, b])


def reflect(i: Value, n: Value) -> InstructionValue:
    """Reflect vector."""
    return _get_builder()._emit(IROp.REFLECT, i.type, [i, n])


def refract(i: Value, n: Value, eta: Value) -> InstructionValue:
    """Refract vector."""
    return _get_builder()._emit(IROp.REFRACT, i.type, [i, n, eta])


def faceforward(n: Value, i: Value, ng: Value) -> InstructionValue:
    """Face forward."""
    return _get_builder()._emit(IROp.FACEFORWARD, n.type, [n, i, ng])


# ============================================================================
# Matrix Functions
# ============================================================================

def transpose(m: Value) -> InstructionValue:
    """Transpose matrix."""
    return _get_builder()._emit(IROp.MATRIX_TRANSPOSE, m.type, [m])


def inverse(m: Value) -> InstructionValue:
    """Compute matrix inverse."""
    return _get_builder()._emit(IROp.MATRIX_INVERSE, m.type, [m])


def determinant(m: Value) -> InstructionValue:
    """Compute matrix determinant."""
    from ..types import float32
    return _get_builder()._emit(IROp.MATRIX_DETERMINANT, float32, [m])
