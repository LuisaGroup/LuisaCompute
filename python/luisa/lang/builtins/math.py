"""
Math builtin functions for the LuisaCompute Python DSL v2.

These functions use the @router decorator to support:
1. Constant folding: sin(1.0 + 2.0) -> constant sin(3.0)
2. Device routing: sin(dsl_var) -> device-side SIN instruction
3. Vector constant folding: normalize(Float3(1,2,3)) -> (1,0,0)
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Callable, Tuple

if TYPE_CHECKING:
    from ...transform.ir import Value, InstructionValue
    from ..types import Type

from ...transform.op import Op
from ...transform.ir import ConstantValue
from ...transform.builder import get_current_builder
from ..types import Float, value_to_type, promote_types
from ..router import router, RoutedFunction, is_constant_value, extract_constant_value, is_vector_constant, extract_vector_components


# ============================================================================
# Host implementations for constant folding
# ============================================================================

# Unary math functions
_sqrt_host = math.sqrt
_abs_host = abs  # Python built-in abs works for numbers
_sin_host = math.sin
_cos_host = math.cos
_tan_host = math.tan
_asin_host = math.asin
_acos_host = math.acos
_atan_host = math.atan
_sinh_host = math.sinh
_cosh_host = math.cosh
_tanh_host = math.tanh
_exp_host = math.exp
_exp2_host = lambda x: 2.0 ** x
_log_host = math.log
_log2_host = lambda x: math.log(x) / math.log(2.0) if x > 0 else float('-inf')
_log10_host = math.log10
_floor_host = math.floor
_ceil_host = math.ceil
_round_host = round  # Python built-in round
_trunc_host = lambda x: int(x) if x >= 0 else int(x) - 1 if x != int(x) else int(x)
_fract_host = lambda x: x - math.floor(x)
_saturate_host = lambda x: max(0.0, min(1.0, x))

# Additional math functions
_rsqrt_host = lambda x: 1.0 / math.sqrt(x) if x > 0 else float('inf')
_exp10_host = lambda x: 10.0 ** x
_asinh_host = lambda x: math.asinh(x)
_acosh_host = lambda x: math.acosh(x) if x >= 1.0 else float('nan')
_atanh_host = lambda x: math.atanh(x) if abs(x) < 1.0 else float('nan')
_isinf_host = lambda x: math.isinf(x)
_isnan_host = lambda x: math.isnan(x)
_copysign_host = lambda x, y: math.copysign(x, y)
_fma_host = lambda a, b, c: a * b + c  # Fused multiply-add


# ============================================================================
# Vector Host Implementations for Constant Folding (using tuples)
# ============================================================================

def _normalize_host(v):
    """Host normalize for vector constants (tuple)."""
    if isinstance(v, (list, tuple)):
        # Compute length
        length_sq = sum(c * c for c in v)
        length = math.sqrt(length_sq)
        if length == 0:
            # Return zero vector for zero-length input
            return tuple([0.0] * len(v))
        # Normalize components
        return tuple(c / length for c in v)
    raise TypeError(f"normalize() constant folding only supports tuples, got {type(v)}")


def _length_host(v):
    """Host length for vector constants (tuple)."""
    if isinstance(v, (list, tuple)):
        length_sq = sum(c * c for c in v)
        return math.sqrt(length_sq)
    raise TypeError(f"length() constant folding only supports tuples, got {type(v)}")


def _length_squared_host(v):
    """Host length_squared for vector constants (tuple)."""
    if isinstance(v, (list, tuple)):
        return sum(c * c for c in v)
    raise TypeError(f"length_squared() constant folding only supports tuples, got {type(v)}")


def _dot_host(a, b):
    """Host dot product for vector constants (tuple)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Vector sizes must match for dot product: {len(a)} vs {len(b)}")
        return sum(ac * bc for ac, bc in zip(a, b))
    raise TypeError(f"dot() constant folding only supports tuples, got {type(a)}, {type(b)}")


def _cross_host(a, b):
    """Host cross product for vector constants (3D only)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != 3 or len(b) != 3:
            raise ValueError(f"Cross product requires 3D vectors, got {len(a)} and {len(b)}")
        ax, ay, az = a
        bx, by, bz = b
        return (ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx)
    raise TypeError(f"cross() constant folding only supports tuples, got {type(a)}, {type(b)}")


def _distance_host(a, b):
    """Host distance for vector constants (tuple)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Vector sizes must match for distance: {len(a)} vs {len(b)}")
        diff_sq = sum((ac - bc) ** 2 for ac, bc in zip(a, b))
        return math.sqrt(diff_sq)
    raise TypeError(f"distance() constant folding only supports tuples, got {type(a)}, {type(b)}")


def _reflect_host(i, n):
    """Host reflect for vector constants (tuple)."""
    if isinstance(i, (list, tuple)) and isinstance(n, (list, tuple)):
        if len(i) != len(n):
            raise ValueError(f"Vector sizes must match for reflect: {len(i)} vs {len(n)}")
        # r = i - 2 * dot(n, i) * n
        dot_ni = sum(nc * ic for nc, ic in zip(n, i))
        factor = 2.0 * dot_ni
        result = tuple(ic - factor * nc for ic, nc in zip(i, n))
        return result
    raise TypeError(f"reflect() constant folding only supports tuples, got {type(i)}, {type(n)}")


import builtins


def _min_host(a, b):
    """Host min implementation."""
    return builtins.min(a, b)


def _max_host(a, b):
    """Host max implementation."""
    return builtins.max(a, b)


def _clamp_host(x, min_val, max_val):
    """Host clamp implementation."""
    return max(min_val, min(x, max_val))


def _lerp_host(a, b, t):
    """Host lerp implementation."""
    return a + (b - a) * t


def _step_host(edge, x):
    """Host step implementation."""
    return 1.0 if x >= edge else 0.0


def _smoothstep_host(edge0, edge1, x):
    """Host smoothstep implementation."""
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


def _pow_host(base, exp):
    """Host pow implementation."""
    return base ** exp


def _atan2_host(y, x):
    """Host atan2 implementation."""
    return math.atan2(y, x)


# Integer bit operations (for host-side constant folding)
def _clz_host(x):
    """Count leading zeros."""
    if x == 0:
        return 32  # Assuming 32-bit integers
    # Use bit_length to find position of highest set bit
    return 32 - x.bit_length()


def _ctz_host(x):
    """Count trailing zeros."""
    if x == 0:
        return 32
    # Count trailing zeros by finding lowest set bit
    return (x & -x).bit_length() - 1


def _popcount_host(x):
    """Count set bits (population count)."""
    return bin(x).count('1')


def _reverse_host(x):
    """Bit reversal (for 32-bit integers)."""
    # Reverse bits
    result = 0
    for i in range(32):
        result = (result << 1) | ((x >> i) & 1)
    return result


# ============================================================================
# Unary Math Functions
# ============================================================================

@router(host_impl=_sqrt_host, device_op=Op.SQRT)
def sqrt(x):
    """Compute square root."""
    pass


@router(host_impl=_abs_host, device_op=Op.ABS)
def abs(x):
    """Compute absolute value."""
    pass


@router(host_impl=_sin_host, device_op=Op.SIN)
def sin(x):
    """Compute sine."""
    pass


@router(host_impl=_cos_host, device_op=Op.COS)
def cos(x):
    """Compute cosine."""
    pass


@router(host_impl=_tan_host, device_op=Op.TAN)
def tan(x):
    """Compute tangent."""
    pass


@router(host_impl=_sinh_host, device_op=Op.SINH)
def sinh(x):
    """Compute hyperbolic sine."""
    pass


@router(host_impl=_cosh_host, device_op=Op.COSH)
def cosh(x):
    """Compute hyperbolic cosine."""
    pass


@router(host_impl=_tanh_host, device_op=Op.TANH)
def tanh(x):
    """Compute hyperbolic tangent."""
    pass


@router(host_impl=_asin_host, device_op=Op.ASIN)
def asin(x):
    """Compute arc sine."""
    pass


@router(host_impl=_acos_host, device_op=Op.ACOS)
def acos(x):
    """Compute arc cosine."""
    pass


@router(host_impl=_atan_host, device_op=Op.ATAN)
def atan(x):
    """Compute arc tangent."""
    pass


@router(host_impl=_exp_host, device_op=Op.EXP)
def exp(x):
    """Compute exponential."""
    pass


@router(host_impl=_exp2_host, device_op=Op.EXP2)
def exp2(x):
    """Compute base-2 exponential."""
    pass


@router(host_impl=_log_host, device_op=Op.LOG)
def log(x):
    """Compute natural logarithm."""
    pass


@router(host_impl=_log2_host, device_op=Op.LOG2)
def log2(x):
    """Compute base-2 logarithm."""
    pass


@router(host_impl=_log10_host, device_op=Op.LOG10)
def log10(x):
    """Compute base-10 logarithm."""
    pass


@router(host_impl=_floor_host, device_op=Op.FLOOR)
def floor(x):
    """Compute floor."""
    pass


@router(host_impl=_ceil_host, device_op=Op.CEIL)
def ceil(x):
    """Compute ceiling."""
    pass


@router(host_impl=_round_host, device_op=Op.ROUND)
def round(x):
    """Round to nearest integer."""
    pass


@router(host_impl=_trunc_host, device_op=Op.TRUNC)
def trunc(x):
    """Truncate to integer."""
    pass


@router(host_impl=_fract_host, device_op=Op.FRACT)
def fract(x):
    """Compute fractional part."""
    pass


@router(host_impl=_saturate_host, device_op=Op.SATURATE)
def saturate(x):
    """Clamp to [0, 1]."""
    pass


# ============================================================================
# Additional Scalar Math Functions
# ============================================================================

@router(host_impl=_rsqrt_host, device_op=Op.RSQRT)
def rsqrt(x):
    """Compute reciprocal square root (1/sqrt(x))."""
    pass


@router(host_impl=_exp10_host, device_op=Op.EXP10)
def exp10(x):
    """Compute base-10 exponential (10^x)."""
    pass


@router(host_impl=_asinh_host, device_op=Op.ASINH)
def asinh(x):
    """Compute inverse hyperbolic sine."""
    pass


@router(host_impl=_acosh_host, device_op=Op.ACOSH)
def acosh(x):
    """Compute inverse hyperbolic cosine."""
    pass


@router(host_impl=_atanh_host, device_op=Op.ATANH)
def atanh(x):
    """Compute inverse hyperbolic tangent."""
    pass


@router(host_impl=_isinf_host, device_op=Op.ISINF)
def isinf(x):
    """Check if value is infinite."""
    pass


@router(host_impl=_isnan_host, device_op=Op.ISNAN)
def isnan(x):
    """Check if value is NaN."""
    pass


@router(host_impl=_copysign_host, device_op=Op.COPYSIGN)
def copysign(x, y):
    """Return x with the sign of y."""
    pass


@router(host_impl=_fma_host, device_op=Op.FMA)
def fma(a, b, c):
    """Fused multiply-add: compute a*b + c with single rounding."""
    pass


# ============================================================================
# Integer Bit Operations
# ============================================================================

@router(host_impl=_clz_host, device_op=Op.CLZ)
def clz(x):
    """Count leading zeros in integer representation."""
    pass


@router(host_impl=_ctz_host, device_op=Op.CTZ)
def ctz(x):
    """Count trailing zeros in integer representation."""
    pass


@router(host_impl=_popcount_host, device_op=Op.POPCOUNT)
def popcount(x):
    """Count number of set bits (population count)."""
    pass


@router(host_impl=_reverse_host, device_op=Op.REVERSE)
def reverse(x):
    """Reverse bits in integer representation."""
    pass


# ============================================================================
# Vector Math Functions with Constant Folding
# ============================================================================

def _normalize_device_wrapper(builder, x):
    """Device-side wrapper for normalize."""
    return builder._emit(Op.NORMALIZE, x.type, [x])


@router(host_impl=_normalize_host, device_wrapper=_normalize_device_wrapper)
def normalize(x):
    """Normalize vector."""
    pass


def _length_device_wrapper(builder, x):
    """Device-side wrapper for length."""
    return builder._emit(Op.LENGTH, Float, [x])


@router(host_impl=_length_host, device_wrapper=_length_device_wrapper)
def length(x):
    """Compute vector length."""
    pass


def _length_squared_device_wrapper(builder, x):
    """Device-side wrapper for length_squared."""
    return builder._emit(Op.LENGTH_SQUARED, Float, [x])


@router(host_impl=_length_squared_host, device_wrapper=_length_squared_device_wrapper)
def length_squared(x):
    """Compute squared vector length."""
    pass


def _dot_device_wrapper(builder, a, b):
    """Device-side wrapper for dot."""
    return builder._emit(Op.DOT, Float, [a, b])


@router(host_impl=_dot_host, device_wrapper=_dot_device_wrapper)
def dot(a, b):
    """Compute dot product."""
    pass


def _cross_device_wrapper(builder, a, b):
    """Device-side wrapper for cross."""
    return builder._emit(Op.CROSS, a.type, [a, b])


@router(host_impl=_cross_host, device_wrapper=_cross_device_wrapper)
def cross(a, b):
    """Compute cross product (3D vectors only)."""
    pass


def _distance_device_wrapper(builder, a, b):
    """Device-side wrapper for distance."""
    return builder._emit(Op.DISTANCE, Float, [a, b])


@router(host_impl=_distance_host, device_wrapper=_distance_device_wrapper)
def distance(a, b):
    """Compute distance between two points."""
    pass


def _reflect_device_wrapper(builder, i, n):
    """Device-side wrapper for reflect."""
    return builder._emit(Op.REFLECT, i.type, [i, n])


@router(host_impl=_reflect_host, device_wrapper=_reflect_device_wrapper)
def reflect(i, n):
    """Reflect vector."""
    pass


def refract(i, n, eta):
    """Refract vector."""
    # Complex to implement host-side, keep device-only for now
    return get_current_builder()._emit(Op.REFRACT, i.type, [i, n, eta])


def faceforward(n, i, ng):
    """Face forward."""
    # Complex to implement host-side, keep device-only for now
    return get_current_builder()._emit(Op.FACEFORWARD, n.type, [n, i, ng])


# ============================================================================
# Binary Math Functions
# ============================================================================

@router(host_impl=_min_host, device_op=Op.MIN)
def min(a, b):
    """Compute minimum."""
    pass


@router(host_impl=_max_host, device_op=Op.MAX)
def max(a, b):
    """Compute maximum."""
    pass


def _clamp_device_wrapper(builder, x, min_val, max_val):
    """Device-side wrapper for clamp (3 args)."""
    return builder._emit(Op.CLAMP, x.type, [x, min_val, max_val])


@router(host_impl=_clamp_host, device_wrapper=_clamp_device_wrapper)
def clamp(x, min_val, max_val):
    """Clamp to range [min_val, max_val]."""
    pass


def _lerp_device_wrapper(builder, a, b, t):
    """Device-side wrapper for lerp (3 args)."""
    return builder._emit(Op.LERP, a.type, [a, b, t])


@router(host_impl=_lerp_host, device_wrapper=_lerp_device_wrapper)
def lerp(a, b, t):
    """Linear interpolation: a + (b - a) * t"""
    pass


def _step_device_wrapper(builder, edge, x):
    """Device-side wrapper for step."""
    return builder._emit(Op.STEP, x.type, [edge, x])


@router(host_impl=_step_host, device_wrapper=_step_device_wrapper)
def step(edge, x):
    """Step function: (x >= edge) ? 1 : 0"""
    pass


def _smoothstep_device_wrapper(builder, edge0, edge1, x):
    """Device-side wrapper for smoothstep."""
    return builder._emit(Op.SMOOTHSTEP, x.type, [edge0, edge1, x])


@router(host_impl=_smoothstep_host, device_wrapper=_smoothstep_device_wrapper)
def smoothstep(edge0, edge1, x):
    """Smooth Hermite interpolation."""
    pass


@router(host_impl=_pow_host, device_op=Op.POW)
def pow(base, exp):
    """Compute power."""
    pass


@router(host_impl=_atan2_host, device_op=Op.ATAN2)
def atan2(y, x):
    """Compute arc tangent of y/x."""
    pass


# ============================================================================
# Matrix Functions
# ============================================================================

def transpose(m):
    """Transpose matrix."""
    return get_current_builder()._emit(Op.MATRIX_TRANSPOSE, m.type, [m])


def inverse(m):
    """Compute matrix inverse."""
    return get_current_builder()._emit(Op.MATRIX_INVERSE, m.type, [m])


def determinant(m):
    """Compute matrix determinant."""
    return get_current_builder()._emit(Op.MATRIX_DETERMINANT, Float, [m])
