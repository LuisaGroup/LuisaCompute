import builtins
from ._internal.logging import \
    set_log_level_verbose, \
    set_log_level_info, \
    set_log_level_warning, \
    set_log_level_error, \
    log_verbose, log_info, log_warning, log_error
from .runtime import Device, \
    ACCEL_BUILD_HINT_FAST_TRACE, \
    ACCEL_BUILD_HINT_FAST_UPDATE, \
    ACCEL_BUILD_HINT_FAST_REBUILD
from .type import Type
from .pixel import *
from .ast import Constant

type = Type.of
tuple = Type.tuple

array = Type.array
vector = Type.vector
matrix = Type.matrix
struct = Type.struct

bool = Type.of(bool)
float = Type.of(float)
int = Type.of(int)
uint = Type.of("uint")

bvec2 = bool2 = vector(bool, 2)
vec2 = float2 = vector(float, 2)
ivec2 = int2 = vector(int, 2)
uvec2 = uint2 = vector(uint, 2)

bvec3 = bool3 = vector(bool, 3)
vec3 = float3 = vector(float, 3)
ivec3 = int3 = vector(int, 3)
uvec3 = uint3 = vector(uint, 3)

bvec4 = bool4 = vector(bool, 4)
vec4 = float4 = vector(float, 4)
ivec4 = int4 = vector(int, 4)
uvec4 = uint4 = vector(uint, 4)

mat2 = float2x2 = matrix(2)
mat3 = float3x3 = matrix(3)
mat4 = float4x4 = matrix(4)

__all__ = [g for g in globals() if g not in dir(builtins)]
