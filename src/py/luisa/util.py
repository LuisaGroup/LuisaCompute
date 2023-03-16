from . import func, StructType
from .mathtypes import *
from .types import *
import math
RandomSampler = StructType(state=int)


@func
def _f(self, init_state):
    self.state = init_state


RandomSampler.add_method(_f, "__init__")
@func
def make_random_sampler(v0, v1):
    s0 = uint()
    for i in range(4):
        s0 += 0x9e3779b9
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4)
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e)
    return RandomSampler(v0)

@func
def sign(x):
    return copysign(1., x)


@func
def fmod(x, y):
    return x - y * trunc(x / y)


@func
def mod(x, y):
    return x - y * floor(x / y)


@func
def _f(self):
    lcg_a = 1664525
    lcg_c = 1013904223
    self.state = lcg_a * self.state + lcg_c
    return float(self.state & 0x00ffffff) * (1.0 / 0x01000000)


RandomSampler.add_method(_f, "next")


@func
def _f(self):
    return float2(self.next(), self.next())


RandomSampler.add_method(_f, "next2f")


@func
def _f(self):
    return float3(self.next(), self.next(), self.next())


RandomSampler.add_method(_f, "next3f")


@func
def smoothstep(left, right, x):
    t = saturate((x - left) / (right - left))
    return t * t * fma(t, -2., 3.)


@func
def ite(a, b, c):
    return select(c, b, a)


@func
def make_float2x2_eye(v: float):
    return float2x2(v, 0,
                    0, v)


@func
def make_float3x3_eye(v: float):
    return float3x3(v, 0, 0,
                    0, v, 0,
                    0, 0, v)


@func
def make_float4x4_eye(v: float):
    return float4x4(v, 0, 0, 0,
                    0, v, 0, 0,
                    0, 0, v, 0,
                    0, 0, 0, v)


@func
def distance(a: float3, b: float3):
    return length(a - b)


@func
def distance_squared(a: float3, b: float3):
    return length_squared(a - b)


@func
def radians(x):
    return x * (math.pi / 180.)


@func
def degree(x):
    return x * (180. / math.pi)
