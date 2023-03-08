from . import func, StructType, int3, float2, float3
from .mathtypes import *
RandomSampler = StructType(state=int)


@func
def _f(self, p: int3):
    PRIME32_2 = 2246822519
    PRIME32_3 = 3266489917
    PRIME32_4 = 668265263
    PRIME32_5 = 374761393
    h32 = p.z + PRIME32_5 + p.x * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 += p.y * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 = PRIME32_2 * (h32 ^ ((h32 >> 15) & 0x0001ffff))
    h32 = PRIME32_3 * (h32 ^ ((h32 >> 13) & 0x0007ffff))
    self.state = h32 ^ ((h32 >> 16) & 0x0000ffff)


RandomSampler.add_method(_f, "__init__")


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
