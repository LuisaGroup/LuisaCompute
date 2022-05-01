from . import StructType, callable_method, int3, ref

RandomSampler = StructType(state=int)

@callable_method(RandomSampler)
def __init__(self: ref(RandomSampler), p: int3):
    PRIME32_2 = 2246822519
    PRIME32_3 = 3266489917
    PRIME32_4 = 668265263
    PRIME32_5 = 374761393
    h32 =  p.z + PRIME32_5 + p.x * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 += p.y * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 = PRIME32_2 * (h32 ^ ((h32 >> 15) & 0x0001ffff))
    h32 = PRIME32_3 * (h32 ^ ((h32 >> 13) & 0x0007ffff))
    self.state = h32 ^ ((h32 >> 16) & 0x0000ffff)

@callable_method(RandomSampler)
def next(self: ref(RandomSampler)):
    lcg_a = 1664525
    lcg_c = 1013904223
    self.state = lcg_a * self.state + lcg_c
    return float(self.state & 0x00ffffff) * (1.0 / 0x01000000)