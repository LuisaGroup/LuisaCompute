import numpy as np
import luisa
from luisa.mathtypes import *


luisa.init()
# ============= test script ================



RandomSampler = luisa.StructType(state=int)

@luisa.callable_method(RandomSampler)
def __init__(self: luisa.ref(RandomSampler), p: int3):
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
    print("WOW!", self.state)

@luisa.callable_method(RandomSampler)
def next(self: luisa.ref(RandomSampler)):
    lcg_a = 1664525
    lcg_c = 1013904223
    self.state = lcg_a * self.state + lcg_c
    print("Okay..", self.state)
    return float(self.state & 0x00ffffff) * (1.0 / 0x01000000)



# @luisa.kernel
# def f():
#     sampler = RandomSampler(make_int3(dispatch_id().xy, 0))
#     a = sampler.next()


# f(dispatch_size=(2,1,1))
# f(dispatch_size=(1024, 1024, 1))
# luisa.synchronize()




from luisa import globalvars, lcapi
from luisa.globalvars import get_global_device
import math
from math import pi


accel = luisa.Accel()
from luisa import Ray


v_buffer = luisa.Buffer(3, float3)
t_buffer = luisa.Buffer(3, int)
v_buffer.copy_from(np.array([0,0,0,0,0,1,2,0,0,2,1,0], dtype=np.float32))
t_buffer.copy_from(np.array([0,1,2], dtype=np.int32))
mesh = luisa.Mesh(v_buffer, t_buffer)

accel.add(mesh)
accel.build()
luisa.globalvars.stream.synchronize()

@luisa.kernel
def test():
    a,b = pi
    r = Ray()
    r.set_origin(make_float3(0,0,0))
    r.t_min = 0.
    r.t_max = 1e5
    r.set_dir(make_float3(1,0,0))
    h = accel.trace_closest(r)


# arr = np.ones(1024*1024*4, dtype=np.uint8)
# img.copy_to(arr)
# print(arr)
# im.fromarray(arr.reshape((1024,1024,4))).save('aaa.png')
# cv2.imwrite("a.hdr", arr.reshape((1024,1024,4)))

