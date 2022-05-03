import numpy as np
import luisa
from luisa.mathtypes import *


luisa.init()
# ============= test script ================



# arr = luisa.BindlessArray()

@luisa.callable
def g(a):
    a += a


@luisa.kernel
def f(a):
    print("Aaaaa")
    print(a)
    g(a)
    print(a)


f(1, dispatch_size=(1,1,1))
print("============================")
f(float3(1), dispatch_size=(1,1,1))





# accel = luisa.Accel()
# from luisa import Ray


# v_buffer = luisa.Buffer(3, float3)
# t_buffer = luisa.Buffer(3, int)
# v_buffer.copy_from(np.array([0,0,0,0,0,1,2,0,0,2,1,0], dtype=np.float32))
# t_buffer.copy_from(np.array([0,1,2], dtype=np.int32))
# mesh = luisa.Mesh(v_buffer, t_buffer)

# accel.add(mesh)
# accel.build()
# luisa.globalvars.stream.synchronize()

# @luisa.kernel
# def test():
#     a,b = pi
#     r = Ray()
#     r.set_origin(make_float3(0,0,0))
#     r.t_min = 0.
#     r.t_max = 1e5
#     r.set_dir(make_float3(1,0,0))
#     h = accel.trace_closest(r)


# arr = np.ones(1024*1024*4, dtype=np.uint8)
# img.copy_to(arr)
# print(arr)
# im.fromarray(arr.reshape((1024,1024,4))).save('aaa.png')
# cv2.imwrite("a.hdr", arr.reshape((1024,1024,4)))

