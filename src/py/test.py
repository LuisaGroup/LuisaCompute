import luisa
import numpy as np
from luisa.builtin import builtin_func
from luisa.mathtypes import *
from PIL import Image as im

luisa.init()
# ============= test script ================

# def test_astgen():
#     lcapi.builder().set_block_size(256,1,1)
#     int_type = lcapi.Type.from_("int")
#     buf_type = lcapi.Type.from_("buffer<int>")
#     buf = lcapi.builder().buffer_binding(buf_type, buffer_handle, 0)
#     idx3 = lcapi.builder().dispatch_id()
#     idx = lcapi.builder().access(int_type, idx3, lcapi.builder().literal(int_type, 0))
#     index = lcapi.builder().literal(int_type, 1)
#     value = lcapi.builder().literal(int_type, 42)
#     lcapi.builder().call(lcapi.CallOp.BUFFER_WRITE, [buf, idx, idx])



# x1 = lcapi.make_float2(6,10)
# m1 = lcapi.make_float2x2(1,2,3,4)

# Arr = luisa.ArrayType(int, 3)


# @luisa.callable
# def g(arr: Arr):
#     if dispatch_id().x < 3:
#         print("in callable! ", dispatch_id())
#     return arr[0] * arr[1] + arr[2]


# @luisa.callable
# def flipsign(x: luisa.ref(int)):
#     x = -x


# @luisa.kernel
# def f(a: int, arr: Arr, b: luisa.BufferType(int)):
#     # x[0] = a
#     a1 = Arr()
#     # a2[4] = 5
#     idx = dispatch_id().x
#     aaa = int(-0.3)
#     for xxx in range(100, 200, 3):
#         aaa += xxx
#     a += g(arr) if True else -1
#     flipsign(aaa)
#     b.write(idx, aaa)
#     if dispatch_id().x < 5:
#         vtmp = make_float4(123., 123., 123., 123.)
#         print("blah", aaa, True, 3.14, vtmp)
#     copysign(-1.,1.)


@luisa.callable
def test_matrix_vector():
    mat1 = make_float2x2(1., 2., 3., 4.)
    mat2 = make_float2x2(5., 6., 7., 8.)
    vec1 = make_float2(9., 10.)

    mat = mat1 + mat2
    mat = mat1 - mat2
    mat = mat1 * mat2
    vec = mat1 * vec1
    # print(vec)
    vec = vec1 * vec1
    vec = vec / vec1
    # vec = vec1 * mat1

    mat3 = make_float3x3(1., 2., 3., 4., 5., 6., 7., 8., 9.)
    # mat = mat1 + mat3


@luisa.callable
def test_arithmetic():
    scalar1 = 1
    scalar2 = 1.
    scalar3 = True

    scalar = scalar1 + scalar2
    # scalar = scalar1 + scalar3
    # scalar = scalar2 + scalar3
    scalar4 = scalar1 << scalar1
    scalar4 = scalar4 >> scalar1

    bits1 = 23
    bits2 = 10
    bits = bits1 | bits2
    bits = bits1 & bits2
    bits = bits1 ^ bits2


@luisa.callable
def test_broadcast():
    scalar1 = 2.
    vec1 = make_float4(2., 3., 4., 5.)

    vec = scalar1 + vec1
    # print(vec)
    vec = vec1 * scalar1
    # print(vec)


@luisa.callable
def test_compare():
    vec1 = make_int2(9, 10)
    # print(vec1)
    vec2 = make_int2(11, 3)
    # print(vec2)


@luisa.callable
def test_div_and_floordiv():
    veci1 = make_int2(5, 10)
    veci2 = make_int2(2, 3)
    veci = veci1 // veci2
    vecf = veci1 / veci2
    print(veci)
    print(vecf)


@luisa.callable
def test_fail():
    vec = make_int2(True, 2.2)
    vec2 = make_float2(vec)
    vec3 = make_float2(1, 2.5)
    print(vec2)
    print(vec3)


@luisa.callable
def test_unary():
    vec = make_float2(1., 2.)
    vec1 = -vec


@luisa.callable
def test_power():
    vec = make_float3(2., 3., 4.)
    scalar = 2
    vec1 = vec ** vec
    vec2 = vec ** scalar
    vec3 = scalar ** vec
    scalar1 = scalar ** scalar
    print(vec1)
    print(vec2)
    print(vec3)
    print(scalar1)


@luisa.kernel
def test_builtin():
    test_matrix_vector()
    test_arithmetic()
    test_broadcast()
    test_compare()
    test_div_and_floordiv()
    test_unary()
    test_fail()
    test_power()


test_builtin(dispatch_size=(5, 1, 1))

# img = luisa.Texture2D(1024, 1024, 4, float, luisa.lcapi.PixelStorage.BYTE4)
# b = luisa.Buffer(1024, int)
#
#
# @luisa.kernel
# def f():
#     # need int/int->float
#     # need cast float3(int3)
#     cx = float(dispatch_id().x) / dispatch_size().x
#     cy = float(dispatch_id().y) / dispatch_size().y
#     cz = float(dispatch_id().z) / dispatch_size().z
#     img.write(dispatch_id().xy, make_float4(cx,cy,cz,1.))
#     # b.write(dispatch_id(), 3)
#
#
# f(dispatch_size=(1024, 1024, 1))
#
# arr = np.ones(1024*1024*4, dtype=np.uint8)
# img.copy_to(arr)
# print(arr)
# im.fromarray(arr.reshape((1024,1024,4))).save('aaa.png')
# cv2.imwrite("a.hdr", arr.reshape((1024,1024,4)))

