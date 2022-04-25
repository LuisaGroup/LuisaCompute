import luisa
import numpy as np
from luisa.builtin import builtin_func

luisa.init()
# ============= test script ================


luisa.init("ispc")

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

Arr = luisa.ArrayType(int, 3)


@luisa.callable
def g(arr: Arr):
    return arr[0] * arr[1] + arr[2]

@luisa.kernel
def f(a: int, arr: Arr, b: luisa.BufferType(int)):
    a1 = Arr()
    # a2[4] = 5
    idx = dispatch_id().x
    aaa = 0
    for xxx in range(100, 200, 3):
        aaa += xxx
    a += g(arr) if True else -1
    b.write(idx, aaa)
    if dispatch_id().x < 5:
        print("blah", aaa, True, 3.14, dispatch_id())


@luisa.callable
def test_matrix_vector():
    mat1 = make_float2x2(1., 2., 3., 4.)
    mat2 = make_float2x2(5., 6., 7., 8.)
    vec1 = make_float2(9., 10.)

    mat = mat1 + mat2
    mat = mat1 - mat2
    mat = mat1 * mat2
    vec = mat1 * vec1
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
    # 不能将 int 赋值给 float?
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
    print(vec)
    vec = vec1 * scalar1
    print(vec)


@luisa.callable
def test_compare():
    vec1 = make_int2(9, 10)
    print(vec1)
    vec2 = make_int2(11, 3)
    print(vec2)
    vec_bool = vec1 > vec2
    print(vec_bool)


@luisa.kernel
def test_builtin():
    test_matrix_vector()
    test_arithmetic()
    test_broadcast()
    test_compare()


test_builtin(dispatch_size=(5, 1, 1))

b = luisa.Buffer(100, int)

arr = np.ones(100, dtype='int32')
arr1 = np.zeros(100, dtype='int32')

b.copy_from(arr)
f(42, Arr([10, 20, 30]), b, dispatch_size=(100, 1, 1))
b.copy_to(arr1)

print(arr1)
