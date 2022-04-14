import luisa
import numpy as np





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

luisa.init('ispc')
b = luisa.Buffer(100, int)
# x1 = lcapi.make_float2(6,10)
# m1 = lcapi.make_float2x2(1,2,3,4)

@luisa.kernel
def f(a: int):
    idx = dispatch_id().x
    # val = b.read(idx)
    # x = make_float2(3,5) * -1 + x1
    a += 2
    b.write(idx, a)
    # m2 = make_float2x2(1,2,3,4,5,6,7)




arr = np.ones(100, dtype='int32')
arr1 = np.zeros(100, dtype='int32')

# upload command
b.async_copy_from(arr)

# dispatch
f(dispatch_size = (100,1,1))

# download command
b.async_copy_to(arr1)

stream.synchronize()

print(arr1)
