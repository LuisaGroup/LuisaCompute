from luisa import *
from luisa.builtin import *
from luisa.types import *
import numpy as np
init()
MyStructType = StructType(my_v0=float3, my_v1=float4)
buffer = Buffer(32, MyStructType)
atomic_buffer = Buffer(1, int)
print("buffer byte size(should be 32 * 32 = 1024): " + str(buffer.bytesize))
# def get_negative_number_with_arg(buffer, atomic_buffer):


@func
def get_negative_number():
    index = dispatch_id().x
    tmp = buffer.read(index)
    tmp.my_v0 = sin(tmp.my_v0)
    tmp.my_v1 = cos(tmp.my_v1)
    buffer.write(index, tmp)
    last_value = atomic_buffer.atomic_fetch_add(0, 1)


array = np.zeros([32 * 8], dtype=np.float32)
for i in range(len(array)):
    array[i] = i
atomic_array = np.zeros(1, dtype=np.int32)
buffer.copy_from(array)
atomic_buffer.copy_from(atomic_array)
get_negative_number(dispatch_size=(32, 1, 1))
# get_negative_number_with_arg(buffer, atomic_buffer, dispatch_size=(32,1,1))

buffer.copy_to(array)
atomic_buffer.copy_to(atomic_array)
result_str = ""
for i in array:
    result_str += str(i) + ' '
print("result: ")
print(result_str)
print("count: " + str(atomic_array[0]))
