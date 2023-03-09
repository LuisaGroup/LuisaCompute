from luisa import *
from luisa.builtin import *
from luisa.types import *
import numpy as np
init()
MyStructType = StructType(my_v0=float3, my_v1=float4)
buffer = Buffer(32, MyStructType)
print("buffer byte size(should be 32 * 32 = 1024): " + str(buffer.bytesize))
# def get_negative_number_with_arg(buffer, atomic_buffer):


@func
def get_negative_number():
    index = dispatch_id().x
    tmp = buffer.read(index)
    tmp.my_v0 = sin(tmp.my_v0)
    tmp.my_v1 = cos(tmp.my_v1)
    buffer.write(index, tmp)


array = np.zeros([32 * 8], dtype=np.float32)
buffer.copy_from(array)
# write array after "copy_from" command is fine here, all commands is scheduled in queue before call execute()
for i in range(len(array)):
    array[i] = i
get_negative_number(dispatch_size=(32, 1, 1))
# all commands start running when execute() is called
execute()
# get_negative_number_with_arg(buffer, atomic_buffer, dispatch_size=(32,1,1))
# host wait all commands finish
synchronize()
buffer.copy_to(array)
result_str = ""
for i in array:
    result_str += str(i) + ' '
print("result: ")
print(result_str)
