import luisa
from luisa.mathtypes import *

print(float2x2(4))
print(make_float2x2(4))

arr_t = luisa.ArrayType(100, bool)
arr_t in luisa.types.basic_type_dict

luisa.init()

@luisa.func
def f():
    print(float2x2(4.0))

f( dispatch_size=(1,1,1))