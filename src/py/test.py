import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()

b = luisa.Buffer.empty(size=4, dtype=luisa.StructType(a=bool, b=float3, c=luisa.ArrayType(size=3, dtype=bool), d=luisa.StructType(alignment=16, a=int, b=float)))

@luisa.func
def f():
	b.write(dispatch_id().x, struct(a=True, b=float3(4), c=array([True,False,True]), d=struct(16, a=dispatch_id().x, b=3.14)))

f(dispatch_size=4)
print(b.to_list())