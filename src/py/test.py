import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()


a = float4(3)
print(a.copy())
y = a
y[1] = 8
x = a.copy()
x[1] = 9
print(a,x)

b = float3x3(2)
print(b.copy())

a = b[0]
a[0] = 0
print(b)

a = b[0].copy()
a[0] = 9
print(b)