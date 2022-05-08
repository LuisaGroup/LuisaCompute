import luisa
from luisa.mathtypes import *
from luisa import array, struct

print(float2x2(4))
print(make_float2x2(4))


luisa.init()

@luisa.func
def f():
    print(array([2,4,6]), struct(a=4,b=0.9))

f(dispatch_size=(1,1,1))
print(array([2,4,6]), struct(a=4,b=0.9))
