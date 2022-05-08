import luisa
from luisa.mathtypes import *
from luisa import array, struct


@luisa.func
def f():
    print(array([2,4,6]), struct(a=4,b=struct(a=4, b=float2(0.5)), c=array([0.9])))

luisa.init()
f(dispatch_size=(1,1,1))
print(array([2,4,6]), struct(a=4,b=struct(a=4, b=float2(0.5)), c=array([0.9])))

