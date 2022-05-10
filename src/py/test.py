import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()
b = luisa.Buffer.zeros(1, dtype=int)
m = luisa.Buffer.zeros(1, dtype=int)

@luisa.func
def add():
    m.lock(0)
    b.write(0, b.read(0) + 1)
    m.unlock(0)

luisa.synchronize()
add(dispatch_size=2)
luisa.synchronize()

print(b.numpy())

# @luisa.func
# def f():
#     print(array([2,4,6]), struct(a=4,b=struct(a=4, b=float2(0.5)), c=array([0.9])))

# luisa.init()
# f(dispatch_size=(1,1,1))
# print(array([2,4,6]), struct(a=4,b=struct(a=4, b=float2(0.5)), c=array([0.9])))


