import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()

# sampler = luisa.RandomSampler(state=45)
s = luisa.struct(a=float3(4), b=luisa.array([3,4]), c=luisa.struct(a=3, b=True))

# print(type(sampler))

@luisa.func
def add():
    # print(sampler.next2f())
    print(s)

add(dispatch_size=1)

