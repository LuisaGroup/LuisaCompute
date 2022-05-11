import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()

sampler = luisa.RandomSampler(state=45)
sampler1 = sampler.copy()

print(luisa.types.dtype_of(sampler))

@luisa.func
def add():
    print(sampler.next2f())
    print(sampler1.next2f())

add(dispatch_size=1)

a = luisa.array([bool2(True,False), bool2(True), bool2(False)])
print(a, a.copy(), a.copy().copy(), a)