import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()

sampler = luisa.RandomSampler(state=45)

print(luisa.types.dtype_of(sampler))

@luisa.func
def add():
    print(sampler.next2f())

add(dispatch_size=1)

