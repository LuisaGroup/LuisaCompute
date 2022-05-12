import luisa
from luisa.mathtypes import *
from luisa import array, struct


luisa.init()


@luisa.func
def f():
    luisa.builtin.set_block_size(4,1,1)

f(dispatch_size=4)
print(b.to_list())