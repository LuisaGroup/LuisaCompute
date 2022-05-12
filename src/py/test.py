import luisa
from luisa.mathtypes import *


luisa.init()

@luisa.func
def f():
	print(int2(3)/int2(2))

f(dispatch_size=1)
print(int2(3)/int2(2))
