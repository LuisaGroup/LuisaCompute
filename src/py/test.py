import luisa
from luisa.mathtypes import *


luisa.init()

@luisa.func
def f(x):
	return abs(x)

@luisa.func
def g():
	print(f(True))

g(dispatch_size=1)
# print(int2(3)/int2(2))
