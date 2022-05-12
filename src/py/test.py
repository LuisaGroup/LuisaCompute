import luisa
from luisa.mathtypes import *


luisa.init()

@luisa.func
def f():
	# print(int2(3)/int2(2))
	return 3.0 < 9

@luisa.func
def g():
	print(f())

g(dispatch_size=1)
# print(int2(3)/int2(2))
