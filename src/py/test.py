import luisa
from luisa.mathtypes import *


luisa.init()

@luisa.func
def f(x):
	b = 423
	return abs(x)
	a = 3
	# 31221
	b = 423

@luisa.func
def g():
	print(f(True))

g(dispatch_size=1)
# print(int2(3)/int2(2))
