import luisa
from luisa.mathtypes import *


luisa.init()

@luisa.func
def f(x):
	a,b = 4,5
	a,b = a+b, a-b
	print(a,b)

@luisa.func
def g():
	f(True)

g(dispatch_size=1)
# print(int2(3)/int2(2))
