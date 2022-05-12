import luisa
# from luisa.mathtypes import *
from luisa import array, struct


luisa.init()

a = array([5,32,6,4,0])

@luisa.func
def f():
	for x in a:
		print(x)

f(dispatch_size=1)
