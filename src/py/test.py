import luisa
from luisa.mathtypes import *


luisa.init()

a = float4

@luisa.func
def f(x):
	b:int = 2

@luisa.func
def g():
	f(True)

g(dispatch_size=1)
# print(int2(3)/int2(2))
