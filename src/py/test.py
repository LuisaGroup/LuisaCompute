import luisa

a = luisa.array([x for x in range(200)])

@luisa.func
def f():
    print(a)

luisa.init()
f(dispatch_size=1000)

