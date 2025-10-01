from luisa import *
from luisa.autodiff import requires_grad, autodiff, backward, grad


@func
def clampa(x, a, b):
    return min(max(x, a), b)


@func
def f(x) -> float:
    return clampa(x, 0., 1.)


@func
def df(x) -> float:
    with autodiff():
        requires_grad(x)
        t = f(x)
        backward(t, 1.0)
        g = grad(x)
    return g

@func
def test_autodiff(x):
    print("testing autodiff (", x, ")")
    eps = 1e-3
    print("ad f(a) =", df(x))
    print("fd f(a) =", (f(x + eps) - f(x - eps)) / (2.0 * eps))


init()
test_autodiff(1.5, dispatch_size=1)
