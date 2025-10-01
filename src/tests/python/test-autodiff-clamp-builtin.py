from luisa import *
from luisa.autodiff import requires_grad, autodiff, backward, grad


@func
def f_builtin(x) -> float:
    return clamp(x, 0., 1.)


@func
def df_builtin(x) -> float:
    with autodiff():
        requires_grad(x)
        t = f_builtin(x)
        backward(t, 1.0)
        g = grad(x)
    return g

@func
def test_autodiff_builtin(x):
    eps = 1e-3
    ad_grad = df_builtin(x)
    fd_grad = (f_builtin(x + eps) - f_builtin(x - eps)) / (2.0 * eps)
    print("x =", x, "| ad =", ad_grad, "| fd =", fd_grad, "| match =", abs(ad_grad - fd_grad) < 1e-2)


init()
print("Testing built-in clamp() function:")
# Test case 1: x > upper bound (gradient should be 0)
test_autodiff_builtin(1.5, dispatch_size=1)

# Test case 2: x inside [a, b] (gradient should be 1)
test_autodiff_builtin(0.5, dispatch_size=1)

# Test case 3: x < lower bound (gradient should be 0)
test_autodiff_builtin(-0.5, dispatch_size=1)

# Test case 4: x at upper bound (gradient should be 0)
test_autodiff_builtin(1.0, dispatch_size=1)

# Test case 5: x at lower bound (gradient should be 1)
test_autodiff_builtin(0.0, dispatch_size=1)
