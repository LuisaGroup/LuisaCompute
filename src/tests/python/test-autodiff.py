from luisa import *
from luisa.autodiff import *
from luisa.types import *
import numpy as np
init()

N = 1024
x_buffer = Buffer(N, float)
y_buffer = Buffer(N, float2)
dx_buffer = Buffer(N, float)
dy_buffer = Buffer(N, float2)

x_values = np.arange(N, dtype=np.float32)
y_values = np.arange(N * 2, dtype=np.float32).reshape(-1, 2)
x_buffer.copy_from(x_values)
y_buffer.copy_from(y_values)

@func
def f(x: float, y: float2):
    return x * x + y.x * y.x + y.y * y.y

@func
def shader():
    i = dispatch_id().x
    x = x_buffer.read(i)
    y = y_buffer.read(i)
    with autodiff():
        requires_grad(x, y)
        z = f(x, y)
        backward(z)
        dx = grad(x)
        dy = grad(y)
    dx_buffer.write(i, dx)
    dy_buffer.write(i, dy)


def fd():
    x = np.array(x_values, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)
    f = lambda x, y: x * x + y[:, 0] * y[:, 0] + y[:, 1] * y[:, 1]
    eps = 1e-4
    dx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    dy0 = (f(x, y + [eps, 0]) - f(x, y - [eps, 0])) / (2 * eps)
    dy1 = (f(x, y + [0, eps]) - f(x, y - [0, eps])) / (2 * eps)
    dy = np.stack([dy0, dy1], axis=1)
    return dx, dy


shader(dispatch_size=N)
dx_result = np.zeros([N], dtype=np.float32)
dy_result = np.zeros([N, 2], dtype=np.float32)
dx_buffer.copy_to(dx_result)
dy_buffer.copy_to(dy_result)
dx_fd, dy_fd = fd()
dx_err = np.abs(dx_result - dx_fd)
dy_err = np.abs(dy_result - dy_fd)
print(f"dx ad:\n{dx_result}")
print(f"dx fd:\n{dx_fd}")
print(f"dx error (max = {np.max(dx_err)}):\n{dx_err}")
print(f"dy ad:\n{dy_result}")
print(f"dy fd:\n{dy_fd}")
print(f"dy error (max = {np.max(dy_err)}):\n{dy_err}")
