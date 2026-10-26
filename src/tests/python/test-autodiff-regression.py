"""Regression: autodiff forward + ``backward()`` on a small polynomial (dx).

Covers the Phase-3 fix in ``src/py/luisa/autodiff.py``: ``backward(z)`` must
attach the seed via ``GRADIENT_MARKER`` and emit a no-operand ``BACKWARD`` call
(the CUDA autodiff path additionally needs the experimental LLVM codegen absent
from this debug build, so this runs on ``dx`` -- exactly why the existing
test-autodiff*.py files are overridden to ``dx`` in scripts/run_py_tests.py).

For ``f(x) = x**3 + 2*x**2 + 5*x`` the analytic gradient is ``3x**2 + 4x + 5``.
We run the device ``backward()`` per element and compare against *central
finite differences* on the host (the same technique test-autodiff.py uses), then
against the analytic derivative.
"""
from luisa import *
from luisa.autodiff import *
from luisa.types import *
import numpy as np
import sys

backend_name = "dx"  # forced: see docstring (CUDA autodiff LLVM codegen not built)
if len(sys.argv) >= 2:
    backend_name = sys.argv[1]
init(backend_name=backend_name)

N = 256
x_buffer = Buffer(N, float)
dx_buffer = Buffer(N, float)

x_values = np.linspace(0.5, 3.0, N, dtype=np.float32)
x_buffer.copy_from(x_values)


@func
def f(x: float):
    return x * x * x + 2.0 * x * x + 5.0 * x


@func
def shader():
    i = dispatch_id().x
    x = x_buffer.read(i)
    with autodiff():
        requires_grad(x)
        z = f(x)
        backward(z)
        dx = grad(x)
    dx_buffer.write(i, dx)


shader(dispatch_size=(N, 1, 1))
dx_result = np.zeros(N, dtype=np.float32)
dx_buffer.copy_to(dx_result)

# central finite differences on the host (float64)
x64 = x_values.astype(np.float64)
fval = lambda x: x ** 3 + 2.0 * x ** 2 + 5.0 * x
eps = 1e-4
dx_fd = (fval(x64 + eps) - fval(x64 - eps)) / (2.0 * eps)
dx_analytic = 3.0 * x64 ** 2 + 4.0 * x64 + 5.0

ad = dx_result.astype(np.float64)
err_fd = np.max(np.abs(ad - dx_fd))
err_ana = np.max(np.abs(ad - dx_analytic))
rel = np.max(np.abs(ad - dx_fd) / np.maximum(1.0, np.abs(dx_fd)))
print(f"  max abs error vs finite differences : {err_fd:.3e}")
print(f"  max abs error vs analytic gradient  : {err_ana:.3e}")
print(f"  max rel error vs finite differences : {rel:.3e}")

fd_ok = err_fd < 1e-2
ana_ok = err_ana < 1e-2
print(("  [OK]  " if fd_ok else "  [FAIL] ") + "backward() gradient matches finite differences")
print(("  [OK]  " if ana_ok else "  [FAIL] ") + "backward() gradient matches analytic derivative")

passed = fd_ok and ana_ok
print("PASS" if passed else "FAIL")
if not passed:
    sys.exit(1)
