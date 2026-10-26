"""Regression: ``IndirectDispatchBuffer(size)`` construction + indirect dispatch.

Covers the Phase-3 fix in ``src/py/luisa/buffer.py``: the C++
``ResourceCreationInfo`` binding (commit e687e22cd, "Make py support external
resource") removed the old ``BufferCreationInfo`` accessors ``size_bytes()`` /
``element_stride()``, so ``IndirectDispatchBuffer.__init__`` must no longer call
them -- it now derives ``bytesize``/``stride`` from the record ``size`` alone
(mirroring ``src/dsl/dispatch_indirect.cpp``).  A plain ``IndirectDispatchBuffer(size)``
must therefore *construct* and drive a real indirect dispatch to completion.

Runs on the ``dx`` backend: the CUDA *debug* build lacks the device runtime
``cudadevrt`` (see scripts/run_py_tests.py -- the existing test-indirect.py is
overridden to dx for the same reason).  This mirrors test-indirect.py's usage:
a per-dispatch kernel ``atomic_fetch_add``s ``dispatch_size().x`` into ``buffer``
at ``kernel_id()``, so record ``i`` (with ``dispatch_size().x == i``) writes
``i * i``.  Expected: ``0 1 4 9 ... 225``.
"""
from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
import numpy as np
import sys

backend_name = "dx"  # forced: see docstring (cudadevrt missing on cuda debug build)
if len(sys.argv) >= 2:
    backend_name = sys.argv[1]
init(backend_name=backend_name)

dispatch_count = 16
kernel_dispatch_size = 64


@func
def dispatch(buffer):
    set_block_size(kernel_dispatch_size, 1, 1)
    old = buffer.atomic_fetch_add(kernel_id(), dispatch_size().x)


# The regression itself: constructing the indirect dispatch buffer from a plain
# int size must not touch the removed C++ accessors.
try:
    dispatch_buffer = IndirectDispatchBuffer(dispatch_count)
    constructed = True
    print(f"  [OK]  IndirectDispatchBuffer({dispatch_count}) constructed "
          f"(size={dispatch_buffer.size}, bytesize={dispatch_buffer.bytesize}, "
          f"stride={dispatch_buffer.stride})")
except Exception as e:
    constructed = False
    print(f"  [FAIL] IndirectDispatchBuffer raised {type(e).__name__}: {e}")


@func
def clear_indirect():
    set_block_size(1, 1, 1)
    dispatch_buffer.set_dispatch_count(dispatch_count)


@func
def emplace_indirect():
    set_block_size(dispatch_count, 1, 1)
    dispatch_buffer.set_kernel(
        dispatch_id().x,
        uint3(kernel_dispatch_size, 1, 1), uint3(dispatch_id().x, 1, 1), dispatch_id().x)


buffer = Buffer(dispatch_count, uint)
arr = np.zeros(dispatch_count, dtype=np.uint32)
out_arr = np.zeros(dispatch_count, dtype=np.uint32)
buffer.copy_from(arr)

clear_indirect(dispatch_size=(1, 1, 1))
emplace_indirect(dispatch_size=(dispatch_count, 1, 1))
dispatch(buffer, dispatch_size=dispatch_buffer)
buffer.copy_to(out_arr)

expected = (np.arange(dispatch_count, dtype=np.uint64) ** 2).astype(np.uint32)
values_ok = np.array_equal(out_arr, expected)
print(f"  result:   {' '.join(map(str, out_arr))}")
print(f"  expected: {' '.join(map(str, expected))}")
print(("  [OK]  " if values_ok else "  [FAIL] ") + "indirect dispatch produced i*i per record")

passed = constructed and values_ok
print("PASS" if passed else "FAIL")
if not passed:
    sys.exit(1)
