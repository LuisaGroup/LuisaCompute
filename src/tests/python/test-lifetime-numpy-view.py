"""GPU ``Buffer`` <-> numpy round-trips and DLPack shared-storage *views*, with
focus on the lifetime coupling between numpy producers / borrowed storage and the
Python ``Buffer`` objects.

The API surface (see ``src/py/luisa/buffer.py``): ``Buffer.from_array`` /
``copy_from`` / ``copy_to`` / ``numpy`` move data, ``__dlpack__`` exports a
capsule that pins the GPU storage, and ``Buffer.from_dlpack`` imports a foreign
tensor by *borrowing* its memory (attaching a ``borrowed_deleter``).

Guarantees asserted (all deterministic, backend via ``sys.argv[1]``):
  (1) ``from_array`` -> ``numpy()`` round-trips exactly while both live;
  (2) ``copy_from`` reads the numpy buffer at ``execute``/``synchronize`` time
      (a *lazy* upload), so mutating the source after enqueue is visible to the
      kernel -- and the source must stay alive until then (the stream keeps it);
  (3) an imported DLPack view *aliases* the GPU storage: a write through the
      view is visible through the originating Buffer;
  (4) the borrowed view stays readable after the originating Python ``Buffer``
      is deleted -- the DLPack deleter, not ``Buffer.__del__``, governs the
      storage lifetime; explicit release (``borrowed_deleter``) is clean;
  (5) feeding a foreign host-CPU numpy tensor to ``Buffer.from_dlpack`` is
      REJECTED with a device mismatch (the API contract).
"""
from luisa import *
from luisa.types import *
import numpy as np
import gc
import sys

backend_name = None
if len(sys.argv) >= 2:
    backend_name = sys.argv[1]
init(backend_name=backend_name)

PASS = True


def check(cond, msg):
    global PASS
    print(("  [OK]  " if cond else "  [FAIL] ") + msg)
    PASS = PASS and bool(cond)


@func
def double(buf):
    i = dispatch_id().x
    buf.write(i, buf.read(i) * 2.0)


# (1) from_array / numpy round-trip ---------------------------------------------
print("[1] Buffer.from_array -> Buffer.numpy round-trip")
src = np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32)
b = Buffer.from_array(src)
execute()
synchronize()
check(np.array_equal(b.numpy(), src), "exact round-trip while both objects are alive")

# (2) lazy upload: mutate numpy AFTER copy_from, BEFORE synchronize --------------
print("[2] upload reads the numpy buffer at execute()/synchronize() time")
c = np.zeros(4, dtype=np.float32)
d = Buffer(4, float)
d.copy_from(c)                 # enqueue; the stream keeps a reference to `c`
c[:] = [10., 20., 30., 40.]    # mutate the source AFTER enqueue
double(d, dispatch_size=(4, 1, 1))
synchronize()
o2 = np.zeros(4, dtype=np.float32)
d.copy_to(o2)
check(np.allclose(o2, [20., 40., 60., 80.]),
      "kernel saw the mutated numpy values (numpy must stay alive until sync)")

# (3) DLPack view aliases the same GPU storage -----------------------------------
print("[3] imported DLPack view aliases the originating Buffer's GPU storage")
A = Buffer(4, float)
A.copy_from(np.array([1., 2., 3., 4.], dtype=np.float32))
execute()
synchronize()
cap = A.__dlpack__()
B = Buffer.from_dlpack(cap)     # borrows A's GPU storage (capsule keeps it alive)
check(B.size == 4 and B.dtype is float, "imported view metadata (size / dtype)")
double(B, dispatch_size=(4, 1, 1))
synchronize()
oA = np.zeros(4, dtype=np.float32)
A.copy_to(oA)                   # read through the ORIGINAL Buffer
check(np.allclose(oA, [2., 4., 6., 8.]),
      "write-through-B is visible via A (aliased GPU storage)")

# (4) borrowed view outlives the originating Python Buffer -----------------------
print("[4] borrowed view stays valid after the originating Buffer is dropped")
del A
gc.collect()                    # A.__del__ runs; the DLPack deleter still owns storage
oB = np.zeros(4, dtype=np.float32)
B.copy_to(oB)                   # the buffer.py guarantee: storage lifetime is bound to
                                # `borrowed_deleter`, not to the Buffer that produced it
check(np.allclose(oB, [2., 4., 6., 8.]),
      "borrowed view readable after originating Buffer died")
if B.borrowed_deleter is not None:
    B.borrowed_deleter()        # explicit, single release
B.borrowed_deleter = None       # avoid a double release in Buffer.__del__
gc.collect()
check(True, "explicit release of the borrowed view did not crash")

# (5) foreign host-CPU tensor rejected by Buffer.from_dlpack ---------------------
print("[5] host-CPU numpy tensor rejected by Buffer.from_dlpack (device mismatch)")
try:
    Buffer.from_dlpack(np.arange(3, dtype=np.float32))
    rejected = False
    print("  (did not raise -- unexpected)")
except RuntimeError as e:
    rejected = "mismatch" in str(e)
    print(f"  raised: {e}")
check(rejected, "Buffer.from_dlpack rejects a CPU tensor with a device mismatch")

print("PASS" if PASS else "FAIL")
if not PASS:
    sys.exit(1)
