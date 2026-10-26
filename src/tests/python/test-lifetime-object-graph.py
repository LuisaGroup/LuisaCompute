"""GPU destruction-order robustness across the object graph (device / stream /
buffers / images) plus in-process re-initialization.

This exercises cross-language lifetime ordering: resources created on a device
may be released in any order relative to each other and relative to the device
itself, and the process must not crash.  It also regression-tests the Phase-4
frontend fix in ``src/py/luisa/__init__.py`` ``del_device()``: it now resets the
module attribute to ``None`` (``globalvars.vars = None``) instead of deleting it,
so a device can be torn down and re-initialized *within the same process* --
previously the next ``init()`` raised ``AttributeError`` because its guard reads
``globalvars.vars is not None``.

Only asserts what the API guarantees: no crash on out-of-order teardown, a
surviving resource stays usable, and re-init produces a working device.
"""
from luisa import *
from luisa.types import *
import numpy as np
import gc
import sys
import luisa.globalvars as gv

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
def paint(img):
    c = dispatch_id().xy
    img.write(c, float4(0.25, 0.5, 0.75, 1.0))


@func
def write_idx(buf):
    i = dispatch_id().x
    buf.write(i, float(i))


# Build a small object graph: several buffers + images on the same device.
bufs = [Buffer(16, float) for _ in range(4)]
imgs = [Image2D(4, 4, 4, float, storage="BYTE") for _ in range(2)]
for b in bufs:
    b.copy_from(np.zeros(16, dtype=np.float32))
execute()
for im in imgs:
    paint(im, dispatch_size=(4, 4, 1))
write_idx(bufs[0], dispatch_size=(16, 1, 1))
synchronize()

o = np.zeros(16, dtype=np.float32)
bufs[0].copy_to(o)
check(np.allclose(o, np.arange(16, dtype=np.float32)), "graph built and read back deterministically")

# Nasty order A: drop images first (device still alive), then buffers.
del imgs
gc.collect()
check(True, "dropping all images while the device is alive did not crash")

del bufs[3]
gc.collect()
check(True, "dropping one buffer did not crash")

del bufs[2]
del bufs[1]
gc.collect()
check(True, "dropping more buffers did not crash")

# The one surviving buffer must still be fully usable.
bufs[0].copy_from(np.ones(16, dtype=np.float32))
synchronize()
r = np.zeros(16, dtype=np.float32)
bufs[0].copy_to(r)
check(np.allclose(r, 1.0), "surviving buffer still usable after siblings were collected")

del bufs[0]
gc.collect()
check(True, "dropping the last buffer did not crash")

# Device-last teardown, then in-process re-init (the del_device fix under test).
del_device()
gc.collect()
check(gv.vars is None and gv.device is None, "del_device() reset vars/device to None (attribute kept)")

reinit_ok = False
try:
    init(backend_name=backend_name)
    reinit_ok = gv.device is not None
except Exception as e:
    print(f"  re-init raised {type(e).__name__}: {e}")
check(reinit_ok, "re-init() in the same process created a fresh device")

if reinit_ok:
    # A kernel defined *after* re-init compiles against the new device; the
    # top-level `write_idx`/`paint` closures are deliberately not reused here
    # because their cached shader handles belong to the destroyed device.
    @func
    def write_idx2(buf):
        i = dispatch_id().x
        buf.write(i, float(i * 2))

    nb = Buffer(8, float)
    nb.copy_from(np.arange(8, dtype=np.float32))
    write_idx2(nb, dispatch_size=(8, 1, 1))   # new shader on the new device
    synchronize()
    no = np.zeros(8, dtype=np.float32)
    nb.copy_to(no)
    check(np.allclose(no, np.arange(8, dtype=np.float32) * 2.0),
          "new device usable after re-init (fresh kernel compiled + correct output)")
    del nb
    gc.collect()

del_device()
gc.collect()
print("PASS" if PASS else "FAIL")
if not PASS:
    sys.exit(1)
