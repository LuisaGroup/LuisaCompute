"""GPU cross-language lifetime for ``Buffer``: the C++ side owns the resource
for the duration of any enqueued command, so a Python ``Buffer`` (and the numpy
producer it copied from) may be collected *before* the device work finishes.

Assertions (deterministic, backend via ``sys.argv[1]`` like the other GPU tests):
  (1) upload: the numpy *source* may be deleted + ``gc``'d right after
      ``copy_from`` -- the stream keeps it alive until ``synchronize``;
  (2) the Python ``Buffer`` may be dropped while its dispatch is pending and the
      command still completes without touching freed memory (no crash), with the
      ``weakref`` callback firing on collection;
  (3) allocator stress: a few hundred short-lived buffers churn without crashing;
  (4) a ``weakref`` with callback fires exactly once on ``Buffer`` deletion;
  (5) ordering on the single default stream is preserved (last enqueued upload
      wins).
"""
from luisa import *
from luisa.types import *
import numpy as np
import gc
import sys
import weakref

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
def scale2(buf):
    i = dispatch_id().x
    buf.write(i, buf.read(i) * 2.0)


@func
def fill_squares(buf):
    i = dispatch_id().x
    buf.write(i, float(i * i))


# (1) numpy source may die right after enqueue ---------------------------------
print("[1] numpy source collected after enqueue")
src = np.arange(8, dtype=np.float32)
buf = Buffer(8, float)
buf.copy_from(src)            # enqueue upload that references `src`
del src
gc.collect()                  # source dies before execute/synchronize
scale2(buf, dispatch_size=(8, 1, 1))
synchronize()                 # upload still valid: the stream holds the numpy buffer
out = np.zeros(8, dtype=np.float32)
buf.copy_to(out)
check(np.array_equal(out, np.arange(8, dtype=np.float32) * 2.0),
      "kernel produced correct values after the numpy source was dropped pre-sync")

# (2) buffer captured by a pending dispatch: drop it, then synchronize ---------
print("[2] Python Buffer dropped while its dispatch is pending")
pending = Buffer(10, float)
fill_squares(pending, dispatch_size=(10, 1, 1))   # async enqueue referencing pending.handle
fired = []
wr = weakref.ref(pending, lambda _r: fired.append(1))
del pending
gc.collect()
check(wr() is None, "Python Buffer object was collected")
check(len(fired) == 1, "weakref callback fired on Buffer deletion")
try:
    synchronize()             # C++ kept the resource alive for the pending command
    check(True, "synchronize() after dropping the Buffer did not crash")
except Exception as e:
    check(False, f"synchronize() raised {type(e).__name__}: {e}")

# (3) allocator stress ----------------------------------------------------------
print("[3] allocator stress: many short-lived buffers")
try:
    for n in range(300):
        t = Buffer(32, float)
        t.copy_from(np.full(32, float(n), dtype=np.float32))
        scale2(t, dispatch_size=(32, 1, 1))
        del t                 # dropped immediately; the enqueued work keeps it alive
        if n % 60 == 59:
            synchronize()
            gc.collect()
    synchronize()
    gc.collect()
    check(True, "300 short-lived buffers churned without crash")
except Exception as e:
    check(False, f"allocator stress raised {type(e).__name__}: {e}")

# (4) weakref callback fires exactly once ---------------------------------------
print("[4] weakref callback on Buffer fires exactly once")
w = Buffer(4, float)
hits = []
_ = weakref.ref(w, lambda _r: hits.append(1))
del w
gc.collect()
check(len(hits) == 1, f"weakref callback fired exactly once (got {len(hits)})")

# (5) ordering on the default stream --------------------------------------------
print("[5] default-stream ordering: last enqueued upload wins")
o = Buffer(1, float)
o.copy_from(np.zeros(1, dtype=np.float32))
for v in (1.0, 2.0, 3.0, 4.0, 5.0):
    o.copy_from(np.array([v], dtype=np.float32))   # 5 uploads, each keeps its temp alive
synchronize()
res = np.zeros(1, dtype=np.float32)
o.copy_to(res)
check(res[0] == 5.0, f"final value is the last enqueued upload (got {res[0]})")

print("PASS" if PASS else "FAIL")
if not PASS:
    sys.exit(1)
