"""Regression: macro-branch folding and `set_block_size` on named int constants.

Covers the two Phase-3 frontend fixes in ``src/py/luisa/astbuilder.py`` /
``src/py/luisa/builtin.py``:

  * Fix 1 -- ``if <python-constant>:`` is resolved on the host by ``host_eval``
    and only the taken branch is emitted *before* name-checking, so the dead
    branch may contain illegal / undeclared identifiers (``vendor_ext.``/ghost
    names) and still compile.
  * Fix 2 -- ``set_block_size`` accepts module-level int *constants* (names that
    resolve to host ints via ``host_eval``), not only numeric literals.

``build_While`` does NOT host-fold (it always emits a real loop), so the ``while``
sub-test exercises a genuine *device-bool* loop instead -- only asserting what
the implementation actually supports.
"""
from luisa import *
from luisa.types import *
import numpy as np
import sys

backend_name = None
if len(sys.argv) >= 2:
    backend_name = sys.argv[1]
init(backend_name=backend_name)

# Module-level host constants consumed by the kernel below.
BLOCK = 16            # fix 2: a named int constant passed to set_block_size
USE_VENDOR_EXT = False  # fix 1: a named host constant driving a macro branch

PASS = True


def check(cond, msg):
    global PASS
    print(("  [OK]  " if cond else "  [FAIL] ") + msg)
    PASS = PASS and bool(cond)


# ---- Fix 1 + Fix 2 together: false macro branch with an illegal dead branch,
#      and set_block_size taking the module-level BLOCK constant. -------------
@func
def macro_false(buf):
    set_block_size(BLOCK, 1, 1)              # fix 2: named int constant
    i = dispatch_id().x
    if USE_VENDOR_EXT:                        # folds to False -> dead branch erased
        vendor_ext.weird_instruction(ghost_name)   # illegal; must never be name-checked
    buf.write(i, uint(i + 1))


buf = Buffer(BLOCK, uint)
macro_false(buf, dispatch_size=(BLOCK, 1, 1))
synchronize()
out = np.zeros(BLOCK, dtype=np.uint32)
buf.copy_to(out)
check(np.array_equal(out, np.arange(1, BLOCK + 1, dtype=np.uint32)),
      "if <False> with an illegal dead branch compiled and ran; set_block_size(BLOCK,..) accepted")


# ---- The true side: `if True:` keeps only the live branch, drops the else. ---
@func
def macro_true(buf):
    i = dispatch_id().x
    if True:                                  # folds to True
        buf.write(i, uint(i * 3))
    else:
        also.illegal(dead_symbol)             # erased before name-check


buf2 = Buffer(8, uint)
macro_true(buf2, dispatch_size=(8, 1, 1))
synchronize()
out2 = np.zeros(8, dtype=np.uint32)
buf2.copy_to(out2)
check(np.array_equal(out2, (np.arange(8) * 3).astype(np.uint32)),
      "if <True> drops the illegal else branch")


# ---- A folded constant expression as the test (`(1+1)==3`), like test-buffer. -
@func
def macro_expr(buf):
    i = dispatch_id().x
    flag = ((1 + 1) == 3)                     # host-constant False (computed)
    if flag:
        ridiculous.wtf_is_this(nothing)       # erased
    buf.write(i, uint(i * i))


buf3 = Buffer(8, uint)
macro_expr(buf3, dispatch_size=(8, 1, 1))
synchronize()
out3 = np.zeros(8, dtype=np.uint32)
buf3.copy_to(out3)
check(np.array_equal(out3, (np.arange(8) ** 2).astype(np.uint32)),
      "if <computed host constant> folded and dead branch erased")


# ---- `while` is NOT a macro branch (build_While always emits a real loop, it
#      never host-folds a constant condition), so `while False:` does NOT erase
#      its body.  We only assert what actually works: a plain device-bool loop.
@func
def while_tri(out_buf):
    i = dispatch_id().x
    n = int(i)
    s = int(0)
    while n > 0:                              # device bool condition -> real loop
        s = s + n
        n = n - 1
    out_buf.write(i, s)


N = 8
outw = Buffer(N, int)
while_tri(outw, dispatch_size=(N, 1, 1))
synchronize()
ow = np.zeros(N, dtype=np.int32)
outw.copy_to(ow)
expected = np.array([sum(range(v + 1)) for v in range(N)], dtype=np.int32)
check(np.array_equal(ow, expected),
      "device-bool `while` loop runs (host `while` is not macro-folded; only real loops asserted)")

print("PASS" if PASS else "FAIL")
if not PASS:
    sys.exit(1)
