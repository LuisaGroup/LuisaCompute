"""Regression: large integer literals round-trip exactly through buffers.

Covers the Phase-3 fix in ``src/py/luisa/astbuilder.py``
(``_int_literal_dtype`` / ``_build_int_literal``): the C++ ``literal`` binding's
``std::variant`` greedily captures python ``int`` objects through its ``float``
(binary32) alternative, so a value not exactly representable in 24 mantissa bits
used to be silently rounded (e.g. ``2654435800`` -> ``2654435840``) and
``0xFFFFFFFF`` overflowed and aborted.  The fix keeps the direct ``literal`` call
for float-exact values and otherwise synthesises the exact value from base-2**16
device-integer digits (constant-folded by codegen).

The literals are written with the buffer element dtype ``_int_literal_dtype``
chooses for each value (int / uint / long / ulong); the exact numeric value is
read back with numpy and compared for equality.  ``2**40+1`` / ``-2**40`` are
spelled as decimal *literals* (an ``**`` BinOp is a device float power, not an
integer literal).  Ordinary small literals are included to confirm no behaviour
change.
"""
from luisa import *
from luisa.types import *
import numpy as np
import sys

backend_name = None
if len(sys.argv) >= 2:
    backend_name = sys.argv[1]
init(backend_name=backend_name)

PASS = True


def check(got, expected, label):
    global PASS
    ok = int(got) == int(expected)
    PASS = PASS and ok
    print(("  [OK]  " if ok else "  [FAIL] ") + f"{label}: expected {expected}, got {got}")


@func
def literals(bi, bu, bl, bul):
    # int32-fit literals stay dtype `int`
    bi.write(0, 0)
    bi.write(1, 1)
    bi.write(2, 42)
    bi.write(3, -7)
    bi.write(4, 2147483647)     # int32 max
    bi.write(5, -2147483647)    # int32 min+1 (as USub of an int literal)
    # > int32 but <= uint32 -> dtype `uint`
    bu.write(0, 0xFFFFFFFF)     # 4294967295 -- previously aborted in the binding
    bu.write(1, 2654435800)     # not float32-exact -- previously corrupted
    # > uint32 but fits int64 -> dtype `long`
    bl.write(0, 1099511627777)  # 2**40 + 1
    bl.write(1, -1099511627776)  # -(2**40)
    # > int64 max but fits uint64 -> dtype `ulong`
    bul.write(0, 18446744073709551615)  # 2**64 - 1
    bul.write(1, 9223372036854775808)   # 2**63

    # exact value round-trips regardless of how it was synthesised
    bi.write(6, 1000000)
    bu.write(2, 4000000000)


bi = Buffer(7, int)
bu = Buffer(3, uint)
bl = Buffer(2, long)
bul = Buffer(2, ulong)
literals(bi, bu, bl, bul, dispatch_size=(1, 1, 1))
synchronize()

oi = np.zeros(7, dtype=np.int32); bi.copy_to(oi)
ou = np.zeros(3, dtype=np.uint32); bu.copy_to(ou)
ol = np.zeros(2, dtype=np.int64); bl.copy_to(ol)
oul = np.zeros(2, dtype=np.uint64); bul.copy_to(oul)

check(oi[0], 0, "int 0")
check(oi[1], 1, "int 1")
check(oi[2], 42, "int 42")
check(oi[3], -7, "int -7")
check(oi[4], 2147483647, "int 2**31-1")
check(oi[5], -2147483647, "int -(2**31-1)")
check(oi[6], 1000000, "int 1000000")
check(ou[0], 0xFFFFFFFF, "uint 0xFFFFFFFF")
check(ou[1], 2654435800, "uint 2654435800")
check(ou[2], 4000000000, "uint 4000000000")
check(ol[0], 2 ** 40 + 1, "long 2**40+1")
check(ol[1], -(2 ** 40), "long -(2**40)")
check(oul[0], 2 ** 64 - 1, "ulong 2**64-1")
check(oul[1], 2 ** 63, "ulong 2**63")

print("PASS" if PASS else "FAIL")
if not PASS:
    sys.exit(1)
