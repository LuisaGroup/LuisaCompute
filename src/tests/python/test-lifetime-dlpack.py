"""Host-only cross-language DLPack *lifetime / ownership* coverage.

Extends the ctypes capsule-interop style of ``test-dlpack-errors.py`` (which
covers error paths) to the object-lifetime questions that matter for the
Python<->C++ boundary, all driven through the ``lcapi`` ``to_dlpack`` /
``from_dlpack`` primitives and the CPython capsule ABI -- **no GPU device is
required**:

  (a) a capsule keeps the *originating* storage valid after the Python producer
      object is deleted (``del`` + ``gc``): consuming it later yields the same
      data pointer / values, and the producer only dies once the consumer
      releases the borrowed memory;
  (b) double-consumption of one capsule raises (the capsule name flips
      ``dltensor`` -> ``used_dltensor``);
  (c) the producer's deleter (``manager_ctx`` counter via ``ctypes``) runs
      *exactly once* even when the originating object dies first, and release is
      idempotent;
  (d) a foreign host-CPU tensor is *handled* at the capsule layer (imported with
      a CPU device) -- while a malformed foreign producer is *rejected* per the
      API contract without running the producer's deleter.

The GPU ``Buffer.from_dlpack`` device-mismatch rejection of a CPU tensor is
covered by the GPU test ``test-lifetime-numpy-view.py``.
"""

import ctypes as ct
import gc
import unittest
import weakref

import numpy as np

from luisa import lcapi

# ---- DLPack ABI structures (mirrors test-dlpack-errors.py) ----
class DLDevice(ct.Structure):
    _fields_ = [("device_type", ct.c_int), ("device_id", ct.c_int32)]


class DLDataType(ct.Structure):
    _fields_ = [("code", ct.c_uint8), ("bits", ct.c_uint8), ("lanes", ct.c_uint16)]


class DLTensor(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("device", DLDevice), ("ndim", ct.c_int32),
                ("dtype", DLDataType), ("shape", ct.POINTER(ct.c_int64)),
                ("strides", ct.POINTER(ct.c_int64)), ("byte_offset", ct.c_uint64)]


class DLManagedTensor(ct.Structure):
    pass


Deleter = ct.CFUNCTYPE(None, ct.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ct.c_void_p),
                            ("deleter", Deleter)]

capsule_new = ct.pythonapi.PyCapsule_New
capsule_new.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p]
capsule_new.restype = ct.py_object
capsule_valid = ct.pythonapi.PyCapsule_IsValid
capsule_valid.argtypes = [ct.py_object, ct.c_char_p]
capsule_valid.restype = ct.c_int

K_DLCPU = 1          # DLDeviceType::kDLCPU
K_DL_FLOAT = 2       # DLDataTypeCode::kDLFloat


class LifetimeDLPack(unittest.TestCase):

    def _host_capsule(self, values, counter, backing, keep):
        """Build a 1-D CPU float32 capsule whose ``manager_ctx`` is ``counter``
        (a ``c_long``) and whose deleter increments it and releases ``backing``
        (the ctypes buffers that hold the DLManagedTensor + payload alive).  The
        returned capsule must be consumed via ``lcapi.from_dlpack``; the producer
        memory lives only as long as ``backing``.

        ``keep`` must hold the returned object alive for the whole test: a struct
        built with ``from_buffer`` stores the ``deleter`` as a raw function
        pointer and cannot keep a *Python* reference to the ``CFUNCTYPE`` object,
        so the caller is responsible for preventing the callback from being
        collected while C++ may still invoke it (the classic ctypes callback
        lifetime pitfall)."""
        @Deleter
        def _deleter(_ptr):
            counter.value += 1
            backing.clear()

        keep.append(_deleter)
        buf = (ct.c_byte * ct.sizeof(DLManagedTensor))()
        t = DLManagedTensor.from_buffer(buf)
        backing.extend([buf, t])
        shape = (ct.c_int64 * 1)(len(values))
        payload = (ct.c_float * len(values))(*[float(v) for v in values])
        backing.extend([shape, payload])
        t.dl_tensor = DLTensor(ct.cast(payload, ct.c_void_p), DLDevice(K_DLCPU, 0), 1,
                               DLDataType(K_DL_FLOAT, 32, 1), shape, None, 0)
        t.manager_ctx = ct.cast(ct.pointer(counter), ct.c_void_p)
        t.deleter = _deleter
        return capsule_new(ct.addressof(t), b"dltensor", None)

    def test_a_capsule_keeps_storage_after_originating_object_dies(self):
        # Producer side: a numpy array stands in for the storage a Buffer owns.
        vals = np.arange(6, dtype=np.float32) * 1.5
        capsule = vals.__dlpack__()
        base_addr = vals.ctypes.data
        producer_ref = weakref.ref(vals)
        del vals
        gc.collect()
        # The capsule alone must keep the producer's storage alive (its DLPack
        # manager_ctx holds the reference): consuming it must NOT fail and must
        # report the very same data pointer -- never freed/reused memory.
        self.assertEqual(capsule_valid(capsule, b"dltensor"), 1)
        self.assertIsNotNone(producer_ref())
        lctype, size, addr, device, release = lcapi.from_dlpack(capsule)
        self.assertEqual(size, 6)
        self.assertEqual(addr, base_addr)
        self.assertEqual(list((ct.c_float * size).from_address(addr)),
                         [0.0, 1.5, 3.0, 4.5, 6.0, 7.5])
        self.assertEqual(device[1], 0)                      # device id 0
        self.assertEqual(int(device[0]), K_DLCPU)           # host CPU
        # Storage is owned by the consumer now; producer stays alive until release.
        self.assertIsNotNone(producer_ref())
        release()
        gc.collect()
        # Once the consumer releases the borrowed memory, the producer can die.
        self.assertIsNone(producer_ref())

    def test_a2_exported_capsule_pins_owner_until_second_from_dlpack_release(self):
        # Export side (the same code path Buffer.__dlpack__ uses): a capsule
        # obtained from an owner keeps that owner alive until it is consumed and
        # released -- so "consume via a (second) from_dlpack" is valid even after
        # the originating object is dropped.
        class Owner:
            pass

        owner = Owner()
        owner_ref = weakref.ref(owner)
        capsule = lcapi.to_dlpack(owner, 0, 0, lcapi.Type.from_("float"), "fallback", 0)
        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref())          # capsule pins the owner
        imported = lcapi.from_dlpack(capsule)
        self.assertEqual(imported[1:3], (0, 0))    # size/native_handle carried through
        imported[-1]()
        gc.collect()
        self.assertIsNone(owner_ref())             # freed only after release

    def test_b_double_consumption_of_one_capsule_raises(self):
        counter = ct.c_long(0)
        backing = []
        keep = []
        capsule = self._host_capsule([1.0, 2.0, 3.0, 4.0], counter, backing, keep)
        self.assertEqual(capsule_valid(capsule, b"dltensor"), 1)
        imported = lcapi.from_dlpack(capsule)      # first consumption succeeds
        self.assertEqual(capsule_valid(capsule, b"dltensor"), 0)
        self.assertEqual(capsule_valid(capsule, b"used_dltensor"), 1)
        # A second from_dlpack on the now-consumed capsule must fail.
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            lcapi.from_dlpack(capsule)
        self.assertEqual(counter.value, 0)         # deleter has not run yet
        imported[-1]()
        self.assertEqual(counter.value, 1)
        imported[-1]()                             # release is idempotent
        self.assertEqual(counter.value, 1)

    def test_c_producer_deleter_runs_exactly_once_even_if_producer_dies_first(self):
        counter = ct.c_long(0)
        backing = []
        keep = []
        capsule = self._host_capsule([7.0, 8.0, 9.0, 10.0], counter, backing, keep)
        imported = lcapi.from_dlpack(capsule)      # C++ takes ownership of manager
        del capsule                                # drop our only capsule reference
        gc.collect()
        # Deleter must NOT have run: the consumer owns the resource, storage is
        # still valid (producer memory is kept alive by `backing` until release).
        self.assertEqual(counter.value, 0)
        self.assertEqual(len(backing), 4)
        lctype, size, addr, device, release = imported
        self.assertEqual(list((ct.c_float * size).from_address(addr)),
                         [7.0, 8.0, 9.0, 10.0])
        release()                                  # consumer releases -> deleter runs once
        self.assertEqual(counter.value, 1)
        release()                                  # a second release is a no-op
        self.assertEqual(counter.value, 1)
        self.assertEqual(backing, [])              # producer freed by its own deleter

    def test_d_foreign_host_cpu_tensor_is_handled_or_rejected(self):
        # Handled at the lcapi (capsule-interop) layer: a foreign host-CPU numpy
        # tensor imports fine and reports a CPU device.
        arr = np.arange(5, dtype=np.float32)
        cpu_import = lcapi.from_dlpack(arr.__dlpack__())
        lctype, size, addr, device, release = cpu_import
        self.assertEqual(size, 5)
        self.assertEqual(int(device[0]), K_DLCPU)
        self.assertEqual(list((ct.c_float * size).from_address(addr)),
                         [0.0, 1.0, 2.0, 3.0, 4.0])
        # The (still-uncalled) deleter is the numpy capsule's own; consume another
        # CPU tensor and compare the device tuples to confirm both are the same
        # host-CPU device (avoids relying on enum/int comparison specifics).
        other = lcapi.from_dlpack((np.arange(2, dtype=np.float32)).__dlpack__())
        self.assertEqual(tuple(device), tuple(other[3]))
        release()
        other[-1]()

        # Rejected per contract: a malformed foreign producer (3-D shape) raises
        # and ownership stays with the producer (its deleter is NOT run, and the
        # capsule remains valid as an unconsumed ``dltensor``).
        releases = []
        deleter = Deleter(lambda _p: releases.append(True))
        shape = (ct.c_int64 * 3)(1, 2, 3)
        tensor = DLManagedTensor(
            DLTensor(None, DLDevice(K_DLCPU, 0), 3, DLDataType(K_DL_FLOAT, 32, 1), shape, None, 0),
            None, deleter)
        capsule = capsule_new(ct.addressof(tensor), b"dltensor", None)
        with self.assertRaisesRegex(RuntimeError, "unsupported shape"):
            lcapi.from_dlpack(capsule)
        self.assertEqual(releases, [])
        self.assertEqual(capsule_valid(capsule, b"dltensor"), 1)
        # keep `tensor`/`shape` referenced until here so the raw pointer stays live
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.dl_tensor.ndim, 3)


if __name__ == "__main__":
    unittest.main()
