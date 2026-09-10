"""Host-only DLPack error and ownership coverage; requires built Python bindings."""

import ctypes as ct
import gc
import unittest
import weakref

from luisa import lcapi


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


class DLPackErrors(unittest.TestCase):
    def test_device_error_and_bytes_backend(self):
        with self.assertRaisesRegex(RuntimeError, "backend unsupported"):
            lcapi.to_dlpack_device("unsupported", 0)
        self.assertEqual(lcapi.to_dlpack_device(b"fallback", 0),
                         lcapi.to_dlpack_device("fallback", 0))

    def test_export_rejects_invalid_input(self):
        dtype = lcapi.Type.from_("float")
        with self.assertRaisesRegex(RuntimeError, "invalid buffer type or size"):
            lcapi.to_dlpack(object(), 0, -1, dtype, "fallback", 0)
        with self.assertRaises(TypeError):
            lcapi.to_dlpack(object(), -1, 0, dtype, "fallback", 0)
        with self.assertRaises(TypeError):
            lcapi.to_dlpack_device("fallback", 1 << 40)

    def test_failed_import_keeps_capsule_ownership(self):
        for shape_values, datatype, message in [
            ((1, 2, 3), DLDataType(2, 32, 1), "unsupported shape"),
            ((1, 2, 2), DLDataType(0, 32, 1), "matrix only supports float"),
            ((1,), DLDataType(2, 32, 2), "lanes != 1"),
        ]:
            with self.subTest(message=message):
                shape = (ct.c_int64 * len(shape_values))(*shape_values)
                releases = []
                deleter = Deleter(lambda _: releases.append(True))
                tensor = DLManagedTensor(DLTensor(None, DLDevice(1, 0), len(shape_values),
                                                   datatype, shape, None, 0), None, deleter)
                capsule = capsule_new(ct.addressof(tensor), b"dltensor", None)
                with self.assertRaisesRegex(RuntimeError, message):
                    lcapi.from_dlpack(capsule)
                self.assertEqual(capsule_valid(capsule, b"dltensor"), 1)
                self.assertEqual(releases, [])
                # Repair and consume the same capsule after the rejected import.
                tensor.dl_tensor.ndim = 1
                tensor.dl_tensor.dtype = DLDataType(2, 32, 1)
                imported = lcapi.from_dlpack(capsule)
                self.assertEqual(imported[1], 1)
                self.assertEqual(capsule_valid(capsule, b"used_dltensor"), 1)
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    lcapi.from_dlpack(capsule)
                imported[-1]()
                imported[-1]()
                self.assertEqual(releases, [True])

    def test_export_keeps_owner_until_release(self):
        class Owner:
            pass

        owner = Owner()
        owner_ref = weakref.ref(owner)
        capsule = lcapi.to_dlpack(owner, 0, 0, lcapi.Type.from_("float"), "fallback", 0)
        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref())
        imported = lcapi.from_dlpack(capsule)
        self.assertEqual(imported[1:3], (0, 0))
        imported[-1]()
        gc.collect()
        self.assertIsNone(owner_ref())


if __name__ == "__main__":
    unittest.main()
