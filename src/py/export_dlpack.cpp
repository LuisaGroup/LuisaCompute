#include <pybind11/pybind11.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <exception>
#include <memory>
#include <optional>
#include <utility>
#include "dlpack.h"

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;

std::optional<DLDevice> get_dldevice(luisa::string_view backend_name, int32_t device_id) {
    DLDevice device{};
    if (backend_name == "cuda") {
        device.device_type = DLDeviceType::kDLCUDA;
    } else if (backend_name == "fallback") {
        device.device_type = DLDeviceType::kDLCPU;
    } else if (backend_name == "vk") {
        device.device_type = DLDeviceType::kDLVulkan;
    } else if (backend_name == "metal") {
        device.device_type = DLDeviceType::kDLMetal;
    } else {
        PyErr_SetString(PyExc_RuntimeError, luisa::format("backend unsupported by dlpack: {}", backend_name).c_str());
        return {};
    }
    device.device_id = device_id;
    return device;
}

PyObject *get_pydldevice(luisa::string_view backend_name, int32_t device_id) {
    auto device = get_dldevice(backend_name, device_id);
    if (!device) { return nullptr; }
    return py::make_tuple(device->device_type, device_id).release().ptr();
}

const Type *scalar_type_from_dldatatype(DLDataType datatype) {
    if (datatype.lanes != 1) {
        PyErr_SetString(PyExc_RuntimeError, "LC doesn't support lanes != 1");
        return nullptr;
    }
#define MATCH(CODE, BITS, REPR)                         \
    if (datatype.code == CODE && datatype.bits == BITS) \
        return Type::from(REPR);
    MATCH(kDLBool, 8, "bool")
    MATCH(kDLInt, 8, "byte")
    MATCH(kDLUInt, 8, "ubyte")
    MATCH(kDLInt, 16, "short")
    MATCH(kDLUInt, 16, "ushort")
    MATCH(kDLInt, 32, "int")
    MATCH(kDLUInt, 32, "uint")
    MATCH(kDLInt, 64, "long")
    MATCH(kDLUInt, 64, "ulong")
    MATCH(kDLFloat, 16, "half")
    MATCH(kDLFloat, 32, "float")
    MATCH(kDLFloat, 64, "double")
#undef MATCH
    PyErr_SetString(PyExc_RuntimeError, "unsupported DLDataType");
    return nullptr;
}

std::optional<DLDataType> get_dldatatype(const Type *type) {
    DLDataType datatype{};
    switch (type->element()->tag()) {
        case Type::Tag::BOOL:
            datatype.code = DLDataTypeCode::kDLBool;
            break;
        case Type::Tag::INT8:
        case Type::Tag::INT16:
        case Type::Tag::INT32:
        case Type::Tag::INT64:
            datatype.code = DLDataTypeCode::kDLInt;
            break;
        case Type::Tag::UINT8:
        case Type::Tag::UINT16:
        case Type::Tag::UINT32:
        case Type::Tag::UINT64:
            datatype.code = DLDataTypeCode::kDLUInt;
            break;
        case Type::Tag::FLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
            datatype.code = DLDataTypeCode::kDLFloat;
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, luisa::format("element type unsupported by dlpack: {}", type->element()->description()).c_str());
            return {};
    }
    datatype.bits = type->element()->size() * 8;
    // Note: multi-lane representations are not supported by pytorch
    datatype.lanes = 1;
    return datatype;
}

const Type *buffer_dtype_from_dltensor(const DLTensor &t) {
    auto fail = [](const char *message) -> const Type * {
        PyErr_SetString(PyExc_RuntimeError, message);
        return nullptr;
    };
    if (t.byte_offset != 0u) { return fail("non-zero byte offset"); }
    if (t.ndim < 1 || t.ndim > 3 || t.shape == nullptr) { return fail("invalid dimension"); }
    if (t.shape[0] < 0) { return fail("negative buffer size"); }
    auto scalar_type = scalar_type_from_dldatatype(t.dtype);
    if (scalar_type == nullptr) { return nullptr; }
    // Buffer of scalars.
    if (t.ndim == 1) {
        if (t.strides == nullptr || t.strides[0] == 1) { return scalar_type; }
        return fail("linear buffer must be compact");
    }
    // Buffer of vectors.
    auto n = t.shape[1];
    if (t.ndim == 2) {
        if (n == 3) {
            if (t.strides != nullptr && t.strides[1] == 1 && t.strides[0] == 4) {
                return Type::vector(scalar_type, n);
            }
            return fail("vector[3] must be 4-aligned");
        }
        if (n == 2 || n == 4) {
            if (t.strides == nullptr || (t.strides[1] == 1 && t.strides[0] == n)) {
                return Type::vector(scalar_type, n);
            }
            return fail("buffer of vector[2/4] must be compact");
        }
        return fail("buffer of unsupported vector size");
    }
    // Buffer of matrices.
    if (n != t.shape[2]) { return fail("unsupported shape"); }
    if (scalar_type != Type::from("float")) { return fail("lc matrix only supports float"); }
    if (n == 3) {
        if (t.strides != nullptr && t.strides[2] == 1 && t.strides[1] == 4 && t.strides[0] == 12) {
            return Type::matrix(n);
        }
        return fail("matrix[3] must be 4-aligned");
    }
    if (n == 2 || n == 4) {
        if (t.strides == nullptr || (t.strides[2] == 1 && t.strides[1] == n && t.strides[0] == n * n)) {
            return Type::matrix(n);
        }
        return fail("buffer of matrix[2/4] must be compact");
    }
    return fail("buffer of unsupported matrix size");
}

int32_t get_dlndim(const Type *type) {
    // TODO: support buffer of array
    if (type->is_scalar())
        return 1;
    if (type->is_vector())
        return 2;
    if (type->is_matrix())
        return 3;
    PyErr_SetString(PyExc_RuntimeError, luisa::format("element type unsupported by dlpack: {}", type->description()).c_str());
    return 0;
}

int64_t *get_dlshape(int64_t buffer_size, const Type *type) {
    int64_t *shape = new int64_t[get_dlndim(type)];
    shape[0] = buffer_size;
    if (type->is_vector())
        shape[1] = type->dimension();
    if (type->is_matrix())
        shape[1] = shape[2] = type->dimension();
    return shape;
}

int64_t *get_dlstrides(const Type *type) {
    int64_t *strides = new int64_t[get_dlndim(type)];
    if (type->is_scalar())
        strides[0] = 1;
    if (type->is_vector()) {
        auto n = type->dimension();
        strides[0] = n == 3 ? 4 : n;
        strides[1] = 1;
    }
    if (type->is_matrix()) {
        auto n = type->dimension();
        strides[0] = n == 3 ? 12 : n * n;
        strides[1] = n == 3 ? 4 : n;
        strides[2] = 1;
    }
    return strides;
}

static void cleanup(DLManagedTensor *t) {
    py::gil_scoped_acquire acquire;
    delete[] t->dl_tensor.shape;
    delete[] t->dl_tensor.strides;
    Py_DECREF(t->manager_ctx);
    delete t;
}

void DLPack_Capsule_Destructor(PyObject *o) {
    if (!PyCapsule_IsValid(o, "dltensor")) {
        // consumed capsules are renamed.
        // PyCapsule_Destructor calls deleter only for capsules whose name is "dltensor";
        return;
    }
    DLManagedTensor *dlmtensor = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(o, "dltensor"));
    dlmtensor->deleter(dlmtensor);
}

PyObject *to_py_dlpack(
    PyObject *owner,
    uint64_t native_handle,
    int64_t buffer_size,
    const Type *type,
    luisa::string_view backend_name,
    int32_t device_id) {
    if (type == nullptr || buffer_size < 0) {
        PyErr_SetString(PyExc_RuntimeError, "invalid buffer type or size");
        return nullptr;
    }
    auto device = get_dldevice(backend_name, device_id);
    if (!device) { return nullptr; }
    auto ndim = get_dlndim(type);
    if (ndim == 0) { return nullptr; }
    auto datatype = get_dldatatype(type);
    if (!datatype) { return nullptr; }
    auto t = std::make_unique<DLManagedTensor>();
    auto shape = std::unique_ptr<int64_t[]>{get_dlshape(buffer_size, type)};
    auto strides = std::unique_ptr<int64_t[]>{get_dlstrides(type)};
    t->dl_tensor.data = reinterpret_cast<void *>(native_handle);
    t->dl_tensor.device = *device;
    t->dl_tensor.ndim = ndim;
    t->dl_tensor.dtype = *datatype;
    t->dl_tensor.shape = shape.get();
    t->dl_tensor.strides = strides.get();
    t->dl_tensor.byte_offset = 0;
    t->manager_ctx = owner;
    t->deleter = cleanup;
    auto capsule = PyCapsule_New(t.get(), "dltensor", DLPack_Capsule_Destructor);
    if (capsule == nullptr) { return nullptr; }
    Py_INCREF(owner);
    static_cast<void>(shape.release());
    static_cast<void>(strides.release());
    static_cast<void>(t.release());
    return capsule;
}

PyObject *from_py_dlpack(PyObject *capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        PyErr_SetString(PyExc_RuntimeError, "DLTensor capsule is invalid or was already consumed!");
        return nullptr;
    }
    auto t = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (t == nullptr) { return nullptr; }
    auto dtype = buffer_dtype_from_dltensor(t->dl_tensor);
    if (dtype == nullptr) { return nullptr; }
    auto size = t->dl_tensor.shape[0];
    auto addr = reinterpret_cast<uint64_t>(t->dl_tensor.data);
    auto device = t->dl_tensor.device;
    auto deleter = [t]() mutable {
        if (auto tensor = std::exchange(t, nullptr); tensor != nullptr && tensor->deleter != nullptr) {
            tensor->deleter(tensor);
        }
    };
    auto result = py::make_tuple(dtype, size, addr, py::make_tuple(device.device_type, device.device_id), py::cpp_function(deleter));
    // Validation and Python allocations must succeed before ownership moves.
    // A rejected capsule remains consumable and still owns its tensor.
    if (PyCapsule_SetName(capsule, "used_dltensor") != 0) { return nullptr; }
    return result.release().ptr();
}

namespace {

// These C API entry points propagate Python's error indicator directly.
// A null py::object returned through m.def would instead become a pybind11
// return-conversion TypeError. Only third-party exceptions cross this boundary.
template<typename F>
PyObject *python_entry(F &&f) noexcept {
    try {
        return f();
    } catch (py::error_already_set &error) {
        error.restore();
    } catch (const py::cast_error &error) {
        PyErr_SetString(PyExc_TypeError, error.what());
    } catch (const std::bad_alloc &) {
        PyErr_NoMemory();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown DLPack binding failure");
    }
    return nullptr;
}

PyObject *to_dlpack_entry(PyObject *, PyObject *args) {
    return python_entry([&]() -> PyObject * {
        PyObject *owner = nullptr;
        PyObject *native_handle_object = nullptr;
        PyObject *size_object = nullptr;
        PyObject *type_object = nullptr;
        PyObject *backend_object = nullptr;
        PyObject *device_id_object = nullptr;
        if (!PyArg_ParseTuple(args, "OOOOOO:to_dlpack", &owner, &native_handle_object, &size_object, &type_object, &backend_object, &device_id_object)) {
            return nullptr;
        }
        auto native_handle = py::cast<uint64_t>(py::handle{native_handle_object});
        auto size = py::cast<int64_t>(py::handle{size_object});
        auto type = py::cast<const Type *>(py::handle{type_object});
        auto backend = py::cast<luisa::string_view>(py::handle{backend_object});
        auto device_id = py::cast<int32_t>(py::handle{device_id_object});
        return to_py_dlpack(owner, native_handle, size, type, backend, device_id);
    });
}

PyObject *device_entry(PyObject *, PyObject *args) {
    return python_entry([&]() -> PyObject * {
        PyObject *backend_object = nullptr;
        PyObject *device_id_object = nullptr;
        if (!PyArg_ParseTuple(args, "OO:to_dlpack_device", &backend_object, &device_id_object)) { return nullptr; }
        auto backend = py::cast<luisa::string_view>(py::handle{backend_object});
        auto device_id = py::cast<int32_t>(py::handle{device_id_object});
        return get_pydldevice(backend, device_id);
    });
}

PyObject *from_dlpack_entry(PyObject *, PyObject *args) {
    return python_entry([&]() -> PyObject * {
        PyObject *capsule = nullptr;
        if (!PyArg_ParseTuple(args, "O:from_dlpack", &capsule)) { return nullptr; }
        if (!PyCapsule_CheckExact(capsule)) {
            PyErr_SetString(PyExc_TypeError, "from_dlpack requires a capsule");
            return nullptr;
        }
        return from_py_dlpack(capsule);
    });
}

}// namespace

void export_dlpack(py::module &m) {
    py::enum_<DLDeviceType>(m, "DLDeviceType");
    static PyMethodDef methods[]{
        {"to_dlpack", to_dlpack_entry, METH_VARARGS, "Export a buffer as a DLPack capsule."},
        {"to_dlpack_device", device_entry, METH_VARARGS, "Return the DLPack device type and index."},
        {"from_dlpack", from_dlpack_entry, METH_VARARGS, "Consume a supported DLPack capsule."},
        {nullptr, nullptr, 0, nullptr}};
    LUISA_ASSERT(PyModule_AddFunctions(m.ptr(), methods) == 0, "Failed to register DLPack Python entry points.");
}
