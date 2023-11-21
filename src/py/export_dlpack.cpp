#include <pybind11/pybind11.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <string>
#include "dlpack.h"

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;


DLDevice get_dldevice(luisa::string_view backend_name, int32_t device_id) {
    DLDevice device;
    if (backend_name == "cuda") {
        device.device_type = DLDeviceType::kDLCUDA;
    }
    else if (backend_name == "cpu") {
        device.device_type = DLDeviceType::kDLCPU;
    }
    else if (backend_name == "vk") {
        device.device_type = DLDeviceType::kDLVulkan;
    }
    else if (backend_name == "metal") {
        device.device_type = DLDeviceType::kDLMetal;
    }
    else {
        throw std::runtime_error("backend unsupported by dlpack: " + std::string(backend_name)); // e.g. dx
    }
    device.device_id = device_id;
    return device;
}

auto get_pydldevice(luisa::string_view backend_name, int32_t device_id) {
    DLDevice device = get_dldevice(backend_name, device_id);
    return py::make_tuple(device.device_type, device_id);
}


const Type* scalar_type_from_dldatatype(DLDataType datatype)
{
    if (datatype.lanes != 1) {
        throw std::runtime_error("LC doesn't support lanes != 1");
    }
    #define MATCH(CODE,BITS,REPR) \
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
    throw std::runtime_error("unsupported DLDataType");
}


DLDataType get_dldatatype(const Type *type) {
    DLDataType datatype;
    switch (type->element()->tag())
    {
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
            throw std::runtime_error("element type unsupported by dlpack: " + std::string(type->element()->description())); // e.g. dx
            break;
    }
    datatype.bits = type->element()->size() * 8;
    // Note: multi-lane representations are not supported by pytorch
    datatype.lanes = 1;
    return datatype;
}


const Type* buffer_dtype_from_dltensor(DLTensor t)
{
    if (t.byte_offset) {
        throw std::runtime_error("non-zero byte offset");
    }
    const Type* scalar_type = scalar_type_from_dldatatype(t.dtype);
    // buffer of scalars
    if (t.ndim == 1) {
        if (!t.strides || t.strides[0]==1)
            return scalar_type;
        else
            throw std::runtime_error("linear buffer must be compact");
    }
    // buffer of vector
    if (t.ndim == 2) {
        auto n = t.shape[1];
        if (n==3) {
            if (t.strides && t.strides[1]==1 && t.strides[0]==4)
                return Type::vector(scalar_type, n);
            else
                throw std::runtime_error("vector[3] must be 4-aligned");
        }
        if (n==2 || n==4) {
            if (!t.strides || t.strides[1]==1 && t.strides[0]==n)
                return Type::vector(scalar_type, n);
            else
                throw std::runtime_error("buffer of vector[2/4] must be compact");
        }
        throw std::runtime_error("buffer of unsupported vector size");
    }
    // buffer of matrices
    if (t.ndim == 3) {
        auto n = t.shape[1];
        if (n != t.shape[2])
            std::runtime_error("unsupported shape");
        if (scalar_type != Type::from("float"))
            std::runtime_error("lc matrix only supports float");
        if (n==3) {
            if (t.strides && t.strides[2]==1 && t.strides[1]==4 && t.strides[0]==12)
                return Type::matrix(n);
            else
                throw std::runtime_error("vector[3] must be 4-aligned");
        }
        if (n==2 || n==4) {
            if (!t.strides || t.strides[2]==1 && t.strides[1]==n && t.strides[0]==n*n)
                return Type::matrix(n);
            else
                throw std::runtime_error("buffer of matrix[2/4] must be compact");
        }
        throw std::runtime_error("buffer of unsupported vector size");
    }
    throw std::runtime_error("invalid dimension");
}


int32_t get_dlndim(const Type *type) {
    // TODO: support buffer of array
    if (type->is_scalar())
        return 1;
    if (type->is_vector())
        return 2;
    if (type->is_matrix())
        return 3;
    throw std::runtime_error("element type unsupported by dlpack: " + std::string(type->element()->description())); // e.g. dx
}

int64_t* get_dlshape(int64_t buffer_size, const Type *type) {
    int64_t* shape = new int64_t[get_dlndim(type)];
    shape[0] = buffer_size;
    if (type->is_vector())
        shape[1] = type->dimension();
    if (type->is_matrix())
        shape[1] = shape[2] = type->dimension();
    return shape;
}

int64_t* get_dlstrides(const Type *type) {
    int64_t* strides = new int64_t[get_dlndim(type)];
    if (type->is_scalar())
        strides[0] = 1;
    if (type->is_vector()) {
        auto n = type->dimension();
        strides[0] = n==3? 4: n;
        strides[1] = 1;
    }
    if (type->is_matrix()) {
        auto n = type->dimension();
        strides[0] = n==3? 12: n*n;
        strides[1] = n==3? 4: n;
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
    DLManagedTensor* dlmtensor = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(o, "dltensor"));
    dlmtensor->deleter(dlmtensor);
}

py::capsule to_py_dlpack(
        const py::object &owner,
        uint64_t native_handle,
        int64_t buffer_size,
        const Type *type,
        luisa::string_view backend_name,
        int32_t device_id)
{
    DLManagedTensor* t = new DLManagedTensor();
    t->dl_tensor.data = reinterpret_cast<void*>(native_handle);
    t->dl_tensor.device = get_dldevice(backend_name, device_id);
    t->dl_tensor.ndim = get_dlndim(type);
    t->dl_tensor.dtype = get_dldatatype(type);
    t->dl_tensor.shape = get_dlshape(buffer_size, type);
    t->dl_tensor.strides = get_dlstrides(type);
    t->dl_tensor.byte_offset = 0;
    t->manager_ctx = owner.ptr();
    Py_INCREF(t->manager_ctx);
    t->deleter = cleanup;
    py::capsule capsule(t, "dltensor", DLPack_Capsule_Destructor);
    return capsule;
}


auto from_py_dlpack(const py::capsule &o) {
    const char *name = PyCapsule_GetName(o.ptr());
    if (strcmp(name, "dltensor") != 0)
        throw std::runtime_error("DLTensor capsule was already consumed!");
    DLManagedTensor* t = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(o.ptr(), "dltensor"));
    PyCapsule_SetName(o.ptr(), "used_dltensor"); // rename to make sure the capsule won't be consumed again
    const Type* dtype = buffer_dtype_from_dltensor(t->dl_tensor);
    int64_t size = t->dl_tensor.shape[0];
    uint64_t addr = reinterpret_cast<uint64_t>(t->dl_tensor.data);
    DLDevice device = t->dl_tensor.device;
    auto deleter = [t](){
        if (t->deleter) {
            t->deleter(t);
        }
    };
    return py::make_tuple(dtype, size, addr, py::make_tuple(device.device_type, device.device_id), py::cpp_function(deleter));
}


void export_dlpack(py::module &m) {
    py::enum_<DLDeviceType>(m, "DLDeviceType");
    m.def("to_dlpack", to_py_dlpack);
    m.def("to_dlpack_device", get_pydldevice);
    m.def("from_dlpack", from_py_dlpack);
}

