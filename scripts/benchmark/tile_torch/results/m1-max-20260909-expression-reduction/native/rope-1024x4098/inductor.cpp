
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1024LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(2049LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(2048LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 4098LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 2049LL*x0), static_cast<int64_t>(4));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(2049LL + x1 + 4098LL*x0), static_cast<int64_t>(4));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 2049LL*x0), static_cast<int64_t>(4));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = tmp0 * tmp4;
                        auto tmp8 = tmp3 * tmp1;
                        auto tmp9 = tmp7 + tmp8;
                        tmp6.store(out_ptr0 + static_cast<int64_t>(x1 + 4098LL*x0));
                        tmp9.store(out_ptr1 + static_cast<int64_t>(x1 + 4098LL*x0));
                    }
                    if(C10_UNLIKELY(x1 >= static_cast<int64_t>(2048LL) && x1 < static_cast<int64_t>(2049LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 4098LL*x0), static_cast<int64_t>(1LL));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 2049LL*x0), static_cast<int64_t>(1LL));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(2049LL + x1 + 4098LL*x0), static_cast<int64_t>(1LL));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 2049LL*x0), static_cast<int64_t>(1LL));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = tmp0 * tmp4;
                        auto tmp8 = tmp3 * tmp1;
                        auto tmp9 = tmp7 + tmp8;
                        tmp6.store(out_ptr0 + static_cast<int64_t>(x1 + 4098LL*x0), static_cast<int64_t>(1LL));
                        tmp9.store(out_ptr1 + static_cast<int64_t>(x1 + 4098LL*x0), static_cast<int64_t>(1LL));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>
#include <cerrno>

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(result == -1 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}
template <> inline float parse_arg<float>(PyObject* args, size_t n) {
    auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
    if(result == -1.0 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected float arg");
    return static_cast<float>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(!PyTuple_CheckExact(args)) [[unlikely]]
            throw std::runtime_error("tuple args required");
        if(PyTuple_GET_SIZE(args) != 5) [[unlikely]]
            throw std::runtime_error("requires 5 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }

    char* endptr = nullptr;
    errno = 0;
    uintptr_t addr = std::strtoull(str_addr, &endptr, 10);
    if(errno != 0 || endptr == str_addr || addr == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to parse _TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
        return nullptr;
    }
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
