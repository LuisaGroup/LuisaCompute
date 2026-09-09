
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(257LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(1538LL));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(384LL));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(1LL));
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1538LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1536LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1538LL*x0), static_cast<int64_t>(4));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0, &welford_helper0);
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(1536LL) && x1 < static_cast<int64_t>(1538LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1538LL*x0), static_cast<int64_t>(2LL));
                            masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, tmp0, static_cast<int64_t>(2LL), &masked_welford_helper0);
                        }
                    }
                }
                tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(1538LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1536LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1538LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(1538.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<int64_t>(x1 + 1538LL*x0));
                    }
                    if(C10_UNLIKELY(x1 >= static_cast<int64_t>(1536LL) && x1 < static_cast<int64_t>(1538LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1538LL*x0), static_cast<int64_t>(2LL));
                        auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(2LL));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(2LL));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(1538.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<int64_t>(x1 + 1538LL*x0), static_cast<int64_t>(2LL));
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
        if(PyTuple_GET_SIZE(args) != 6) [[unlikely]]
            throw std::runtime_error("requires 6 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4), parse_arg<float*>(args, 5)); Py_RETURN_NONE;
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
