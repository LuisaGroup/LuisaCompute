
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(65LL);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(17LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(65LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64LL)))
                        {
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 65LL*x0), static_cast<int64_t>(4));
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                            auto tmp3 = x0;
                            auto tmp4 = c10::convert<int64_t>(tmp3);
                            auto tmp5 = at::vec::VectorizedN<int64_t,2>(tmp4);
                            auto tmp6 = at::vec::VecMask<int64_t,2>(tmp2 <= tmp5);
                            auto tmp8 = static_cast<float>(-1.0000000150474662e+30);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp6.template cast<float,1>());
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp10);
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(64LL) && x1 < static_cast<int64_t>(65LL)))
                        {
                            for (int64_t x1_tail = static_cast<int64_t>(64LL);x1_tail < static_cast<int64_t>(65LL); x1_tail++)
                            {
                                auto tmp5 = in_ptr0[static_cast<int64_t>(x1_tail + 65LL*x0)];
                                auto tmp0 = x1_tail;
                                auto tmp1 = c10::convert<int64_t>(tmp0);
                                auto tmp2 = x0;
                                auto tmp3 = c10::convert<int64_t>(tmp2);
                                auto tmp4 = tmp1 <= tmp3;
                                auto tmp6 = static_cast<float>(-1.0000000150474662e+30);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                            }
                        }
                    }
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(65LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64LL)))
                        {
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 65LL*x0), static_cast<int64_t>(4));
                            auto tmp11 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                            auto tmp3 = x0;
                            auto tmp4 = c10::convert<int64_t>(tmp3);
                            auto tmp5 = at::vec::VectorizedN<int64_t,2>(tmp4);
                            auto tmp6 = at::vec::VecMask<int64_t,2>(tmp2 <= tmp5);
                            auto tmp8 = static_cast<float>(-1.0000000150474662e+30);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp6.template cast<float,1>());
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 - tmp12;
                            auto tmp14 = tmp13.exp();
                            tmp14.store(local_buffer_data_0 + static_cast<int64_t>(x1));
                            tmp_acc0_vec = tmp_acc0_vec + tmp14;
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(64LL) && x1 < static_cast<int64_t>(65LL)))
                        {
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 65LL*x0), static_cast<int64_t>(1LL));
                            auto tmp11 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                            auto tmp3 = x0;
                            auto tmp4 = c10::convert<int64_t>(tmp3);
                            auto tmp5 = at::vec::VectorizedN<int64_t,2>(tmp4);
                            auto tmp6 = at::vec::VecMask<int64_t,2>(tmp2 <= tmp5);
                            auto tmp8 = static_cast<float>(-1.0000000150474662e+30);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp6.template cast<float,1>());
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 - tmp12;
                            auto tmp14 = tmp13.exp();
                            tmp14.store(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(1LL));
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp14, static_cast<int64_t>(1LL));
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(65LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(out_ptr2 + static_cast<int64_t>(x1 + 65LL*x0));
                    }
                    if(C10_UNLIKELY(x1 >= static_cast<int64_t>(64LL) && x1 < static_cast<int64_t>(65LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(1LL));
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(out_ptr2 + static_cast<int64_t>(x1 + 65LL*x0), static_cast<int64_t>(1LL));
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
        if(PyTuple_GET_SIZE(args) != 4) [[unlikely]]
            throw std::runtime_error("requires 4 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3)); Py_RETURN_NONE;
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
