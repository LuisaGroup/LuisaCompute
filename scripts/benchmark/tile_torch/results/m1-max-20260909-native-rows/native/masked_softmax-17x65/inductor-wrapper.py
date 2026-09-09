# AOT ID: ['8_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__softmax_arange_le_remainder_scalar_tensor_unsqueeze_where_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], r'''
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
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (17, 65), (65, 1), 'input')
        buf0 = empty_strided_cpu((17, 1), (1, 17), torch.float32)
        buf2 = empty_strided_cpu((17, 1), (1, 17), torch.float32)
        buf3 = empty_strided_cpu((17, 65), (65, 1), torch.float32)
        cpp_fused__softmax_arange_le_remainder_scalar_tensor_unsqueeze_where_0(arg0_1, buf0, buf2, buf3)
        del arg0_1
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((17, 65), (65, 1), device='cpu', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
