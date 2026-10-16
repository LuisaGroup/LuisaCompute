# AOT ID: ['23_inference']
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


cpp_fused_add_mul_split_sub_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*', 'float*'], r'''
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((1024, 4098), (1024, 2049), (1024, 2049)), ((4098, 1), (2049, 1), (2049, 1)), 'input')
        buf2 = empty_strided_cpu((1024, 4098), (4098, 1), torch.float32)
        buf0 = reinterpret_tensor(buf2, (1024, 2049), (4098, 1), 0)  # alias
        buf1 = reinterpret_tensor(buf2, (1024, 2049), (4098, 1), 2049)  # alias
        cpp_fused_add_mul_split_sub_0(arg0_1, arg1_1, arg2_1, buf0, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1024, 4098), (4098, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 2049), (2049, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 2049), (2049, 1), device='cpu', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
