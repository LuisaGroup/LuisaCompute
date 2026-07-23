#pragma once
#include <luisa/backends/ext/cuda/lcub/dcub/dcub_common.h>
#include <cub/version.cuh>
#include <cub/thread/thread_operators.cuh>

namespace luisa::compute::cuda::dcub {

namespace cub_compat {

#if CUB_VERSION < 300000
using MaxOp = ::cub::Max;
using MinOp = ::cub::Min;
using EqualityOp = ::cub::Equality;
#else
struct MaxOp {
    template<typename T>
    __host__ __device__ __forceinline__ T
    operator()(const T &a, const T &b) const noexcept { return a < b ? b : a; }
};
struct MinOp {
    template<typename T>
    __host__ __device__ __forceinline__ T
    operator()(const T &a, const T &b) const noexcept { return b < a ? b : a; }
};
struct EqualityOp {
    template<typename T>
    __host__ __device__ __forceinline__ bool
    operator()(const T &a, const T &b) const noexcept { return a == b; }
};
#endif

}// namespace cub_compat

template<typename F>
inline cudaError_t op_mapper(BinaryOperator op, F &&f) noexcept {
    switch (op) {
        case BinaryOperator::Max:
            return f(cub_compat::MaxOp{});
        case BinaryOperator::Min:
            return f(cub_compat::MinOp{});
        default:
            return f(cub_compat::MaxOp{});
    }
}

struct Difference {
    template<typename T>
    __host__ __device__
        __forceinline__ T
        operator()(const T &lhs, const T &rhs) const noexcept { return lhs - rhs; }
};
}// namespace luisa::compute::cuda::dcub
