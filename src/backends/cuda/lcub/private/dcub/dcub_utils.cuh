#pragma once
#include <luisa/backends/ext/cuda/lcub/dcub/dcub_common.h>
#include <cub/thread/thread_operators.cuh>

namespace luisa::compute::cuda::dcub {
template<typename F>
inline cudaError_t op_mapper(BinaryOperator op, F &&f) noexcept {
    switch (op) {
        case BinaryOperator::Max:
            return f(cub::Max{});
        case BinaryOperator::Min:
            return f(cub::Min{});
        default:
            return f(cub::Max{});
    }
}

struct Difference {
    template<typename T>
    __host__ __device__
        __forceinline__ T
        operator()(const T &lhs, const T &rhs) const noexcept { return lhs - rhs; }
};
}// namespace dcub