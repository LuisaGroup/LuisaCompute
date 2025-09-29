//
// Created by mike on 9/19/25.
//

#pragma once

#include <cstdint>

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig {

    enum struct OptLevel : uint8_t {
        LEVEL_NONE = 0,
        LEVEL_LESS = 1,
        LEVEL_DEFAULT = 2,
        LEVEL_AGGRESSIVE = 3,
    };

    uint32_t cuda_arch{};
    OptLevel opt_level{OptLevel::LEVEL_AGGRESSIVE};
    bool enable_fast_math{true};
    bool enable_debug_info{false};
    bool enable_ray_tracing{false};
};

}// namespace luisa::compute::cuda
