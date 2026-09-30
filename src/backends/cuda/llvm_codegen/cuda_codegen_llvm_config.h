//
// Created by mike on 9/19/25.
//

#pragma once

#include <cstdint>
#include <array>

#include <luisa/core/stl/string.h>
#include <luisa/ast/function.h>

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig {

    enum struct OptLevel : uint8_t {
        LEVEL_NONE = 0,
        LEVEL_LESS = 1,
        LEVEL_DEFAULT = 2,
        LEVEL_AGGRESSIVE = 3,
    };

    enum struct OutputFormat : uint8_t {
        PTX,
        OPTIX_IR,
    };

    luisa::string source_file{};
    luisa::span<const Function::Binding> bindings{};
    std::array<uint32_t, 3> block_size{};// {0, 0, 0} if dynamic, must be constant for now
    uint32_t cuda_arch{};
    OptLevel opt_level{OptLevel::LEVEL_AGGRESSIVE};
    bool enable_fast_math{true};
    bool enable_debug_info{false};
    bool requires_ray_tracing{false};
    bool requires_ray_query{false};
    OutputFormat output_format{OutputFormat::PTX};
};

}// namespace luisa::compute::cuda
