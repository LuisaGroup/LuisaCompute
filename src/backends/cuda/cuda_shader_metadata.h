#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/optional.h>
#include <luisa/ast/usage.h>

namespace luisa::compute::cuda {

struct CUDAShaderMetadata {

    enum struct Kind : uint32_t {
        UNKNOWN,
        COMPUTE,
        RAY_TRACING,
    };

    uint64_t checksum;
    Kind kind;
    bool enable_debug;
    bool requires_trace_closest;
    bool requires_trace_any;
    bool requires_ray_query;
    uint3 block_size;
    luisa::vector<luisa::string> argument_types;
    luisa::vector<Usage> argument_usages;

    [[nodiscard]] auto operator==(const CUDAShaderMetadata &rhs) const noexcept {
        return checksum == rhs.checksum &&
               kind == rhs.kind &&
               enable_debug == rhs.enable_debug &&
               requires_trace_closest == rhs.requires_trace_closest &&
               requires_trace_any == rhs.requires_trace_any &&
               requires_ray_query == rhs.requires_ray_query &&
               all(block_size == rhs.block_size) &&
               argument_types == rhs.argument_types &&
               argument_usages == rhs.argument_usages;
    }
};

[[nodiscard]] luisa::string serialize_cuda_shader_metadata(const CUDAShaderMetadata &metadata) noexcept;
[[nodiscard]] luisa::optional<CUDAShaderMetadata> deserialize_cuda_shader_metadata(luisa::string_view metadata) noexcept;

}// namespace luisa::compute::cuda
