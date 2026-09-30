#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/curve_basis.h>
#include <luisa/ast/usage.h>

namespace luisa::compute::cuda {

struct CUDAShaderMetadata {

    enum struct Kind : uint8_t {
        UNKNOWN,
        COMPUTE,
        RAY_TRACING,
        TILE,
    };

    enum struct CodeFormat : uint8_t {
        PTX,
        OPTIX_IR,
    };

    uint64_t checksum;
    CurveBasisSet curve_bases;

    Kind kind;
    CodeFormat code_format{CodeFormat::PTX};
    bool enable_debug;
    bool requires_trace_closest;
    bool requires_trace_any;
    bool requires_ray_query;
    uint32_t ray_query_payload_count{2u};// AST uses 2 words; LLVM declares its actual module maximum
    bool requires_printing;
    bool requires_motion_blur;
    uint max_register_count;
    uint3 block_size;
    luisa::vector<luisa::string> argument_types;
    luisa::vector<Usage> argument_usages;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;

    [[nodiscard]] auto operator==(const CUDAShaderMetadata &rhs) const noexcept {
        return checksum == rhs.checksum &&
               curve_bases == rhs.curve_bases &&
               kind == rhs.kind &&
               code_format == rhs.code_format &&
               enable_debug == rhs.enable_debug &&
               requires_trace_closest == rhs.requires_trace_closest &&
               requires_trace_any == rhs.requires_trace_any &&
               requires_ray_query == rhs.requires_ray_query &&
               ray_query_payload_count == rhs.ray_query_payload_count &&
               requires_printing == rhs.requires_printing &&
               requires_motion_blur == rhs.requires_motion_blur &&
               max_register_count == rhs.max_register_count &&
               all(block_size == rhs.block_size) &&
               argument_types == rhs.argument_types &&
               argument_usages == rhs.argument_usages &&
               format_types == rhs.format_types;
    }
};

// Checks the serialization envelope, not the semantics of PTX or bitcode.
[[nodiscard]] bool cuda_shader_code_matches_format(luisa::span<const std::byte> code,
                                                   CUDAShaderMetadata::CodeFormat format) noexcept;

[[nodiscard]] luisa::string serialize_cuda_shader_metadata(const CUDAShaderMetadata &metadata) noexcept;
[[nodiscard]] luisa::optional<CUDAShaderMetadata> deserialize_cuda_shader_metadata(luisa::string_view metadata) noexcept;

}// namespace luisa::compute::cuda
