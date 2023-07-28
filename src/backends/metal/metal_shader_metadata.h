#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/optional.h>
#include <luisa/ast/usage.h>

namespace luisa::compute::metal {

struct MetalShaderMetadata {

    uint64_t checksum;
    uint3 block_size;
    luisa::vector<luisa::string> argument_types;
    luisa::vector<Usage> argument_usages;

    [[nodiscard]] auto operator==(const MetalShaderMetadata &rhs) const noexcept {
        return checksum == rhs.checksum &&
               all(block_size == rhs.block_size) &&
               argument_types == rhs.argument_types &&
               argument_usages == rhs.argument_usages;
    }
};

[[nodiscard]] luisa::string serialize_metal_shader_metadata(const MetalShaderMetadata &metadata) noexcept;
[[nodiscard]] luisa::optional<MetalShaderMetadata> deserialize_metal_shader_metadata(luisa::string_view metadata) noexcept;

}// namespace luisa::compute::metal

