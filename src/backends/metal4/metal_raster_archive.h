#pragma once

#include <cstdint>

#include <luisa/ast/usage.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/raster/raster_state.h>

namespace luisa::compute::metal {

inline constexpr uint64_t metal_raster_archive_checksum_seed = 0x8f4d1a23c7b965e0ull;

enum class MetalRasterArchiveStage : uint8_t {
    VERTEX,
    FRAGMENT,
};

struct MetalRasterArchiveArgument {
    luisa::string type;
    Usage usage;
    MetalRasterArchiveStage stage;
};

struct MetalRasterArchive {
    MeshFormat mesh_format;
    luisa::vector<MetalRasterArchiveArgument> arguments;
    luisa::vector<std::byte> library;
    size_t root_argument_size{0u};
    uint32_t fragment_output_count{0u};
};

[[nodiscard]] bool validate_metal_raster_mesh_format(
    const MeshFormat &mesh_format,
    luisa::string *reason = nullptr) noexcept;

[[nodiscard]] luisa::vector<std::byte>
serialize_metal_raster_archive(const MetalRasterArchive &archive) noexcept;

[[nodiscard]] luisa::optional<MetalRasterArchive>
deserialize_metal_raster_archive(luisa::span<const std::byte> data) noexcept;

}// namespace luisa::compute::metal
