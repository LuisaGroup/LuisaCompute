#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::cuda {

struct NativeTileBinary {
    luisa::vector<std::byte> cubin;
    luisa::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty() && !cubin.empty(); }
};

[[nodiscard]] bool native_tile_ir_compiler_available() noexcept;

// No PTX handling, persistent cache or user archives. Only the explicit native
// Tile route calls this helper; the ordinary CUDA compiler remains unchanged.
[[nodiscard]] NativeTileBinary compile_native_tile_ir(
    const luisa::filesystem::path &runtime_directory, luisa::string_view source,
    uint32_t architecture, bool debug_info) noexcept;

}// namespace luisa::compute::cuda
