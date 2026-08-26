#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class MeshFormat;
struct ShaderOption;
namespace xir {
class Module;
}// namespace xir
}// namespace luisa::compute

namespace luisa::compute::metal {

struct MetalAIRCodegenResult {
    luisa::vector<std::byte> library;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
};

struct MetalAIRRasterCodegenResult {
    luisa::vector<std::byte> library;
    size_t root_argument_size{0u};
    uint32_t fragment_output_count{0u};
    luisa::string vertex_entry;
    luisa::string fragment_entry;
};

[[nodiscard]] MetalAIRCodegenResult
metal_codegen_air(const xir::Module &module, const ShaderOption &option) noexcept;

[[nodiscard]] MetalAIRRasterCodegenResult
metal_codegen_air(
    const xir::Module &vertex_module,
    const xir::Module &fragment_module,
    const MeshFormat &mesh_format,
    const ShaderOption &option) noexcept;

}// namespace luisa::compute::metal
