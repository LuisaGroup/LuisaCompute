#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "llvm_codegen/metal_codegen_llvm.h"

namespace luisa::compute {
class MeshFormat;
struct ShaderOption;
namespace xir {
class Module;
}// namespace xir
}// namespace luisa::compute

namespace luisa::compute::metal {

struct MetalAIRTarget {
    MetalAIRPlatform platform{MetalAIRPlatform::MACOS};
    MetalAIRVersion operating_system_version{14u, 0u, 0u};
    MetalAIRVersion sdk_version{14u, 0u, 0u};
};

[[nodiscard]] MetalAIRTarget metal_air_target_for_ios(
    MetalAIRVersion operating_system_version,
    MetalAIRVersion sdk_version) noexcept;

[[nodiscard]] MetalAIRTarget
metal_air_target_for_current_device() noexcept;

[[nodiscard]] MetalCodegenLLVMConfig metal_air_codegen_config(
    const MetalAIRTarget &target,
    luisa::string source_file = {}) noexcept;

struct MetalAIRCodegenResult {
    luisa::vector<std::byte> library;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
    luisa::vector<luisa::string> intersection_functions;
    size_t root_argument_size{0u};
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

[[nodiscard]] MetalAIRCodegenResult
metal_codegen_air(
    const xir::Module &module, const ShaderOption &option,
    const MetalAIRTarget &target) noexcept;

[[nodiscard]] MetalAIRRasterCodegenResult
metal_codegen_air(
    const xir::Module &vertex_module,
    const xir::Module &fragment_module,
    const MeshFormat &mesh_format,
    const ShaderOption &option) noexcept;

}// namespace luisa::compute::metal
