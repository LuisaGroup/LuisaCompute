#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/raster/vertex_attribute.h>

namespace llvm {
class LLVMContext;
class Module;
}// namespace llvm

namespace luisa::compute::xir {
class Argument;
class Module;
}// namespace luisa::compute::xir

namespace luisa::compute::metal {

struct MetalAIRVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

enum class MetalAIRKernelEntry : uint8_t {
    DIRECT,
    INDIRECT,
};

enum class MetalAIRProgram : uint8_t {
    COMPUTE,
    RASTER_VERTEX,
    RASTER_FRAGMENT,
};

struct MetalAIRRasterVertexAttribute {
    VertexAttributeType semantic;
    PixelFormat format;
};

struct MetalAIRRasterConfig {
    luisa::vector<const xir::Argument *> root_arguments;
    luisa::vector<MetalAIRRasterVertexAttribute> vertex_attributes;
    size_t stage_root_argument_offset{0u};
};

struct MetalCodegenLLVMConfig {
    MetalAIRVersion macos_version{14u, 0u, 0u};
    MetalAIRVersion sdk_version{14u, 0u, 0u};
    MetalAIRVersion air_version{2u, 6u, 0u};
    MetalAIRVersion metal_version{3u, 1u, 0u};
    luisa::string source_file;
    luisa::string native_include;
    bool enable_fast_math{false};
    MetalAIRKernelEntry entry{MetalAIRKernelEntry::DIRECT};
    MetalAIRProgram program{MetalAIRProgram::COMPUTE};
    MetalAIRRasterConfig raster;
};

struct MetalCodegenLLVMResult {
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> module;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
    size_t root_argument_size{0u};
    uint32_t fragment_output_count{0u};

    MetalCodegenLLVMResult() noexcept;
    MetalCodegenLLVMResult(MetalCodegenLLVMResult &&) noexcept;
    MetalCodegenLLVMResult &operator=(MetalCodegenLLVMResult &&) noexcept;
    ~MetalCodegenLLVMResult() noexcept;

    MetalCodegenLLVMResult(const MetalCodegenLLVMResult &) = delete;
    MetalCodegenLLVMResult &operator=(const MetalCodegenLLVMResult &) = delete;

    [[nodiscard]] luisa::string ir() const noexcept;
    [[nodiscard]] luisa::vector<std::byte> bitcode() const noexcept;
};

[[nodiscard]] MetalCodegenLLVMResult luisa_compute_metal_codegen_llvm(
    const xir::Module &xir_module,
    const MetalCodegenLLVMConfig &config) noexcept;

[[nodiscard]] bool luisa_compute_metal_codegen_llvm_supported(
    const xir::Module &xir_module,
    luisa::string *reason = nullptr) noexcept;

[[nodiscard]] bool luisa_compute_metal_codegen_llvm_supported(
    const xir::Module &xir_module,
    const MetalCodegenLLVMConfig &config,
    luisa::string *reason = nullptr) noexcept;

}// namespace luisa::compute::metal
