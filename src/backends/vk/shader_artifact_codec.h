#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <luisa/core/binary_io.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/vstl/md5.h>

#include "shader_binary_contract.h"
#include "shader_interface_plan.h"

namespace lc::vk::detail {

enum class ShaderArtifactSpirvStage : uint8_t {
    COMPUTE,
    VERTEX,
    FRAGMENT
};

enum class ShaderArtifactCodecError : uint8_t {
    NONE,
    TRUNCATED_HEADER,
    INVALID_HEADER,
    IDENTITY_MISMATCH,
    CODEGEN_DIALECT_MISMATCH,
    INVALID_SECTION_SIZES,
    SECTION_DIGEST_MISMATCH,
    INVALID_SHADER_INTERFACE,
    INVALID_PRINTER_PAYLOAD,
    INVALID_SPIRV,
    NATIVE_FEATURE_MASK_UNAVAILABLE,
    NATIVE_FEATURE_MASK_MISMATCH
};

[[nodiscard]] const char *shader_artifact_codec_error_name(
    ShaderArtifactCodecError error) noexcept;

struct SpirvArtifactModuleValidationResult {
    ShaderArtifactCodecError error{ShaderArtifactCodecError::NONE};
    spirv::SpirvTargetFeatureMask reconciled_features{};
    size_t failed_module_index{};
    luisa::string diagnostics;
    bool has_warning{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == ShaderArtifactCodecError::NONE;
    }
};

// Validates complete modules for the Vulkan 1.2 environment, verifies that
// each module contains the expected "main" entry point, and reconciles the
// final capability-owned feature requirements for native XIR artifacts.
// Non-native dialects deliberately preserve their persisted feature mask.
[[nodiscard]] SpirvArtifactModuleValidationResult
validate_spirv_artifact_modules(
    ShaderCodegenDialect dialect,
    spirv::SpirvTargetFeatureMask persisted_features,
    luisa::span<const luisa::span<const uint32_t>> modules,
    luisa::span<const ShaderArtifactSpirvStage> stages);

struct ComputeShaderArtifactEncodeInfo {
    luisa::span<const hlsl::Property> properties{};
    luisa::span<const SavedArgument> arguments{};
    vstd::MD5 shader_md5{};
    vstd::MD5 type_md5{};
    std::array<uint32_t, 3u> block_size{};
    luisa::span<const uint32_t> spirv{};
    luisa::span<const std::pair<luisa::string, luisa::compute::Type const *>> printers{};
    luisa::span<const std::byte> constant_ubo_data{};
    uint32_t validation_count{};
    uint32_t required_subgroup_size{};
    bool use_buffer_bindless{};
    bool use_tex2d_bindless{};
    bool use_tex3d_bindless{};
    ShaderCodegenDialect codegen_dialect{ShaderCodegenDialect::HLSL_SPIRV};
    spirv::SpirvTargetFeatureMask required_spirv_features{};
};

struct RasterShaderArtifactEncodeInfo {
    luisa::span<const hlsl::Property> properties{};
    luisa::span<const SavedArgument> arguments{};
    vstd::MD5 shader_md5{};
    vstd::MD5 type_md5{};
    luisa::span<const uint32_t> vertex_spirv{};
    luisa::span<const uint32_t> pixel_spirv{};
    luisa::span<const std::pair<luisa::string, luisa::compute::Type const *>> printers{};
    uint32_t validation_count{};
    bool use_buffer_bindless{};
    bool use_tex2d_bindless{};
    bool use_tex3d_bindless{};
    ShaderCodegenDialect codegen_dialect{ShaderCodegenDialect::HLSL_SPIRV};
    spirv::SpirvTargetFeatureMask required_spirv_features{};
};

struct ComputeShaderArtifact {
    ShaderSerHeader header{};
    luisa::vector<hlsl::Property> properties;
    luisa::vector<SavedArgument> arguments;
    luisa::vector<uint32_t> spirv;
    luisa::vector<std::pair<luisa::string, luisa::compute::Type const *>> printers;
    luisa::vector<std::byte> constant_ubo_data;
};

struct RasterShaderArtifact {
    RasterSerHeader header{};
    luisa::vector<hlsl::Property> properties;
    luisa::vector<SavedArgument> arguments;
    luisa::vector<uint32_t> vertex_spirv;
    luisa::vector<uint32_t> pixel_spirv;
    luisa::vector<std::pair<luisa::string, luisa::compute::Type const *>> printers;
};

struct ComputeShaderArtifactDecodeResult {
    ComputeShaderArtifact artifact;
    ShaderArtifactCodecError error{ShaderArtifactCodecError::NONE};
    size_t failed_spirv_module_index{};
    luisa::string diagnostics;
    bool has_spirv_warning{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == ShaderArtifactCodecError::NONE;
    }
};

struct RasterShaderArtifactDecodeResult {
    RasterShaderArtifact artifact;
    ShaderArtifactCodecError error{ShaderArtifactCodecError::NONE};
    size_t failed_spirv_module_index{};
    luisa::string diagnostics;
    bool has_spirv_warning{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == ShaderArtifactCodecError::NONE;
    }
};

[[nodiscard]] luisa::vector<std::byte>
encode_compute_shader_artifact(
    const ComputeShaderArtifactEncodeInfo &info);

[[nodiscard]] luisa::vector<std::byte>
encode_raster_shader_artifact(
    const RasterShaderArtifactEncodeInfo &info);

[[nodiscard]] ComputeShaderArtifactDecodeResult
decode_compute_shader_artifact(
    luisa::BinaryStream &stream,
    luisa::optional<vstd::MD5> expected_shader_md5 = luisa::nullopt,
    luisa::optional<vstd::MD5> expected_type_md5 = luisa::nullopt,
    luisa::optional<ShaderCodegenDialect> expected_codegen_dialect =
        luisa::nullopt);

[[nodiscard]] RasterShaderArtifactDecodeResult
decode_raster_shader_artifact(
    luisa::BinaryStream &stream,
    luisa::optional<vstd::MD5> expected_shader_md5 = luisa::nullopt,
    luisa::optional<vstd::MD5> expected_type_md5 = luisa::nullopt,
    luisa::optional<ShaderCodegenDialect> expected_codegen_dialect =
        luisa::nullopt);

}// namespace lc::vk::detail
