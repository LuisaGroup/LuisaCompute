#pragma once
#include <luisa/vstl/md5.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include "../common/hlsl/shader_property.h"
#include "../common/spirv/spirv_codegen/kernel_argument_role.h"
#include "../common/spirv/spirv_codegen/target_feature_mask.h"
#include "device.h"
#include "serde_type.h"
#include "shader.h"
namespace luisa {
class BinaryIO;
}// namespace luisa
namespace lc::vk {
using namespace luisa;
class RasterShader;
class Shader;
class ComputeShader;
// Strong API boundary around the persistent uint64 mask. In particular, a
// legacy bool argument must not silently become feature bit zero.
struct SpirvArtifactFeatureRequirements {
    spirv::SpirvTargetFeatureMask mask{};

    constexpr SpirvArtifactFeatureRequirements() noexcept = default;
    explicit constexpr SpirvArtifactFeatureRequirements(
        spirv::SpirvTargetFeatureMask value) noexcept
        : mask{value} {}
};

// Every cache consumer must state the artifact identity it relies on. An
// empty field means that dimension is intentionally unconstrained (for
// example generic user-supplied AOT loading), rather than being omitted by an
// overload default.
struct ComputeShaderArtifactLoadRequirements {
    luisa::optional<vstd::MD5> shader_md5;
    luisa::optional<vstd::MD5> type_md5;
    luisa::optional<detail::ShaderCodegenDialect> codegen_dialect;
};

class ShaderSerializer {
    ShaderSerializer() = delete;
    ~ShaderSerializer() = delete;
public:
    static void serialize_raster(
        vstd::span<const hlsl::Property> binds,
        vstd::span<const SavedArgument> saved_args,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        vstd::string_view file_name,
        vstd::span<const uint> vert_spv_code,
        vstd::span<const uint> pixel_spv_code,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::span<std::pair<vstd::string, luisa::compute::Type const *> const> printers,
        uint validation_count = 0,
        SpirvArtifactFeatureRequirements required_spirv_features = {},
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV);
    static void serialize_bytecode(
        vstd::span<const hlsl::Property> binds,
        vstd::span<const SavedArgument> saved_args,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        uint3 block_size,
        vstd::string_view file_name,
        vstd::span<const uint> spv_code,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::span<std::pair<vstd::string, luisa::compute::Type const *> const> printers,
        uint validation_count = 0,
        luisa::span<const std::byte> constant_ubo_data = {},
        luisa::optional<uint8_t> required_subgroup_size = luisa::nullopt,
        SpirvArtifactFeatureRequirements required_spirv_features = {},
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV);
    static bool require_recompile(
        vstd::string_view file_name,
        ComputeShaderArtifactLoadRequirements requirements,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static bool require_recompile_raster(
        vstd::string_view file_name,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static void serialize_pso(
        Device *device,
        Shader const *shader,
        vstd::MD5 shader_md5,
        BinaryIO const *bin_io);

    struct DeserResult {
        Shader *shader;
        vstd::MD5 type_md5;
    };
    struct RasterDeserResult {
        RasterShader *shader;
        vstd::MD5 type_md5;
    };
    static DeserResult try_deser_compute(
        Device *device,
        ComputeShaderArtifactLoadRequirements requirements,
        vstd::vector<Argument> &&captured,
        vstd::string_view file_name,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        uint32_t push_constant_size = 32u,
        bool enable_driver_optimization = true);
    static RasterDeserResult try_deser_raster(
        Device *device,
        // invalid md5 for AOT
        vstd::optional<vstd::MD5> shader_md5,
        luisa::optional<vstd::MD5> expected_type_md5,
        vstd::vector<Argument> &&captured,
        vstd::string_view file_name,
        SerdeType serde_type,
        BinaryIO const *bin_io);
    static vstd::vector<SavedArgument> serialize_saved_args(Function kernel);
    static vstd::vector<SavedArgument> serialize_saved_args(
        luisa::span<const std::pair<Variable, Usage>> arguments,
        bool enable_buffer_metadata = false,
        luisa::span<const spirv::SpirvKernelArgumentRoleMask>
            native_argument_roles = {});
    static vstd::vector<SavedArgument> serialize_saved_args(vstd::IRange<std::pair<Variable, Usage>> &arguments);
};
}// namespace lc::vk
