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
    luisa::optional<uint8_t> required_subgroup_size;
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

    struct DeserResult {
        Shader *shader;
        vstd::MD5 type_md5;
    };
    struct RasterDeserResult {
        RasterShader *shader;
        vstd::MD5 type_md5;
    };
    // Host-side product of one compute-shader cache load (plan P5.1): everything a
    // `ComputeShader` needs, plus the name of the legacy per-shader PSO blob that
    // still has to be folded into the device cache. It holds no Vulkan handle, so a
    // caller can load it on a worker thread and construct later, on the thread that
    // owns pipeline creation.
    struct ComputeShaderArtifactLoad {
        uint3 block_size{};
        uint32_t push_constant_size{32u};
        luisa::optional<uint8_t> required_subgroup_size;
        uint32_t validation_count{0u};
        vstd::MD5 type_md5{};
        detail::ShaderCodegenDialect codegen_dialect{detail::ShaderCodegenDialect::HLSL_SPIRV};
        bool use_tex2d_bindless{false};
        bool use_tex3d_bindless{false};
        bool use_buffer_bindless{false};
        vstd::vector<hlsl::Property> properties;
        vstd::vector<SavedArgument> arguments;
        std::vector<uint32_t> spirv;
        vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> printers;
        luisa::vector<std::byte> constant_ubo_data;
        // Non-empty when the pre-P1.3 per-shader PSO blob for this shader must be
        // merged into the device cache before construction (plan P1.3: legacy blobs
        // are add-only inputs). Merging is a Vulkan call, so it belongs to the
        // constructing thread, not to the load.
        luisa::string pso_seed_name;
    };
    // Copies one artifact out of the BinaryIO. The copy is what lets a caller hold
    // its device-wide BinaryIO lock for the read only and decode/validate the bytes
    // without it (plan P5.1: a worker must not serialize the cache).
    [[nodiscard]] static luisa::vector<std::byte> read_shader_artifact(
        SerdeType serde_type, BinaryIO const *bin_io,
        vstd::string_view file_name) noexcept;
    // Byte-level core of `require_recompile` / `load_compute_artifact`: the caller
    // already holds the artifact's bytes, so nothing here touches the BinaryIO.
    [[nodiscard]] static bool require_recompile_from_bytes(
        luisa::span<const std::byte> artifact,
        ComputeShaderArtifactLoadRequirements requirements) noexcept;
    [[nodiscard]] static bool load_compute_artifact_from_bytes(
        Device *device,
        ComputeShaderArtifactLoadRequirements requirements,
        luisa::span<const std::byte> artifact,
        vstd::string_view file_name,
        uint32_t push_constant_size,
        bool enable_driver_optimization,
        ComputeShaderArtifactLoad &out) noexcept;
    // Reads and validates one compute-shader artifact without touching Vulkan.
    // Returns false (and leaves `out` unspecified) exactly where
    // `try_deser_compute` would answer "cache miss".
    [[nodiscard]] static bool load_compute_artifact(
        Device *device,
        ComputeShaderArtifactLoadRequirements requirements,
        vstd::string_view file_name,
        SerdeType serde_type,
        BinaryIO const *bin_io,
        uint32_t push_constant_size,
        bool enable_driver_optimization,
        ComputeShaderArtifactLoad &out) noexcept;
    // Folds the legacy per-shader PSO blob `pso_seed_name` into the device cache.
    // `file_name` only appears in the warning text. No-op for an empty name.
    static void merge_legacy_pso_seed(
        Device *device, vstd::string_view file_name,
        vstd::string_view pso_seed_name) noexcept;
    // Encodes the bytecode artifact a serial compile would write, without writing
    // it. The returned bytes are exactly what `serialize_bytecode` passes to the
    // BinaryIO for the same arguments, so a batch can encode on a worker thread and
    // write on the publishing thread without changing a single byte.
    [[nodiscard]] static luisa::vector<std::byte> encode_bytecode_artifact(
        vstd::span<const hlsl::Property> binds,
        vstd::span<const SavedArgument> saved_args,
        vstd::MD5 shader_md5,
        vstd::MD5 type_md5,
        uint3 block_size,
        vstd::span<const uint> spv_code,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::span<std::pair<vstd::string, luisa::compute::Type const *> const> printers,
        uint validation_count,
        luisa::span<const std::byte> constant_ubo_data,
        luisa::optional<uint8_t> required_subgroup_size,
        SpirvArtifactFeatureRequirements required_spirv_features,
        detail::ShaderCodegenDialect codegen_dialect) noexcept;
    // Writes one already-encoded artifact through the BinaryIO.
    static void write_bytecode_artifact(
        SerdeType serde_type, BinaryIO const *bin_io,
        vstd::string_view file_name,
        luisa::span<const std::byte> artifact) noexcept;
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
