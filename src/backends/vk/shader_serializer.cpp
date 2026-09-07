#include "shader_serializer.h"
#include "shader.h"
#include "../common/hlsl/shader_property.h"
#include "../common/hlsl/hlsl_codegen.h"
#include "compute_shader.h"
#include "raster_shader.h"
#include "shader_artifact_codec.h"
#include "shader_binary_contract.h"
#include "descriptor_interface_plan.h"
#include "shader_interface_plan.h"

namespace lc::vk {
namespace detail {
struct PSODataPackage {
    VkPipelineCacheHeaderVersionOne header;
    std::byte md5[sizeof(vstd::MD5)];
    uint64_t pipeline_cache_identity;
    uint64_t contract_version;
};
static_assert(
    sizeof(PSODataPackage) ==
    sizeof(VkPipelineCacheHeaderVersionOne) +
        sizeof(vstd::MD5) + sizeof(uint64_t) * 2u);
inline constexpr uint64_t pso_identity_contract_version = 1u;
}// namespace detail
class StringViewBinaryStream : public BinaryStream {

public:
    luisa::string_view strv;
    explicit StringViewBinaryStream(luisa::string_view strv) : strv(strv) {}
    [[nodiscard]] size_t length() const noexcept override { return strv.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override {
        if (dst.empty()) { return; }
        if (_pos > strv.size() || dst.size() > strv.size() - _pos) [[unlikely]] {
            // BinaryStream::read cannot report failure. Zeroing the requested
            // destination makes the enclosing version/size checks reject the
            // truncated embedded blob without reading beyond its storage.
            std::memset(dst.data(), 0, dst.size());
            _pos = strv.size();
            return;
        }
        std::memcpy(dst.data(), strv.data() + _pos, dst.size());
        _pos += dst.size();
    }
    ~StringViewBinaryStream() noexcept override = default;

private:
    size_t _pos{};
};

namespace {

[[nodiscard]] bool valid_shader_properties(
    const Device *device, luisa::span<const hlsl::Property> properties,
    detail::DescriptorInterfaceStageMask stage_mask,
    bool has_constant_ubo_payload, bool use_buffer_bindless,
    bool use_tex2d_bindless, bool use_tex3d_bindless) noexcept {
    auto plan = detail::plan_descriptor_interface(
        {.properties = properties,
         .stage_mask = stage_mask,
         .bindless_heap_capacity = device->bindless_heap_capacity(),
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = has_constant_ubo_payload,
         .acceleration_structure_available = device->enable_raytracing(),
         .sampled_image_update_after_bind_enabled = device->enable_bindless(),
         .storage_buffer_update_after_bind_enabled = device->enable_bindless()},
        detail::descriptor_interface_limits_from(
            device->properties().limits,
            device->descriptor_indexing_properties(),
            device->acceleration_structure_properties()));
    return static_cast<bool>(plan);
}

[[nodiscard]] bool validate_spirv_target_feature_requirements(
    const Device *device,
    spirv::SpirvTargetFeatureMask required_features,
    luisa::string_view file_name,
    luisa::string_view shader_kind) noexcept {
    auto check = spirv::check_spirv_target_feature_requirements(
        required_features, device->enabled_spirv_artifact_features());
    if (check.unknown_required_bits != 0u) {
        LUISA_WARNING(
            "Rejecting Vulkan {} shader '{}' because its persisted SPIR-V "
            "target-feature mask contains unknown bits 0x{:016x}.",
            shader_kind, file_name, check.unknown_required_bits);
    }
    if (check.missing_required_bits != 0u) {
        for (auto feature : spirv::list_spirv_target_features(
                 check.missing_required_bits)) {
            LUISA_WARNING(
                "Rejecting Vulkan {} shader '{}' because it requires SPIR-V "
                "target feature '{}', which is not enabled on this logical device.",
                shader_kind, file_name, feature.name);
        }
    }
    return static_cast<bool>(check);
}

[[nodiscard]] luisa::vector<std::byte>
serialize_pipeline_cache_artifact(
    luisa::span<const std::byte> payload) {
    using namespace detail;
    LUISA_ASSERT(
        payload.size_bytes() >=
                sizeof(VkPipelineCacheHeaderVersionOne) &&
            payload.size_bytes() <= max_pipeline_cache_byte_size,
        "Vulkan pipeline cache payload has invalid size {}.",
        payload.size_bytes());
    PipelineCacheArtifactHeader header{
        .payload_size = payload.size_bytes(),
        .payload_md5 = pipeline_cache_payload_md5(payload)};
    auto total_size = sizeof(header) + payload.size_bytes();
    luisa::vector<std::byte> artifact(total_size);
    std::memcpy(artifact.data(), &header, sizeof(header));
    std::memcpy(artifact.data() + sizeof(header),
                payload.data(), payload.size_bytes());
    return artifact;
}

}// namespace

luisa::unique_ptr<luisa::BinaryStream> read_binary_io(SerdeType type, luisa::BinaryIO const *bin_io, luisa::string_view file_name) {
    switch (type) {
        case SerdeType::kCache:
            return bin_io->read_shader_cache(file_name);
        case SerdeType::kBuiltin: {
            auto internal_data = hlsl::CodegenUtility::ReadInternalHLSLFile(file_name);
            if (!internal_data.empty()) {
                return luisa::make_unique<StringViewBinaryStream>(internal_data);
            }
            return bin_io->read_internal_shader(file_name);
        }
        case SerdeType::kByteCode:
            return bin_io->read_shader_bytecode(file_name);
    }
    return luisa::unique_ptr<luisa::BinaryStream>{};
}
void ShaderSerializer::serialize_raster(
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
    vstd::span<std::pair<vstd::string, Type const *> const> printers,
    uint validation_count,
    SpirvArtifactFeatureRequirements required_spirv_features,
    detail::ShaderCodegenDialect codegen_dialect) {
    auto artifact = detail::encode_raster_shader_artifact(
        {.properties = binds,
         .arguments = saved_args,
         .shader_md5 = shader_md5,
         .type_md5 = type_md5,
         .vertex_spirv = vert_spv_code,
         .pixel_spirv = pixel_spv_code,
         .printers = printers,
         .validation_count = validation_count,
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .codegen_dialect = codegen_dialect,
         .required_spirv_features = required_spirv_features.mask});

    switch (serde_type) {
        case SerdeType::kCache:
            static_cast<void>(bin_io->write_shader_cache(file_name, artifact));
            break;
        case SerdeType::kBuiltin:
            static_cast<void>(bin_io->write_internal_shader(file_name, artifact));
            break;
        case SerdeType::kByteCode:
            static_cast<void>(bin_io->write_shader_bytecode(file_name, artifact));
            break;
    }
}

bool ShaderSerializer::require_recompile(
    vstd::string_view file_name,
    ComputeShaderArtifactLoadRequirements requirements,
    SerdeType serde_type,
    BinaryIO const *bin_io) {
    using namespace detail;
    auto read_stream = read_binary_io(serde_type, bin_io, file_name);
    if (!read_stream) return true;
    return !decode_compute_shader_artifact(
        *read_stream, requirements.shader_md5,
        requirements.type_md5, requirements.codegen_dialect);
}

bool ShaderSerializer::require_recompile_raster(
    vstd::string_view file_name,
    vstd::MD5 shader_md5,
    vstd::MD5 type_md5,
    SerdeType serde_type,
    BinaryIO const *bin_io) {
    using namespace detail;
    auto read_stream = read_binary_io(serde_type, bin_io, file_name);
    if (!read_stream) return true;
    return !decode_raster_shader_artifact(
        *read_stream, shader_md5, type_md5);
}
void ShaderSerializer::serialize_bytecode(
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
    vstd::span<std::pair<vstd::string, Type const *> const> printers,
    uint validation_count,
    luisa::span<const std::byte> constant_ubo_data,
    luisa::optional<uint8_t> required_subgroup_size,
    SpirvArtifactFeatureRequirements required_spirv_features,
    detail::ShaderCodegenDialect codegen_dialect) {
    auto artifact = detail::encode_compute_shader_artifact(
        {.properties = binds,
         .arguments = saved_args,
         .shader_md5 = shader_md5,
         .type_md5 = type_md5,
         .block_size = {block_size.x, block_size.y, block_size.z},
         .spirv = spv_code,
         .printers = printers,
         .constant_ubo_data = constant_ubo_data,
         .validation_count = validation_count,
         .required_subgroup_size = required_subgroup_size.value_or(0u),
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .codegen_dialect = codegen_dialect,
         .required_spirv_features = required_spirv_features.mask});

    switch (serde_type) {
        case SerdeType::kCache:
            bin_io->write_shader_cache(file_name, artifact);
            break;
        case SerdeType::kBuiltin:
            bin_io->write_internal_shader(file_name, artifact);
            break;
        case SerdeType::kByteCode:
            bin_io->write_shader_bytecode(file_name, artifact);
            break;
    }
}
void ShaderSerializer::serialize_pso(
    Device *device,
    Shader const *shader,
    vstd::MD5 shader_md5,
    BinaryIO const *bin_io) {
    vstd::vector<std::byte> pso_data;
    if (!shader->serialize_pso(pso_data)) return;
    if (pso_data.size() > detail::max_pipeline_cache_byte_size) {
        LUISA_WARNING(
            "Ignoring an implausibly large Vulkan pipeline cache blob ({} bytes).",
            pso_data.size());
        return;
    }
    using namespace detail;
    PSODataPackage package{
        .header = device->pso_header(),
        .pipeline_cache_identity =
            shader->pipeline_cache_identity(),
        .contract_version = pso_identity_contract_version};
    memcpy(package.md5, &shader_md5, sizeof(vstd::MD5));
    vstd::MD5 pso_md5{
        {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
    auto file_name = pso_md5.to_string(false) + ".vk";

    auto artifact = serialize_pipeline_cache_artifact(pso_data);
    bin_io->write_shader_cache(file_name, artifact);
}
auto ShaderSerializer::try_deser_raster(
    Device *device,
    // invalid md5 for AOT
    vstd::optional<vstd::MD5> shader_md5,
    luisa::optional<vstd::MD5> expected_type_md5,
    vstd::vector<Argument> &&captured,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io) -> RasterDeserResult {
    RasterDeserResult result{
        .shader = nullptr};
    using namespace detail;
    auto read_stream = read_binary_io(serde_type, bin_io, file_name);
    if (!read_stream) return result;
    auto decoded = decode_raster_shader_artifact(
        *read_stream,
        shader_md5 ? luisa::optional<vstd::MD5>{*shader_md5} : luisa::nullopt,
        expected_type_md5,
        detail::ShaderCodegenDialect::HLSL_SPIRV);
    if (!decoded) {
        if (decoded.error == ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH ||
            decoded.error == ShaderArtifactCodecError::INVALID_SPIRV ||
            decoded.error == ShaderArtifactCodecError::NATIVE_FEATURE_MASK_UNAVAILABLE ||
            decoded.error == ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH) {
            LUISA_WARNING(
                "Rejecting Vulkan raster shader '{}' ({}): {}",
                file_name, shader_artifact_codec_error_name(decoded.error),
                decoded.diagnostics);
        }
        return result;
    }
    if (decoded.has_spirv_warning && !decoded.diagnostics.empty()) {
        LUISA_WARNING(
            "Vulkan raster shader '{}' SPIR-V validation diagnostics:\n{}",
            file_name, decoded.diagnostics);
    }
    auto artifact = std::move(decoded.artifact);
    auto header = artifact.header;
    auto &limits = device->properties().limits;
    if (header.validation_count >
            limits.maxStorageBufferRange / sizeof(uint) ||
        !valid_shader_properties(
            device, artifact.properties,
            DescriptorInterfaceStageMask::RASTER, false,
            header.use_bindless_buffer != 0u,
            header.use_bindless_tex2d != 0u,
            header.use_bindless_tex3d != 0u) ||
        !validate_spirv_target_feature_requirements(
            device, header.required_spirv_features,
            file_name, "raster")) {
        return result;
    }
    result.type_md5 = header.type_md5;
    result.shader = new RasterShader(
        device,
        std::move(captured),
        std::move(artifact.arguments),
        artifact.properties,
        {},
        std::move(artifact.vertex_spirv),
        std::move(artifact.pixel_spirv),
        header.use_bindless_tex2d != 0u,
        header.use_bindless_tex3d != 0u,
        header.use_bindless_buffer != 0u,
        header.validation_count,
        static_cast<detail::ShaderCodegenDialect>(
            header.codegen_dialect));
    return result;
}

ShaderSerializer::DeserResult ShaderSerializer::try_deser_compute(
    Device *device,
    ComputeShaderArtifactLoadRequirements requirements,
    vstd::vector<Argument> &&captured,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io,
    uint32_t push_constant_size,
    bool enable_driver_optimization) {
    using namespace detail;
    DeserResult result{
        .shader = nullptr};
    auto read_stream = read_binary_io(serde_type, bin_io, file_name);
    if (!read_stream) return result;
    auto decoded = decode_compute_shader_artifact(
        *read_stream,
        requirements.shader_md5,
        requirements.type_md5,
        requirements.codegen_dialect);
    if (!decoded) {
        if (decoded.error == ShaderArtifactCodecError::CODEGEN_DIALECT_MISMATCH ||
            decoded.error == ShaderArtifactCodecError::INVALID_SPIRV ||
            decoded.error == ShaderArtifactCodecError::NATIVE_FEATURE_MASK_UNAVAILABLE ||
            decoded.error == ShaderArtifactCodecError::NATIVE_FEATURE_MASK_MISMATCH) {
            LUISA_WARNING(
                "Rejecting Vulkan compute shader '{}' ({}): {}",
                file_name, shader_artifact_codec_error_name(decoded.error),
                decoded.diagnostics);
        }
        return result;
    }
    if (decoded.has_spirv_warning && !decoded.diagnostics.empty()) {
        LUISA_WARNING(
            "Vulkan compute shader '{}' SPIR-V validation diagnostics:\n{}",
            file_name, decoded.diagnostics);
    }
    auto artifact = std::move(decoded.artifact);
    auto header = artifact.header;
    result.type_md5 = header.type_md5;
    auto block_size = uint3(
        header.block_size[0], header.block_size[1], header.block_size[2]);
    luisa::optional<uint8_t> required_subgroup_size;
    auto &limits = device->properties().limits;
    auto block_invocations =
        static_cast<uint64_t>(header.block_size[0]) *
        static_cast<uint64_t>(header.block_size[1]) *
        static_cast<uint64_t>(header.block_size[2]);
    if (!detail::valid_shader_constant_payload_size(
            header.constant_ubo_size,
            limits.maxUniformBufferRange) ||
        header.validation_count >
            limits.maxStorageBufferRange / sizeof(uint) ||
        header.block_size[0] > limits.maxComputeWorkGroupSize[0] ||
        header.block_size[1] > limits.maxComputeWorkGroupSize[1] ||
        header.block_size[2] > limits.maxComputeWorkGroupSize[2] ||
        block_invocations > limits.maxComputeWorkGroupInvocations ||
        !valid_shader_properties(
            device, artifact.properties,
            detail::DescriptorInterfaceStageMask::COMPUTE,
            !artifact.constant_ubo_data.empty(),
            header.use_bindless_buffer != 0u,
            header.use_bindless_tex2d != 0u,
            header.use_bindless_tex3d != 0u) ||
        !validate_spirv_target_feature_requirements(
            device, header.required_spirv_features,
            file_name, "compute")) {
        return result;
    }
    if (header.required_subgroup_size != 0u) {
        auto subgroup_size = header.required_subgroup_size;
        auto &subgroup = device->subgroup_size_control_properties();
        if (!device->subgroup_size_control_enabled ||
            (subgroup.requiredSubgroupSizeStages &
             VK_SHADER_STAGE_COMPUTE_BIT) == 0u ||
            subgroup_size < subgroup.minSubgroupSize ||
            subgroup_size > subgroup.maxSubgroupSize) {
            return result;
        }
        required_subgroup_size = static_cast<uint8_t>(subgroup_size);
    }
    vstd::vector<std::byte> pso_data;
    vstd::string pso_name;
    {
        PSODataPackage package{
            .header = device->pso_header(),
            .pipeline_cache_identity =
                ComputeShader::pipeline_create_flags(
                    enable_driver_optimization),
            .contract_version =
                pso_identity_contract_version};
        if (!requirements.shader_md5) {
            requirements.shader_md5.emplace(file_name);
        }
        std::memcpy(package.md5, &*requirements.shader_md5,
                    sizeof(vstd::MD5));
        vstd::MD5 pso_md5{
            {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
        pso_name = pso_md5.to_string(false) + ".vk";
        auto read_stream = bin_io->read_shader_cache(pso_name);
        if (read_stream) {
            auto stream_len = read_stream->length();
            if (stream_len >= sizeof(PipelineCacheArtifactHeader)) {
                PipelineCacheArtifactHeader artifact_header{};
                read_stream->read({reinterpret_cast<std::byte *>(&artifact_header),
                                   sizeof(artifact_header)});
                if (valid_pipeline_cache_artifact_framing(
                        artifact_header, stream_len) &&
                    artifact_header.payload_size >=
                        sizeof(VkPipelineCacheHeaderVersionOne)) {
                    luisa::enlarge_by(
                        pso_data, artifact_header.payload_size);
                    read_stream->read(pso_data);
                    VkPipelineCacheHeaderVersionOne cache_header{};
                    std::memcpy(&cache_header, pso_data.data(),
                                sizeof(cache_header));
                    if (read_stream->pos() != stream_len ||
                        !valid_pipeline_cache_artifact_payload(
                            artifact_header, pso_data) ||
                        !device->is_pso_same(cache_header)) {
                        pso_data.clear();
                    }
                }
            }
        }
    }
    auto shader = new ComputeShader{
        device,
        block_size,
        artifact.properties,
        std::move(artifact.arguments),
        artifact.spirv,
        std::move(captured),
        pso_data,
        header.use_bindless_tex2d != 0u,
        header.use_bindless_tex3d != 0u,
        header.use_bindless_buffer != 0u,
        std::move(artifact.printers),
        artifact.constant_ubo_data,
        header.validation_count,
        required_subgroup_size,
        push_constant_size,
        static_cast<detail::ShaderCodegenDialect>(
            header.codegen_dialect),
        enable_driver_optimization};
    if (pso_data.empty() &&
        shader->serialize_pso(pso_data)) {
        auto artifact = serialize_pipeline_cache_artifact(pso_data);
        bin_io->write_shader_cache(pso_name, artifact);
    }
    result.shader = shader;
    return result;
}
vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(Function kernel) {
    LUISA_ASSUME(kernel.tag() != Function::Tag::CALLABLE);
    auto &&args = kernel.arguments();
    vstd::vector<SavedArgument> result;
    vstd::push_back_func(result, args.size(), [&](size_t i) {
        auto &&var = args[i];
        return SavedArgument(kernel, var);
    });
    return result;
}

vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(
    luisa::span<const std::pair<Variable, Usage>> arguments,
    bool enable_buffer_metadata,
    luisa::span<const spirv::SpirvKernelArgumentRoleMask>
        native_argument_roles) {
    LUISA_ASSERT(
        native_argument_roles.empty() ||
            native_argument_roles.size() == arguments.size(),
        "Vulkan native argument-role table has {} entries for {} kernel "
        "arguments.",
        native_argument_roles.size(), arguments.size());
    vstd::vector<SavedArgument> result;
    result.reserve(arguments.size());
    auto next_buffer_metadata_index = 0u;
    for (auto index = 0u; index < arguments.size(); ++index) {
        auto &&argument = arguments[index];
        auto &saved = result.emplace_back(
            argument.second, argument.first);
        if (enable_buffer_metadata &&
            argument.first.type()->is_buffer()) {
            saved.set_buffer_metadata_index(
                next_buffer_metadata_index++);
        }
        if (!native_argument_roles.empty()) {
            auto roles = native_argument_roles[index];
            if (argument.first.type()->is_accel()) {
                LUISA_ASSERT(
                    (roles &
                     ~spirv::kernel_argument_role::accel_known_mask) == 0u,
                    "Vulkan native accel argument {} has unknown role bits "
                    "0x{:08x}.",
                    index, roles);
                saved.set_native_accel_roles(roles);
            } else if (argument.first.type()->is_buffer()) {
                LUISA_ASSERT(
                    (roles &
                     ~spirv::kernel_argument_role::buffer_known_mask) == 0u,
                    "Vulkan native buffer argument {} has unknown role bits "
                    "0x{:08x}.",
                    index, roles);
                saved.set_native_buffer_roles(roles);
            } else if (argument.first.type()->is_bindless_array()) {
                LUISA_ASSERT(
                    (roles &
                     ~spirv::kernel_argument_role::bindless_known_mask) == 0u,
                    "Vulkan native bindless argument {} has unknown role bits "
                    "0x{:08x}.",
                    index, roles);
                saved.set_native_bindless_roles(roles);
            } else {
                LUISA_ASSERT(
                    roles == spirv::kernel_argument_role::none,
                    "Vulkan native non-resource argument {} has role bits "
                    "0x{:08x}.",
                    index, roles);
            }
        }
    }
    return result;
}

vstd::vector<SavedArgument> ShaderSerializer::serialize_saved_args(
    vstd::IRange<std::pair<Variable, Usage>> &arguments) {
    vstd::vector<SavedArgument> result;
    for (auto &&i : arguments) {
        result.emplace_back(i.second, i.first);
    }
    return result;
}

}// namespace lc::vk
