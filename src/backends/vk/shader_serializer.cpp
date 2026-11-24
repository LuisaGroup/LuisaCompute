#include "shader_serializer.h"
#include "pipeline_cache.h"
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
         // Keep the persisted-property check in step with the pipeline
         // layout of this device: with the software fallback the
         // acceleration-structure slot is a storage buffer, so a program
         // that is reloaded here must be planned the same way it was
         // created.
         .fallback_rtx = device->use_fallback_rtx(),
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

luisa::vector<std::byte> ShaderSerializer::read_shader_artifact(
    SerdeType serde_type, BinaryIO const *bin_io,
    vstd::string_view file_name) noexcept {
    auto stream = read_binary_io(serde_type, bin_io, file_name);
    if (!stream) { return {}; }
    auto length = stream->length();
    if (length == 0u) { return {}; }
    luisa::vector<std::byte> bytes(length);
    stream->read(luisa::span<std::byte>{bytes.data(), bytes.size()});
    return bytes;
}

bool ShaderSerializer::require_recompile_from_bytes(
    luisa::span<const std::byte> artifact,
    ComputeShaderArtifactLoadRequirements requirements) noexcept {
    using namespace detail;
    if (artifact.empty()) { return true; }
    StringViewBinaryStream stream{luisa::string_view{
        reinterpret_cast<const char *>(artifact.data()), artifact.size()}};
    auto decoded = decode_compute_shader_artifact(
        stream, requirements.shader_md5,
        requirements.type_md5, requirements.codegen_dialect);
    if (!decoded) return true;
    if (requirements.required_subgroup_size &&
        decoded.artifact.header.required_subgroup_size !=
            *requirements.required_subgroup_size) {
        return true;
    }
    return false;
}

bool ShaderSerializer::require_recompile(
    vstd::string_view file_name,
    ComputeShaderArtifactLoadRequirements requirements,
    SerdeType serde_type,
    BinaryIO const *bin_io) {
    return require_recompile_from_bytes(
        read_shader_artifact(serde_type, bin_io, file_name), requirements);
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
luisa::vector<std::byte> ShaderSerializer::encode_bytecode_artifact(
    vstd::span<const hlsl::Property> binds,
    vstd::span<const SavedArgument> saved_args,
    vstd::MD5 shader_md5,
    vstd::MD5 type_md5,
    uint3 block_size,
    vstd::span<const uint> spv_code,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless,
    vstd::span<std::pair<vstd::string, Type const *> const> printers,
    uint validation_count,
    luisa::span<const std::byte> constant_ubo_data,
    luisa::optional<uint8_t> required_subgroup_size,
    SpirvArtifactFeatureRequirements required_spirv_features,
    detail::ShaderCodegenDialect codegen_dialect) noexcept {
    return detail::encode_compute_shader_artifact(
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
}

void ShaderSerializer::write_bytecode_artifact(
    SerdeType serde_type, BinaryIO const *bin_io,
    vstd::string_view file_name,
    luisa::span<const std::byte> artifact) noexcept {
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
    // Encode and write are separate steps (plan P5.1): a batch encodes on a worker
    // thread and writes on the publishing thread, and both use these two helpers so
    // the bytes cannot drift from the serial path.
    auto artifact = encode_bytecode_artifact(
        binds, saved_args, shader_md5, type_md5, block_size, spv_code,
        use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless,
        printers, validation_count, constant_ubo_data,
        required_subgroup_size, required_spirv_features, codegen_dialect);
    write_bytecode_artifact(serde_type, bin_io, file_name, artifact);
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

bool ShaderSerializer::load_compute_artifact(
    Device *device,
    ComputeShaderArtifactLoadRequirements requirements,
    vstd::string_view file_name,
    SerdeType serde_type,
    BinaryIO const *bin_io,
    uint32_t push_constant_size,
    bool enable_driver_optimization,
    ComputeShaderArtifactLoad &out) noexcept {
    return load_compute_artifact_from_bytes(
        device, requirements,
        read_shader_artifact(serde_type, bin_io, file_name),
        file_name, push_constant_size, enable_driver_optimization, out);
}

bool ShaderSerializer::load_compute_artifact_from_bytes(
    Device *device,
    ComputeShaderArtifactLoadRequirements requirements,
    luisa::span<const std::byte> artifact_bytes,
    vstd::string_view file_name,
    uint32_t push_constant_size,
    bool enable_driver_optimization,
    ComputeShaderArtifactLoad &out) noexcept {
    using namespace detail;
    if (artifact_bytes.empty()) { return false; }
    StringViewBinaryStream read_stream{luisa::string_view{
        reinterpret_cast<const char *>(artifact_bytes.data()),
        artifact_bytes.size()}};
    auto decoded = decode_compute_shader_artifact(
        read_stream,
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
        return false;
    }
    if (decoded.has_spirv_warning && !decoded.diagnostics.empty()) {
        LUISA_WARNING(
            "Vulkan compute shader '{}' SPIR-V validation diagnostics:\n{}",
            file_name, decoded.diagnostics);
    }
    auto artifact = std::move(decoded.artifact);
    auto header = artifact.header;
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
        return false;
    }
    if (header.required_subgroup_size != 0u) {
        auto subgroup_size = header.required_subgroup_size;
        auto &subgroup = device->subgroup_size_control_properties();
        if (!device->subgroup_size_control_enabled ||
            (subgroup.requiredSubgroupSizeStages &
             VK_SHADER_STAGE_COMPUTE_BIT) == 0u ||
            subgroup_size < subgroup.minSubgroupSize ||
            subgroup_size > subgroup.maxSubgroupSize) {
            return false;
        }
        required_subgroup_size = static_cast<uint8_t>(subgroup_size);
    }
    if (requirements.required_subgroup_size &&
        header.required_subgroup_size != *requirements.required_subgroup_size) {
        return false;
    }
    // Identity of the pre-P1.3 per-shader PSO blob this shader would have written.
    // The blob itself is read and merged by `merge_legacy_pso_seed`, on the thread
    // that creates pipelines: reading it here would put a Vulkan call on a worker.
    {
        PSODataPackage package{
            .header = device->pso_header(),
            .pipeline_cache_identity =
                ComputeShader::pipeline_cache_identity_for(
                    enable_driver_optimization, required_subgroup_size),
            .contract_version =
                pso_identity_contract_version};
        auto shader_md5 = requirements.shader_md5;
        if (!shader_md5) {
            shader_md5.emplace(file_name);
        }
        std::memcpy(package.md5, &*shader_md5, sizeof(vstd::MD5));
        vstd::MD5 pso_md5{
            {reinterpret_cast<uint8_t const *>(&package), sizeof(PSODataPackage)}};
        out.pso_seed_name = pso_md5.to_string(false) + ".vk";
    }
    out.block_size = block_size;
    out.push_constant_size = push_constant_size;
    out.required_subgroup_size = required_subgroup_size;
    out.validation_count = header.validation_count;
    out.type_md5 = header.type_md5;
    out.codegen_dialect = static_cast<detail::ShaderCodegenDialect>(
        header.codegen_dialect);
    out.use_tex2d_bindless = header.use_bindless_tex2d != 0u;
    out.use_tex3d_bindless = header.use_bindless_tex3d != 0u;
    out.use_buffer_bindless = header.use_bindless_buffer != 0u;
    out.properties = std::move(artifact.properties);
    out.arguments = std::move(artifact.arguments);
    // The codec stores `luisa::vector` (eastl when the bundled STL is on) and the
    // shader wants a `std::vector`; copy instead of moving across containers.
    out.spirv.assign(artifact.spirv.begin(), artifact.spirv.end());
    out.printers = std::move(artifact.printers);
    out.constant_ubo_data = std::move(artifact.constant_ubo_data);
    return true;
}

void ShaderSerializer::merge_legacy_pso_seed(
    Device *device, vstd::string_view file_name,
    vstd::string_view pso_seed_name) noexcept {
    using namespace detail;
    if (pso_seed_name.empty()) { return; }
    // Pre-P1.3 per-shader PSO blobs are add-only inputs now (plan P1.3): the blob
    // is validated exactly like before (framing, sizes, MD5 and the
    // VkPipelineCacheHeaderVersionOne identity) and folded into the device-scoped
    // cache, which is the only cache shaders use.
    // Release `_shader_cache_mtx` before calling `merge_seed()` so cache I/O and
    // `DevicePipelineCache::_mtx` never nest.
    auto seed = [&]() noexcept {
        if (device != nullptr) {
            std::lock_guard cache_lock{device->shader_cache_mutex()};
            return read_pso_artifact(device, pso_seed_name);
        }
        return read_pso_artifact(device, pso_seed_name);
    }();
    if (seed.status == PsoArtifactStatus::kAccepted) {
        static_cast<void>(
            device->pipeline_cache().merge_seed(device, seed.payload));
    } else if (seed.status == PsoArtifactStatus::kRejected) {
        device->pipeline_cache().note_rejected_seed();
        LUISA_WARNING(
            "Vulkan compute shader '{}': ignoring the legacy per-shader PSO "
            "blob '{}' ({}); the device pipeline cache stays as it is.",
            file_name, pso_seed_name, seed.reason);
    }
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
    DeserResult result{.shader = nullptr};
    ComputeShaderArtifactLoad load;
    const bool loaded = [&]() noexcept {
        if (device != nullptr) {
            std::lock_guard cache_lock{device->shader_cache_mutex()};
            return load_compute_artifact(
                device, requirements, file_name, serde_type, bin_io,
                push_constant_size, enable_driver_optimization, load);
        }
        return load_compute_artifact(
            device, requirements, file_name, serde_type, bin_io,
            push_constant_size, enable_driver_optimization, load);
    }();
    if (!loaded) {
        return result;
    }
    result.type_md5 = load.type_md5;
    merge_legacy_pso_seed(device, file_name, load.pso_seed_name);
    result.shader = new ComputeShader{
        device,
        load.block_size,
        load.properties,
        std::move(load.arguments),
        load.spirv,
        std::move(captured),
        load.use_tex2d_bindless,
        load.use_tex3d_bindless,
        load.use_buffer_bindless,
        std::move(load.printers),
        load.constant_ubo_data,
        load.validation_count,
        load.required_subgroup_size,
        load.push_constant_size,
        load.codegen_dialect,
        enable_driver_optimization};
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
