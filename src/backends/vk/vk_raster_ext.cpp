#include "vk_raster_ext.h"
#include "device.h"
#include "../common/hlsl/shader_compiler.h"
#include "../common/hlsl/hlsl_codegen.h"
#include "shader_serializer.h"
#include "raster_shader.h"
#include "sampler_anisotropy.h"
#include "texture.h"
#include "../common/backend_print_code.h"
namespace lc::vk {
ResourceCreationInfo VkRasterExt::create_raster_shader(
    Function vert,
    Function pixel,
    const ShaderOption &option) noexcept {
    LUISA_ASSERT(option.compile_only, "Raster only allow AOT.");
    LUISA_ASSERT(!option.name.empty(), "Raster shader name must not be empty.");
    auto vert_sampler_usage = detail::analyze_sampler_usage(vert);
    auto pixel_sampler_usage = detail::analyze_sampler_usage(pixel);
    if (vert_sampler_usage.has_invalid_filter ||
        pixel_sampler_usage.has_invalid_filter) [[unlikely]] {
        LUISA_ERROR(
            "Vulkan raster shader '{}' contains an explicit texture sampler "
            "filter outside the valid [0, 4) selector range.",
            option.name);
    }
    auto has_dynamic_sampler_filter =
        vert_sampler_usage.has_dynamic_filter ||
        pixel_sampler_usage.has_dynamic_filter;
    auto unrestricted_native_sampler_access =
        !option.native_include.empty();
    auto requires_sampler_anisotropy =
        vert_sampler_usage.requires_anisotropy ||
        pixel_sampler_usage.requires_anisotropy ||
        unrestricted_native_sampler_access;
    if (!detail::sampler_requirement_is_supported(
            requires_sampler_anisotropy,
            _device->enable_sampler_anisotropy())) [[unlikely]] {
        if (unrestricted_native_sampler_access) {
            LUISA_ERROR(
                "Vulkan raster shader '{}' includes unrestricted native "
                "HLSL, which may access anisotropic sampler-heap entries, "
                "but samplerAnisotropy is not enabled on this logical device.",
                option.name);
        }
        if (has_dynamic_sampler_filter) {
            LUISA_ERROR(
                "Vulkan raster shader '{}' dynamically selects a texture "
                "sampler filter, which may select ANISOTROPIC, but "
                "samplerAnisotropy is not enabled on this logical device.",
                option.name);
        }
        LUISA_ERROR(
            "Vulkan raster shader '{}' selects ANISOTROPIC texture filtering, "
            "but samplerAnisotropy is not enabled on this logical device.",
            option.name);
    }
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }
    auto code = hlsl::CodegenUtility{}.RasterCodegen(vert, pixel, option.native_include, mask, true, false, option.enable_debug_info);
    if (luisa::compute::backend_print_code_enabled()) {
        auto dump_name = option.name;
        auto dump_file_name = luisa::format("hlsl_output_{}.hlsl", dump_name);
        auto f = fopen(dump_file_name.c_str(), "wb");
        if (f) {
            fwrite(code.result.data(), code.result.size(), 1, f);
            fclose(f);
        }
    }
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    auto comp_result = Device::compiler()->compile_raster(code.result.view(), !option.enable_debug_info, kShaderModel, option.enable_fast_math, true, option.enable_debug_info);
    if (comp_result.vertex.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile vertex-shader error: {}", *comp_result.vertex.try_get<vstd::string>());
    }
    if (comp_result.pixel.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile pixel-shader error: {}", *comp_result.pixel.try_get<vstd::string>());
    }
    auto kernel_args = [&]() {
        auto vert_span = vert.arguments();
        auto vert_args =
            vstd::range_linker{
                vstd::make_ite_range(vert_span.subspan(1)),
                vstd::transform_range{
                    [&](Variable const &var) {
                        return std::pair<Variable, Usage>{var, vert.variable_usage(var.uid())};
                    }}};
        auto pixel_span = pixel.arguments();
        auto pixel_args =
            vstd::range_linker{
                vstd::make_ite_range(pixel_span.subspan(1)),
                vstd::transform_range{
                    [&](Variable const &var) {
                        return std::pair<Variable, Usage>{var, pixel.variable_usage(var.uid())};
                    }}};
        auto args = vstd::tuple_range(std::move(vert_args), std::move(pixel_args)).i_range();
        return ShaderSerializer::serialize_saved_args(args);
    }();
    auto &&vert_buffer = comp_result.vertex.get<0>();
    auto &&pixel_buffer = comp_result.pixel.get<0>();
    auto required_spirv_features =
        _device->enabled_spirv_artifact_features();
    if (requires_sampler_anisotropy) {
        required_spirv_features |=
            spirv::target_feature::sampler_anisotropy;
    }
    auto feature_check = spirv::check_spirv_target_feature_requirements(
        required_spirv_features,
        _device->enabled_spirv_artifact_features());
    LUISA_ASSERT(
        static_cast<bool>(feature_check),
        "Vulkan raster SPIR-V artifact requires unavailable features "
        "0x{:016x} or unknown features 0x{:016x}.",
        feature_check.missing_required_bits,
        feature_check.unknown_required_bits);
    ShaderSerializer::serialize_raster(
        code.properties,
        kernel_args,
        check_md5,
        code.typeMD5,
        option.name,
        {reinterpret_cast<const uint *>(vert_buffer->GetBufferPointer()), vert_buffer->GetBufferSize() / sizeof(uint)},
        {reinterpret_cast<const uint *>(pixel_buffer->GetBufferPointer()), pixel_buffer->GetBufferSize() / sizeof(uint)},
        SerdeType::kByteCode,
        _device->binary_io(),
        code.useTex2DBindless,
        code.useTex3DBindless,
        code.useBufferBindless,
        code.printers,
        code.validation_count,
        SpirvArtifactFeatureRequirements{required_spirv_features},
        detail::ShaderCodegenDialect::HLSL_SPIRV);
    return ResourceCreationInfo::make_invalid();
}

ResourceCreationInfo VkRasterExt::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto expected_type_md5 = hlsl::CodegenUtility::GetTypeMD5(types);
    auto deser_result = ShaderSerializer::try_deser_raster(
        _device, {}, expected_type_md5, {}, ser_path,
        SerdeType::kByteCode, _device->binary_io());
    if (!deser_result.shader)
        return ResourceCreationInfo::make_invalid();
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(deser_result.shader);
    return info;
}

VkRasterExt::VkRasterExt(Device *device) {
    _device = device;
}

void VkRasterExt::destroy_raster_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<RasterShader *>(handle);
}

// depth buffer
ResourceCreationInfo VkRasterExt::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    ResourceCreationInfo r{};
    auto tex = new Texture(_device, format, uint2(width, height));
    r.handle = reinterpret_cast<uint64_t>(tex);
    r.native_handle = tex->vk_image();
    return r;
}
void VkRasterExt::destroy_depth_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<Texture *>(handle);
}
}// namespace lc::vk
