#include "native_shader_ext.h"

#include "device.h"
#include "native_shader.h"
#include "glslang_compiler.h"
#include "stream.h"
#include "../../common/native_shader/native_shader_reflection.h"

#ifndef LC_NO_HLSL_BUILTIN
#include "../../common/hlsl/shader_compiler.h"
#endif

namespace lc::vk {

namespace {

[[nodiscard]] const char *kind_name(NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer: return "ConstantBuffer";
        case NativeShaderResourceKind::Sampler: return "Sampler";
        case NativeShaderResourceKind::StructuredBuffer: return "StructuredBuffer";
        case NativeShaderResourceKind::RWStructuredBuffer: return "RWStructuredBuffer";
        case NativeShaderResourceKind::ByteAddressBuffer: return "ByteAddressBuffer";
        case NativeShaderResourceKind::RWByteAddressBuffer: return "RWByteAddressBuffer";
        case NativeShaderResourceKind::TypedBuffer: return "TypedBuffer";
        case NativeShaderResourceKind::RWTypedBuffer: return "RWTypedBuffer";
        case NativeShaderResourceKind::Texture2D: return "Texture2D";
        case NativeShaderResourceKind::RWTexture2D: return "RWTexture2D";
        case NativeShaderResourceKind::Texture3D: return "Texture3D";
        case NativeShaderResourceKind::RWTexture3D: return "RWTexture3D";
        case NativeShaderResourceKind::AccelerationStructure: return "AccelerationStructure";
    }
    return "Unknown";
}

// Native Vulkan binding classes this iteration can bind.
[[nodiscard]] bool is_supported_kind(NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer:
        case NativeShaderResourceKind::StructuredBuffer:
        case NativeShaderResourceKind::RWStructuredBuffer:
        case NativeShaderResourceKind::ByteAddressBuffer:
        case NativeShaderResourceKind::RWByteAddressBuffer:
        case NativeShaderResourceKind::TypedBuffer:
        case NativeShaderResourceKind::RWTypedBuffer:
            return true;
        default: return false;
    }
}

// R12: fail closed when the module requires optional features the logical
// device did not enable, instead of letting pipeline creation fail opaquely.
[[nodiscard]] bool check_required_features(
    Device *device, const native_shader_reflection::SpirvReflection &reflection,
    NativeShaderCompileResult &result) noexcept {
    auto check = lc::spirv::check_spirv_target_feature_requirements(
        reflection.required_features, device->enabled_spirv_artifact_features());
    if (check) { return true; }
    auto names = native_shader_reflection::detail::describe_required_features(
        check.missing_required_bits | check.unknown_required_bits);
    result.error = luisa::format(
        "The native shader module requires features the Vulkan device did not "
        "enable (0x{:016x}); missing or unknown requirements: {}. Enable them "
        "through the device configuration or use a shader without them.",
        check.missing_required_bits | check.unknown_required_bits, names);
    return false;
}

[[nodiscard]] NativeShaderCompileResult compile_hlsl(
    Device *device, const NativeShaderCompileInfo &info) noexcept {
    NativeShaderCompileResult result;
#ifdef LC_NO_HLSL_BUILTIN
    static_cast<void>(device);
    static_cast<void>(info);
    result.error = "The Vulkan backend was built without DXC compatibility, so "
                   "it cannot compile HLSL; use GLSL instead.";
#else
    auto compiler = Device::compiler();
    if (compiler == nullptr) {
        result.error = "The Vulkan backend has no DXC compiler instance.";
        return result;
    }
    auto compiled = compiler->compile_compute(
        info.source, info.optimize, info.shader_model,
        info.enable_fast_math, /*spirv*/ true, info.enable_debug_info,
        info.entry_point);
    if (compiled.is_type_of<vstd::string>()) {
        result.error = luisa::string{compiled.get<1>()};
        return result;
    }
    auto &blob = compiled.get<0>();
    if (blob == nullptr || blob->GetBufferSize() == 0u) {
        result.error = "DXC returned an empty SPIR-V module.";
        return result;
    }
    result.binary.resize(blob->GetBufferSize());
    std::memcpy(result.binary.data(), blob->GetBufferPointer(),
                blob->GetBufferSize());
#endif
    return result;
}

[[nodiscard]] NativeShaderCompileResult compile_glsl(
    const NativeShaderCompileInfo &info) noexcept {
    NativeShaderCompileResult result;
    auto compiled = compile_glsl_to_spirv(
        info.source, info.entry_point, info.optimize, info.enable_debug_info);
    if (!compiled.ok()) {
        result.error = std::move(compiled.error);
        return result;
    }
    result.binary = std::move(compiled.spirv);
    return result;
}

}// namespace

VkNativeShaderExt::VkNativeShaderExt(Device *device) noexcept
    : NativeShaderExt{device}, _vk_device{device} {}

VkNativeShaderExt::~VkNativeShaderExt() noexcept {
    luisa::vector<std::pair<uint64_t, NativeShader *>> shaders;
    {
        std::lock_guard lock{_mutex};
        for (auto &&entry : _shaders) { shaders.emplace_back(entry); }
        _shaders.clear();
    }
    if (!shaders.empty()) {
        LUISA_WARNING(
            "The Vulkan native-shader extension is being destroyed with {} "
            "shader instance(s) still alive; a native shader must be destroyed "
            "(`NativeShader::reset()` / `destroy_shader()`) before its device.",
            shaders.size());
    }
    for (auto &&[handle, shader] : shaders) {
        static_cast<void>(handle);
        delete shader;
    }
}

NativeShader *VkNativeShaderExt::find(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _shaders.find(handle); it != _shaders.end()) {
        return it->second;
    }
    return nullptr;
}

NativeShaderCompileResult VkNativeShaderExt::compile(
    const NativeShaderCompileInfo &info) noexcept {
    NativeShaderCompileResult result;
    result.shader_model = info.shader_model;
    result.language = info.language;
    if (info.source.empty()) {
        result.error = "Native shader compile requires a non-empty source.";
        return result;
    }
    result = info.language == NativeShaderLanguage::HLSL ?
                 compile_hlsl(_vk_device, info) :
                 compile_glsl(info);
    result.language = info.language;
    if (!result.error.empty()) { return result; }
    // The SPIR-V module is the authoritative reflection source (R2).
    auto reflection = native_shader_reflection::parse_spirv(result.binary);
    if (!reflection.ok()) {
        result.error = std::move(reflection.error);
        result.binary.clear();
        return result;
    }
    result.bindings = std::move(reflection.bindings);
    result.block_size = reflection.block_size;
    if (result.block_size.x == 0u || result.block_size.y == 0u ||
        result.block_size.z == 0u) {
        if (info.block_size.x > 0u && info.block_size.y > 0u &&
            info.block_size.z > 0u) {
            result.block_size = info.block_size;
        } else {
            result.error = "Could not determine the native shader's workgroup "
                           "size from the SPIR-V module; declare "
                           "`layout(local_size_...)`/`[numthreads]` or pass "
                           "NativeShaderCompileInfo::block_size.";
            result.binary.clear();
            return result;
        }
    } else if (info.block_size.x > 0u || info.block_size.y > 0u ||
               info.block_size.z > 0u) {
        if (info.block_size.x != result.block_size.x ||
            info.block_size.y != result.block_size.y ||
            info.block_size.z != result.block_size.z) {
            result.error = luisa::format(
                "Declared block size ({}, {}, {}) does not match the shader's "
                "workgroup size ({}, {}, {}).",
                info.block_size.x, info.block_size.y, info.block_size.z,
                result.block_size.x, result.block_size.y, result.block_size.z);
            result.binary.clear();
            return result;
        }
    }
    if (info.push_constant_size != 0u) {
        if (!reflection.has_push_constant) {
            result.error = luisa::format(
                "NativeShaderCompileInfo::push_constant_size is {} bytes, but "
                "the module declares no push-constant block.",
                info.push_constant_size);
            result.binary.clear();
            return result;
        }
        result.push_constant_size = info.push_constant_size;
    } else {
        result.push_constant_size = reflection.push_constant_size;
    }
    if (result.push_constant_size != 0u &&
        result.push_constant_size % sizeof(uint32_t) != 0u) {
        result.error = "The native shader push-constant block must be a multiple "
                       "of 4 bytes.";
        result.binary.clear();
        return result;
    }
    if (!check_required_features(_vk_device, reflection, result)) {
        result.binary.clear();
        return result;
    }
    return result;
}

NativeShaderMetadata VkNativeShaderExt::load(
    const NativeShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    NativeShaderMetadata metadata;
    metadata.language = result.language;
    metadata.push_constant_size = result.push_constant_size;
    if (!result.ok()) {
        LUISA_WARNING("Native shader load() called with a failed compile "
                      "result: {}",
                      result.error);
        return metadata;
    }
    if (!usage_override.empty() &&
        usage_override.size() != result.bindings.size()) {
        LUISA_WARNING("Native shader load() received {} usage overrides for {} "
                      "reflected bindings.",
                      usage_override.size(), result.bindings.size());
        return metadata;
    }
    metadata.bindings = result.bindings;
    for (auto i = 0u; i < metadata.bindings.size(); i++) {
        auto &&binding = metadata.bindings[i];
        if (!usage_override.empty()) { binding.usage = usage_override[i]; }
        if (!is_supported_kind(binding.kind)) {
            LUISA_WARNING(
                "Native shader binding (set {}, binding {}) has kind {}, which "
                "the Vulkan native-shader route cannot bind yet; only constant "
                "buffers and buffers are supported on vk.",
                binding.space_index, binding.register_index, kind_name(binding.kind));
            return metadata;
        }
        if (binding.array_size != 1u) {
            LUISA_WARNING(
                "Native shader binding (set {}, binding {}) is an array of {} "
                "descriptors, which the Vulkan native-shader route does not "
                "support yet.",
                binding.space_index, binding.register_index, binding.array_size);
            return metadata;
        }
        auto writes = (luisa::to_underlying(binding.usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (binding.usage == Usage::NONE ||
            (!is_uav && writes)) {
            LUISA_WARNING(
                "Native shader binding (set {}, binding {}) declares an invalid "
                "usage ({}); read-only bindings must be READ and writable "
                "bindings WRITE or READ_WRITE.",
                binding.space_index, binding.register_index,
                luisa::to_underlying(binding.usage));
            return metadata;
        }
    }
    // The module's entry point is the name the pipeline must be created with.
    auto reflection = native_shader_reflection::parse_spirv(result.binary);
    if (!reflection.ok()) {
        LUISA_WARNING("Native shader SPIR-V re-parse failed: {}",
                      reflection.error);
        return metadata;
    }
    auto *shader = new NativeShader(
        _vk_device, result.block_size, metadata.bindings, result.binary,
        reflection.entry_point.empty() ? luisa::string_view{"main"} :
                                         luisa::string_view{reflection.entry_point},
        result.push_constant_size);
    auto handle = reinterpret_cast<uint64_t>(shader);
    {
        std::lock_guard lock{_mutex};
        _shaders.try_emplace(handle, shader);
    }
    metadata.handle = handle;
    metadata.block_size = result.block_size;
    return metadata;
}

void VkNativeShaderExt::destroy_shader(uint64_t handle) noexcept {
    NativeShader *shader = nullptr;
    {
        std::lock_guard lock{_mutex};
        if (auto it = _shaders.find(handle); it != _shaders.end()) {
            shader = it->second;
            _shaders.erase(it);
        }
    }
    delete shader;
}

void encode_native_shader_dispatch(
    Device *device, CommandBufferState *state, VkCommandBuffer cmdbuffer,
    NativeShaderDispatchCommand const *cmd) noexcept {
    auto *ext = static_cast<VkNativeShaderExt *>(
        device->extension(NativeShaderExt::name));
    if (ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The Vulkan device has no NativeShaderExt; native shader "
            "dispatches cannot be encoded.");
        return;
    }
    auto *shader = ext->find(cmd->shader_handle());
    if (shader == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Native shader dispatch references an unknown shader instance "
            "(handle {}).",
            cmd->shader_handle());
        return;
    }
    shader->encode(state, cmdbuffer, cmd);
}

}// namespace lc::vk
