#include "native_shader_ext.h"

#include "device.h"
#include "native_shader.h"
#include "native_raster_shader.h"
#include "glslang_compiler.h"
#include "stream.h"
#include "../../common/native_shader/native_shader_reflection.h"

#ifndef LC_NO_HLSL_BUILTIN
#include "../../common/hlsl/shader_compiler.h"
#endif

namespace lc::vk {

// The helpers live in a file-unique namespace rather than an anonymous one: the
// backend is built with unity (jumbo) batches, so two translation units of one
// batch share their anonymous namespace and two same-named file-local helpers
// would collide.
namespace native_shader_ext_detail {

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
    luisa::string source_storage;
    luisa::string_view source;
    luisa::vector<luisa::filesystem::path> include_dirs;
    if (!luisa::compute::detail::native_shader_resolve_source_and_include_dirs(
            info, source_storage, source, include_dirs, result.error)) {
        return result;
    }
    auto compiled = compiler->compile_compute(
        source, info.optimize, info.shader_model,
        info.enable_fast_math, /*spirv*/ true, info.enable_debug_info,
        info.entry_point, include_dirs);
    if (compiled.is_type_of<vstd::string>()) {
        result.error = luisa::string{compiled.get<1>()};
        LUISA_WARNING("Vulkan native shader compile failed (HLSL -> SPIR-V via "
                      "DXC, entry point '{}'): {}",
                      info.entry_point, result.error);
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
    luisa::string source_storage;
    luisa::string_view source;
    luisa::vector<luisa::filesystem::path> include_dirs;
    if (!luisa::compute::detail::native_shader_resolve_source_and_include_dirs(
            info, source_storage, source, include_dirs, result.error)) {
        return result;
    }
    auto compiled = compile_glsl_to_spirv(
        source, info.entry_point, info.optimize, info.enable_debug_info,
        include_dirs);
    if (!compiled.ok()) {
        result.error = std::move(compiled.error);
        LUISA_WARNING("Vulkan native shader compile failed (GLSL -> SPIR-V via "
                      "glslang, entry point 'main'): {}",
                      result.error);
        return result;
    }
    result.binary = std::move(compiled.spirv);
    return result;
}

// ---- raster -------------------------------------------------------------

[[nodiscard]] const char *stage_name(bool vertex) noexcept {
    return vertex ? "vertex" : "pixel";
}

// Compiles one stage of a native raster pair to SPIR-V.
//
// Mirrors `compile_hlsl` / `compile_glsl` (same source and include-directory
// resolution, same warnings) but selects the front end's stage: DXC's `vs_<sm>` /
// `ps_<sm>` profile through `compile_graphics_stage`, glslang's
// `EShLangVertex` / `EShLangFragment`. The two stages are *independent*
// translation units - one source per stage - so they are never routed through
// `compile_raster`, whose `/DVS`/`/DPS` contract belongs to the DSL codegen.
[[nodiscard]] NativeShaderCompileResult compile_raster_stage(
    Device *device, const NativeShaderCompileInfo &info, bool vertex) noexcept {
    NativeShaderCompileResult result;
    result.language = info.language;
    result.shader_model = info.shader_model;
    result.stage = vertex ? NativeShaderStage::Vertex : NativeShaderStage::Pixel;
    auto *stage = stage_name(vertex);
    if (info.source.empty()) {
        result.error = luisa::format(
            "A native raster shader {} stage requires a non-empty source.", stage);
        return result;
    }
    luisa::string source_storage;
    luisa::string_view source;
    luisa::vector<luisa::filesystem::path> include_dirs;
    if (!luisa::compute::detail::native_shader_resolve_source_and_include_dirs(
            info, source_storage, source, include_dirs, result.error)) {
        return result;
    }
    switch (info.language) {
        case NativeShaderLanguage::HLSL: {
#ifdef LC_NO_HLSL_BUILTIN
            static_cast<void>(device);
            static_cast<void>(source);
            static_cast<void>(include_dirs);
            result.error = luisa::format(
                "The Vulkan backend was built without DXC compatibility, so it "
                "cannot compile the HLSL {} stage of a native raster shader; use "
                "GLSL instead.",
                stage);
#else
            auto compiler = Device::compiler();
            if (compiler == nullptr) {
                result.error = "The Vulkan backend has no DXC compiler instance.";
                return result;
            }
            auto compiled = compiler->compile_graphics_stage(
                source, vertex, info.optimize, info.shader_model,
                info.enable_fast_math, /*spirv*/ true, info.enable_debug_info,
                info.entry_point, include_dirs);
            if (compiled.is_type_of<vstd::string>()) {
                result.error = luisa::string{compiled.get<1>()};
                LUISA_WARNING("Vulkan native raster shader compile failed (HLSL "
                              "-> SPIR-V via DXC, {} stage, entry point '{}'): {}",
                              stage, info.entry_point, result.error);
                return result;
            }
            auto &blob = compiled.get<0>();
            if (blob == nullptr || blob->GetBufferSize() == 0u) {
                result.error = luisa::format(
                    "DXC returned an empty SPIR-V module for the {} stage.", stage);
                return result;
            }
            result.binary.resize(blob->GetBufferSize());
            std::memcpy(result.binary.data(), blob->GetBufferPointer(),
                        blob->GetBufferSize());
#endif
        } break;
        case NativeShaderLanguage::GLSL: {
            auto compiled = compile_glsl_to_spirv(
                source, info.entry_point, info.optimize, info.enable_debug_info,
                include_dirs,
                vertex ? GlslShaderStage::Vertex : GlslShaderStage::Fragment);
            if (!compiled.ok()) {
                result.error = std::move(compiled.error);
                LUISA_WARNING("Vulkan native raster shader compile failed (GLSL "
                              "-> SPIR-V via glslang, {} stage): {}",
                              stage, result.error);
                return result;
            }
            result.binary = std::move(compiled.spirv);
        } break;
        case NativeShaderLanguage::CUDA_NVRTC: {
            result.error = luisa::format(
                "The Vulkan backend cannot compile CUDA C++; the {} stage of a "
                "native raster shader must be HLSL or GLSL.",
                stage);
        } break;
    }
    return result;
}

// Reflects one compiled raster stage and applies to it exactly the restrictions
// the compute route applies (a supported resource kind, one descriptor per
// binding), plus the graphics-stage contract: the module must declare the
// execution model that was asked for, which is what catches a vertex source
// handed in as the pixel stage (and the other way round).
//
// `reflected_push_constant_size` receives the module's push-constant block size.
// That block is the *whole* block the shader declares, i.e. it includes the
// object-id slot when the application asked for one, which is why the caller can
// only compare it against `object_id_bytes + uniform bytes`.
[[nodiscard]] bool reflect_raster_stage(
    Device *device, NativeShaderCompileResult &stage_result, bool vertex,
    uint32_t &reflected_push_constant_size) noexcept {
    auto *stage = stage_name(vertex);
    auto reflection = native_shader_reflection::parse_spirv(stage_result.binary);
    if (!reflection.ok()) {
        stage_result.error = luisa::format(
            "The native raster shader {} stage is not a readable SPIR-V module: {}",
            stage, reflection.error);
        return false;
    }
    auto expected_model = vertex ?
                              native_shader_reflection::spirv::ExecutionModelVertex :
                              native_shader_reflection::spirv::ExecutionModelFragment;
    // A module whose OpEntryPoint uses a model the reflection does not know
    // (ray generation, mesh, ...) leaves both `entry_point` empty and
    // `execution_model` at its GLCompute default, so the empty name is checked
    // first - it is the only signal that names the real problem.
    if (reflection.entry_point.empty()) {
        stage_result.error = luisa::format(
            "The native raster shader {} stage declares no usable entry point: a "
            "vertex stage needs an OpEntryPoint with the Vertex execution model and "
            "a pixel stage one with Fragment, but this module declares neither (is "
            "it a compute or ray-tracing shader?).",
            stage);
        return false;
    }
    if (reflection.execution_model != expected_model) {
        stage_result.error = luisa::format(
            "The native raster shader {} stage compiled to a SPIR-V module whose "
            "entry point has execution model {} instead of {}; are the vertex and "
            "pixel sources swapped?",
            stage, reflection.execution_model,
            static_cast<uint32_t>(expected_model));
        return false;
    }
    if (!check_required_features(device, reflection, stage_result)) {
        stage_result.error = luisa::format(
            "The native raster shader {} stage: {}", stage, stage_result.error);
        return false;
    }
    for (auto &&binding : reflection.bindings) {
        if (!is_supported_kind(binding.kind)) {
            stage_result.error = luisa::format(
                "The native raster shader {} stage declares a {} at set {} "
                "binding {}, which the Vulkan native-shader route cannot bind "
                "yet; only constant buffers and buffers are supported on vk.",
                stage, kind_name(binding.kind),
                binding.space_index, binding.register_index);
            return false;
        }
        if (binding.array_size != 1u) {
            stage_result.error = luisa::format(
                "The native raster shader {} stage declares an array of {} "
                "descriptors at set {} binding {}, which the Vulkan native-shader "
                "route does not support yet.",
                stage, binding.array_size,
                binding.space_index, binding.register_index);
            return false;
        }
    }
    reflected_push_constant_size = reflection.push_constant_size;
    stage_result.bindings = std::move(reflection.bindings);
    // The pipeline is created with this name (`pName`), so it must be the
    // reflected one; a module without an entry-point name falls back to main.
    stage_result.entry_point = reflection.entry_point.empty() ?
                                   luisa::string{"main"} :
                                   std::move(reflection.entry_point);
    // A graphics module has no workgroup size, so `block_size` stays {0,0,0}.
    return true;
}

// Merges the two stages' reflected descriptor tables into the single table one
// pipeline layout can describe: both stages share the layout, so a (set,
// binding) pair declared by both is one descriptor, and the two stages must agree
// on its resource kind. The result is in the canonical (space, register, kind)
// order, which is also the order `NativeShaderLauncher` binds by index.
[[nodiscard]] bool merge_raster_bindings(
    NativeRasterShaderCompileResult &result) noexcept {
    struct Entry {
        NativeShaderResourceBinding binding;
        bool vertex;
    };
    luisa::vector<Entry> entries;
    entries.reserve(result.vertex.bindings.size() + result.pixel.bindings.size());
    for (auto &&binding : result.vertex.bindings) {
        entries.emplace_back(Entry{binding, true});
    }
    for (auto &&binding : result.pixel.bindings) {
        entries.emplace_back(Entry{binding, false});
    }
    // Both tables arrive sorted; a stable sort by the shared canonical
    // comparator merges them without disturbing the vertex-before-pixel order
    // that makes an error message deterministic.
    std::stable_sort(
        entries.begin(), entries.end(),
        [](Entry const &a, Entry const &b) noexcept {
            return luisa::compute::detail::native_shader_binding_less(
                a.binding, b.binding);
        });
    luisa::vector<Entry> merged;
    merged.reserve(entries.size());
    for (auto &&entry : entries) {
        auto &&binding = entry.binding;
        if (!merged.empty()) {
            auto &&kept = merged.back().binding;
            if (kept.space_index == binding.space_index &&
                kept.register_index == binding.register_index) {
                if (kept.kind != binding.kind) {
                    // The sort orders by kind, so the retained entry is not
                    // necessarily the vertex stage's; report each side's stage.
                    result.error = luisa::format(
                        "The native raster shader stages disagree at set {} "
                        "binding {}: the {} stage declares a {}, the {} stage "
                        "declares a {}. One pipeline layout cannot satisfy both.",
                        binding.space_index, binding.register_index,
                        merged.back().vertex ? "vertex" : "pixel",
                        kind_name(kept.kind),
                        entry.vertex ? "vertex" : "pixel", kind_name(binding.kind));
                    return false;
                }
                // The same descriptor in both stages: it is bound once, with the
                // kind's default usage, and keeps the largest of the two reported
                // sizes (both describe the same buffer).
                kept.size_bytes = std::max(kept.size_bytes, binding.size_bytes);
                kept.stride = std::max(kept.stride, binding.stride);
                continue;
            }
        }
        merged.emplace_back(entry);
        // Reflect the default usage explicitly: it is what `load_raster()`
        // validates and what the launcher cross-checks its arguments against.
        merged.back().binding.usage = native_shader_default_usage(binding.kind);
    }
    result.bindings.clear();
    result.bindings.reserve(merged.size());
    for (auto &&entry : merged) {
        result.bindings.emplace_back(entry.binding);
    }
    return true;
}

}// namespace native_shader_ext_detail

VkNativeShaderExt::VkNativeShaderExt(Device *device) noexcept
    : NativeShaderExt{device}, _vk_device{device} {}

VkNativeShaderExt::~VkNativeShaderExt() noexcept {
    luisa::vector<std::pair<uint64_t, NativeShader *>> shaders;
    luisa::vector<std::pair<uint64_t, NativeRasterShader *>> raster_shaders;
    {
        std::lock_guard lock{_mutex};
        for (auto &&entry : _shaders) { shaders.emplace_back(entry); }
        _shaders.clear();
        for (auto &&entry : _raster_shaders) { raster_shaders.emplace_back(entry); }
        _raster_shaders.clear();
    }
    // Two warnings, not one: the two instance kinds are created through
    // different entry points, so the message has to name the `reset()`/destroy
    // call the application forgot.
    if (!shaders.empty()) {
        LUISA_WARNING(
            "The Vulkan native-shader extension is being destroyed with {} "
            "shader instance(s) still alive; a native shader must be destroyed "
            "(`NativeShader::reset()` / `destroy_shader()`) before its device.",
            shaders.size());
    }
    if (!raster_shaders.empty()) {
        LUISA_WARNING(
            "The Vulkan native-shader extension is being destroyed with {} "
            "native raster shader instance(s) still alive; a native raster "
            "shader must be destroyed (`NativeRasterShader::reset()` / "
            "`destroy_raster_shader()`) before its device.",
            raster_shaders.size());
    }
    for (auto &&[handle, shader] : shaders) {
        static_cast<void>(handle);
        delete shader;
    }
    for (auto &&[handle, shader] : raster_shaders) {
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

NativeRasterShader *VkNativeShaderExt::find_raster(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _raster_shaders.find(handle); it != _raster_shaders.end()) {
        return it->second;
    }
    return nullptr;
}

NativeShaderCompileResult VkNativeShaderExt::compile(
    const NativeShaderCompileInfo &info) noexcept {
    using namespace native_shader_ext_detail;
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
    using namespace native_shader_ext_detail;
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

NativeRasterShaderCompileResult VkNativeShaderExt::compile_raster(
    const NativeRasterShaderCompileInfo &info) noexcept {
    using namespace native_shader_ext_detail;
    NativeRasterShaderCompileResult result;
    // A failed compile reports empty binaries and a message, never a partially
    // usable pair (see the contract note in native_shader_ext.h).
    auto clear_on_error = vstd::scope_exit([&result] {
        if (result.error.empty()) { return; }
        result.vertex.binary.clear();
        result.pixel.binary.clear();
    });
    // Both errors are reported together: a user who swapped the two sources
    // should not have to recompile twice to see why the second stage failed.
    auto combine_stage_errors = [&result]() noexcept {
        luisa::string message;
        if (!result.vertex.error.empty()) {
            message += luisa::format("The vertex stage failed: {}",
                                     result.vertex.error);
        }
        if (!result.pixel.error.empty()) {
            if (!message.empty()) { message += ' '; }
            message += luisa::format("The pixel stage failed: {}",
                                     result.pixel.error);
        }
        return message;
    };

    // ---- 1. compile both stages ------------------------------------------
    result.vertex = compile_raster_stage(_vk_device, info.vertex, true);
    result.pixel = compile_raster_stage(_vk_device, info.pixel, false);
    if (!result.vertex.error.empty() || !result.pixel.error.empty()) {
        result.error = combine_stage_errors();
        LUISA_WARNING("Vulkan native raster shader compile failed: {}",
                      result.error);
        return result;
    }
    // ---- 2..4. reflect, check features, check the stage contract ---------
    auto vertex_push_constant_size = 0u;
    auto pixel_push_constant_size = 0u;
    auto vertex_ok = reflect_raster_stage(
        _vk_device, result.vertex, true, vertex_push_constant_size);
    auto pixel_ok = reflect_raster_stage(
        _vk_device, result.pixel, false, pixel_push_constant_size);
    if (!vertex_ok || !pixel_ok) {
        result.error = combine_stage_errors();
        LUISA_WARNING("Vulkan native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // ---- 5. merge the two descriptor tables ------------------------------
    if (!merge_raster_bindings(result)) {
        LUISA_WARNING("Vulkan native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // ---- 6. push constants -----------------------------------------------
    // The object-id slot sits in front of the application's uniform bytes, so a
    // module's push-constant block is `object_id_bytes + uniform bytes` long.
    result.object_id_bytes = info.push_object_id ? 4u : 0u;
    auto stage_max = std::max(vertex_push_constant_size, pixel_push_constant_size);
    auto uniform_bytes = std::max(info.vertex.push_constant_size,
                                  info.pixel.push_constant_size);
    if (uniform_bytes == 0u) {
        // Not declared by the application: take the largest reflected block.
        if (stage_max < result.object_id_bytes) {
            result.error = luisa::format(
                "NativeRasterShaderCompileInfo::push_object_id reserves {} "
                "push-constant bytes in front of the application's uniform "
                "block, but neither stage declares a push-constant block "
                "(reflected sizes: vertex {}, pixel {}).",
                result.object_id_bytes, vertex_push_constant_size,
                pixel_push_constant_size);
            LUISA_WARNING("Vulkan native raster shader reflection failed: {}",
                          result.error);
            return result;
        }
        uniform_bytes = stage_max - result.object_id_bytes;
    }
    if (vertex_push_constant_size > result.object_id_bytes + uniform_bytes ||
        pixel_push_constant_size > result.object_id_bytes + uniform_bytes) {
        result.error = luisa::format(
            "The native raster shader push-constant block is {} bytes in the "
            "vertex stage and {} bytes in the pixel stage, but the requested "
            "size covers only {} ({} object-id + {} uniform bytes).",
            vertex_push_constant_size, pixel_push_constant_size,
            result.object_id_bytes + uniform_bytes,
            result.object_id_bytes, uniform_bytes);
        LUISA_WARNING("Vulkan native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    if (uniform_bytes % sizeof(uint32_t) != 0u) {
        result.error = luisa::format(
            "The native raster shader uniform block must be a multiple of 4 "
            "bytes (got {}).",
            uniform_bytes);
        LUISA_WARNING("Vulkan native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    result.push_constant_size = uniform_bytes;
    // Mirror the shared block into both stages so that the result is
    // self-contained, exactly like the compute route's single-stage result.
    result.vertex.push_constant_size = uniform_bytes;
    result.pixel.push_constant_size = uniform_bytes;
    return result;
}

NativeShaderMetadata VkNativeShaderExt::load_raster(
    const NativeRasterShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    using namespace native_shader_ext_detail;
    NativeShaderMetadata metadata;
    metadata.language = result.vertex.language;
    metadata.push_constant_size = result.push_constant_size;
    metadata.object_id_bytes = result.object_id_bytes;
    // One instance holds both stages, so the field names the kind of shader
    // (`NativeShaderMetadata::is_raster`), not one of the two modules.
    metadata.stage = NativeShaderStage::Vertex;
    if (!result.ok()) {
        LUISA_WARNING("Native raster shader load_raster() called with a failed "
                      "compile result: {}",
                      result.error);
        return metadata;
    }
    // `NativeRasterShader` asserts on this; reject it here so that a
    // hand-assembled result fails closed instead of aborting.
    if (result.object_id_bytes != 0u && result.object_id_bytes != 4u) {
        LUISA_WARNING(
            "Native raster shader load_raster() received object_id_bytes = {}; "
            "the object-id slot is either absent (0) or one 32-bit word (4).",
            result.object_id_bytes);
        return metadata;
    }
    if (!usage_override.empty() &&
        usage_override.size() != result.bindings.size()) {
        LUISA_WARNING("Native raster shader load_raster() received {} usage "
                      "overrides for {} merged bindings.",
                      usage_override.size(), result.bindings.size());
        return metadata;
    }
    metadata.bindings = result.bindings;
    for (auto i = 0u; i < metadata.bindings.size(); i++) {
        auto &&binding = metadata.bindings[i];
        if (!usage_override.empty()) { binding.usage = usage_override[i]; }
        if (!is_supported_kind(binding.kind)) {
            LUISA_WARNING(
                "Native raster shader binding (set {}, binding {}) has kind {}, "
                "which the Vulkan native-shader route cannot bind yet; only "
                "constant buffers and buffers are supported on vk.",
                binding.space_index, binding.register_index, kind_name(binding.kind));
            return metadata;
        }
        if (binding.array_size != 1u) {
            LUISA_WARNING(
                "Native raster shader binding (set {}, binding {}) is an array "
                "of {} descriptors, which the Vulkan native-shader route does "
                "not support yet.",
                binding.space_index, binding.register_index, binding.array_size);
            return metadata;
        }
        // The same usage contract as `load()`: a read-only binding declared
        // writable is always invalid, and Usage::NONE is never valid. A
        // UAV-class binding declared read-only is rejected by
        // `NativeShaderLauncher::_plan()` at draw time, like on the compute
        // route, because only the launcher knows whether the application
        // allowed the override.
        auto writes = (luisa::to_underlying(binding.usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (binding.usage == Usage::NONE || (!is_uav && writes)) {
            LUISA_WARNING(
                "Native raster shader binding (set {}, binding {}) declares an "
                "invalid usage ({}); read-only bindings must be READ and "
                "writable bindings WRITE or READ_WRITE.",
                binding.space_index, binding.register_index,
                luisa::to_underlying(binding.usage));
            return metadata;
        }
    }
    auto entry_point_of = [](luisa::string const &name) noexcept {
        return name.empty() ? luisa::string_view{"main"} : luisa::string_view{name};
    };
    auto *shader = new NativeRasterShader(
        _vk_device, metadata.bindings,
        result.vertex.binary, entry_point_of(result.vertex.entry_point),
        result.pixel.binary, entry_point_of(result.pixel.entry_point),
        result.push_constant_size, result.object_id_bytes);
    auto handle = reinterpret_cast<uint64_t>(shader);
    {
        std::lock_guard lock{_mutex};
        _raster_shaders.try_emplace(handle, shader);
    }
    metadata.handle = handle;
    // A raster instance has no workgroup size: `block_size` stays {0,0,0}.
    return metadata;
}

void VkNativeShaderExt::destroy_raster_shader(uint64_t handle) noexcept {
    NativeRasterShader *shader = nullptr;
    {
        std::lock_guard lock{_mutex};
        if (auto it = _raster_shaders.find(handle); it != _raster_shaders.end()) {
            shader = it->second;
            _raster_shaders.erase(it);
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

VkRenderPass encode_native_raster_draw(
    Device *device, CommandBufferState *state, VkCommandBuffer cmdbuffer,
    DrawRasterSceneCommand const *cmd, luisa::span<uint const> strides,
    NativeRasterShader **shader) noexcept {
    auto *ext = static_cast<VkNativeShaderExt *>(
        device->extension(NativeShaderExt::name));
    if (ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The Vulkan device has no NativeShaderExt; native raster draws "
            "cannot be encoded.");
        return VK_NULL_HANDLE;
    }
    // Resolve the handle through the registry: a native raster draw's handle is a
    // `NativeRasterShader` instance, and an unregistered one must be reported
    // rather than dereferenced.
    auto *native_shader = ext->find_raster(cmd->handle());
    if (native_shader == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Native raster draw references an unknown shader instance (handle "
            "{}).",
            cmd->handle());
        return VK_NULL_HANDLE;
    }
    if (shader != nullptr) { *shader = native_shader; }
    // Binds the pipeline and this draw's descriptor sets, and pushes the
    // application's uniform block; VK_NULL_HANDLE means "already reported".
    return native_shader->encode_draw_state(state, cmdbuffer, cmd, strides);
}

}// namespace lc::vk
