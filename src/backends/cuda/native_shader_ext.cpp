#include "native_shader_ext.h"

#include <utility>

#include <luisa/core/logging.h>

#include "cuda_device.h"
#include "cuda_command_encoder.h"
#include "native_shader.h"
#include "native_shader_reflection.h"

namespace luisa::compute::cuda {

namespace {

// Native CUDA shader binding classes this route can express: a kernel parameter
// is either a buffer pointer or a scalar, so only the buffer classes appear.
[[nodiscard]] bool is_supported_kind(NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::StructuredBuffer:
        case NativeShaderResourceKind::RWStructuredBuffer:
        case NativeShaderResourceKind::ByteAddressBuffer:
        case NativeShaderResourceKind::RWByteAddressBuffer:
            return true;
        default: return false;
    }
}

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

}// namespace

CUDANativeShaderExt::CUDANativeShaderExt(CUDADevice *device) noexcept
    : NativeShaderExt{device}, _device{device} {}

CUDANativeShaderExt::~CUDANativeShaderExt() noexcept {
    luisa::vector<CUDANativeShader *> shaders;
    {
        std::lock_guard lock{_mutex};
        for (auto &&entry : _shaders) { shaders.emplace_back(entry.second); }
        _shaders.clear();
    }
    if (!shaders.empty()) {
        LUISA_WARNING(
            "The CUDA native-shader extension is being destroyed with {} "
            "shader instance(s) still alive; a native shader must be destroyed "
            "(`NativeShader::reset()` / `destroy_shader()`) before its device.",
            shaders.size());
    }
    if (!shaders.empty()) {
        // The extension outlives the device's CUDA context (CUDADevice declares
        // its `Handle` first), so the modules can still be unloaded here.
        _device->with_handle([&shaders] {
            for (auto *shader : shaders) { delete_with_allocator(shader); }
        });
    }
}

CUDANativeShader *CUDANativeShaderExt::_find(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _shaders.find(handle); it != _shaders.end()) {
        return it->second;
    }
    return nullptr;
}

NativeShaderCompileResult CUDANativeShaderExt::compile(
    const NativeShaderCompileInfo &info) noexcept {
    NativeShaderCompileResult result;
    result.language = NativeShaderLanguage::CUDA_NVRTC;
    if (info.language != NativeShaderLanguage::CUDA_NVRTC) {
        result.error =
            "The CUDA backend compiles CUDA C++ source only; pass "
            "NativeShaderLanguage::CUDA_NVRTC (HLSL and GLSL are not supported "
            "on this backend).";
        return result;
    }
    if (info.source.empty()) {
        result.error = "Native shader compile requires a non-empty source.";
        return result;
    }
    auto compiler = _device->compiler();
    if (compiler == nullptr) {
        result.error = "The CUDA backend has no NVRTC compiler instance.";
        return result;
    }
    // The parameter roles (buffer vs scalar, read-only vs writable) only exist
    // in the source, so parse it first: the entry-point name this returns also
    // selects the kernel in the compiled module.
    auto reflection = native_shader::reflect_source(info.source, info.entry_point);
    if (!reflection.ok()) {
        result.error = reflection.error;
        return result;
    }
    // NVRTC options. The generated PTX is loaded by the driver of the very same
    // device, so the architecture is the device's compute capability.
    luisa::vector<luisa::string> option_storage;
    option_storage.reserve(8u);// keep the `c_str()` pointers below stable
    option_storage.emplace_back(luisa::format(
        "-arch=compute_{}", _device->handle().compute_capability()));
    option_storage.emplace_back("--std=c++17");
    if (!info.optimize) { option_storage.emplace_back("-G"); }
    if (info.enable_fast_math) { option_storage.emplace_back("-use_fast_math"); }
    if (info.enable_debug_info) { option_storage.emplace_back("-lineinfo"); }
    luisa::vector<const char *> options;
    options.reserve(option_storage.size());
    for (auto &&option : option_storage) { options.emplace_back(option.c_str()); }
    auto filename = info.file_name.empty() ?
                        luisa::string{"native_shader.cu"} :
                        luisa::string{info.file_name};
    auto ptx = compiler->compile(luisa::string{info.source}, filename,
                                 options, nullptr);
    if (ptx.empty()) {
        result.error = luisa::format(
            "NVRTC produced an empty PTX module for kernel '{}'.",
            reflection.entry_point);
        return result;
    }
    // `cuModuleLoadData` requires a NUL-terminated PTX image (NVRTC already
    // terminates its output; be explicit so a hand-built result is accepted too).
    if (ptx.back() != std::byte{0}) { ptx.emplace_back(std::byte{0}); }
    auto ptx_text = luisa::string_view{
        reinterpret_cast<const char *>(ptx.data()), ptx.size()};
    auto module = native_shader::reflect_ptx(ptx_text, reflection.entry_point);
    if (!module.ok()) {
        result.error = module.error;
        return result;
    }
    auto merged = native_shader::merge(std::move(module), reflection);
    if (!merged.ok()) {
        result.error = merged.error;
        return result;
    }
    // workgroup size: declared, or the kernel's `__launch_bounds__(N)`
    auto block_size = info.block_size;
    auto declares_block_size = block_size.x != 0u || block_size.y != 0u ||
                               block_size.z != 0u;
    if (declares_block_size) {
        if (block_size.x == 0u || block_size.y == 0u || block_size.z == 0u) {
            result.error = luisa::format(
                "Declared block size ({}, {}, {}) is incomplete; all three "
                "components must be nonzero.",
                block_size.x, block_size.y, block_size.z);
            return result;
        }
        if (any(merged.block_size != make_uint3(0u)) &&
            any(merged.block_size != block_size)) {
            result.error = luisa::format(
                "Declared block size ({}, {}, {}) does not match the kernel's "
                "`__launch_bounds__` ({}, {}, {}).",
                block_size.x, block_size.y, block_size.z,
                merged.block_size.x, merged.block_size.y, merged.block_size.z);
            return result;
        }
    } else {
        block_size = merged.block_size;
        if (any(block_size == make_uint3(0u))) {
            result.error = luisa::format(
                "Could not determine the workgroup size of kernel '{}'; "
                "declare `__launch_bounds__(N)` or pass "
                "NativeShaderCompileInfo::block_size.",
                merged.entry_point);
            return result;
        }
    }
    if (info.push_constant_size != 0u &&
        info.push_constant_size != merged.uniform_bytes) {
        result.error = luisa::format(
            "Declared push-constant size ({} bytes) does not match the {} "
            "byte(s) of scalar kernel parameters of '{}'.",
            info.push_constant_size, merged.uniform_bytes, merged.entry_point);
        return result;
    }
    // the reflection: one binding per pointer parameter, in declaration order
    for (auto i = 0u; i < merged.parameters.size(); i++) {
        auto &&parameter = merged.parameters[i];
        if (!parameter.is_buffer) { continue; }
        NativeShaderResourceBinding binding;
        binding.kind = parameter.read_only ?
                           NativeShaderResourceKind::StructuredBuffer :
                           NativeShaderResourceKind::RWStructuredBuffer;
        binding.register_index = i;
        binding.space_index = 0u;
        binding.array_size = 1u;
        binding.usage = native_shader_default_usage(binding.kind);
        result.bindings.emplace_back(binding);
    }
    // canonical order == binding order == buffer-parameter order
    detail::native_shader_canonicalize_bindings(result.bindings);
    result.block_size = block_size;
    result.push_constant_size = merged.uniform_bytes;
    result.entry_point = std::move(merged.entry_point);
    result.binary = std::move(ptx);
    return result;
}

NativeShaderMetadata CUDANativeShaderExt::load(
    const NativeShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    NativeShaderMetadata metadata;
    metadata.language = NativeShaderLanguage::CUDA_NVRTC;
    metadata.push_constant_size = result.push_constant_size;
    if (!result.ok()) {
        LUISA_WARNING("Native shader load() called with a failed compile "
                      "result: {}",
                      result.error);
        return metadata;
    }
    if (result.language != NativeShaderLanguage::CUDA_NVRTC) {
        LUISA_WARNING("Native shader load() received a result that was not "
                      "compiled by the CUDA route (language {}).",
                      luisa::to_underlying(result.language));
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
                "Native CUDA shader binding (parameter {}) has kind {}, which "
                "the CUDA native-shader route cannot bind; only buffers are "
                "supported on this route.",
                binding.register_index, kind_name(binding.kind));
            return metadata;
        }
        if (binding.array_size != 1u) {
            LUISA_WARNING(
                "Native CUDA shader binding (parameter {}) is an array of {} "
                "resources, which the CUDA native-shader route does not support.",
                binding.register_index, binding.array_size);
            return metadata;
        }
        auto writes = (luisa::to_underlying(binding.usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (binding.usage == Usage::NONE || (is_uav ? false : writes)) {
            LUISA_WARNING(
                "Native CUDA shader binding (parameter {}) declares an invalid "
                "usage ({}); read-only buffer pointers must be READ and "
                "writable ones must be WRITE or READ_WRITE.",
                binding.register_index, luisa::to_underlying(binding.usage));
            return metadata;
        }
    }
    // Re-reflect the module for the launch-time parameter layout (the PTX is the
    // authority for the sizes the driver copies into the kernel).
    auto ptx_text = luisa::string_view{
        reinterpret_cast<const char *>(result.binary.data()),
        result.binary.size()};
    auto module = native_shader::reflect_ptx(ptx_text, result.entry_point);
    if (!module.ok()) {
        LUISA_WARNING("Native CUDA shader load() could not reflect the PTX "
                      "module: {}",
                      module.error);
        return metadata;
    }
    luisa::vector<CUDANativeShader::Parameter> parameters;
    parameters.reserve(module.parameters.size());
    for (auto i = 0u; i < module.parameters.size(); i++) {
        auto is_buffer = false;
        for (auto &&binding : metadata.bindings) {
            if (binding.space_index == 0u && binding.register_index == i) {
                is_buffer = true;
                break;
            }
        }
        if (is_buffer && module.parameters[i].size != sizeof(uint64_t)) {
            LUISA_WARNING(
                "Native CUDA shader parameter {} is a reflected buffer but "
                "occupies {} bytes in the compiled PTX; the CUDA route requires "
                "64-bit device addresses.",
                i, module.parameters[i].size);
            return metadata;
        }
        parameters.emplace_back(CUDANativeShader::Parameter{
            module.parameters[i].size,
            module.parameters[i].alignment,
            is_buffer});
    }
    auto uniform_bytes = 0u;
    for (auto &&parameter : parameters) {
        if (!parameter.is_buffer) { uniform_bytes += parameter.size; }
    }
    if (uniform_bytes != result.push_constant_size) {
        LUISA_WARNING(
            "Native CUDA shader '{}' has {} byte(s) of scalar kernel parameters "
            "but the compile result declares {}; load() refuses the mismatch.",
            module.entry_point, uniform_bytes, result.push_constant_size);
        return metadata;
    }
    auto block_size = result.block_size;
    if (block_size.x == 0u || block_size.y == 0u || block_size.z == 0u) {
        LUISA_WARNING("Native CUDA shader '{}' has no block size.", module.entry_point);
        return metadata;
    }
    auto shader = _device->with_handle([&]() noexcept -> CUDANativeShader * {
        luisa::string error;
        auto p = CUDANativeShader::create(
            result.binary, module.entry_point, block_size,
            std::move(parameters), error);
        if (p == nullptr) {
            LUISA_WARNING("Native CUDA shader '{}' could not be loaded: {}",
                          module.entry_point, error);
        }
        return p;
    });
    if (shader == nullptr) { return metadata; }
    metadata.block_size = block_size;
    metadata.handle = reinterpret_cast<uint64_t>(shader);
    {
        std::lock_guard lock{_mutex};
        _shaders.emplace(metadata.handle, shader);
    }
    LUISA_VERBOSE("Loaded native CUDA shader '{}' ({} buffer binding(s), {} "
                  "byte(s) of scalar parameters, block size {}x{}x{}).",
                  shader->entry(), shader->buffer_parameter_count(),
                  shader->uniform_bytes(), block_size.x, block_size.y,
                  block_size.z);
    return metadata;
}

void CUDANativeShaderExt::destroy_shader(uint64_t handle) noexcept {
    CUDANativeShader *shader = nullptr;
    {
        std::lock_guard lock{_mutex};
        if (auto it = _shaders.find(handle); it != _shaders.end()) {
            shader = it->second;
            _shaders.erase(it);
        }
    }
    if (shader == nullptr) {
        LUISA_WARNING("destroy_shader() called with an unknown native CUDA "
                      "shader handle 0x{:016x}.",
                      handle);
        return;
    }
    _device->with_handle([shader] { delete_with_allocator(shader); });
}

void CUDANativeShaderExt::encode(
    CUDACommandEncoder *encoder,
    NativeShaderDispatchCommand const *command) noexcept {
    auto shader = _find(command->shader_handle());
    if (shader == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Native shader dispatch references an unknown CUDA shader handle "
            "0x{:016x}; the shader may have been destroyed already.",
            command->shader_handle());
    }
    shader->launch(*encoder, command);
}

void encode_native_shader_dispatch(
    CUDADevice *device, CUDACommandEncoder *encoder,
    NativeShaderDispatchCommand const *command) noexcept {
    auto extension = device->extension(NativeShaderExt::name);
    if (extension == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The CUDA device has no NativeShaderExt.");
    }
    static_cast<CUDANativeShaderExt *>(extension)->encode(encoder, command);
}

}// namespace luisa::compute::cuda
