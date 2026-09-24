#include "native_shader_ext.h"

#include <Windows.h>
#include <d3dx12.h>
#include <d3d12shader.h>

#include <limits>

#include <DXApi/LCDevice.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <luisa/core/logging.h>
#include "../../common/hlsl/shader_compiler.h"

namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] NativeShaderResourceKind map_resource_kind(
    D3D_SHADER_INPUT_TYPE type, D3D_SRV_DIMENSION dimension) noexcept {
    switch (type) {
        case D3D_SIT_CBUFFER: return NativeShaderResourceKind::ConstantBuffer;
        case D3D_SIT_TBUFFER: return NativeShaderResourceKind::TypedBuffer;
        case D3D_SIT_TEXTURE:
            switch (dimension) {
                case D3D_SRV_DIMENSION_TEXTURE3D: return NativeShaderResourceKind::Texture3D;
                case D3D_SRV_DIMENSION_BUFFER: return NativeShaderResourceKind::TypedBuffer;
                default: return NativeShaderResourceKind::Texture2D;
            }
        case D3D_SIT_SAMPLER: return NativeShaderResourceKind::Sampler;
        case D3D_SIT_UAV_RWTYPED: return NativeShaderResourceKind::RWTypedBuffer;
        case D3D_SIT_STRUCTURED: return NativeShaderResourceKind::StructuredBuffer;
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
            return NativeShaderResourceKind::RWStructuredBuffer;
        case D3D_SIT_BYTEADDRESS: return NativeShaderResourceKind::ByteAddressBuffer;
        case D3D_SIT_UAV_RWBYTEADDRESS: return NativeShaderResourceKind::RWByteAddressBuffer;
        case D3D_SIT_RTACCELERATIONSTRUCTURE:
            return NativeShaderResourceKind::AccelerationStructure;
        default: return NativeShaderResourceKind::TypedBuffer;
    }
}

// Native DX binding classes this iteration can bind.
[[nodiscard]] bool is_supported_kind(NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer:
        case NativeShaderResourceKind::StructuredBuffer:
        case NativeShaderResourceKind::RWStructuredBuffer:
        case NativeShaderResourceKind::ByteAddressBuffer:
        case NativeShaderResourceKind::RWByteAddressBuffer:
            return true;
        default: return false;
    }
}

[[nodiscard]] hlsl::ShaderVariableType map_property_type(
    NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer: return hlsl::ShaderVariableType::ConstantBuffer;
        case NativeShaderResourceKind::RWStructuredBuffer:
        case NativeShaderResourceKind::RWByteAddressBuffer:
            return hlsl::ShaderVariableType::RWStructuredBuffer;
        default: return hlsl::ShaderVariableType::StructuredBuffer;
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

[[nodiscard]] uint32_t ceil_div(uint32_t value, uint32_t divisor) noexcept {
    return (value + divisor - 1u) / divisor;
}

}// namespace

DxNativeShaderExt::DxNativeShaderExt(LCDevice *device) noexcept
    : NativeShaderExt{device}, _lc_device{device} {}

DxNativeShaderExt::~DxNativeShaderExt() noexcept {
    luisa::vector<ComputeShader *> shaders;
    {
        std::lock_guard lock{_mutex};
        for (auto &&entry : _shaders) { shaders.emplace_back(entry.second.shader); }
        _shaders.clear();
    }
    if (!shaders.empty()) {
        LUISA_WARNING(
            "The DirectX native-shader extension is being destroyed with {} "
            "shader instance(s) still alive; a native shader must be destroyed "
            "(`NativeShader::reset()` / `destroy_shader()`) before its device.",
            shaders.size());
    }
    for (auto *shader : shaders) { delete shader; }
}

DxNativeShaderExt::Entry *DxNativeShaderExt::_find(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _shaders.find(handle); it != _shaders.end()) {
        return &it->second;
    }
    return nullptr;
}

NativeShaderCompileResult DxNativeShaderExt::compile(
    const NativeShaderCompileInfo &info) noexcept {
    NativeShaderCompileResult result;
    result.shader_model = info.shader_model;
    result.push_constant_size = info.push_constant_size;
    if (info.language == NativeShaderLanguage::GLSL) {
        result.error = "GLSL is not supported by the DirectX backend; compile "
                       "the shader to HLSL or use the Vulkan backend.";
        return result;
    }
    if (info.source.empty()) {
        result.error = "Native shader compile requires a non-empty source.";
        return result;
    }
    if (info.push_constant_size % sizeof(uint32_t) != 0u) {
        result.error = "Native shader push-constant size must be a multiple of "
                       "4 bytes (a root 32-bit constant is 4 bytes wide).";
        return result;
    }
    if (info.push_constant_size > 64u * sizeof(uint32_t)) {
        result.error = "Native shader uniform block exceeds the DirectX "
                       "root-constant limit of 64 32-bit values (256 bytes).";
        return result;
    }
    auto compiler = Device::compiler();
    if (compiler == nullptr) {
        result.error = "DirectX backend has no DXC compiler instance.";
        return result;
    }
    auto compiled = compiler->compile_compute(
        info.source, info.optimize, info.shader_model,
        info.enable_fast_math, /*spirv*/ false, info.enable_debug_info,
        info.entry_point);
    if (compiled.is_type_of<vstd::string>()) {
        result.error = luisa::string{compiled.get<1>()};
        return result;
    }
    auto &blob = compiled.get<0>();
    if (blob == nullptr || blob->GetBufferSize() == 0u) {
        result.error = "DXC returned an empty DXIL blob.";
        return result;
    }
    // Reflection (R5: `d3d12shader.h` ships with the Windows/DX SDK the
    // backend already builds against).
    DxcBuffer dxc_buffer{blob->GetBufferPointer(), blob->GetBufferSize(), DXC_CP_ACP};
    Microsoft::WRL::ComPtr<ID3D12ShaderReflection> reflection;
    auto hr = compiler->utils()->CreateReflection(
        &dxc_buffer, IID_PPV_ARGS(reflection.GetAddressOf()));
    if (FAILED(hr) || reflection == nullptr) {
        result.error = "DXC could not reflect the compiled DXIL module; the "
                       "module may not carry a reflection part.";
        return result;
    }
    D3D12_SHADER_DESC description{};
    hr = reflection->GetDesc(&description);
    if (FAILED(hr)) {
        result.error = "ID3D12ShaderReflection::GetDesc failed.";
        return result;
    }
    // R15: the workgroup size must be known so that the launcher cannot
    // silently disagree with `[numthreads]`.
    UINT size_x = 0u, size_y = 0u, size_z = 0u;
    reflection->GetThreadGroupSize(&size_x, &size_y, &size_z);
    result.block_size = uint3{size_x, size_y, size_z};
    if (size_x == 0u || size_y == 0u || size_z == 0u) {
        if (info.block_size.x > 0u && info.block_size.y > 0u &&
            info.block_size.z > 0u) {
            result.block_size = info.block_size;
        } else {
            result.error = "Could not determine the native shader's workgroup "
                           "size; declare `[numthreads(x,y,z)]` or pass "
                           "NativeShaderCompileInfo::block_size.";
            return result;
        }
    }
    if (info.block_size.x > 0u || info.block_size.y > 0u ||
        info.block_size.z > 0u) {
        if (info.block_size.x != result.block_size.x ||
            info.block_size.y != result.block_size.y ||
            info.block_size.z != result.block_size.z) {
            result.error = luisa::format(
                "Declared block size ({}, {}, {}) does not match the shader's "
                "[numthreads] ({}, {}, {}).",
                info.block_size.x, info.block_size.y, info.block_size.z,
                result.block_size.x, result.block_size.y, result.block_size.z);
            return result;
        }
    }
    for (auto i = 0u; i < description.BoundResources; i++) {
        D3D12_SHADER_INPUT_BIND_DESC binding{};
        if (FAILED(reflection->GetResourceBindingDesc(i, &binding))) {
            continue;
        }
        NativeShaderResourceBinding reflected;
        reflected.kind = map_resource_kind(binding.Type, binding.Dimension);
        reflected.register_index = binding.BindPoint;
        reflected.space_index = binding.Space;
        reflected.array_size = binding.BindCount == 0u ? 1u : binding.BindCount;
        reflected.usage = native_shader_default_usage(reflected.kind);
        result.bindings.emplace_back(reflected);
    }
    // Canonical order == root-parameter order == launcher positional order.
    luisa::compute::detail::native_shader_canonicalize_bindings(result.bindings);
    result.binary.resize(blob->GetBufferSize());
    std::memcpy(result.binary.data(), blob->GetBufferPointer(),
                blob->GetBufferSize());
    return result;
}

NativeShaderMetadata DxNativeShaderExt::load(
    const NativeShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    NativeShaderMetadata metadata;
    metadata.language = NativeShaderLanguage::HLSL;
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
                "Native shader binding (register {}, space {}) has kind {}, "
                "which the DirectX native-shader route cannot bind yet; only "
                "constant buffers, structured buffers and byte-address buffers "
                "are supported on dx.",
                binding.register_index, binding.space_index,
                kind_name(binding.kind));
            return metadata;
        }
        if (binding.array_size != 1u) {
            LUISA_WARNING(
                "Native shader binding (register {}, space {}) is an array of "
                "{} resources, which the DirectX native-shader route does not "
                "support yet.",
                binding.register_index, binding.space_index, binding.array_size);
            return metadata;
        }
        auto writes = (luisa::to_underlying(binding.usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (binding.usage == Usage::NONE || (is_uav ? false : writes)) {
            LUISA_WARNING(
                "Native shader binding (register {}, space {}) declares an "
                "invalid usage ({}); SRV/CBV bindings must be READ and UAV "
                "bindings must be WRITE or READ_WRITE.",
                binding.register_index, binding.space_index,
                luisa::to_underlying(binding.usage));
            return metadata;
        }
        if (luisa::to_underlying(binding.usage) ==
                luisa::to_underlying(Usage::READ_WRITE) &&
            !is_uav) {
            LUISA_WARNING(
                "Native shader binding (register {}, space {}) is an SRV/CBV "
                "bound as READ_WRITE; read-only resources must be declared "
                "Usage::READ.",
                binding.register_index, binding.space_index);
            return metadata;
        }
    }

    auto device = &_lc_device->native_device;
    uint32_t uniform_property_index = ~0u;
    if (result.push_constant_size > 0u) {
        auto value_count = static_cast<uint32_t>(result.push_constant_size /
                                                 sizeof(uint32_t));
        if (value_count > 64u) {
            LUISA_WARNING(
                "Native shader uniform block ({} bytes) exceeds the DirectX "
                "root-constant limit of 64 32-bit values (256 bytes).",
                result.push_constant_size);
            return metadata;
        }
        // The launcher's uniform blob replaces the shader's uniform block: the
        // shader declares `cbuffer ... : register(b0)` and the root signature
        // feeds it with root 32-bit constants (which is why the reflected
        // constant buffer must not also be bound as a CBV, and why it is not
        // reported as a resource binding).
        auto uniform_binding = metadata.bindings.end();
        for (auto it = metadata.bindings.begin(); it != metadata.bindings.end(); ++it) {
            if (it->kind == NativeShaderResourceKind::ConstantBuffer &&
                it->register_index == 0u && it->space_index == 0u) {
                uniform_binding = it;
                break;
            }
        }
        if (uniform_binding == metadata.bindings.end()) {
            LUISA_WARNING(
                "Native shader declares a uniform block of {} bytes "
                "(NativeShaderCompileInfo::push_constant_size) but no "
                "`cbuffer ... : register(b0)`; the launcher's add_uniform "
                "values would have nowhere to go on dx. Declare the block at "
                "register(b0) or pass push_constant_size = 0 and bind a "
                "buffer to the reflected constant buffer instead.",
                result.push_constant_size);
            return metadata;
        }
        if (uniform_binding->size_bytes > result.push_constant_size) {
            LUISA_WARNING(
                "Native shader uniform block declares {} bytes but the "
                "DirectX root-constant block is only {} bytes.",
                uniform_binding->size_bytes, result.push_constant_size);
            return metadata;
        }
        metadata.bindings.erase(uniform_binding);
    }

    vstd::vector<hlsl::Property> properties;
    properties.reserve(metadata.bindings.size() + 1u);
    for (auto &&binding : metadata.bindings) {
        hlsl::Property property{
            map_property_type(binding.kind),
            binding.space_index,
            binding.register_index,
            binding.array_size};
        properties.emplace_back(property);
    }
    if (result.push_constant_size > 0u) {
        auto value_count = static_cast<uint32_t>(result.push_constant_size /
                                                 sizeof(uint32_t));
        uniform_property_index = static_cast<uint32_t>(properties.size());
        // `SerializeRootSig` reads `space_index` as the 32-bit value count and
        // `register_index` as the shader register (see the ConstantValue case).
        hlsl::Property uniform_property{
            hlsl::ShaderVariableType::ConstantValue,
            value_count,
            0u,
            1u};
        properties.emplace_back(uniform_property);
    }

    auto *shader = new ComputeShader(
        result.block_size,
        std::move(properties),
        vstd::vector<SavedArgument>{},// the native encode path binds itself
        luisa::span<std::byte const>{
            result.binary.data(), result.binary.size()},
        vstd::vector<luisa::compute::Argument>{},
        vstd::vector<std::pair<vstd::string, Type const *>>{},
        0u,
        device);
    auto handle = reinterpret_cast<uint64_t>(shader);
    {
        std::lock_guard lock{_mutex};
        _shaders.try_emplace(
            handle, Entry{shader, uniform_property_index,
                          result.push_constant_size});
    }
    metadata.handle = handle;
    metadata.block_size = result.block_size;
    return metadata;
}

void DxNativeShaderExt::destroy_shader(uint64_t handle) noexcept {
    ComputeShader *shader = nullptr;
    {
        std::lock_guard lock{_mutex};
        if (auto it = _shaders.find(handle); it != _shaders.end()) {
            shader = it->second.shader;
            _shaders.erase(it);
        }
    }
    if (shader != nullptr) { delete shader; }
}

void DxNativeShaderExt::encode(
    CommandBufferBuilder *builder,
    NativeShaderDispatchCommand const *cmd) noexcept {
    auto *entry = _find(cmd->shader_handle());
    if (entry == nullptr || entry->shader == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Native shader dispatch references an unknown shader instance "
            "(handle {}).",
            cmd->shader_handle());
        return;
    }
    auto *shader = entry->shader;
    auto cmd_list = builder->get_cb()->cmd_list();
    cmd_list->SetComputeRootSignature(shader->root_sig());
    cmd_list->SetPipelineState(shader->pso());
    // Every uniform argument of the launcher shares one root 32-bit constant
    // block (the shader's `cbuffer ... : register(b0)`), so the whole block is
    // written once, from the first uniform argument.
    auto uniform_property = entry->uniform_property_index;
    if (uniform_property != ~0u) {
        auto uniform_begin = std::numeric_limits<uint64_t>::max();
        auto uniform_end = uint64_t{0u};
        for (auto &&argument : cmd->arguments()) {
            if (argument.tag == Argument::Tag::UNIFORM) {
                uniform_begin = std::min<uint64_t>(uniform_begin,
                                                   argument.uniform.offset);
                uniform_end = std::max<uint64_t>(uniform_end,
                                                 argument.uniform.offset +
                                                     argument.uniform.size);
            }
        }
        if (uniform_end > uniform_begin) {
            auto bytes = uniform_end - uniform_begin;
            LUISA_ASSERT(bytes % sizeof(uint32_t) == 0u,
                         "Native shader uniform payload ({} bytes) is not a "
                         "multiple of 4 bytes.",
                         bytes);
            LUISA_ASSERT(bytes <= entry->uniform_size,
                         "Native shader uniform payload ({} bytes) exceeds the "
                         "declared uniform block size ({} bytes).",
                         bytes, entry->uniform_size);
            auto payload = cmd->uniform(Argument::Uniform{
                static_cast<size_t>(uniform_begin),
                static_cast<size_t>(bytes), 4u});
            cmd_list->SetComputeRoot32BitConstants(
                uniform_property,
                static_cast<UINT>(bytes / sizeof(uint32_t)),
                payload.data(), 0u);
        }
    }
    auto arguments = cmd->arguments();
    for (auto i = 0u; i < arguments.size(); i++) {
        auto &&argument = arguments[i];
        switch (argument.tag) {
            case Argument::Tag::BUFFER: {
                shader->set_compute_resource(
                    i, builder,
                    BufferView{reinterpret_cast<Buffer const *>(argument.buffer.handle),
                               argument.buffer.offset,
                               argument.buffer.size});
            } break;
            case Argument::Tag::UNIFORM:
                // Written once for the whole block above.
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Native shader dispatch argument {} is not supported by "
                    "the DirectX encoding path (textures, bindless arrays and "
                    "acceleration structures are rejected at load()).",
                    i);
        }
    }
    auto block_size = cmd->block_size();
    auto threads = cmd->dispatch_size();
    cmd_list->Dispatch(ceil_div(threads.x, block_size.x),
                       ceil_div(threads.y, block_size.y),
                       ceil_div(threads.z, block_size.z));
}

void encode_native_shader_dispatch(
    LCDevice *device, CommandBufferBuilder *builder,
    NativeShaderDispatchCommand const *cmd) noexcept {
    auto *ext = static_cast<DxNativeShaderExt *>(
        device->extension(NativeShaderExt::name));
    if (ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The DirectX device has no NativeShaderExt; native shader "
            "dispatches cannot be encoded.");
        return;
    }
    ext->encode(builder, cmd);
}

}// namespace lc::dx
