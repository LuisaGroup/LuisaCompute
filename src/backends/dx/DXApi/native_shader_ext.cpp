#include "native_shader_ext.h"

#include <Windows.h>
#include <d3dx12.h>
#include <d3d12shader.h>

#include <cstring>
#include <limits>

#include <DXApi/LCDevice.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/Device.h>
#include <Resource/Buffer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <Shader/ShaderSerializer.h>
#include <luisa/core/logging.h>
#include "../../common/hlsl/shader_compiler.h"

namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;

// File-local helpers. A named detail namespace (not an anonymous one): this
// translation unit is merged with sibling files by the unity build, where an
// anonymous namespace is shared with them.
namespace native_shader_ext_detail {

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

// Copies a compile result's binary into the `vstd::vector<std::byte>` a
// `RasterShader` keeps as its stage blob.
[[nodiscard]] vstd::vector<std::byte> to_binary_vector(
    luisa::vector<std::byte> const &binary) noexcept {
    vstd::vector<std::byte> result;
    luisa::enlarge_by(result, binary.size());
    if (!binary.empty()) {
        std::memcpy(result.data(), binary.data(), binary.size());
    }
    return result;
}

[[nodiscard]] const char *stage_name(bool vertex) noexcept {
    return vertex ? "vertex" : "pixel";
}

// One DXIL module's reflection, in the shape both native routes need. The two
// routes differ only in what they do with it: `compile()` reports every binding
// (including the `register(b0)` block, which `load()` later turns into root 32-bit
// constants), while `compile_raster()` needs the block's size up front, because the
// two stages have to agree on one root-constant width before the merged table is
// built.
// One entry of a module's input/output signature. `D3D12_SIGNATURE_PARAMETER_DESC`'s
// `SemanticName` points into the reflection object, which dies with it, so the name
// is copied into an owned string.
struct SignatureParameter {
    D3D12_SIGNATURE_PARAMETER_DESC desc{};
    luisa::string semantic_name;
};

struct ReflectedModule {
    // Every reflected resource binding, in canonical (space, register, kind) order
    // (`native_shader_canonicalize_bindings`), with `size_bytes` filled for
    // constant-buffer bindings.
    vstd::vector<NativeShaderResourceBinding> bindings;
    // The `cbuffer ... : register(b0, space0)` block, i.e. the one the launcher's
    // uniform values feed through root 32-bit constants.
    bool has_uniform_block{false};
    // Bytes the shader *reads* from that block: the highest `offset + size` over
    // its variables. `D3D12_SHADER_BUFFER_DESC::Size` rounds the block up to a
    // 16-byte register, so comparing it against the size the application declares
    // would reject a block that holds exactly what the application pushes (e.g.
    // `cbuffer { float4 tint; float scale; }` reads 20 bytes and allocates 32).
    uint32_t uniform_block_used_size{0u};
    // `D3D12_SHADER_BUFFER_DESC::Size` of that block (0 when there is none).
    uint32_t uniform_block_allocated_size{0u};
    // The module's own input and output signatures. The raster route uses them to
    // tell whether the two sources were handed in the right slots, and to check the
    // stage-to-stage varying linkage, which is a much clearer diagnostic than the
    // PSO-creation failure it would otherwise cause.
    luisa::vector<SignatureParameter> inputs;
    luisa::vector<SignatureParameter> outputs;
    luisa::string error;
};

// Creates a reflection object for a compiled DXIL blob. `-Qstrip_reflect` is never
// passed to DXC, so the DXIL container keeps its reflection part.
[[nodiscard]] bool create_dxil_reflection(
    IDxcUtils *utils, IDxcBlob *blob,
    Microsoft::WRL::ComPtr<ID3D12ShaderReflection> &reflection,
    luisa::string &error) noexcept {
    DxcBuffer dxc_buffer{blob->GetBufferPointer(), blob->GetBufferSize(), DXC_CP_ACP};
    auto hr = utils->CreateReflection(
        &dxc_buffer, IID_PPV_ARGS(reflection.GetAddressOf()));
    if (FAILED(hr) || reflection == nullptr) {
        error = "DXC could not reflect the compiled DXIL module; the "
                "module may not carry a reflection part.";
        return false;
    }
    return true;
}

// Fills `result` from a DXIL reflection: the resource bindings (with the
// constant-buffer sizes the SPIR-V route reads from its decorations), the
// `register(b0)` block's size, and the module's input/output signatures.
[[nodiscard]] bool reflect_dxil_module(
    ID3D12ShaderReflection *reflection,
    ReflectedModule &result) noexcept {
    D3D12_SHADER_DESC description{};
    if (FAILED(reflection->GetDesc(&description))) {
        result.error = "ID3D12ShaderReflection::GetDesc failed.";
        return false;
    }
    // The bindings, in declaration order for now (canonicalized at the end). The
    // names are borrowed from the reflection object, which outlives this call.
    luisa::vector<luisa::string_view> binding_names;
    // Parallel to `binding_names`: whether the cbuffer pass below already explained
    // that binding, so a second cbuffer of the same name (`$Globals` is emitted once
    // per space that declares globals) does not overwrite the first one's size.
    luisa::vector<uint8_t> binding_explained;
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
        binding_names.emplace_back(
            binding.Name == nullptr ? luisa::string_view{} :
                                      luisa::string_view{binding.Name});
        binding_explained.emplace_back(0u);
    }
    // Constant buffers: fill the binding's `size_bytes` with the bytes the shader
    // reads, and remember the `register(b0, space0)` block. DXIL reflection lists
    // cbuffers separately from resource bindings - a `D3D12_SHADER_BUFFER_DESC`
    // carries no register or space - so the two can only be matched by name, and
    // only the first unused candidate is taken.
    for (auto i = 0u; i < description.ConstantBuffers; i++) {
        auto *cbuffer = reflection->GetConstantBufferByIndex(i);
        if (cbuffer == nullptr) { continue; }
        D3D12_SHADER_BUFFER_DESC cbuffer_desc{};
        if (FAILED(cbuffer->GetDesc(&cbuffer_desc))) { continue; }
        auto used_size = 0u;
        for (auto v = 0u; v < cbuffer_desc.Variables; v++) {
            auto *variable = cbuffer->GetVariableByIndex(v);
            if (variable == nullptr) { continue; }
            D3D12_SHADER_VARIABLE_DESC variable_desc{};
            if (FAILED(variable->GetDesc(&variable_desc))) { continue; }
            used_size = std::max(used_size, variable_desc.StartOffset +
                                                variable_desc.Size);
        }
        auto name = luisa::string_view{cbuffer_desc.Name == nullptr ?
                                           "" :
                                           cbuffer_desc.Name};
        for (auto b = 0u; b < binding_names.size(); b++) {
            if (binding_explained[b] != 0u || binding_names[b] != name) { continue; }
            auto &&binding = result.bindings[b];
            if (binding.kind != NativeShaderResourceKind::ConstantBuffer) { continue; }
            binding_explained[b] = 1u;
            binding.size_bytes = used_size;
            if (binding.space_index == 0u && binding.register_index == 0u) {
                result.has_uniform_block = true;
                result.uniform_block_used_size = used_size;
                result.uniform_block_allocated_size = cbuffer_desc.Size;
            }
            break;
        }
    }
    // Canonical order == root-parameter order == launcher positional order.
    luisa::compute::detail::native_shader_canonicalize_bindings(result.bindings);
    auto read_signature = [reflection](auto get, UINT count,
                                       luisa::vector<SignatureParameter> &into) {
        into.reserve(count);
        for (auto i = 0u; i < count; i++) {
            SignatureParameter parameter;
            if (FAILED(get(i, &parameter.desc))) { continue; }
            if (parameter.desc.SemanticName != nullptr) {
                parameter.semantic_name = parameter.desc.SemanticName;
            }
            into.emplace_back(std::move(parameter));
        }
    };
    read_signature(
        [reflection](UINT i, D3D12_SIGNATURE_PARAMETER_DESC *d) {
            return reflection->GetInputParameterDesc(i, d);
        },
        description.InputParameters, result.inputs);
    read_signature(
        [reflection](UINT i, D3D12_SIGNATURE_PARAMETER_DESC *d) {
            return reflection->GetOutputParameterDesc(i, d);
        },
        description.OutputParameters, result.outputs);
    return true;
}

// A graphics stage must write the system value its slot owns: `SV_Position` for
// the vertex stage, `SV_Target` - or a depth output, for a depth-only pipeline with
// no render target - for the pixel stage. Both are what the fixed pipeline reads, so
// a module that declares neither was compiled for the other slot, which is what a
// swapped pair looks like once the per-stage `-T` profile happens to accept the
// source.
[[nodiscard]] bool check_raster_stage_signature(
    ReflectedModule const &reflection, bool vertex,
    luisa::string &error) noexcept {
    auto writes = [&reflection](D3D_NAME name) noexcept {
        for (auto &&parameter : reflection.outputs) {
            if (parameter.desc.SystemValueType == name) { return true; }
        }
        return false;
    };
    auto *stage = stage_name(vertex);
    if (vertex) {
        if (!writes(D3D_NAME_POSITION)) {
            error = luisa::format(
                "The native raster shader {} stage does not write SV_Position; a "
                "vertex stage must (are the vertex and pixel sources swapped?).",
                stage);
            return false;
        }
    } else if (!writes(D3D_NAME_TARGET) && !writes(D3D_NAME_DEPTH) &&
               !writes(D3D_NAME_DEPTH_GREATER_EQUAL) &&
               !writes(D3D_NAME_DEPTH_LESS_EQUAL)) {
        error = luisa::format(
            "The native raster shader {} stage writes neither SV_Target nor "
            "SV_Depth; a pixel stage must write one of them (are the vertex and "
            "pixel sources swapped?).",
            stage);
        return false;
    }
    return true;
}

// DirectX semantic names are case-insensitive (`COLOR`, `Color` and `color` all name
// the same varying), so the stage-to-stage comparison has to fold case: a pair that
// spells a varying differently in its two stages links and renders correctly.
[[nodiscard]] bool semantic_name_equal(luisa::string const &a,
                                       luisa::string const &b) noexcept {
    if (a.size() != b.size()) { return false; }
    // ASCII-only folding, which is all a semantic name can contain.
    auto fold = [](char c) noexcept {
        auto u = static_cast<unsigned char>(c);
        constexpr auto upper_a = static_cast<unsigned char>('A');
        constexpr auto upper_z = static_cast<unsigned char>('Z');
        constexpr auto lower_bit = static_cast<unsigned char>('a' - 'A');
        return u >= upper_a && u <= upper_z ?
                   static_cast<unsigned char>(u | lower_bit) :
                   u;
    };
    for (auto i = 0u; i < a.size(); i++) {
        if (fold(a[i]) != fold(b[i])) { return false; }
    }
    return true;
}

// DirectX links the vertex stage's outputs to the pixel stage's inputs by
// *hardware register*, not by semantic location: D3D12 rejects a PSO whose
// signatures "are incompatible ... Semantic 'X' is defined for mismatched hardware
// registers between the output stage and input stage", with a bare E_INVALIDARG.
// Since DXC assigns those registers in declaration order, two HLSL stages agree only
// when they declare the same interpolators in the same order (the DirectX
// counterpart of an explicit `[[vk::location]]`).
//
// Only user varyings are compared: system values (`SV_Position`, `SV_ClipDistance`,
// ...) are fed by the rasterizer or the fixed-function stage, not by the linkage,
// and a pixel stage may of course consume fewer varyings than the vertex stage
// writes.
[[nodiscard]] luisa::string check_raster_varying_linkage(
    ReflectedModule const &vertex, ReflectedModule const &pixel) noexcept {
    for (auto &&input : pixel.inputs) {
        auto &&in = input.desc;
        if (in.SystemValueType != D3D_NAME_UNDEFINED) { continue; }
        auto *matched = static_cast<D3D12_SIGNATURE_PARAMETER_DESC const *>(nullptr);
        for (auto &&output : vertex.outputs) {
            auto &&out = output.desc;
            if (out.SystemValueType != D3D_NAME_UNDEFINED) { continue; }
            if (semantic_name_equal(output.semantic_name, input.semantic_name) &&
                out.SemanticIndex == in.SemanticIndex) {
                matched = &out;
                break;
            }
        }
        if (matched == nullptr) {
            return luisa::format(
                "The native raster shader pixel stage reads the varying '{}' "
                "(semantic index {}) but the vertex stage does not write it.",
                input.semantic_name, in.SemanticIndex);
        }
        if (matched->Register != in.Register) {
            return luisa::format(
                "The native raster shader stages disagree on the hardware register "
                "of the varying '{}' (semantic index {}): the vertex stage writes it "
                "at register {} and the pixel stage reads it at register {}. DirectX "
                "links the stages by register, so both must declare their "
                "interpolators in the same order (listing `SV_Position` first in the "
                "pixel stage's input struct is the usual fix).",
                input.semantic_name, in.SemanticIndex, matched->Register, in.Register);
        }
    }
    return {};
}

// Compiles one stage of a native raster pair to DXIL and reflects it. The two
// stages are *independent* translation units - one source per stage - so they are
// never routed through `compile_raster` (the DSL codegen's `/DVS`/`/DPS` contract).
[[nodiscard]] bool compile_raster_stage(
    NativeShaderCompileInfo const &info, bool vertex,
    NativeShaderCompileResult &stage_result,
    ReflectedModule &reflection) noexcept {
    auto *stage = stage_name(vertex);
    stage_result.language = info.language;
    stage_result.shader_model = info.shader_model;
    stage_result.stage = vertex ? NativeShaderStage::Vertex : NativeShaderStage::Pixel;
    switch (info.language) {
        case NativeShaderLanguage::HLSL: break;
        case NativeShaderLanguage::GLSL:
            stage_result.error = luisa::format(
                "GLSL is not supported by the DirectX backend; the {} stage of a "
                "native raster shader must be HLSL (the Vulkan backend compiles "
                "GLSL as well).",
                stage);
            return false;
        case NativeShaderLanguage::CUDA_NVRTC:
            stage_result.error = luisa::format(
                "The DirectX backend cannot compile CUDA C++; the {} stage of a "
                "native raster shader must be HLSL.",
                stage);
            return false;
    }
    if (info.source.empty()) {
        stage_result.error = luisa::format(
            "A native raster shader {} stage requires a non-empty source.", stage);
        return false;
    }
    luisa::string source_storage;
    luisa::string_view source;
    luisa::vector<luisa::filesystem::path> include_dirs;
    if (!luisa::compute::detail::native_shader_resolve_source_and_include_dirs(
            info, source_storage, source, include_dirs, stage_result.error)) {
        return false;
    }
    auto compiler = Device::compiler();
    if (compiler == nullptr) {
        stage_result.error = "The DirectX backend has no DXC compiler instance.";
        return false;
    }
    auto compiled = compiler->compile_graphics_stage(
        source, vertex, info.optimize, info.shader_model,
        info.enable_fast_math, /*spirv*/ false, info.enable_debug_info,
        info.entry_point, include_dirs);
    if (compiled.is_type_of<vstd::string>()) {
        stage_result.error = luisa::string{compiled.get<1>()};
        LUISA_WARNING("DirectX native raster shader compile failed (HLSL -> DXIL "
                      "via DXC, {} stage, entry point '{}'): {}",
                      stage, info.entry_point, stage_result.error);
        return false;
    }
    auto &blob = compiled.get<0>();
    if (blob == nullptr || blob->GetBufferSize() == 0u) {
        stage_result.error = luisa::format(
            "DXC returned an empty DXIL blob for the {} stage.", stage);
        return false;
    }
    Microsoft::WRL::ComPtr<ID3D12ShaderReflection> shader_reflection;
    if (!create_dxil_reflection(compiler->utils(), blob.get(),
                                shader_reflection, reflection.error) ||
        !reflect_dxil_module(shader_reflection.Get(), reflection)) {
        stage_result.error = reflection.error;
        return false;
    }
    if (!check_raster_stage_signature(reflection, vertex, stage_result.error)) {
        return false;
    }
    // The pipeline is built from the module itself (the DXIL container carries the
    // entry point), so unlike on Vulkan this name is reporting only: it is the name
    // the caller asked for, or the compiler's default `main`. DXIL reflection does
    // not expose the entry point name (`D3D12_SHADER_DESC` has no field for it).
    stage_result.entry_point = info.entry_point.empty() ?
                                   luisa::string{"main"} :
                                   luisa::string{info.entry_point};
    stage_result.binary.resize(blob->GetBufferSize());
    std::memcpy(stage_result.binary.data(), blob->GetBufferPointer(),
                blob->GetBufferSize());
    return true;
}

// Merges the two stages' reflected binding tables into the single table one root
// signature can describe: both stages share the pipeline, so a (space, register)
// pair declared by both is one root parameter, and the two stages must agree on its
// resource kind. The result is in the canonical (space, register, kind) order,
// which is also the order `NativeShaderLauncher` binds by index. Mirrors the Vulkan
// route, with the DirectX register/space wording.
//
// `uniform_block` says whether the modules' `register(b0, space0)` cbuffer is
// consumed as the shader's root-constant block: when it is, it is *not* a bindable
// resource and is dropped from both stages' tables (a stage that does not declare
// it simply reads no uniforms), because the root signature reserves that register
// for the launcher's values.
[[nodiscard]] bool merge_raster_bindings(
    NativeRasterShaderCompileResult &result, bool uniform_block) noexcept {
    struct Entry {
        NativeShaderResourceBinding binding;
        bool vertex;
    };
    auto is_uniform_block = [uniform_block](NativeShaderResourceBinding const &b) noexcept {
        return uniform_block &&
               b.kind == NativeShaderResourceKind::ConstantBuffer &&
               b.space_index == 0u && b.register_index == 0u;
    };
    luisa::vector<Entry> entries;
    entries.reserve(result.vertex.bindings.size() + result.pixel.bindings.size());
    for (auto &&binding : result.vertex.bindings) {
        if (!is_uniform_block(binding)) { entries.emplace_back(Entry{binding, true}); }
    }
    for (auto &&binding : result.pixel.bindings) {
        if (!is_uniform_block(binding)) { entries.emplace_back(Entry{binding, false}); }
    }
    // Both tables arrive sorted; a stable sort by the shared canonical comparator
    // merges them without disturbing the vertex-before-pixel order that makes an
    // error message deterministic.
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
                    // necessarily the vertex stage's; report each side's stage. Two
                    // entries of *one* stage happen because a DirectX shader has
                    // separate register namespaces (`t0` and `u0` are both bind point
                    // 0), which the binding table cannot distinguish.
                    if (merged.back().vertex == entry.vertex) {
                        result.error = luisa::format(
                            "The native raster shader {} stage declares both a {} and "
                            "a {} at space {} register {}: DirectX register namespaces "
                            "are not part of the binding table, so one of the two has "
                            "to use another register.",
                            entry.vertex ? "vertex" : "pixel",
                            kind_name(kept.kind), kind_name(binding.kind),
                            binding.space_index, binding.register_index);
                    } else {
                        result.error = luisa::format(
                            "The native raster shader stages disagree at space {} "
                            "register {}: the {} stage declares a {}, the {} stage "
                            "declares a {}. One root signature cannot satisfy both.",
                            binding.space_index, binding.register_index,
                            merged.back().vertex ? "vertex" : "pixel",
                            kind_name(kept.kind),
                            entry.vertex ? "vertex" : "pixel", kind_name(binding.kind));
                    }
                    return false;
                }
                // The same resource in both stages: it is bound once, with the
                // kind's default usage, and keeps the largest of the two reported
                // sizes (both describe the same buffer).
                kept.size_bytes = std::max(kept.size_bytes, binding.size_bytes);
                // `stride` needs no reconciliation: DXIL reflection reports no
                // element stride, so both stages leave it at 0.
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

DxNativeShaderExt::DxNativeShaderExt(LCDevice *device) noexcept
    : NativeShaderExt{device}, _lc_device{device} {}

DxNativeShaderExt::~DxNativeShaderExt() noexcept {
    luisa::vector<ComputeShader *> shaders;
    luisa::vector<RasterShader *> raster_shaders;
    {
        std::lock_guard lock{_mutex};
        for (auto &&entry : _shaders) { shaders.emplace_back(entry.second.shader); }
        _shaders.clear();
        for (auto &&entry : _raster_shaders) { raster_shaders.emplace_back(entry.second.shader); }
        _raster_shaders.clear();
    }
    // Two warnings, not one: the two instance kinds are created through different
    // entry points, so the message has to name the call the application forgot.
    if (!shaders.empty()) {
        LUISA_WARNING(
            "The DirectX native-shader extension is being destroyed with {} "
            "shader instance(s) still alive; a native shader must be destroyed "
            "(`NativeShader::reset()` / `destroy_shader()`) before its device.",
            shaders.size());
    }
    if (!raster_shaders.empty()) {
        LUISA_WARNING(
            "The DirectX native-shader extension is being destroyed with {} "
            "native raster shader instance(s) still alive; a native raster shader "
            "must be destroyed (`NativeRasterShader::reset()` / "
            "`destroy_raster_shader()`) before its device.",
            raster_shaders.size());
    }
    for (auto *shader : shaders) { delete shader; }
    for (auto *shader : raster_shaders) { delete shader; }
}

DxNativeShaderExt::Entry *DxNativeShaderExt::_find(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _shaders.find(handle); it != _shaders.end()) {
        return &it->second;
    }
    return nullptr;
}

DxNativeShaderExt::RasterEntry *DxNativeShaderExt::_find_raster(uint64_t handle) noexcept {
    std::lock_guard lock{_mutex};
    if (auto it = _raster_shaders.find(handle); it != _raster_shaders.end()) {
        return &it->second;
    }
    return nullptr;
}

bool DxNativeShaderExt::find_raster(
    uint64_t handle, NativeRasterShaderState &state) noexcept {
    auto *entry = _find_raster(handle);
    if (entry == nullptr || entry->shader == nullptr) { return false; }
    state = *entry;
    return true;
}

NativeShaderCompileResult DxNativeShaderExt::compile(
    const NativeShaderCompileInfo &info) noexcept {
    using namespace native_shader_ext_detail;
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
    // Resolve FilePath sources: read the file into `source_storage`; its
    // directory leads the include search so quoted includes next to the file work.
    luisa::string source_storage;
    luisa::string_view source;
    luisa::vector<luisa::filesystem::path> include_dirs;
    if (!luisa::compute::detail::native_shader_resolve_source_and_include_dirs(
            info, source_storage, source, include_dirs, result.error)) {
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
        source, info.optimize, info.shader_model,
        info.enable_fast_math, /*spirv*/ false, info.enable_debug_info,
        info.entry_point, include_dirs);
    if (compiled.is_type_of<vstd::string>()) {
        result.error = luisa::string{compiled.get<1>()};
        LUISA_WARNING("DirectX native shader compile failed (HLSL -> DXIL via "
                      "DXC, entry point '{}'): {}",
                      info.entry_point, result.error);
        return result;
    }
    auto &blob = compiled.get<0>();
    if (blob == nullptr || blob->GetBufferSize() == 0u) {
        result.error = "DXC returned an empty DXIL blob.";
        return result;
    }
    // Reflection (R5: `d3d12shader.h` ships with the Windows/DX SDK the
    // backend already builds against).
    Microsoft::WRL::ComPtr<ID3D12ShaderReflection> reflection;
    if (!create_dxil_reflection(compiler->utils(), blob.get(), reflection,
                                result.error)) {
        return result;
    }
    ReflectedModule reflected;
    if (!reflect_dxil_module(reflection.Get(), reflected)) {
        result.error = std::move(reflected.error);
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
    result.bindings = std::move(reflected.bindings);
    result.binary.resize(blob->GetBufferSize());
    std::memcpy(result.binary.data(), blob->GetBufferPointer(),
                blob->GetBufferSize());
    return result;
}

NativeShaderMetadata DxNativeShaderExt::load(
    const NativeShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    using namespace native_shader_ext_detail;
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

NativeRasterShaderCompileResult DxNativeShaderExt::compile_raster(
    const NativeRasterShaderCompileInfo &info) noexcept {
    using namespace native_shader_ext_detail;
    NativeRasterShaderCompileResult result;
    // A failed compile reports empty binaries and a message, never a partially
    // usable pair.
    auto clear_on_error = vstd::scope_exit([&result] {
        if (result.error.empty()) { return; }
        result.vertex.binary.clear();
        result.pixel.binary.clear();
    });
    // Both errors are reported together: a user who swapped the two sources should
    // not have to recompile twice to see why the second stage failed.
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

    // ---- 1. compile and reflect both stages -------------------------------
    ReflectedModule vertex_reflection;
    ReflectedModule pixel_reflection;
    auto vertex_ok = compile_raster_stage(
        info.vertex, true, result.vertex, vertex_reflection);
    auto pixel_ok = compile_raster_stage(
        info.pixel, false, result.pixel, pixel_reflection);

    if (!vertex_ok || !pixel_ok) {
        result.error = combine_stage_errors();
        LUISA_WARNING("DirectX native raster shader compile failed: {}",
                      result.error);
        return result;
    }
    // ---- 2. the compute route's binding-class restrictions ----------------
    for (auto vertex : {true, false}) {
        auto &reflection = vertex ? vertex_reflection : pixel_reflection;
        auto &stage_result = vertex ? result.vertex : result.pixel;
        for (auto &&binding : reflection.bindings) {
            if (!is_supported_kind(binding.kind)) {
                stage_result.error = luisa::format(
                    "The native raster shader {} stage declares a {} at register "
                    "{}, space {}, which the DirectX native-shader route cannot "
                    "bind yet; only constant buffers, structured buffers and "
                    "byte-address buffers are supported on dx.",
                    stage_name(vertex), kind_name(binding.kind),
                    binding.register_index, binding.space_index);
                result.error = combine_stage_errors();
                LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                              result.error);
                return result;
            }
            if (binding.array_size != 1u) {
                stage_result.error = luisa::format(
                    "The native raster shader {} stage declares an array of {} "
                    "resources at register {}, space {}, which the DirectX "
                    "native-shader route does not support yet.",
                    stage_name(vertex), binding.array_size,
                    binding.register_index, binding.space_index);
                result.error = combine_stage_errors();
                LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                              result.error);
                return result;
            }
        }
    }
    // The stages' own tables become part of the result (the merge below reads
    // them, and a per-stage table is what a caller reads to see which stage
    // declared what).
    result.vertex.bindings = std::move(vertex_reflection.bindings);
    result.pixel.bindings = std::move(pixel_reflection.bindings);
    // ---- 2b. the stage-to-stage varying linkage --------------------------
    // DirectX links the two stages by *hardware register*: a varying the pixel
    // stage reads must be one the vertex stage writes at the same register, which
    // for HLSL means declaring the interpolators in the same order on both sides.
    // A mismatch is not caught by the per-stage compile - it makes
    // `CreateGraphicsPipelineState` fail with a bare E_INVALIDARG - so it is
    // reported here instead.
    if (auto linkage_error = check_raster_varying_linkage(
            vertex_reflection, pixel_reflection);
        !linkage_error.empty()) {
        result.error = std::move(linkage_error);
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // ---- 3. the root-constant block --------------------------------------
    // Its width is `object_id_bytes + uniform bytes`, so a module's reflected
    // block is the *whole* block, object-id slot included.
    result.object_id_bytes = info.push_object_id ? 4u : 0u;
    auto stage_max = std::max(vertex_reflection.uniform_block_used_size,
                              pixel_reflection.uniform_block_used_size);
    auto uniform_bytes = std::max(info.vertex.push_constant_size,
                                  info.pixel.push_constant_size);
    if (uniform_bytes == 0u) {
        // Not declared by the application: take the largest reflected block.
        if (stage_max < result.object_id_bytes) {
            result.error = luisa::format(
                "NativeRasterShaderCompileInfo::push_object_id reserves {} bytes "
                "in front of the application's uniform block, but neither stage "
                "declares a `cbuffer ... : register(b0)` block (reflected sizes: "
                "vertex {}, pixel {}).",
                result.object_id_bytes,
                vertex_reflection.uniform_block_used_size,
                pixel_reflection.uniform_block_used_size);
            LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                          result.error);
            return result;
        }
        uniform_bytes = stage_max - result.object_id_bytes;
    }
    if (vertex_reflection.uniform_block_used_size >
            result.object_id_bytes + uniform_bytes ||
        pixel_reflection.uniform_block_used_size >
            result.object_id_bytes + uniform_bytes) {
        result.error = luisa::format(
            "The native raster shader uniform block is {} bytes in the vertex "
            "stage and {} bytes in the pixel stage (as allocated by DirectX: {} "
            "and {}), but the requested size covers only {} ({} object-id + {} "
            "uniform bytes).",
            vertex_reflection.uniform_block_used_size,
            pixel_reflection.uniform_block_used_size,
            vertex_reflection.uniform_block_allocated_size,
            pixel_reflection.uniform_block_allocated_size,
            result.object_id_bytes + uniform_bytes,
            result.object_id_bytes, uniform_bytes);
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    if (uniform_bytes % sizeof(uint32_t) != 0u) {
        result.error = luisa::format(
            "The native raster shader uniform block must be a multiple of 4 "
            "bytes (got {}).",
            uniform_bytes);
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // Capacity: one root-parameter slot per binding plus the uniform block, and
    // the hardware's 64-dword root-constant budget for the block itself.
    auto uniform_block = result.object_id_bytes + uniform_bytes > 0u;
    if (result.object_id_bytes + uniform_bytes > 64u * sizeof(uint32_t)) {
        result.error = luisa::format(
            "The native raster shader root-constant block ({} object-id + {} "
            "uniform bytes) exceeds the DirectX limit of 64 32-bit values "
            "(256 bytes).",
            result.object_id_bytes, uniform_bytes);
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    if (uniform_block &&
        (!vertex_reflection.has_uniform_block &&
         !pixel_reflection.has_uniform_block)) {
        result.error = luisa::format(
            "The native raster shader declares {} uniform bytes "
            "(NativeRasterShaderCompileInfo::push_constant_size, plus the "
            "object-id slot) but neither stage declares a "
            "`cbuffer ... : register(b0)`; the launcher's add_uniform() values "
            "would have nowhere to go on dx. Declare the block at register(b0) or "
            "pass push_constant_size = 0.",
            uniform_bytes);
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // ---- 4. merge the two binding tables ---------------------------------
    if (!merge_raster_bindings(result, uniform_block)) {
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    // `SerializeRootSig` holds the root parameters in a `fixed_vector<..., 32>`, and
    // D3D12 allows 64; the tighter of the two is the serializer's.
    if (result.bindings.size() + (uniform_block ? 1u : 0u) > 32u) {
        result.error = luisa::format(
            "The native raster shader declares {} root parameters ({} bindings "
            "plus the uniform block), but a DirectX root signature built by this "
            "backend holds at most 32.",
            result.bindings.size() + (uniform_block ? 1u : 0u),
            result.bindings.size());
        LUISA_WARNING("DirectX native raster shader reflection failed: {}",
                      result.error);
        return result;
    }
    result.push_constant_size = uniform_bytes;
    // Mirror the shared block into both stages so that the result is self-contained,
    // exactly like the compute route's single-stage result.
    result.vertex.push_constant_size = uniform_bytes;
    result.pixel.push_constant_size = uniform_bytes;
    return result;
}

NativeShaderMetadata DxNativeShaderExt::load_raster(
    const NativeRasterShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    using namespace native_shader_ext_detail;
    NativeShaderMetadata metadata;
    metadata.language = NativeShaderLanguage::HLSL;
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
    // 0 or one 32-bit word: anything else is a hand-assembled result.
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
                "Native raster shader binding (register {}, space {}) has kind "
                "{}, which the DirectX native-shader route cannot bind yet; only "
                "constant buffers, structured buffers and byte-address buffers "
                "are supported on dx.",
                binding.register_index, binding.space_index,
                kind_name(binding.kind));
            return metadata;
        }
        if (binding.array_size != 1u) {
            LUISA_WARNING(
                "Native raster shader binding (register {}, space {}) is an "
                "array of {} resources, which the DirectX native-shader route "
                "does not support yet.",
                binding.register_index, binding.space_index, binding.array_size);
            return metadata;
        }
        // The same usage contract as `load()`: a read-only binding declared
        // writable is always invalid, and Usage::NONE is never valid. A UAV-class
        // binding declared read-only is rejected by `NativeShaderLauncher::_plan()`
        // at draw time, like on the compute route, because only the launcher knows
        // whether the application allowed the override.
        auto writes = (luisa::to_underlying(binding.usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (binding.usage == Usage::NONE || (!is_uav && writes)) {
            LUISA_WARNING(
                "Native raster shader binding (register {}, space {}) declares "
                "an invalid usage ({}); read-only bindings must be READ and "
                "writable bindings WRITE or READ_WRITE.",
                binding.register_index, binding.space_index,
                luisa::to_underlying(binding.usage));
            return metadata;
        }
    }
    auto device = _lc_device->native_device.device.Get();
    // One root parameter per binding, in the merged table's order, plus the
    // root-constant block (whose `space_index` is its 32-bit word count and whose
    // `register_index` is the b-register — the serializer's documented inversion).
    vstd::vector<hlsl::Property> properties;
    properties.reserve(metadata.bindings.size() + 1u);
    for (auto &&binding : metadata.bindings) {
        properties.emplace_back(hlsl::Property{
            map_property_type(binding.kind),
            binding.space_index,
            binding.register_index,
            binding.array_size});
    }
    auto uniform_dwords = static_cast<uint32_t>(
        (result.object_id_bytes + result.push_constant_size) / sizeof(uint32_t));
    auto uniform_property_index = ~0u;
    if (uniform_dwords > 0u) {
        if (uniform_dwords > 64u) {
            LUISA_WARNING(
                "Native raster shader root-constant block ({} bytes) exceeds the "
                "DirectX limit of 64 32-bit values (256 bytes).",
                result.object_id_bytes + result.push_constant_size);
            return metadata;
        }
        if (properties.size() >= 32u) {
            LUISA_WARNING(
                "Native raster shader declares {} bindings, which leaves no room "
                "for its uniform block in a DirectX root signature built by this "
                "backend (32 parameters).",
                properties.size());
            return metadata;
        }
        uniform_property_index = static_cast<uint32_t>(properties.size());
        properties.emplace_back(hlsl::Property{
            hlsl::ShaderVariableType::ConstantValue,
            uniform_dwords,
            0u,
            1u});
    }
    // `isRasterShader` is what sets ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT (without it
    // CreateGraphicsPipelineState rejects the PSO), while `append_object_id` is
    // false because this route reserves the object-id dword *inside* the block
    // above instead of appending a second root parameter for `register(b0)`.
    ComPtr<ID3D12RootSignature> root_sig;
    {
        auto serialized = ShaderSerializer::SerializeRootSig(
            properties, /*isRasterShader*/ true, /*append_object_id*/ false);
        if (serialized == nullptr ||
            FAILED(device->CreateRootSignature(
                0,
                serialized->GetBufferPointer(),
                serialized->GetBufferSize(),
                IID_PPV_ARGS(root_sig.GetAddressOf()))) ||
            root_sig == nullptr) {
            LUISA_WARNING(
                "The DirectX device rejected the native raster shader's root "
                "signature ({} root parameter(s)); the shader cannot be created.",
                properties.size());
            return metadata;
        }
    }
    // The md5 names the PSO disk-cache entry (`RasterShader::get_pso`), so it has to
    // be content-derived: two native pairs must not share one.
    luisa::vector<uint8_t> md5_source;
    auto append_binary = [&md5_source](luisa::vector<std::byte> const &binary) {
        auto offset = md5_source.size();
        luisa::enlarge_by(md5_source, binary.size());
        std::memcpy(md5_source.data() + offset, binary.data(), binary.size());
    };
    append_binary(result.vertex.binary);
    append_binary(result.pixel.binary);
    auto md5 = vstd::MD5{vstd::span<uint8_t const>{md5_source.data(),
                                                   md5_source.size()}};
    auto *shader = RasterShader::create_native_raster_shader(
        &_lc_device->native_device,
        md5,
        std::move(properties),
        std::move(root_sig),
        to_binary_vector(result.vertex.binary),
        to_binary_vector(result.pixel.binary));
    auto handle = reinterpret_cast<uint64_t>(shader);
    {
        std::lock_guard lock{_mutex};
        auto inserted = _raster_shaders.try_emplace(
            handle, RasterEntry{shader, uniform_property_index,
                                result.push_constant_size,
                                result.object_id_bytes});
        // A `RasterShader` is hand-allocated, so a live handle is unique; a
        // collision would mean a double insert (and a leaked one of the two).
        LUISA_ASSERT(inserted.second, "Duplicate native raster shader handle {}",
                     handle);
    }
    metadata.handle = handle;
    // A raster instance has no workgroup size: `block_size` stays {0,0,0}.
    return metadata;
}

void DxNativeShaderExt::destroy_raster_shader(uint64_t handle) noexcept {
    RasterShader *shader = nullptr;
    {
        std::lock_guard lock{_mutex};
        if (auto it = _raster_shaders.find(handle); it != _raster_shaders.end()) {
            shader = it->second.shader;
            _raster_shaders.erase(it);
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

bool find_native_raster_shader(
    LCDevice *device, uint64_t handle,
    NativeRasterShaderState &state) noexcept {
    auto *ext = static_cast<DxNativeShaderExt *>(
        device->extension(NativeShaderExt::name));
    if (ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "The DirectX device has no NativeShaderExt; native raster draws "
            "cannot be encoded.");
        return false;
    }
    if (!ext->find_raster(handle, state)) {
        LUISA_ERROR_WITH_LOCATION(
            "Native raster draw references an unknown shader instance (handle "
            "{}).",
            handle);
        return false;
    }
    return true;
}

bool encode_native_raster_draw(
    LCDevice *device, CommandBufferBuilder *builder,
    DrawRasterSceneCommand const *cmd, ID3D12PipelineState *pso) noexcept {
    NativeRasterShaderState state;
    // Resolving here as well (the caller had to resolve it to build the PSO) keeps
    // this entry point fail-closed on its own.
    if (!find_native_raster_shader(device, cmd->handle(), state)) { return false; }
    auto *shader = state.shader;
    if (shader == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Native raster draw references a shader instance that was already "
            "destroyed (handle {}).",
            cmd->handle());
        return false;
    }
    // Root parameter index == index in the property vector, and the property vector
    // is [one per merged binding][the root-constant block], so the resource
    // arguments (which the launcher packs in binding order) map 1:1 onto the first
    // `binding_count` properties.
    auto property_count = shader->properties().size();
    auto binding_count = state.uniform_property_index == ~0u ?
                             property_count :
                             static_cast<size_t>(state.uniform_property_index);
    auto resource_count = size_t{0u};
    auto arguments = cmd->arguments();
    for (auto &&argument : arguments) {
        if (argument.tag != Argument::Tag::UNIFORM) { resource_count++; }
    }
    if (resource_count != binding_count) {
        LUISA_ERROR_WITH_LOCATION(
            "A native raster draw carries {} resource argument(s) for a shader "
            "with {} root resource parameter(s); the command was not built by "
            "NativeShaderLauncher::draw().",
            resource_count, binding_count);
        return false;
    }
    auto cmd_list = builder->get_cb()->cmd_list();
    cmd_list->SetGraphicsRootSignature(shader->root_sig());
    cmd_list->SetPipelineState(pso);
    auto binding_index = 0u;
    for (auto i = 0u; i < arguments.size(); i++) {
        auto &&argument = arguments[i];
        switch (argument.tag) {
            case Argument::Tag::BUFFER: {
                auto *buffer = reinterpret_cast<Buffer const *>(argument.buffer.handle);
                if (buffer == nullptr) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Native raster draw argument {} has a null buffer handle.",
                        i);
                    return false;
                }
                shader->set_raster_resource(
                    binding_index, builder,
                    BufferView{buffer, argument.buffer.offset,
                               argument.buffer.size});
                binding_index++;
            } break;
            case Argument::Tag::UNIFORM:
                // Written once for the whole block below.
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Native raster draw argument {} is not supported by the "
                    "DirectX encoding path (textures, bindless arrays and "
                    "acceleration structures are rejected at load_raster()).",
                    i);
                return false;
        }
    }
    // The launcher packs the uniform values last, one argument per add_uniform()
    // call, so the block is their [min, end) span in the command's argument buffer.
    if (state.uniform_property_index != ~0u) {
        auto uniform_begin = std::numeric_limits<uint64_t>::max();
        auto uniform_end = uint64_t{0u};
        for (auto &&argument : arguments) {
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
            if (bytes % sizeof(uint32_t) != 0u) {
                LUISA_ERROR_WITH_LOCATION(
                    "A native raster draw's uniform payload ({} bytes) is not a "
                    "multiple of 4 bytes, so it cannot be bound as root 32-bit "
                    "constants.",
                    bytes);
                return false;
            }
            if (bytes > state.uniform_size) {
                LUISA_ERROR_WITH_LOCATION(
                    "A native raster draw's uniform payload ({} bytes) exceeds "
                    "the shader's declared uniform block ({} bytes).",
                    bytes, state.uniform_size);
                return false;
            }
            auto payload = cmd->uniform(Argument::Uniform{
                static_cast<size_t>(uniform_begin),
                static_cast<size_t>(bytes), 4u});
            // The reserved object-id dword (when there is one) sits in front of the
            // application's bytes, exactly like the Vulkan route's push-constant
            // range.
            cmd_list->SetGraphicsRoot32BitConstants(
                state.uniform_property_index,
                static_cast<UINT>(bytes / sizeof(uint32_t)),
                payload.data(),
                static_cast<UINT>(state.object_id_bytes / sizeof(uint32_t)));
        }
    }
    return true;
}

void encode_native_raster_object_id(
    LCDevice *device, CommandBufferBuilder *builder,
    DrawRasterSceneCommand const *cmd, uint32_t object_id) noexcept {
    NativeRasterShaderState state;
    if (!find_native_raster_shader(device, cmd->handle(), state)) { return; }
    // A shader that reserved no slot reads no object id; writing one anyway would
    // overwrite the first uniform value the application pushed.
    if (state.object_id_bytes == 0u || state.uniform_property_index == ~0u) { return; }
    builder->get_cb()->cmd_list()->SetGraphicsRoot32BitConstant(
        state.uniform_property_index, object_id, 0u);
}

}// namespace lc::dx
