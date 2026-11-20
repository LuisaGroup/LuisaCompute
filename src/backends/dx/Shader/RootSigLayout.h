#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <luisa/vstl/common.h>
#include <luisa/ast/function.h>
#include "../../common/hlsl/shader_property.h"

namespace lc::dx {

using namespace luisa;
using namespace luisa::compute;

struct SavedArgument {
    Type::Tag tag{};
    Usage var_usage{Usage::NONE};
    bool is_raw_byte_buffer{false};
    uint struct_size{0u};

    SavedArgument() = default;
    SavedArgument(Type const *type) {
        tag = type->tag();
        is_raw_byte_buffer = false;
        if (type->tag() == Type::Tag::BUFFER) {
            if (auto ele = type->element()) {
                struct_size = static_cast<uint>(ele->size());
            } else {
                struct_size = 1u;// byte buffer
                is_raw_byte_buffer = true;
            }
        } else if (luisa::to_underlying(type->tag()) < luisa::to_underlying(Type::Tag::BUFFER)) {
            struct_size = static_cast<uint>(type->size());
        }
    }
    SavedArgument(Usage usage, Variable const &var)
        : SavedArgument(var.type()) {
        var_usage = usage;
    }
    SavedArgument(Function kernel, Variable const &var)
        : SavedArgument(var.type()) {
        var_usage = kernel.variable_usage(var.uid());
    }
};

struct RootSigBinding {
    enum class Kind : uint8_t {
        Direct,
        SpilledSRV,
        SpilledUAV
    };
    Kind kind{Kind::Direct};
    uint index{0u};
    uint element_stride{0u};
    bool force_raw{false};
};

struct RootSigLayout {
    bool spilled_tables{false};
    bool mapping_complete{true};
    uint spilled_srv_root_param{0u};
    uint spilled_uav_root_param{0u};
    uint spilled_srv_count{0u};
    uint spilled_uav_count{0u};
    size_t kernel_arg_prop_offset{0u};
    vstd::vector<RootSigBinding> bindings;
};

[[nodiscard]] inline bool needs_descriptor_table_spill(
    vstd::span<hlsl::Property const> properties,
    bool append_object_id = false) noexcept {
    uint dwords = append_object_id ? 1u : 0u;
    for (auto &&var : properties) {
        switch (var.type) {
            case hlsl::ShaderVariableType::ConstantValue:
                dwords += std::max<uint>(var.space_index, 1u);
                break;
            case hlsl::ShaderVariableType::ConstantBuffer:
            case hlsl::ShaderVariableType::StructuredBuffer:
            case hlsl::ShaderVariableType::RWStructuredBuffer:
                dwords += 2u;
                break;
            case hlsl::ShaderVariableType::SRVTextureHeap:
            case hlsl::ShaderVariableType::UAVTextureHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
            case hlsl::ShaderVariableType::UAVBufferHeap:
            case hlsl::ShaderVariableType::CBVBufferHeap:
            case hlsl::ShaderVariableType::SamplerHeap:
                dwords += 1u;
                break;
            default:
                break;
        }
    }
    return dwords > 64u;
}

// Keep (space=0, reg=0) StructuredBuffer as a 2-DWORD direct Root SRV:
// PreprocessCodegenProperties emits the kernel uniform argument buffer
// (_Global) at t0, space0 (which has no entry in _kernel_arguments and does
// not need StructureByteStride as a root SRV). Keeping t0, space0 direct in
// both SerializeRootSig and build_root_sig_layout guarantees exact root-sig
// agreement without needing _kernel_arguments during root-sig serialization.
// Single textures stay direct 1-DWORD descriptor tables (avoiding illegal
// CopyDescriptorsSimple from shader-visible global_heap).
[[nodiscard]] inline bool is_spilled_srv(hlsl::Property const &var) noexcept {
    return var.space_index == 0u &&
           var.type == hlsl::ShaderVariableType::StructuredBuffer &&
           var.register_index != 0u;
}

[[nodiscard]] inline bool is_spilled_uav(hlsl::Property const &var) noexcept {
    return var.space_index == 0u &&
           var.type == hlsl::ShaderVariableType::RWStructuredBuffer;
}

namespace detail {

[[nodiscard]] inline bool is_uniform_arg_buffer_prop(hlsl::Property const &p) noexcept {
    return p.type == hlsl::ShaderVariableType::StructuredBuffer &&
           p.space_index == 0u &&
           p.register_index == 0u &&
           p.array_size == 1u;
}

[[nodiscard]] inline bool is_bindless_heap_prop(hlsl::Property const &p) noexcept {
    return p.array_size == std::numeric_limits<uint>::max();
}

// Walks kernel resource arguments + optional printer properties starting from
// `prop_idx` in exact CodegenProperties order. Returns the ending property
// index, or `~size_t{0}` on mismatch/overflow.
[[nodiscard]] inline size_t map_kernel_arg_properties(
    vstd::span<hlsl::Property const> properties,
    size_t prop_idx,
    vstd::span<SavedArgument const> kernel_arguments,
    bool has_printers,
    RootSigBinding *out_bindings = nullptr) noexcept {
    const size_t n = properties.size();
    for (auto const &arg : kernel_arguments) {
        switch (arg.tag) {
            case Type::Tag::BUFFER: {
                if (prop_idx >= n) { return ~size_t{0}; }
                auto const t = properties[prop_idx].type;
                if (t != hlsl::ShaderVariableType::StructuredBuffer &&
                    t != hlsl::ShaderVariableType::RWStructuredBuffer) {
                    return ~size_t{0};
                }
                if (out_bindings != nullptr) {
                    out_bindings[prop_idx].element_stride = arg.struct_size;
                    out_bindings[prop_idx].force_raw = arg.is_raw_byte_buffer;
                }
                ++prop_idx;
            } break;
            case Type::Tag::TEXTURE: {
                if (prop_idx >= n) { return ~size_t{0}; }
                auto const t = properties[prop_idx].type;
                if (t != hlsl::ShaderVariableType::SRVTextureHeap &&
                    t != hlsl::ShaderVariableType::UAVTextureHeap) {
                    return ~size_t{0};
                }
                ++prop_idx;
            } break;
            case Type::Tag::BINDLESS_ARRAY: {
                if (prop_idx >= n) { return ~size_t{0}; }
                if (properties[prop_idx].type != hlsl::ShaderVariableType::StructuredBuffer) {
                    return ~size_t{0};
                }
                if (out_bindings != nullptr) {
                    out_bindings[prop_idx].element_stride = 4u;
                    out_bindings[prop_idx].force_raw = false;
                }
                ++prop_idx;
            } break;
            case Type::Tag::ACCEL: {
                const bool writable = (luisa::to_underlying(arg.var_usage) &
                                       luisa::to_underlying(Usage::WRITE)) != 0u;
                if (writable) {
                    if (prop_idx >= n) { return ~size_t{0}; }
                    if (properties[prop_idx].type != hlsl::ShaderVariableType::RWStructuredBuffer) {
                        return ~size_t{0};
                    }
                    const bool is_fallback =
                        (prop_idx + 1u < n) &&
                        properties[prop_idx + 1u].type == hlsl::ShaderVariableType::ConstantValue;
                    if (out_bindings != nullptr) {
                        out_bindings[prop_idx].element_stride = is_fallback ? 16u : 64u;
                        out_bindings[prop_idx].force_raw = false;
                    }
                    prop_idx += is_fallback ? 2u : 1u;
                } else {
                    if (prop_idx + 1u >= n) { return ~size_t{0}; }
                    if (properties[prop_idx].type != hlsl::ShaderVariableType::StructuredBuffer ||
                        properties[prop_idx + 1u].type != hlsl::ShaderVariableType::StructuredBuffer) {
                        return ~size_t{0};
                    }
                    const bool is_fallback =
                        (prop_idx + 2u < n) &&
                        properties[prop_idx + 2u].type == hlsl::ShaderVariableType::ConstantValue;
                    if (is_fallback) {
                        if (out_bindings != nullptr) {
                            out_bindings[prop_idx].element_stride = 4u;
                            out_bindings[prop_idx].force_raw = false;
                            out_bindings[prop_idx + 1u].element_stride = 16u;
                            out_bindings[prop_idx + 1u].force_raw = false;
                        }
                        prop_idx += 3u;
                    } else {
                        if (out_bindings != nullptr) {
                            // prop_idx is TopAccel (handled via TopAccel* in bind_resources)
                            out_bindings[prop_idx].element_stride = 0u;
                            out_bindings[prop_idx].force_raw = false;
                            out_bindings[prop_idx + 1u].element_stride = 64u;
                            out_bindings[prop_idx + 1u].force_raw = false;
                        }
                        prop_idx += 2u;
                    }
                }
            } break;
            case Type::Tag::CUSTOM: {
                if (prop_idx < n &&
                    (properties[prop_idx].type == hlsl::ShaderVariableType::RWStructuredBuffer ||
                     properties[prop_idx].type == hlsl::ShaderVariableType::StructuredBuffer)) {
                    if (out_bindings != nullptr) {
                        out_bindings[prop_idx].element_stride = 4u;
                        out_bindings[prop_idx].force_raw = false;
                    }
                    ++prop_idx;
                }
            } break;
            default:
                break;
        }
    }
    if (has_printers) {
        if (prop_idx + 1u >= n) { return ~size_t{0}; }
        if (properties[prop_idx].type != hlsl::ShaderVariableType::RWStructuredBuffer ||
            properties[prop_idx + 1u].type != hlsl::ShaderVariableType::RWStructuredBuffer) {
            return ~size_t{0};
        }
        if (out_bindings != nullptr) {
            out_bindings[prop_idx].element_stride = 4u;
            out_bindings[prop_idx].force_raw = false;
            out_bindings[prop_idx + 1u].element_stride = 0u;
            out_bindings[prop_idx + 1u].force_raw = true;
        }
        prop_idx += 2u;
    }
    return prop_idx;
}

}// namespace detail

// Computes the exact index in `properties` where CodegenProperties kernel
// resource bindings begin, skipping the PreprocessCodegenProperties prefix:
//   [ConstantValue (dsp_c)] [SamplerHeap] [StructuredBuffer _Global (t0,s0)] [bindless heaps...]
[[nodiscard]] inline size_t find_kernel_arg_property_offset(
    vstd::span<hlsl::Property const> properties,
    vstd::span<SavedArgument const> kernel_arguments,
    bool has_printers) noexcept {
    const size_t n = properties.size();
    size_t prop_idx = 0u;
    if (prop_idx < n && properties[prop_idx].type == hlsl::ShaderVariableType::ConstantValue) {
        ++prop_idx;
    }
    if (prop_idx < n && properties[prop_idx].type == hlsl::ShaderVariableType::SamplerHeap) {
        ++prop_idx;
    }
    if (prop_idx < n && properties[prop_idx].type == hlsl::ShaderVariableType::ConstantBuffer) {
        ++prop_idx;
    }
    if (prop_idx < n && detail::is_uniform_arg_buffer_prop(properties[prop_idx])) {
        size_t bindless_end = prop_idx + 1u;
        while (bindless_end < n && detail::is_bindless_heap_prop(properties[bindless_end])) {
            ++bindless_end;
        }
        if (bindless_end > prop_idx + 1u) {
            // GenerateBindless always runs immediately after the uniform
            // StructuredBuffer (_Global), so any t0/space0 StructuredBuffer
            // preceding bindless heaps is unambiguously _Global.
            return bindless_end;
        }
        // No bindless heaps follow: disambiguate _Global at t0/space0 vs a
        // first kernel SRV argument at t0/space0 by exact property count.
        if (detail::map_kernel_arg_properties(properties, prop_idx + 1u, kernel_arguments, has_printers) == n) {
            return prop_idx + 1u;
        }
        return prop_idx;
    }
    while (prop_idx < n && detail::is_bindless_heap_prop(properties[prop_idx])) {
        ++prop_idx;
    }
    return prop_idx;
}

[[nodiscard]] inline RootSigLayout build_root_sig_layout(
    vstd::span<hlsl::Property const> properties,
    vstd::span<SavedArgument const> kernel_arguments,
    bool has_printers,
    bool append_object_id = false) {
    RootSigLayout layout{};
    layout.spilled_tables = needs_descriptor_table_spill(properties, append_object_id);
    if (!layout.spilled_tables) {
        layout.mapping_complete = true;
        return layout;
    }

    layout.bindings.resize(properties.size());
    uint direct_root_idx = 0u;
    for (size_t i = 0u; i < properties.size(); ++i) {
        auto const &var = properties[i];
        if (is_spilled_srv(var)) {
            layout.bindings[i].kind = RootSigBinding::Kind::SpilledSRV;
            layout.bindings[i].index = layout.spilled_srv_count++;
        } else if (is_spilled_uav(var)) {
            layout.bindings[i].kind = RootSigBinding::Kind::SpilledUAV;
            layout.bindings[i].index = layout.spilled_uav_count++;
        } else {
            layout.bindings[i].kind = RootSigBinding::Kind::Direct;
            layout.bindings[i].index = direct_root_idx++;
        }
    }
    if (layout.spilled_srv_count > 0u) {
        layout.spilled_srv_root_param = direct_root_idx++;
    }
    if (layout.spilled_uav_count > 0u) {
        layout.spilled_uav_root_param = direct_root_idx++;
    }

    layout.kernel_arg_prop_offset =
        find_kernel_arg_property_offset(properties, kernel_arguments, has_printers);
    const size_t mapped_end = detail::map_kernel_arg_properties(
        properties,
        layout.kernel_arg_prop_offset,
        kernel_arguments,
        has_printers,
        layout.bindings.data());
    layout.mapping_complete = (mapped_end == properties.size());
    return layout;
}

}// namespace lc::dx
