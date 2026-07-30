#pragma once

#include <cstdint>

#include "descriptor_interface_plan.h"
#include "saved_argument_contract.h"

namespace lc::vk::detail {

// The local descriptor binding sequence is shared by shader codegen and the
// command encoder. Persist the producer dialect explicitly instead of
// inferring it from incidental property-table details: native XIR shaders use
// per-buffer metadata descriptors that HLSL/LLVM artifacts do not, while the
// backend's builtins have a small externally-bound interface of their own.
enum class ShaderCodegenDialect : uint8_t {
    HLSL_SPIRV = 0u,
    XIR_SPIRV = 1u,
    LLVM_SPIRV = 2u,
    VULKAN_BUILTIN = 3u
};

[[nodiscard]] constexpr bool valid_shader_codegen_dialect(
    uint8_t value) noexcept {
    return value <=
           static_cast<uint8_t>(ShaderCodegenDialect::VULKAN_BUILTIN);
}

enum class ShaderInterfaceError : uint8_t {
    NONE,
    INVALID_DIALECT,
    DIALECT_STAGE_MISMATCH,
    INVALID_DESCRIPTOR_INTERFACE,
    INVALID_SAVED_ARGUMENT_INTERFACE,
    DIALECT_ARGUMENT_TRAILER_MISMATCH,
    ARGUMENT_BUFFER_BINDING_MISMATCH,
    CONSTANT_UBO_BINDING_MISMATCH,
    UNSUPPORTED_LOCAL_BUFFER_ROLE,
    RESOURCE_BINDING_MISMATCH,
    ACCEL_ROLE_MISMATCH,
    NATIVE_BINDLESS_METADATA_MISMATCH,
    INDIRECT_BINDING_MISMATCH,
    PRINTER_TAIL_MISMATCH,
    BUILTIN_INTERFACE_MISMATCH,
    LOCAL_BINDING_COUNT_MISMATCH
};

[[nodiscard]] constexpr const char *shader_interface_error_name(
    ShaderInterfaceError error) noexcept {
    switch (error) {
        case ShaderInterfaceError::NONE: return "none";
        case ShaderInterfaceError::INVALID_DIALECT:
            return "invalid codegen dialect";
        case ShaderInterfaceError::DIALECT_STAGE_MISMATCH:
            return "codegen dialect is not supported by this shader stage";
        case ShaderInterfaceError::INVALID_DESCRIPTOR_INTERFACE:
            return "invalid descriptor interface";
        case ShaderInterfaceError::INVALID_SAVED_ARGUMENT_INTERFACE:
            return "invalid saved-argument interface";
        case ShaderInterfaceError::DIALECT_ARGUMENT_TRAILER_MISMATCH:
            return "argument metadata/trailer does not match codegen dialect";
        case ShaderInterfaceError::ARGUMENT_BUFFER_BINDING_MISMATCH:
            return "argument-buffer binding mismatch";
        case ShaderInterfaceError::CONSTANT_UBO_BINDING_MISMATCH:
            return "constant-UBO binding mismatch";
        case ShaderInterfaceError::UNSUPPORTED_LOCAL_BUFFER_ROLE:
            return "unsupported local buffer-heap role";
        case ShaderInterfaceError::RESOURCE_BINDING_MISMATCH:
            return "resource binding sequence mismatch";
        case ShaderInterfaceError::ACCEL_ROLE_MISMATCH:
            return "accel argument role mismatch";
        case ShaderInterfaceError::NATIVE_BINDLESS_METADATA_MISMATCH:
            return "native bindless metadata binding mismatch";
        case ShaderInterfaceError::INDIRECT_BINDING_MISMATCH:
            return "indirect-dispatch binding mismatch";
        case ShaderInterfaceError::PRINTER_TAIL_MISMATCH:
            return "printer descriptor tail mismatch";
        case ShaderInterfaceError::BUILTIN_INTERFACE_MISMATCH:
            return "backend builtin interface mismatch";
        case ShaderInterfaceError::LOCAL_BINDING_COUNT_MISMATCH:
            return "local descriptor binding count mismatch";
    }
    return "unknown shader-interface error";
}

struct ShaderInterfaceRequest {
    luisa::span<const hlsl::Property> properties{};
    luisa::span<const SavedArgument> arguments{};
    DescriptorInterfaceStageMask stage_mask{};
    ShaderCodegenDialect dialect{};
    uint32_t printer_count{};
    uint32_t validation_count{};
    bool use_buffer_bindless{};
    bool use_tex2d_bindless{};
    bool use_tex3d_bindless{};
    bool has_constant_ubo_payload{};
};

struct ShaderInterfacePlan {
    ShaderInterfaceError error{ShaderInterfaceError::NONE};
    DescriptorInterfacePlan descriptor_interface{};
    uint32_t argument_buffer_binding_count{};
    uint32_t constant_ubo_binding_count{};
    uint32_t resource_binding_count{};
    uint32_t indirect_binding_count{};
    uint32_t printer_binding_count{};
    uint32_t local_binding_count{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return error == ShaderInterfaceError::NONE;
    }
};

namespace shader_interface_detail {

[[nodiscard]] constexpr bool usage_reads(Usage usage) noexcept {
    return usage == Usage::NONE ||
           (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::READ)) != 0u;
}

[[nodiscard]] constexpr bool usage_writes(Usage usage) noexcept {
    return (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::WRITE)) != 0u;
}

[[nodiscard]] constexpr bool is_runtime_resource_tag(
    Type::Tag tag) noexcept {
    return tag == Type::Tag::BUFFER || tag == Type::Tag::TEXTURE ||
           tag == Type::Tag::BINDLESS_ARRAY || tag == Type::Tag::ACCEL ||
           tag == Type::Tag::CUSTOM;
}

[[nodiscard]] constexpr bool is_unsupported_local_buffer_role(
    hlsl::ShaderVariableType type) noexcept {
    return type == hlsl::ShaderVariableType::SRVBufferHeap ||
           type == hlsl::ShaderVariableType::UAVBufferHeap ||
           type == hlsl::ShaderVariableType::CBVBufferHeap;
}

}// namespace shader_interface_detail

// A texture argument owns a sampled-image descriptor for reads and a
// storage-image descriptor for writes, in that order. Derive these roles from
// the persisted argument usage rather than from the next property type: an
// SRV immediately followed by a UAV may describe either one read/write
// argument or two adjacent read-only/write-only arguments.
struct TextureDescriptorRoles {
    bool sampled{};
    bool storage{};
};

[[nodiscard]] constexpr TextureDescriptorRoles texture_descriptor_roles(
    Usage usage) noexcept {
    return {
        .sampled = shader_interface_detail::usage_reads(usage),
        .storage = shader_interface_detail::usage_writes(usage)};
}

// Validate the complete persisted/runtime descriptor ABI. This is the single
// owner of local binding order; device-specific numeric limits remain in
// plan_descriptor_interface and command encoding only consumes the checked
// local binding count produced here.
[[nodiscard]] inline ShaderInterfacePlan plan_shader_interface(
    const ShaderInterfaceRequest &request) noexcept {
    using namespace shader_interface_detail;
    auto plan = ShaderInterfacePlan{};
    auto fail = [&](ShaderInterfaceError error) noexcept {
        plan.error = error;
        return plan;
    };

    if (!valid_shader_codegen_dialect(
            static_cast<uint8_t>(request.dialect))) {
        return fail(ShaderInterfaceError::INVALID_DIALECT);
    }
    plan.descriptor_interface = plan_persisted_descriptor_interface(
        request.properties, request.stage_mask,
        request.use_buffer_bindless, request.use_tex2d_bindless,
        request.use_tex3d_bindless,
        request.has_constant_ubo_payload);
    if (!plan.descriptor_interface) {
        return fail(ShaderInterfaceError::INVALID_DESCRIPTOR_INTERFACE);
    }
    if (!plan_saved_argument_contract(
            request.arguments, request.validation_count)) {
        return fail(ShaderInterfaceError::INVALID_SAVED_ARGUMENT_INTERFACE);
    }

    auto dialect = request.dialect;
    auto is_native = dialect == ShaderCodegenDialect::XIR_SPIRV;
    auto is_builtin = dialect == ShaderCodegenDialect::VULKAN_BUILTIN;

    if (dialect != ShaderCodegenDialect::HLSL_SPIRV &&
        request.stage_mask != DescriptorInterfaceStageMask::COMPUTE) {
        return fail(ShaderInterfaceError::DIALECT_STAGE_MISMATCH);
    }
    // Raster dispatch currently has no printer descriptor/copyback path. Do
    // not accept an artifact merely because its two tail bindings are
    // structurally valid.
    if (request.stage_mask == DescriptorInterfaceStageMask::RASTER &&
        request.printer_count != 0u) {
        return fail(ShaderInterfaceError::PRINTER_TAIL_MISMATCH);
    }

    auto has_uniform_argument = false;
    auto has_buffer_argument = false;
    auto has_any_buffer_metadata = false;
    for (auto argument : request.arguments) {
        has_uniform_argument |= !is_runtime_resource_tag(argument.tag);
        has_buffer_argument |= argument.tag == Type::Tag::BUFFER;
        has_any_buffer_metadata |= argument.has_buffer_metadata();
        if (is_native && argument.tag == Type::Tag::BUFFER &&
            !argument.has_buffer_metadata()) {
            return fail(
                ShaderInterfaceError::DIALECT_ARGUMENT_TRAILER_MISMATCH);
        }
        if (argument.tag == Type::Tag::ACCEL) {
            if (is_native !=
                argument.has_explicit_native_accel_roles()) {
                return fail(ShaderInterfaceError::ACCEL_ROLE_MISMATCH);
            }
            if (is_native) {
                auto usage = luisa::to_underlying(argument.var_usage);
                auto reads =
                    (usage & luisa::to_underlying(Usage::READ)) != 0u;
                auto writes =
                    (usage & luisa::to_underlying(Usage::WRITE)) != 0u;
                auto traversal =
                    argument.native_accel_uses_traversal();
                auto instance =
                    argument.native_accel_uses_instance_buffer();
                auto used = reads || writes;
                if (used != (traversal || instance) ||
                    (traversal && !reads) ||
                    (writes && !instance)) {
                    return fail(
                        ShaderInterfaceError::ACCEL_ROLE_MISMATCH);
                }
            }
        }
    }
    if ((is_native && request.validation_count != 0u) ||
        (!is_native && has_any_buffer_metadata)) {
        return fail(
            ShaderInterfaceError::DIALECT_ARGUMENT_TRAILER_MISMATCH);
    }

    auto persisted_native_bindless_metadata_count = uint32_t{};
    for (auto property : request.properties) {
        if (property.space_index == 0u && property.array_size == 1u &&
            is_unsupported_local_buffer_role(property.type)) {
            return fail(ShaderInterfaceError::UNSUPPORTED_LOCAL_BUFFER_ROLE);
        }
    }

    if (is_builtin) {
        if (!request.arguments.empty() || request.validation_count != 0u ||
            request.printer_count != 0u ||
            request.has_constant_ubo_payload ||
            request.use_buffer_bindless || request.use_tex2d_bindless ||
            request.use_tex3d_bindless ||
            plan.descriptor_interface.indirect_dispatch_binding_count != 0u ||
            plan.descriptor_interface.local_binding_count != 2u) {
            return fail(ShaderInterfaceError::BUILTIN_INTERFACE_MISMATCH);
        }
        auto *input = find_local_descriptor_property(
            request.properties, 0u);
        auto *output = find_local_descriptor_property(
            request.properties, 1u);
        if (input == nullptr || output == nullptr ||
            input->type != hlsl::ShaderVariableType::StructuredBuffer ||
            output->type != hlsl::ShaderVariableType::RWStructuredBuffer) {
            return fail(ShaderInterfaceError::BUILTIN_INTERFACE_MISMATCH);
        }
        plan.resource_binding_count = 2u;
        plan.local_binding_count = 2u;
        return plan;
    }

    plan.constant_ubo_binding_count =
        static_cast<uint32_t>(request.has_constant_ubo_payload);

    auto argument_descriptor_count = uint32_t{};
    for (auto argument : request.arguments) {
        auto reads = usage_reads(argument.var_usage);
        auto writes = usage_writes(argument.var_usage);
        switch (argument.tag) {
            case Type::Tag::BUFFER:
            case Type::Tag::CUSTOM:
                ++argument_descriptor_count;
                break;
            case Type::Tag::TEXTURE:
                argument_descriptor_count +=
                    static_cast<uint32_t>(reads) +
                    static_cast<uint32_t>(writes);
                break;
            case Type::Tag::BINDLESS_ARRAY:
                ++argument_descriptor_count;
                break;
            case Type::Tag::ACCEL:
                // Usage::READ cannot distinguish native traversal from a
                // read-only instance-buffer operation, and LLVM artifacts may
                // expose only the traversal descriptor. Count the persisted
                // role descriptors below, then validate their per-argument
                // order against the dialect.
                break;
            default: break;
        }
    }
    for (auto property : request.properties) {
        if (property.space_index != 0u || property.array_size != 1u) {
            continue;
        }
        switch (property.type) {
            case hlsl::ShaderVariableType::SPIRVAccel:
            case hlsl::ShaderVariableType::SPIRVAccelInstance:
            case hlsl::ShaderVariableType::SPIRVAccelInstanceRW:
                ++argument_descriptor_count;
                break;
            case hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata:
                ++argument_descriptor_count;
                ++persisted_native_bindless_metadata_count;
                break;
            default: break;
        }
    }
    if (is_native && request.use_buffer_bindless &&
        persisted_native_bindless_metadata_count == 0u) {
        return fail(
            ShaderInterfaceError::NATIVE_BINDLESS_METADATA_MISMATCH);
    }
    plan.resource_binding_count = argument_descriptor_count;
    plan.indirect_binding_count =
        plan.descriptor_interface.indirect_dispatch_binding_count;
    plan.printer_binding_count = request.printer_count == 0u ? 0u : 2u;

    if ((is_native && plan.indirect_binding_count != 1u) ||
        (!is_native && plan.indirect_binding_count != 0u)) {
        return fail(ShaderInterfaceError::INDIRECT_BINDING_MISMATCH);
    }

    // Infer the physical argument-buffer slot from the complete local
    // descriptor count. This handles HLSL debug artifacts whose generated
    // `_Global` block is empty (validation_count == 0) without confusing the
    // first read-only user buffer for that internal descriptor.
    auto local_without_argument_buffer =
        plan.constant_ubo_binding_count + plan.resource_binding_count +
        plan.indirect_binding_count + plan.printer_binding_count;
    auto local_binding_count =
        plan.descriptor_interface.local_binding_count;
    auto has_argument_buffer = false;
    if (local_binding_count == local_without_argument_buffer + 1u) {
        has_argument_buffer = true;
    } else if (local_binding_count != local_without_argument_buffer) {
        return fail(ShaderInterfaceError::LOCAL_BINDING_COUNT_MISMATCH);
    }
    auto argument_buffer_required =
        has_uniform_argument || (is_native && has_buffer_argument) ||
        (dialect == ShaderCodegenDialect::HLSL_SPIRV &&
         request.validation_count != 0u);
    if (argument_buffer_required && !has_argument_buffer) {
        return fail(ShaderInterfaceError::ARGUMENT_BUFFER_BINDING_MISMATCH);
    }
    // The only legal empty internal argument buffer is the one emitted by the
    // HLSL debug path. Native XIR and LLVM artifacts have no such placeholder.
    if (!argument_buffer_required && has_argument_buffer &&
        dialect != ShaderCodegenDialect::HLSL_SPIRV) {
        return fail(ShaderInterfaceError::ARGUMENT_BUFFER_BINDING_MISMATCH);
    }
    plan.argument_buffer_binding_count =
        static_cast<uint32_t>(has_argument_buffer);

    auto expected_indirect_binding =
        plan.argument_buffer_binding_count +
        plan.constant_ubo_binding_count +
        plan.resource_binding_count;
    if (plan.indirect_binding_count != 0u) {
        if (!is_native) {
            return fail(ShaderInterfaceError::INDIRECT_BINDING_MISMATCH);
        }
        auto *indirect = find_local_descriptor_property(
            request.properties, expected_indirect_binding);
        if (indirect == nullptr ||
            indirect->type !=
                hlsl::ShaderVariableType::SPIRVIndirectDispatch) {
            return fail(ShaderInterfaceError::INDIRECT_BINDING_MISMATCH);
        }
    }

    if (request.has_constant_ubo_payload) {
        auto expected_constant_binding =
            plan.argument_buffer_binding_count;
        auto *constant = find_local_descriptor_property(
            request.properties, expected_constant_binding);
        if (constant == nullptr ||
            constant->type != hlsl::ShaderVariableType::ConstantBuffer) {
            return fail(
                ShaderInterfaceError::CONSTANT_UBO_BINDING_MISMATCH);
        }
    }

    auto binding = uint32_t{};
    auto expect_binding = [&](hlsl::ShaderVariableType expected,
                              ShaderInterfaceError mismatch) noexcept {
        auto *property = find_local_descriptor_property(
            request.properties, binding);
        if (property == nullptr || property->type != expected) {
            if (property != nullptr &&
                is_unsupported_local_buffer_role(property->type)) {
                plan.error =
                    ShaderInterfaceError::UNSUPPORTED_LOCAL_BUFFER_ROLE;
            } else if (property != nullptr &&
                       property->type ==
                           hlsl::ShaderVariableType::SPIRVIndirectDispatch) {
                plan.error = ShaderInterfaceError::INDIRECT_BINDING_MISMATCH;
            } else {
                plan.error = mismatch;
            }
            return false;
        }
        ++binding;
        return true;
    };

    if (has_argument_buffer &&
        !expect_binding(
            hlsl::ShaderVariableType::StructuredBuffer,
            ShaderInterfaceError::ARGUMENT_BUFFER_BINDING_MISMATCH)) {
        return plan;
    }
    if (request.has_constant_ubo_payload &&
        !expect_binding(
            hlsl::ShaderVariableType::ConstantBuffer,
            ShaderInterfaceError::CONSTANT_UBO_BINDING_MISMATCH)) {
        return plan;
    }

    auto native_bindless_metadata_role_count = uint32_t{};
    for (auto argument : request.arguments) {
        auto reads = usage_reads(argument.var_usage);
        auto writes = usage_writes(argument.var_usage);
        switch (argument.tag) {
            case Type::Tag::BUFFER:
            case Type::Tag::CUSTOM:
                if (!expect_binding(
                        writes ?
                            hlsl::ShaderVariableType::RWStructuredBuffer :
                            hlsl::ShaderVariableType::StructuredBuffer,
                        ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                    return plan;
                }
                break;
            case Type::Tag::TEXTURE:
                if (auto roles = texture_descriptor_roles(argument.var_usage);
                    roles.sampled &&
                    !expect_binding(
                        hlsl::ShaderVariableType::SRVTextureHeap,
                        ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                    return plan;
                }
                if (auto roles = texture_descriptor_roles(argument.var_usage);
                    roles.storage &&
                    !expect_binding(
                        hlsl::ShaderVariableType::UAVTextureHeap,
                        ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                    return plan;
                }
                break;
            case Type::Tag::BINDLESS_ARRAY:
                if (!expect_binding(
                        hlsl::ShaderVariableType::StructuredBuffer,
                        ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                    return plan;
                }
                if (is_native) {
                    if (auto *property = find_local_descriptor_property(
                            request.properties, binding);
                        property != nullptr &&
                        property->type == hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata) {
                        ++binding;
                        ++native_bindless_metadata_role_count;
                    }
                }
                break;
            case Type::Tag::ACCEL:
                if (is_native) {
                    // Native SavedArgument roles delimit the optional
                    // descriptors exactly. Looking at the next property's type
                    // is ambiguous when adjacent accel arguments use different
                    // role subsets.
                    if (argument.native_accel_uses_traversal() &&
                        !expect_binding(
                            hlsl::ShaderVariableType::SPIRVAccel,
                            ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                        return plan;
                    }
                    auto expected_instance = writes ?
                                                 hlsl::ShaderVariableType::SPIRVAccelInstanceRW :
                                                 hlsl::ShaderVariableType::SPIRVAccelInstance;
                    if (argument.native_accel_uses_instance_buffer() &&
                        !expect_binding(
                            expected_instance,
                            ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                        return plan;
                    }
                } else {
                    // Preserve the established HLSL/LLVM contract: READ means
                    // traversal. HLSL may additionally expose the instance
                    // buffer; LLVM currently exposes only the AS descriptor.
                    if (reads &&
                        !expect_binding(
                            hlsl::ShaderVariableType::SPIRVAccel,
                            ShaderInterfaceError::RESOURCE_BINDING_MISMATCH)) {
                        return plan;
                    }
                    auto expected_instance = writes ?
                                                 hlsl::ShaderVariableType::SPIRVAccelInstanceRW :
                                                 hlsl::ShaderVariableType::SPIRVAccelInstance;
                    if (auto *property = find_local_descriptor_property(
                            request.properties, binding);
                        property != nullptr &&
                        property->type == expected_instance) {
                        ++binding;
                    } else if (writes) {
                        return fail(
                            ShaderInterfaceError::RESOURCE_BINDING_MISMATCH);
                    }
                }
                break;
            default: break;
        }
    }
    if (is_native && request.use_buffer_bindless &&
        native_bindless_metadata_role_count == 0u) {
        return fail(
            ShaderInterfaceError::NATIVE_BINDLESS_METADATA_MISMATCH);
    }

    if (plan.indirect_binding_count != 0u &&
        !expect_binding(
            hlsl::ShaderVariableType::SPIRVIndirectDispatch,
            ShaderInterfaceError::INDIRECT_BINDING_MISMATCH)) {
        return plan;
    }
    if (request.printer_count != 0u) {
        if (!expect_binding(
                hlsl::ShaderVariableType::RWStructuredBuffer,
                ShaderInterfaceError::PRINTER_TAIL_MISMATCH) ||
            !expect_binding(
                hlsl::ShaderVariableType::RWStructuredBuffer,
                ShaderInterfaceError::PRINTER_TAIL_MISMATCH)) {
            return plan;
        }
    }

    plan.local_binding_count = binding;
    if (binding != plan.descriptor_interface.local_binding_count) {
        return fail(request.printer_count != 0u ?
                        ShaderInterfaceError::PRINTER_TAIL_MISMATCH :
                        ShaderInterfaceError::LOCAL_BINDING_COUNT_MISMATCH);
    }
    return plan;
}

}// namespace lc::vk::detail
