#pragma once

#include <cstdint>
#include <limits>

#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/core/stl/memory.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/vector.h>

#include "../hlsl/shader_property.h"

namespace lc::llvm_codegen {

using namespace luisa::compute;

constexpr uint32_t llvm_vulkan_sampler_count = 16u;

// Backend-independent input to the Vulkan descriptor-property ABI. Keeping
// this separate from LLVM IR construction makes the producer/runtime handoff
// directly testable even when the experimental LLVM target is not built.
struct LLVMVulkanBindingArgument {
    Type::Tag tag{};
    Usage usage{};
    bool indirect_dispatch_buffer{};
};

struct LLVMVulkanBindingPropertyRequest {
    luisa::span<const LLVMVulkanBindingArgument> arguments{};
    bool use_buffer_bindless{};
    bool use_tex2d_bindless{};
    bool use_tex3d_bindless{};
    uint32_t printer_count{};
};

struct LLVMVulkanBindingPropertyPlan {
    vstd::vector<hlsl::Property> properties;
    uint32_t local_binding_count{};
    bool has_argument_buffer{};
};

namespace llvm_vulkan_binding_detail {

[[nodiscard]] constexpr bool is_runtime_resource_tag(
    Type::Tag tag) noexcept {
    return tag == Type::Tag::BUFFER || tag == Type::Tag::TEXTURE ||
           tag == Type::Tag::BINDLESS_ARRAY || tag == Type::Tag::ACCEL ||
           tag == Type::Tag::CUSTOM;
}

[[nodiscard]] constexpr bool usage_reads(Usage usage) noexcept {
    return usage == Usage::NONE ||
           (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::READ)) != 0u;
}

[[nodiscard]] constexpr bool usage_writes(Usage usage) noexcept {
    return (luisa::to_underlying(usage) &
            luisa::to_underlying(Usage::WRITE)) != 0u;
}

}// namespace llvm_vulkan_binding_detail

// Describe the exact Vulkan runtime ABI the LLVM producer must implement.
// This does not claim that today's LLVM IR resource lowering supports every
// role; validate_llvm_vulkan_resource_model below is the explicit capability
// boundary used before code emission.
[[nodiscard]] inline LLVMVulkanBindingPropertyPlan
plan_llvm_vulkan_binding_properties(
    const LLVMVulkanBindingPropertyRequest &request) {
    using namespace llvm_vulkan_binding_detail;
    LLVMVulkanBindingPropertyPlan plan{};
    for (auto argument : request.arguments) {
        plan.has_argument_buffer |=
            !is_runtime_resource_tag(argument.tag);
    }

    auto local_binding = uint32_t{};
    auto add_local = [&](hlsl::ShaderVariableType type) {
        plan.properties.emplace_back(hlsl::Property{
            type, 0u, local_binding++, 1u});
    };
    auto add_global = [&](hlsl::ShaderVariableType type,
                          uint32_t set) {
        plan.properties.emplace_back(hlsl::Property{
            type, set, 0u, std::numeric_limits<uint32_t>::max()});
    };

    // Set 1 exists for every Vulkan shader, even when no sample instruction
    // survives optimization. The runtime binds the immutable sampler table
    // unconditionally.
    plan.properties.emplace_back(hlsl::Property{
        hlsl::ShaderVariableType::SamplerHeap,
        1u, 0u, llvm_vulkan_sampler_count});

    if (plan.has_argument_buffer) {
        add_local(hlsl::ShaderVariableType::StructuredBuffer);
    }
    auto global_set = 2u;
    if (request.use_buffer_bindless) {
        add_global(hlsl::ShaderVariableType::SRVBufferHeap, global_set++);
    }
    if (request.use_tex2d_bindless) {
        add_global(hlsl::ShaderVariableType::SRVTextureHeap, global_set++);
    }
    if (request.use_tex3d_bindless) {
        add_global(hlsl::ShaderVariableType::SRVTextureHeap, global_set++);
    }

    for (auto argument : request.arguments) {
        auto reads = usage_reads(argument.usage);
        auto writes = usage_writes(argument.usage);
        switch (argument.tag) {
            case Type::Tag::BUFFER:
                add_local(writes ?
                              hlsl::ShaderVariableType::RWStructuredBuffer :
                              hlsl::ShaderVariableType::StructuredBuffer);
                break;
            case Type::Tag::CUSTOM:
                if (argument.indirect_dispatch_buffer) {
                    add_local(writes ?
                                  hlsl::ShaderVariableType::RWStructuredBuffer :
                                  hlsl::ShaderVariableType::StructuredBuffer);
                }
                break;
            case Type::Tag::TEXTURE:
                if (reads) {
                    add_local(hlsl::ShaderVariableType::SRVTextureHeap);
                }
                if (writes) {
                    add_local(hlsl::ShaderVariableType::UAVTextureHeap);
                }
                break;
            case Type::Tag::BINDLESS_ARRAY:
                add_local(hlsl::ShaderVariableType::StructuredBuffer);
                break;
            case Type::Tag::ACCEL:
                if (reads) {
                    add_local(hlsl::ShaderVariableType::SPIRVAccel);
                }
                add_local(writes ?
                              hlsl::ShaderVariableType::SPIRVAccelInstanceRW :
                              hlsl::ShaderVariableType::SPIRVAccelInstance);
                break;
            default: break;
        }
    }
    if (request.printer_count != 0u) {
        add_local(hlsl::ShaderVariableType::RWStructuredBuffer);
        add_local(hlsl::ShaderVariableType::RWStructuredBuffer);
    }
    plan.local_binding_count = local_binding;
    return plan;
}

enum class LLVMVulkanResourceModelError : uint8_t {
    NONE,
    PRINTING_NOT_IMPLEMENTED,
    BINDLESS_RESOURCES_NOT_IMPLEMENTED,
    TEXTURE_RESOURCES_NOT_IMPLEMENTED,
    ACCEL_RESOURCES_NOT_IMPLEMENTED,
    INDIRECT_DISPATCH_BUFFER_NOT_IMPLEMENTED,
    CUSTOM_ARGUMENT_NOT_IMPLEMENTED,
    DIRECT_BUFFER_DESCRIPTORS_NOT_IMPLEMENTED,
    ARGUMENT_BUFFER_DESCRIPTOR_NOT_IMPLEMENTED
};

[[nodiscard]] constexpr const char *llvm_vulkan_resource_model_error_name(
    LLVMVulkanResourceModelError error) noexcept {
    switch (error) {
        case LLVMVulkanResourceModelError::NONE: return "none";
        case LLVMVulkanResourceModelError::PRINTING_NOT_IMPLEMENTED:
            return "shader printing descriptors are not implemented";
        case LLVMVulkanResourceModelError::BINDLESS_RESOURCES_NOT_IMPLEMENTED:
            return "bindless descriptor resources are not implemented";
        case LLVMVulkanResourceModelError::TEXTURE_RESOURCES_NOT_IMPLEMENTED:
            return "direct texture descriptors are not implemented";
        case LLVMVulkanResourceModelError::ACCEL_RESOURCES_NOT_IMPLEMENTED:
            return "acceleration-structure descriptors are not implemented";
        case LLVMVulkanResourceModelError::INDIRECT_DISPATCH_BUFFER_NOT_IMPLEMENTED:
            return "indirect-dispatch buffer descriptors are not implemented";
        case LLVMVulkanResourceModelError::CUSTOM_ARGUMENT_NOT_IMPLEMENTED:
            return "custom argument descriptors other than indirect dispatch are not implemented";
        case LLVMVulkanResourceModelError::DIRECT_BUFFER_DESCRIPTORS_NOT_IMPLEMENTED:
            return "direct storage-buffer descriptors are not implemented";
        case LLVMVulkanResourceModelError::ARGUMENT_BUFFER_DESCRIPTOR_NOT_IMPLEMENTED:
            return "uniform argument-buffer descriptors are not implemented";
    }
    return "unknown LLVM Vulkan resource-model error";
}

struct LLVMVulkanResourceModelSupport {
    LLVMVulkanResourceModelError error{
        LLVMVulkanResourceModelError::NONE};
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return error == LLVMVulkanResourceModelError::NONE;
    }
};

// The current AST->LLVM implementation lowers arguments to ordinary
// CrossWorkgroup globals instead of LLVM's SPIR-V resource-handle intrinsics.
// Such globals cannot honestly be paired with DescriptorSet/Binding metadata.
// Fail before LLVM IR/SPIR-V emission until each resource family is lowered to
// the binding model described by plan_llvm_vulkan_binding_properties.
[[nodiscard]] constexpr LLVMVulkanResourceModelSupport
validate_llvm_vulkan_resource_model(
    const LLVMVulkanBindingPropertyRequest &request) noexcept {
    if (request.printer_count != 0u) {
        return {LLVMVulkanResourceModelError::PRINTING_NOT_IMPLEMENTED};
    }
    if (request.use_buffer_bindless || request.use_tex2d_bindless ||
        request.use_tex3d_bindless) {
        return {
            LLVMVulkanResourceModelError::BINDLESS_RESOURCES_NOT_IMPLEMENTED};
    }
    for (auto argument : request.arguments) {
        switch (argument.tag) {
            case Type::Tag::BINDLESS_ARRAY:
                return {LLVMVulkanResourceModelError::BINDLESS_RESOURCES_NOT_IMPLEMENTED};
            case Type::Tag::TEXTURE:
                return {LLVMVulkanResourceModelError::TEXTURE_RESOURCES_NOT_IMPLEMENTED};
            case Type::Tag::ACCEL:
                return {LLVMVulkanResourceModelError::ACCEL_RESOURCES_NOT_IMPLEMENTED};
            case Type::Tag::CUSTOM:
                return {
                    argument.indirect_dispatch_buffer ?
                        LLVMVulkanResourceModelError::INDIRECT_DISPATCH_BUFFER_NOT_IMPLEMENTED :
                        LLVMVulkanResourceModelError::CUSTOM_ARGUMENT_NOT_IMPLEMENTED};
            case Type::Tag::BUFFER:
                return {LLVMVulkanResourceModelError::DIRECT_BUFFER_DESCRIPTORS_NOT_IMPLEMENTED};
            default:
                return {LLVMVulkanResourceModelError::ARGUMENT_BUFFER_DESCRIPTOR_NOT_IMPLEMENTED};
        }
    }
    return {};
}

}// namespace lc::llvm_codegen
