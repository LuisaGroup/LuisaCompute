#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../common/hlsl/shader_property.h"
#include "../common/indirect_dispatch_layout.h"

namespace lc::vk::detail {

enum class VulkanBuiltinKernel : uint8_t {
    INDIRECT_PREPARE,
    ACCEL_PROCESS,
    BINDLESS_UPLOAD
};

struct VulkanBuiltinKernelContract {
    uint32_t block_size_x;
    uint32_t push_constant_size;
};

struct VulkanAccelUpdateLayout {
#define LC_VULKAN_ACCEL_UPDATE_LAYOUT(shader_name, cpp_name, value) \
    static constexpr uint32_t cpp_name = value;
#include "builtin/vulkan_accel_update_layout.def"
#undef LC_VULKAN_ACCEL_UPDATE_LAYOUT
    static constexpr uint32_t flag_opaque =
        flag_opaque_on | flag_opaque_off;

    [[nodiscard]] static constexpr uint32_t
    updated_contribution_offset_flags(
        uint32_t current, uint32_t update_flags) noexcept {
        auto result = current & high_byte_mask;
        if ((update_flags & flag_opaque) != 0u) {
            auto instance_flags =
                (update_flags & flag_opaque_on) != 0u ?
                    instance_force_opaque :
                    instance_force_no_opaque;
            result = instance_flags << high_byte_shift;
        }
        return result;
    }
};

struct alignas(16) VulkanAccelUpdateInput {
    std::array<float, 12u> affine;
    std::array<uint32_t, 2u> mesh;
    uint32_t index_visibility;
    uint32_t user_id_flags;

    [[nodiscard]] static constexpr uint32_t pack_index_visibility(
        uint32_t index, uint32_t visibility) noexcept {
        return (index & VulkanAccelUpdateLayout::index_mask) |
               ((visibility & VulkanAccelUpdateLayout::byte_mask) <<
                VulkanAccelUpdateLayout::high_byte_shift);
    }

    [[nodiscard]] static constexpr uint32_t pack_user_id_flags(
        uint32_t user_id, uint32_t flags) noexcept {
        return (user_id & VulkanAccelUpdateLayout::index_mask) |
               ((flags & VulkanAccelUpdateLayout::byte_mask) <<
                VulkanAccelUpdateLayout::high_byte_shift);
    }

    [[nodiscard]] static constexpr std::array<uint32_t, 2u>
    device_address_words(uint64_t address) noexcept {
        return {static_cast<uint32_t>(address),
                static_cast<uint32_t>(address >> 32u)};
    }
};

static_assert(std::is_standard_layout_v<VulkanAccelUpdateInput>);
static_assert(sizeof(VulkanAccelUpdateInput) == 64u);
static_assert(alignof(VulkanAccelUpdateInput) == 16u);
static_assert(offsetof(VulkanAccelUpdateInput, affine) == 0u);
static_assert(offsetof(VulkanAccelUpdateInput, mesh) == 48u);
static_assert(offsetof(VulkanAccelUpdateInput, index_visibility) == 56u);
static_assert(offsetof(VulkanAccelUpdateInput, user_id_flags) == 60u);

[[nodiscard]] constexpr VulkanBuiltinKernelContract
vulkan_builtin_kernel_contract(VulkanBuiltinKernel kernel) noexcept {
    switch (kernel) {
        case VulkanBuiltinKernel::INDIRECT_PREPARE:
            return {IndirectDispatchLayout::prepare_block_size,
                    sizeof(IndirectDispatchPrepareConstants)};
        case VulkanBuiltinKernel::ACCEL_PROCESS:
            return {VulkanAccelUpdateLayout::block_size,
                    2u * sizeof(uint32_t)};
        case VulkanBuiltinKernel::BINDLESS_UPLOAD:
            return {VulkanAccelUpdateLayout::bindless_upload_block_size,
                    sizeof(uint32_t)};
    }
    return {};
}

inline constexpr std::array vulkan_builtin_buffer_properties{
    hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer,
                   0u, 0u, 1u},
    hlsl::Property{hlsl::ShaderVariableType::RWStructuredBuffer,
                   0u, 1u, 1u},
    hlsl::Property{hlsl::ShaderVariableType::SamplerHeap,
                   1u, 0u, 16u}};

}// namespace lc::vk::detail
