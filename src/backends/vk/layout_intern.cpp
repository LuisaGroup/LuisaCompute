// Vulkan side of the layout interning facility (plan P1.1). See
// layout_intern.h for the design, the key contents and the ownership contract.
#include "layout_intern.h"
#include "submit_profile.h"
#include "log.h"
#include "device.h"

namespace lc::vk::detail {

namespace {

[[nodiscard]] VkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device, const DescriptorSetLayoutKey &key) noexcept {
    DescriptorSetLayoutCreateInfoBuilder builder{key};
    auto layout = VkDescriptorSetLayout{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &builder.create_info(), Device::alloc_callbacks(), &layout));
    if (layout == VK_NULL_HANDLE) { return VK_NULL_HANDLE; }// fail closed, nothing was created
    // The profiler counts actual driver creations, not interning lookups: a
    // compile that reuses an interned layout creates nothing and must not move
    // this counter (plan P1.1 acceptance).
    submit_profile::record_descriptor_set_layout_create();
    return layout;
}

[[nodiscard]] VkPipelineLayout create_pipeline_layout(
    VkDevice device, const PipelineLayoutKey &key,
    luisa::span<const VkDescriptorSetLayout> set_layouts) noexcept {
    LUISA_ASSERT(
        set_layouts.size() == key.descriptor_set_layouts.size(),
        "Vulkan pipeline layout key lists {} descriptor sets but {} handles "
        "were resolved.",
        key.descriptor_set_layouts.size(), set_layouts.size());
    vstd::vector<VkPushConstantRange> ranges;
    ranges.reserve(key.push_constant_ranges.size());
    for (auto &&range : key.push_constant_ranges) {
        ranges.emplace_back(VkPushConstantRange{
            .stageFlags = range.stage_flags,
            .offset = range.offset,
            .size = range.size});
    }
    VkPipelineLayoutCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .flags = key.flags,
        .setLayoutCount = static_cast<uint32_t>(set_layouts.size()),
        .pSetLayouts = set_layouts.empty() ? nullptr : set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(ranges.size()),
        .pPushConstantRanges = ranges.empty() ? nullptr : ranges.data()};
    auto layout = VkPipelineLayout{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreatePipelineLayout(
        device, &create_info, Device::alloc_callbacks(), &layout));
    if (layout == VK_NULL_HANDLE) { return VK_NULL_HANDLE; }
    submit_profile::record_pipeline_layout_create();
    return layout;
}

}// namespace

InternedLayout LayoutInterner::acquire_descriptor_set_layout(
    const DescriptorSetLayoutKey &key, VkDevice device) noexcept {
    auto result = _descriptor_set_layouts.acquire(
        key, [device](const DescriptorSetLayoutKey &k) noexcept {
            return handle_to_uint64(create_descriptor_set_layout(device, k));
        });
    if (result.orphaned_handle != 0u) {
        // Lost an insertion race: an equal structure is already interned and the
        // handle this call created is not the one the device owns. Destroy it
        // here (this table is the only owner of such objects) so a lost race
        // cannot leak a VkDescriptorSetLayout.
        vkDestroyDescriptorSetLayout(
            device, uint64_to_handle<VkDescriptorSetLayout>(result.orphaned_handle),
            Device::alloc_callbacks());
    }
    return result.layout;
}

InternedLayout LayoutInterner::acquire_pipeline_layout(
    const PipelineLayoutKey &key, VkDevice device) noexcept {
    auto result = _pipeline_layouts.acquire(
        key, [this, device](const PipelineLayoutKey &k) noexcept {
            vstd::vector<VkDescriptorSetLayout> set_layouts;
            set_layouts.reserve(k.descriptor_set_layouts.size());
            for (auto id : k.descriptor_set_layouts) {
                auto handle = _descriptor_set_layouts.handle_of(id);
                if (handle == 0u) {
                    // The set this pipeline layout names is gone. Fail closed
                    // instead of creating a layout over a dangling set.
                    return uint64_t{0u};
                }
                set_layouts.emplace_back(uint64_to_handle<VkDescriptorSetLayout>(handle));
            }
            auto layout = create_pipeline_layout(device, k, luisa::span{set_layouts});
            return handle_to_uint64(layout);
        });
    if (result.orphaned_handle != 0u) {
        vkDestroyPipelineLayout(
            device, uint64_to_handle<VkPipelineLayout>(result.orphaned_handle),
            Device::alloc_callbacks());
    }
    return result.layout;
}

uint64_t LayoutInterner::release_descriptor_set_layout(
    uint64_t id, VkDevice device) noexcept {
    auto handle = _descriptor_set_layouts.release(id);
    if (handle == 0u) { return 0u; }
    vkDestroyDescriptorSetLayout(
        device, uint64_to_handle<VkDescriptorSetLayout>(handle),
        Device::alloc_callbacks());
    return handle;
}

uint64_t LayoutInterner::release_pipeline_layout(
    uint64_t id, VkDevice device) noexcept {
    auto handle = _pipeline_layouts.release(id);
    if (handle == 0u) { return 0u; }
    vkDestroyPipelineLayout(
        device, uint64_to_handle<VkPipelineLayout>(handle),
        Device::alloc_callbacks());
    return handle;
}

void LayoutInterner::destroy_all(VkDevice device) noexcept {
    // Pipeline layouts first: every one of them references descriptor-set
    // layouts this table owns.
    vstd::vector<uint64_t> pipeline_layouts;
    _pipeline_layouts.drain(pipeline_layouts);
    vstd::vector<uint64_t> descriptor_set_layouts;
    _descriptor_set_layouts.drain(descriptor_set_layouts);
    for (auto handle : pipeline_layouts) {
        vkDestroyPipelineLayout(
            device, uint64_to_handle<VkPipelineLayout>(handle),
            Device::alloc_callbacks());
    }
    for (auto handle : descriptor_set_layouts) {
        vkDestroyDescriptorSetLayout(
            device, uint64_to_handle<VkDescriptorSetLayout>(handle),
            Device::alloc_callbacks());
    }
}

}// namespace lc::vk::detail