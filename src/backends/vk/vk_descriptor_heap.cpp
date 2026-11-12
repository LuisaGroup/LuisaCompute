#include "vk_descriptor_heap.h"
#include "device.h"
#include <luisa/core/logging.h>

namespace lc::vk {

// The member definitions below spell the planner types unqualified; they all live
// in `lc::vk::detail`.
using detail::DescriptorClass;
using detail::DescriptorHeapPlan;

// File-local helpers. The vulkan backend is built in unity (jumbo) batches, so a
// name declared here must live in a file-unique namespace to stay distinct from
// the identically-named helpers of a sibling translation unit.
namespace vk_descriptor_heap_detail {

using detail::DescriptorHeapError;

// VkBufferUsageFlags is 32 bits wide, and VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT
// is bit 28, so the 32-bit create-time mask can carry it. The 64-bit
// VK_BUFFER_USAGE_2_* spelling names the same bit and would additionally require
// chaining VkBufferUsageFlags2CreateInfoKHR, which buys nothing here.
static_assert(static_cast<uint64_t>(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT) ==
              static_cast<uint64_t>(VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT));
static_assert(static_cast<uint64_t>(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT) <=
              static_cast<uint64_t>(0xffffffffu));

[[nodiscard]] const char *heap_name(bool sampler_heap) noexcept {
    return sampler_heap ? "sampler heap" : "resource heap";
}

}// namespace vk_descriptor_heap_detail

DescriptorHeap::DescriptorHeap(Device *device, const Config &config) noexcept
    : _device{device} {
    LUISA_ASSERT(_device != nullptr, "A Vulkan descriptor heap needs a device.");
    if (!_device->optional_extension_enabled(detail::OptionalExtension::DESCRIPTOR_HEAP)) {
        // The single gate of this class. `optional_extension_enabled` is false when
        // the extension is unsupported, switched off through
        // VulkanDeviceConfigExt, or the logical device was imported, and in all
        // three cases the pre-existing descriptor-set path is the correct one.
        LUISA_ERROR(
            "VK_EXT_descriptor_heap is not enabled on this Vulkan device; a "
            "DescriptorHeap must not be created. Use "
            "VulkanDeviceConfigExt::enable_descriptor_heap() on a device that "
            "supports the extension.");
    }
    LUISA_ASSERT(
        _device->enable_device_address(),
        "VK_EXT_descriptor_heap binds a descriptor heap through a device address "
        "range, so the buffer device address feature must be enabled.");
    if (vkWriteResourceDescriptorsEXT == nullptr || vkWriteSamplerDescriptorsEXT == nullptr ||
        vkCmdBindResourceHeapEXT == nullptr || vkCmdBindSamplerHeapEXT == nullptr ||
        vkCmdPushDataEXT == nullptr) {
        // Volk only resolves entry points the physical device advertises. An
        // enabled extension without them means the loader/ICD disagrees with
        // itself, which must never be papered over by a null call.
        LUISA_ERROR(
            "VK_EXT_descriptor_heap is enabled but one of its entry points "
            "(vkWriteResourceDescriptorsEXT/vkWriteSamplerDescriptorsEXT/"
            "vkCmdBindResourceHeapEXT/vkCmdBindSamplerHeapEXT/vkCmdPushDataEXT) "
            "was not resolved by the loader.");
    }

    _limits = query_limits(_device);

    detail::DescriptorHeapRequest resource_request{};
    resource_request.extension_enabled = true;
    resource_request.sampler_heap = false;
    resource_request.heap_size_limit = config.resource_heap_size_limit;
    resource_request.region_count = config.resource_region_count;
    LUISA_ASSERT(config.resource_region_count <= max_regions,
                 "At most {} resource-heap regions are supported, {} requested.",
                 max_regions, config.resource_region_count);
    for (uint32_t i = 0u; i < config.resource_region_count; i++) {
        resource_request.regions[i] = detail::DescriptorHeapRegionRequest{
            .klass = config.resource_regions[i].klass,
            .slots = config.resource_regions[i].slots};
    }
    _resource_plan = detail::plan_descriptor_heap(_limits, resource_request);
    if (!_resource_plan.usable) {
        LUISA_ERROR(
            "Vulkan resource descriptor heap is not usable: {} (requested {} "
            "regions, heap size limit {}).",
            detail::descriptor_heap_error_name(_resource_plan.error),
            resource_request.region_count, _resource_plan.heap_size_limit);
    }

    detail::DescriptorHeapRequest sampler_request{};
    sampler_request.extension_enabled = true;
    sampler_request.sampler_heap = true;
    sampler_request.embedded_sampler_count = config.embedded_sampler_count;
    sampler_request.heap_size_limit = config.sampler_heap_size_limit;
    sampler_request.region_count = config.sampler_region_count;
    LUISA_ASSERT(config.sampler_region_count <= max_regions,
                 "At most {} sampler-heap regions are supported, {} requested.",
                 max_regions, config.sampler_region_count);
    for (uint32_t i = 0u; i < config.sampler_region_count; i++) {
        sampler_request.regions[i] = detail::DescriptorHeapRegionRequest{
            .klass = config.sampler_regions[i].klass,
            .slots = config.sampler_regions[i].slots};
    }
    if (sampler_request.region_count != 0u || sampler_request.embedded_sampler_count != 0u) {
        _sampler_plan = detail::plan_descriptor_heap(_limits, sampler_request);
        if (!_sampler_plan.usable) {
            LUISA_ERROR(
                "Vulkan sampler descriptor heap is not usable: {} (requested {} "
                "regions, {} embedded samplers, heap size limit {}).",
                detail::descriptor_heap_error_name(_sampler_plan.error),
                sampler_request.region_count, sampler_request.embedded_sampler_count,
                _sampler_plan.heap_size_limit);
        }
    }

    // One first-fit pool per planned region, indexed like `plan.regions`. A region's
    // slots are handed out from its `first_slot` upward, so a fresh heap behaves like
    // a bump allocator and a released slot is reused by the next allocation.
    for (uint32_t i = 0u; i < _resource_plan.region_count; i++) {
        _resource_pools[i].reset(_resource_plan.regions[i].first_slot,
                                 _resource_plan.regions[i].slots);
    }
    for (uint32_t i = 0u; i < _sampler_plan.region_count; i++) {
        _sampler_pools[i].reset(_sampler_plan.regions[i].first_slot,
                                _sampler_plan.regions[i].slots);
    }

    // Creates one VkBuffer with its own VkDeviceMemory. The heap buffers are
    // device local and additionally carry TRANSFER_DST so the staging upload of
    // `record_upload` can copy the host-written descriptors in; the staging buffers
    // are host visible and carry TRANSFER_SRC.
    auto create_storage = [this](Storage &storage, VkDeviceSize size,
                                 VkBufferUsageFlags usage, bool host_visible,
                                 const char *what) noexcept {
        LUISA_ASSERT(size != 0u, "Vulkan {} has a zero size.", what);
        auto allocation = _device->allocator().allocate_native_buffer(
            static_cast<size_t>(size), usage, 0u, host_visible,
            MemoryInitPolicy::kUndefined);
        storage.buffer = allocation.buffer;
        storage.memory = allocation.memory;
        storage.size = size;
        auto vk_device = _device->logic_device();
        if (host_visible) {
            void *mapped = nullptr;
            auto map_result = vkMapMemory(vk_device, allocation.memory, 0u,
                                          VK_WHOLE_SIZE, 0u, &mapped);
            if (map_result != VK_SUCCESS) {
                LUISA_ERROR("Failed to map the Vulkan {} staging memory: VkResult {}.",
                            what, static_cast<int>(map_result));
            }
            storage.mapped = mapped;
        }
        VkBufferDeviceAddressInfo address_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = allocation.buffer};
        storage.address = vkGetBufferDeviceAddress(vk_device, &address_info);
    };

    // Every heap range must start on a multiple of the advertised heap alignment
    // (VUID-vkCmdBindResourceHeapEXT-pBindInfo-11235 /
    // VUID-vkCmdBindSamplerHeapEXT-pBindInfo-11226). A backend-owned allocation
    // cannot move its own base address, so an unaligned device address is surfaced
    // instead of silently mis-binding the heap.
    constexpr auto heap_usage = static_cast<VkBufferUsageFlags>(
        VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    create_storage(_resource, _resource_plan.heap_size, heap_usage, false, "resource heap");
    LUISA_ASSERT(
        _resource.address % _resource_plan.heap_alignment == 0u,
        "The resource heap's device address {:#x} is not a multiple of its "
        "reservedRangeOffset alignment (resourceHeapAlignment {}); the heap "
        "cannot be bound.",
        static_cast<uint64_t>(_resource.address), _resource_plan.heap_alignment);
    if (_sampler_plan.usable) {
        create_storage(_sampler, _sampler_plan.heap_size, heap_usage, false, "sampler heap");
        LUISA_ASSERT(
            _sampler.address % _sampler_plan.heap_alignment == 0u,
            "The sampler heap's device address {:#x} is not a multiple of its "
            "reservedRangeOffset alignment (samplerHeapAlignment {}); the heap "
            "cannot be bound.",
            static_cast<uint64_t>(_sampler.address), _sampler_plan.heap_alignment);
    }
    // One host-visible staging buffer per heap: a descriptor's byte offset inside
    // its heap is also its offset inside that heap's staging copy, and two separate
    // buffers keep the resource and sampler staging bytes from aliasing each other.
    create_storage(_resource_staging, _resource.size,
                   static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
                   true, "resource heap staging buffer");
    LUISA_ASSERT(_resource_staging.mapped != nullptr,
                 "The resource descriptor heap staging buffer must stay mapped.");
    if (_sampler.size != 0u) {
        create_storage(_sampler_staging, _sampler.size,
                       static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
                       true, "sampler heap staging buffer");
        LUISA_ASSERT(_sampler_staging.mapped != nullptr,
                     "The sampler descriptor heap staging buffer must stay mapped.");
    }
}

DescriptorHeap::~DescriptorHeap() noexcept {
    if (_device == nullptr) { return; }
    auto vk_device = _device->logic_device();
    auto destroy_storage = [this, vk_device](Storage &storage) noexcept {
        if (storage.mapped != nullptr) {
            vkUnmapMemory(vk_device, storage.memory);
            storage.mapped = nullptr;
        }
        if (storage.buffer == VK_NULL_HANDLE && storage.memory == VK_NULL_HANDLE) { return; }
        // `destroy_native_buffer` only reads the two handles; the size and the
        // zero-initialization flag are bookkeeping of the allocation itself.
        NativeBufferAllocation allocation{
            .buffer = storage.buffer,
            .memory = storage.memory,
            .allocation_size = storage.size,
            .driver_zeroed = false};
        _device->allocator().destroy_native_buffer(allocation);
        storage = Storage{};
    };
    destroy_storage(_sampler_staging);
    destroy_storage(_resource_staging);
    destroy_storage(_sampler);
    destroy_storage(_resource);
}

bool DescriptorHeap::is_available(const Device *device) noexcept {
    return device != nullptr &&
           device->optional_extension_enabled(detail::OptionalExtension::DESCRIPTOR_HEAP) &&
           device->enable_device_address();
}

detail::DescriptorHeapLimits DescriptorHeap::query_limits(Device *device) noexcept {
    if (device == nullptr ||
        !device->optional_extension_enabled(detail::OptionalExtension::DESCRIPTOR_HEAP)) {
        return detail::DescriptorHeapLimits{};
    }
    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
        .pNext = nullptr};
    VkPhysicalDeviceDescriptorHeapTensorPropertiesARM tensor_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_TENSOR_PROPERTIES_ARM,
        .pNext = nullptr};
    // The tensor descriptor properties come from VK_ARM_tensors; chaining a
    // structure of an extension that is not enabled is not allowed, so they are
    // only queried when the registry really enabled it.
    auto const tensors = device->optional_extension_enabled(detail::OptionalExtension::TENSOR_ARM);
    if (tensors) { heap_properties.pNext = &tensor_properties; }
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &heap_properties};
    vkGetPhysicalDeviceProperties2(device->physical_device(), &properties);
    if (tensors) { return detail::descriptor_heap_limits(heap_properties, tensor_properties); }
    return detail::descriptor_heap_limits(heap_properties);
}

VkDeviceSize DescriptorHeap::descriptor_size_for_type(Device *device,
                                                      VkDescriptorType type) noexcept {
    if (device == nullptr ||
        !device->optional_extension_enabled(detail::OptionalExtension::DESCRIPTOR_HEAP)) {
        return 0u;
    }
    if (vkGetPhysicalDeviceDescriptorSizeEXT == nullptr) { return 0u; }
    return vkGetPhysicalDeviceDescriptorSizeEXT(device->physical_device(), type);
}

VkDeviceSize DescriptorHeap::_exact_descriptor_size(VkDescriptorType type) const noexcept {
    // VUID-vkGetPhysicalDeviceDescriptorSizeEXT-type-11362 lists the legal types.
    // The two that come from optional extensions are only queried when that
    // extension is really enabled on this device.
    switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            if (!_device->enable_raytracing()) { return 0u; }
            break;
        case VK_DESCRIPTOR_TYPE_TENSOR_ARM:
            if (!_device->optional_extension_enabled(detail::OptionalExtension::TENSOR_ARM)) {
                return 0u;
            }
            break;
        default: return 0u;
    }
    return descriptor_size_for_type(_device, type);
}

detail::DescriptorSlotPool *DescriptorHeap::_pool(const DescriptorHeapPlan &plan,
                                                  detail::DescriptorSlotPool *pools,
                                                  DescriptorClass klass) noexcept {
    for (uint32_t i = 0u; i < plan.region_count; i++) {
        if (plan.regions[i].klass == klass) { return pools + i; }
    }
    return nullptr;
}

const detail::DescriptorSlotPool *DescriptorHeap::_pool(const DescriptorHeapPlan &plan,
                                                        const detail::DescriptorSlotPool *pools,
                                                        DescriptorClass klass) const noexcept {
    for (uint32_t i = 0u; i < plan.region_count; i++) {
        if (plan.regions[i].klass == klass) { return pools + i; }
    }
    return nullptr;
}

uint64_t DescriptorHeap::allocate_slot(DescriptorClass klass) noexcept {
    auto *pool = _pool(_resource_plan, _resource_pools, klass);
    if (pool != nullptr) { return pool->allocate(); }
    pool = _pool(_sampler_plan, _sampler_pools, klass);
    if (pool != nullptr) { return pool->allocate(); }
    // A class without a region can never be allocated: the caller asked for a
    // descriptor family this heap does not own.
    return detail::kInvalidDescriptorSlot;
}

bool DescriptorHeap::release_slot(DescriptorClass klass, uint64_t slot) noexcept {
    if (auto *pool = _pool(_resource_plan, _resource_pools, klass)) {
        return pool->release(slot);
    }
    if (auto *pool = _pool(_sampler_plan, _sampler_pools, klass)) {
        return pool->release(slot);
    }
    return false;
}

uint64_t DescriptorHeap::allocated_slots(DescriptorClass klass) const noexcept {
    if (auto *pool = _pool(_resource_plan, _resource_pools, klass)) {
        return pool->allocated();
    }
    if (auto *pool = _pool(_sampler_plan, _sampler_pools, klass)) {
        return pool->allocated();
    }
    return 0u;
}

uint64_t DescriptorHeap::slot_capacity(DescriptorClass klass) const noexcept {
    if (auto *pool = _pool(_resource_plan, _resource_pools, klass)) {
        return pool->capacity();
    }
    if (auto *pool = _pool(_sampler_plan, _sampler_pools, klass)) {
        return pool->capacity();
    }
    return 0u;
}

detail::DescriptorSlot DescriptorHeap::slot_location(DescriptorClass klass,
                                                      uint64_t slot) const noexcept {
    if (auto location = detail::descriptor_slot(_resource_plan, klass, slot); location.valid) {
        return location;
    }
    return detail::descriptor_slot(_sampler_plan, klass, slot);
}

VkHostAddressRangeEXT DescriptorHeap::_slot_range(const DescriptorHeapPlan &plan,
                                                  DescriptorClass klass, uint64_t slot,
                                                  const char *what) noexcept {
    auto location = detail::descriptor_slot(plan, klass, slot);
    if (!location.valid) {
        LUISA_ERROR(
            "{}: descriptor slot {} is outside the {} region of the {}.",
            what, slot, detail::descriptor_class_name(klass),
            vk_descriptor_heap_detail::heap_name(plan.sampler_heap));
    }
    auto &staging = plan.sampler_heap ? _sampler_staging : _resource_staging;
    LUISA_ASSERT(staging.mapped != nullptr,
                 "A descriptor heap staging buffer must stay mapped.");
    LUISA_ASSERT(location.byte_offset + location.descriptor_size <= staging.size,
                 "{}: descriptor slot {} ends at byte {} but the {} staging buffer is "
                 "only {} bytes.",
                 what, slot,
                 static_cast<uint64_t>(location.byte_offset + location.descriptor_size),
                 vk_descriptor_heap_detail::heap_name(plan.sampler_heap),
                 static_cast<uint64_t>(staging.size));
    return VkHostAddressRangeEXT{
        .address = static_cast<uint8_t *>(staging.mapped) + location.byte_offset,
        .size = static_cast<size_t>(location.descriptor_size)};
}

void DescriptorHeap::_write_resource_descriptor(const DescriptorHeapPlan &plan,
                                                DescriptorClass klass, uint64_t slot,
                                                VkResourceDescriptorInfoEXT &info,
                                                const char *what) noexcept {
    auto range = _slot_range(plan, klass, slot, what);
    // Exactly what the implementation writes for this descriptor type. The plan's
    // descriptor size is the advertised upper bound, so the exact size may only be
    // smaller.
    auto exact_size = _exact_descriptor_size(info.type);
    if (exact_size != 0u) {
        LUISA_ASSERT(exact_size <= range.size,
                     "{}: the implementation writes {} bytes for descriptor type "
                     "{} but the slot only reserves {} bytes.",
                     what, static_cast<uint64_t>(exact_size),
                     static_cast<int>(info.type), static_cast<uint64_t>(range.size));
        range.size = static_cast<size_t>(exact_size);
    }
    auto result = vkWriteResourceDescriptorsEXT(_device->logic_device(), 1u, &info, &range);
    if (result != VK_SUCCESS) {
        LUISA_ERROR("{}: vkWriteResourceDescriptorsEXT failed with VkResult {}.",
                    what, static_cast<int>(result));
    }
    auto &staging = plan.sampler_heap ? _sampler_staging : _resource_staging;
    auto offset = static_cast<VkDeviceSize>(
        static_cast<uint8_t *>(range.address) - static_cast<uint8_t *>(staging.mapped));
    if (plan.sampler_heap) {
        _sampler_pending.mark(offset, range.size);
    } else {
        _resource_pending.mark(offset, range.size);
    }
}

void DescriptorHeap::write_buffer_descriptor(uint64_t slot, VkDeviceAddress address,
                                             VkDeviceSize size,
                                             bool uniform_buffer) noexcept {
    LUISA_ASSERT(size != 0u,
                 "A uniform/storage buffer descriptor needs a non-zero size "
                 "(VUID-VkResourceDescriptorInfoEXT-type-11433).");
    // VUID-VkResourceDescriptorInfoEXT-type-12350/12351.
    auto const &limits = _device->properties().limits;
    auto alignment = uniform_buffer ?
                         limits.minUniformBufferOffsetAlignment :
                         limits.minStorageBufferOffsetAlignment;
    if (alignment != 0u && address % alignment != 0u) {
        LUISA_ERROR(
            "write_buffer_descriptor: the {} buffer address {:#x} is not a "
            "multiple of the required alignment {}.",
            uniform_buffer ? "uniform" : "storage",
            static_cast<uint64_t>(address), static_cast<uint64_t>(alignment));
    }
    VkDeviceAddressRangeEXT address_range{
        .address = address,
        .size = size};
    VkResourceDescriptorInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type = uniform_buffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .data = {.pAddressRange = &address_range}};
    _write_resource_descriptor(_resource_plan, DescriptorClass::kBuffer, slot, info,
                               "write_buffer_descriptor");
}

void DescriptorHeap::write_texel_buffer_descriptor(uint64_t slot, VkFormat format,
                                                    VkDeviceAddress address,
                                                    VkDeviceSize size,
                                                    bool storage) noexcept {
    VkTexelBufferDescriptorInfoEXT texel_info{
        .sType = VK_STRUCTURE_TYPE_TEXEL_BUFFER_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .format = format,
        .addressRange = {.address = address, .size = size}};
    VkResourceDescriptorInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER :
                          VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
        .data = {.pTexelBuffer = &texel_info}};
    // SPIR-V declares a texel buffer as OpTypeImage, so its heap offset must
    // respect imageDescriptorAlignment; the slot therefore lives in an image
    // region.
    _write_resource_descriptor(_resource_plan, DescriptorClass::kImage, slot, info,
                               "write_texel_buffer_descriptor");
}

void DescriptorHeap::write_image_descriptor(uint64_t slot, const VkImageViewCreateInfo &view,
                                            VkImageLayout layout, bool storage) noexcept {
    VkImageDescriptorInfoEXT image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .pView = &view,
        .layout = layout};
    VkResourceDescriptorInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .data = {.pImage = &image_info}};
    _write_resource_descriptor(_resource_plan, DescriptorClass::kImage, slot, info,
                               "write_image_descriptor");
}

void DescriptorHeap::write_acceleration_structure_descriptor(uint64_t slot,
                                                             VkDeviceAddress address) noexcept {
    // The range size is unused by an acceleration-structure descriptor
    // (VUID-VkResourceDescriptorInfoEXT-type-11484), and its address must be a
    // multiple of 256 (VUID-VkResourceDescriptorInfoEXT-type-11454).
    LUISA_ASSERT(address % 256u == 0u,
                 "An acceleration-structure descriptor address {:#x} must be a "
                 "multiple of 256.",
                 static_cast<uint64_t>(address));
    VkDeviceAddressRangeEXT address_range{
        .address = address,
        .size = 0u};
    VkResourceDescriptorInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .data = {.pAddressRange = &address_range}};
    _write_resource_descriptor(_resource_plan, DescriptorClass::kBuffer, slot, info,
                               "write_acceleration_structure_descriptor");
}

void DescriptorHeap::write_tensor_descriptor(uint64_t slot,
                                             const VkTensorViewCreateInfoARM &view) noexcept {
    if (!_device->optional_extension_enabled(detail::OptionalExtension::TENSOR_ARM)) {
        LUISA_ERROR(
            "A tensor descriptor requires VK_ARM_tensors, which is not enabled on "
            "this Vulkan device.");
    }
    VkResourceDescriptorInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        .data = {.pTensorARM = &view}};
    _write_resource_descriptor(_resource_plan, DescriptorClass::kTensor, slot, info,
                               "write_tensor_descriptor");
}

void DescriptorHeap::write_sampler_descriptor(uint64_t slot,
                                              const VkSamplerCreateInfo &info) noexcept {
    // VUID-vkWriteSamplerDescriptorsEXT-pSamplers-11204: a sampler descriptor can
    // never carry a VkSamplerYcbcrConversionInfo; a YCbCr sampler must be embedded
    // through the shader binding mapping instead.
    for (auto next = static_cast<const VkBaseInStructure *>(info.pNext); next != nullptr;
         next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO) {
            LUISA_ERROR(
                "write_sampler_descriptor: a sampler descriptor must not carry a "
                "VkSamplerYcbcrConversionInfo; embed the YCbCr sampler through "
                "VkShaderDescriptorSetAndBindingMappingInfoEXT instead.");
        }
    }
    // VUID-vkWriteSamplerDescriptorsEXT-borderColor-11444/11205: a custom border
    // colour is addressed by index on heaps, not by value.
    if (info.borderColor == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT ||
        info.borderColor == VK_BORDER_COLOR_INT_CUSTOM_EXT) {
        LUISA_ASSERT(
            _limits.max_descriptor_heap_embedded_samplers != 0u,
            "write_sampler_descriptor: a custom border colour needs a "
            "VkSamplerCustomBorderColorIndexCreateInfoEXT in the sampler's pNext "
            "chain; the caller must supply it.");
    }
    auto range = _slot_range(_sampler_plan, DescriptorClass::kSampler, slot,
                             "write_sampler_descriptor");
    auto exact_size = _exact_descriptor_size(VK_DESCRIPTOR_TYPE_SAMPLER);
    if (exact_size != 0u) {
        LUISA_ASSERT(exact_size <= range.size,
                     "write_sampler_descriptor: the implementation writes {} bytes "
                     "for a sampler descriptor but the slot only reserves {} bytes.",
                     static_cast<uint64_t>(exact_size), static_cast<uint64_t>(range.size));
        range.size = static_cast<size_t>(exact_size);
    }
    auto result = vkWriteSamplerDescriptorsEXT(_device->logic_device(), 1u, &info, &range);
    if (result != VK_SUCCESS) {
        LUISA_ERROR("write_sampler_descriptor: vkWriteSamplerDescriptorsEXT failed "
                    "with VkResult {}.",
                    static_cast<int>(result));
    }
    auto offset = static_cast<VkDeviceSize>(
        static_cast<uint8_t *>(range.address) - static_cast<uint8_t *>(_sampler_staging.mapped));
    _sampler_pending.mark(offset, range.size);
}

bool DescriptorHeap::record_upload(VkCommandBuffer command_buffer) noexcept {
    LUISA_ASSERT(command_buffer != VK_NULL_HANDLE,
                 "record_upload needs a live command buffer.");
    auto uploaded = false;
    if (!_resource_pending.empty()) {
        _record_copy(command_buffer, _resource_staging, _resource, _resource_pending);
        _record_barrier(command_buffer, _resource.buffer, 0u, _resource.size);
        _resource_pending.clear();
        uploaded = true;
    }
    if (!_sampler_pending.empty()) {
        _record_copy(command_buffer, _sampler_staging, _sampler, _sampler_pending);
        _record_barrier(command_buffer, _sampler.buffer, 0u, _sampler.size);
        _sampler_pending.clear();
        uploaded = true;
    }
    return uploaded;
}

void DescriptorHeap::_record_copy(VkCommandBuffer command_buffer,
                                  const Storage &staging, const Storage &destination,
                                  const PendingRange &range) const noexcept {
    VkBufferCopy copy{
        .srcOffset = range.begin,
        .dstOffset = range.begin,
        .size = range.end - range.begin};
    LUISA_ASSERT(staging.buffer != VK_NULL_HANDLE && destination.buffer != VK_NULL_HANDLE,
                 "record_upload on a heap that owns no staging or destination buffer.");
    LUISA_ASSERT(range.end <= staging.size,
                 "Staged descriptor range [{}, {}) exceeds the {}-byte staging "
                 "buffer.",
                 static_cast<uint64_t>(range.begin), static_cast<uint64_t>(range.end),
                 static_cast<uint64_t>(staging.size));
    LUISA_ASSERT(range.end <= destination.size,
                 "Staged descriptor range [{}, {}) exceeds the {}-byte destination "
                 "heap.",
                 static_cast<uint64_t>(range.begin), static_cast<uint64_t>(range.end),
                 static_cast<uint64_t>(destination.size));
    vkCmdCopyBuffer(command_buffer, staging.buffer, destination.buffer, 1u, &copy);
}

void DescriptorHeap::_record_barrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                                     VkDeviceSize offset, VkDeviceSize size) noexcept {
    // The 32-bit legacy access mask is enough here and keeps the barrier
    // independent of synchronization2: heap bytes are only ever read by shaders.
    VkBufferMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = offset,
        .size = size};
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0u, 0u, nullptr, 1u, &barrier, 0u, nullptr);
}

void DescriptorHeap::record_bind_resource_heap(VkCommandBuffer command_buffer) const noexcept {
    LUISA_ASSERT(vkCmdBindResourceHeapEXT != nullptr,
                 "vkCmdBindResourceHeapEXT was not resolved; VK_EXT_descriptor_heap "
                 "is not enabled.");
    VkBindHeapInfoEXT bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .pNext = nullptr,
        .heapRange = {.address = _resource.address, .size = _resource.size},
        .reservedRangeOffset = _resource_plan.reserved_range_offset,
        .reservedRangeSize = _resource_plan.reserved_range_size};
    vkCmdBindResourceHeapEXT(command_buffer, &bind_info);
}

void DescriptorHeap::record_bind_sampler_heap(VkCommandBuffer command_buffer) const noexcept {
    LUISA_ASSERT(vkCmdBindSamplerHeapEXT != nullptr,
                 "vkCmdBindSamplerHeapEXT was not resolved; VK_EXT_descriptor_heap "
                 "is not enabled.");
    LUISA_ASSERT(_sampler.buffer != VK_NULL_HANDLE,
                 "This descriptor heap owns no sampler heap to bind.");
    VkBindHeapInfoEXT bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .pNext = nullptr,
        .heapRange = {.address = _sampler.address, .size = _sampler.size},
        .reservedRangeOffset = _sampler_plan.reserved_range_offset,
        .reservedRangeSize = _sampler_plan.reserved_range_size};
    vkCmdBindSamplerHeapEXT(command_buffer, &bind_info);
}

void DescriptorHeap::record_push_data(VkCommandBuffer command_buffer, uint32_t offset,
                                      const void *data, VkDeviceSize size) const noexcept {
    LUISA_ASSERT(vkCmdPushDataEXT != nullptr,
                 "vkCmdPushDataEXT was not resolved; VK_EXT_descriptor_heap is not "
                 "enabled.");
    // The class cannot exist on a device without the extension, so the planner's
    // `extension_enabled` input is unconditionally true here.
    auto plan = detail::plan_push_data(true, _limits.max_push_data_size, offset, size);
    if (!plan.valid) {
        LUISA_ERROR("record_push_data: {} (offset {}, size {}, maxPushDataSize {}).",
                    detail::descriptor_heap_error_name(plan.error),
                    static_cast<uint64_t>(offset), static_cast<uint64_t>(size),
                    _limits.max_push_data_size);
    }
    VkPushDataInfoEXT push_data{
        .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
        .pNext = nullptr,
        .offset = plan.offset,
        .data = {.address = data, .size = static_cast<size_t>(plan.size)}};
    vkCmdPushDataEXT(command_buffer, &push_data);
}

}// namespace lc::vk
