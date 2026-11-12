#include "opacity_micromap.h"
#include "device.h"
#include "stream.h"
#include "log.h"
#include "command_buffer_sync.h"
#include <cstring>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>

namespace lc::vk {

namespace opacity_micromap_detail {

// VkMicromapCreateInfoEXT::offset "must be a multiple of 256"; every buffer this
// file allocates starts at offset 0, which trivially satisfies it, but the value
// is asserted instead of assumed because a future sub-allocation would break it
// silently.
[[nodiscard]] constexpr bool storage_offset_legal(VkDeviceSize offset) noexcept {
    return offset % detail::opacity_micromap_storage_offset_alignment == 0u;
}

// Maps the plan's Vulkan-free enums onto the real Vk enums. Keeping the mapping
// in one function is what lets the unit test prove the decision without Vulkan.
[[nodiscard]] VkIndexType to_vk_index_type(uint8_t index_type) noexcept {
    switch (static_cast<detail::OpacityMicromapIndexType>(index_type)) {
        case detail::OpacityMicromapIndexType::UINT8: return VK_INDEX_TYPE_UINT8;
        case detail::OpacityMicromapIndexType::UINT16: return VK_INDEX_TYPE_UINT16;
        case detail::OpacityMicromapIndexType::UINT32: return VK_INDEX_TYPE_UINT32;
        case detail::OpacityMicromapIndexType::NONE:
        default: return VK_INDEX_TYPE_NONE_KHR;
    }
}

[[nodiscard]] constexpr VkOpacityMicromapFormatEXT to_vk_format(
    detail::OpacityMicromapFormat format) noexcept {
    return format == detail::OpacityMicromapFormat::FOUR_STATE ?
               VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT :
               VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT;
}

}// namespace opacity_micromap_detail

bool OpacityMicromap::available(Device *device) noexcept {
    return device != nullptr &&
           device->optional_extension_enabled(
               detail::OptionalExtension::OPACITY_MICROMAP);
}

bool OpacityMicromap::ext_alias_negotiated(Device *device) noexcept {
    if (device == nullptr) { return false; }
    for (auto &&state : device->optional_extensions()) {
        if (state.id != detail::OptionalExtension::OPACITY_MICROMAP) { continue; }
        return state.name != nullptr &&
               std::strcmp(state.name, VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME) == 0;
    }
    return false;
}

detail::OpacityMicromapBuildRequest OpacityMicromap::make_build_request(
    Device *device, uint32_t triangle_count,
    detail::OpacityMicromapFormat format, uint8_t subdivision_level,
    bool public_api_request, bool host_build_preferred) noexcept {
    detail::OpacityMicromapBuildRequest request;
    request.triangle_count = triangle_count;
    request.format = format;
    request.subdivision_level = subdivision_level;
    request.public_api_request = public_api_request;
    request.host_build_preferred = host_build_preferred;
    if (device == nullptr) { return request; }
    request.extension_enabled = available(device);
    if (!request.extension_enabled) {
        // Nothing below may be queried: the properties/features structures of a
        // disabled extension are not legal pNext members.
        return request;
    }
    // Object model: whichever name the registry resolved. The KHR revision of
    // this spec represents a micromap as a VkAccelerationStructureKHR built
    // through VK_KHR_device_address_commands, and only the EXT spelling (the
    // registry's alias) still defines the VkMicromapEXT object model implemented
    // by this class.
    request.object_model = ext_alias_negotiated(device) ?
                               detail::MicromapObjectModel::EXT_MICROMAP_OBJECT :
                               detail::MicromapObjectModel::KHR_ACCELERATION_STRUCTURE;
    // The registry enables only the `micromap` bit of
    // VkPhysicalDeviceOpacityMicromapFeaturesEXT (see
    // vk_extension_registry.cpp: `check_single_bit<..., &...::micromap>`), so
    // `micromapHostCommands` is VK_FALSE on every logical device this backend
    // creates and vkBuildMicromapsEXT is not callable. The plan turns that into
    // the device-build fallback.
    request.host_commands_enabled = false;
    request.device_address_supported = device->enable_device_address();
    // This class always allocates mapped, host-visible buffer memory.
    request.storage_host_visible = true;
    // Property query, only reachable with the extension enabled.
    if (request.object_model == detail::MicromapObjectModel::EXT_MICROMAP_OBJECT) {
        VkPhysicalDeviceOpacityMicromapPropertiesEXT properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &properties;
        vkGetPhysicalDeviceProperties2(device->physical_device(), &properties2);
        request.max_two_state_subdivision_level = static_cast<uint8_t>(
            properties.maxOpacity2StateSubdivisionLevel > 0xFFu ?
                0xFFu :
                properties.maxOpacity2StateSubdivisionLevel);
        request.max_four_state_subdivision_level = static_cast<uint8_t>(
            properties.maxOpacity4StateSubdivisionLevel > 0xFFu ?
                0xFFu :
                properties.maxOpacity4StateSubdivisionLevel);
    } else {
        VkPhysicalDeviceOpacityMicromapPropertiesKHR properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &properties;
        vkGetPhysicalDeviceProperties2(device->physical_device(), &properties2);
        request.max_two_state_subdivision_level = static_cast<uint8_t>(
            properties.maxOpacity2StateSubdivisionLevel > 0xFFu ?
                0xFFu :
                properties.maxOpacity2StateSubdivisionLevel);
        request.max_four_state_subdivision_level = static_cast<uint8_t>(
            properties.maxOpacity4StateSubdivisionLevel > 0xFFu ?
                0xFFu :
                properties.maxOpacity4StateSubdivisionLevel);
    }
    request.scratch_offset_alignment =
        device->acceleration_structure_properties()
            .minAccelerationStructureScratchOffsetAlignment;
    return request;
}

OpacityMicromap::OpacityMicromap(
    Device *device,
    detail::OpacityMicromapBuildPlan const &plan) noexcept
    : _device{device}, _plan{plan} {}

OpacityMicromap::MappedBuffer OpacityMicromap::create_mapped_buffer(
    VkBufferUsageFlags usage, size_t size) noexcept {
    MappedBuffer result;
    result.size = size;
    VkBufferCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    // The device build reads/writes through device addresses; the host build
    // reads through the mapped pointer. Only ask for the address-capable usage
    // when the device enabled buffer device addresses, otherwise the flag is an
    // invalid create-info bit.
    create_info.usage = usage |
                        (_device->enable_device_address() ?
                             static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) :
                             VkBufferUsageFlags{0u});
    _device->allocator().apply_queue_sharing(create_info);
    VmaAllocationCreateInfo allocation_info{};
    // Host-visible on purpose: one allocation serves the host build (host
    // address) and the device build (device address).
    allocation_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo mapped_info{};
    VK_CHECK_RESULT(vmaCreateBuffer(
        _device->allocator().allocator(), &create_info, &allocation_info,
        &result.buffer, &result.allocation, &mapped_info));
    result.mapped = mapped_info.pMappedData;
    if (_device->enable_device_address()) {
        VkBufferDeviceAddressInfo address_info{};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = result.buffer;
        result.address = vkGetBufferDeviceAddress(_device->logic_device(), &address_info);
    }
    return result;
}

void OpacityMicromap::destroy_mapped_buffer(MappedBuffer &buffer) noexcept {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_device->allocator().allocator(),
                         buffer.buffer, buffer.allocation);
        buffer = {};
    }
}

OpacityMicromap *OpacityMicromap::create(
    Device *device,
    detail::OpacityMicromapBuildRequest const &request,
    luisa::span<std::byte const> data,
    luisa::span<detail::OpacityMicromapTriangleRecord const> triangles) noexcept {
    auto plan = detail::plan_opacity_micromap_build(request);
    if (!plan.build) {
        // Every rejection - the extension being off, the public API being unable
        // to request a micromap, an unsupported object model, an illegal level or
        // count - stops here, before a single Vulkan entry point is called.
        return nullptr;
    }
    LUISA_ASSERT(available(device),
                 "an opacity micromap build plan was committed for a device "
                 "without VK_KHR_opacity_micromap");
    LUISA_ASSERT(plan.object_model == detail::MicromapObjectModel::EXT_MICROMAP_OBJECT,
                 "only the VkMicromapEXT object model is implemented");
    LUISA_ASSERT(data.size() >= plan.data_bytes,
                 "the micromap data buffer is {} bytes, but {} triangles at "
                 "subdivision level {} need {} bytes",
                 data.size(), plan.triangle_count,
                 static_cast<uint32_t>(plan.subdivision_level), plan.data_bytes);
    LUISA_ASSERT(triangles.size() >= plan.triangle_count,
                 "{} triangle records were supplied for a micromap of {} triangles",
                 triangles.size(), plan.triangle_count);
    auto micromap = new_with_allocator<OpacityMicromap>(device, plan);
    micromap->_usage = VkMicromapUsageEXT{
        .count = plan.usage_count,
        .subdivisionLevel = plan.usage_subdivision_level,
        .format = static_cast<uint32_t>(opacity_micromap_detail::to_vk_format(plan.usage_format))};
    // 1. Sizes. `vkGetMicromapBuildSizesEXT` ignores dstMicromap/mode/data/
    //    scratchData/triangleArray, so the usage counts are all it needs.
    VkMicromapBuildInfoEXT build_info{};
    build_info.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
    build_info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    build_info.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    build_info.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    build_info.dstMicromap = VK_NULL_HANDLE;
    build_info.usageCountsCount = plan.usage_count;
    build_info.pUsageCounts = &micromap->_usage;
    build_info.ppUsageCounts = nullptr;
    build_info.triangleArrayStride = plan.triangle_array_stride;
    VkMicromapBuildSizesInfoEXT sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT;
    vkGetMicromapBuildSizesEXT(
        device->logic_device(),
        plan.mode == detail::MicromapBuildMode::HOST ?
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR :
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &sizes);
    micromap->_sizes = sizes;
    micromap->_discardable = sizes.discardable == VK_TRUE;
    // 2. Buffers. A zero-sized allocation is illegal, so a driver that reports
    //    zero scratch keeps a one-byte block that is never read.
    micromap->_storage = micromap->create_mapped_buffer(
        VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT,
        sizes.micromapSize == 0u ? 1u : static_cast<size_t>(sizes.micromapSize));
    micromap->_data = micromap->create_mapped_buffer(
        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
        plan.data_bytes == 0u ? 1u : static_cast<size_t>(plan.data_bytes));
    micromap->_triangles = micromap->create_mapped_buffer(
        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
        plan.triangle_array_bytes == 0u ? 1u : static_cast<size_t>(plan.triangle_array_bytes));
    micromap->_scratch = micromap->create_mapped_buffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizes.buildScratchSize == 0u ? 1u : static_cast<size_t>(sizes.buildScratchSize));
    // 3. Content. The micromap data is packed 1 or 2 bits per micro triangle and
    //    the triangle records are the caller's (VkMicromapTriangleEXT) layout,
    //    which the plan's POD mirrors byte for byte.
    if (micromap->_data.mapped != nullptr) {
        std::memcpy(micromap->_data.mapped, data.data(),
                    static_cast<size_t>(plan.data_bytes));
    }
    if (micromap->_triangles.mapped != nullptr) {
        std::memcpy(micromap->_triangles.mapped, triangles.data(),
                    static_cast<size_t>(plan.triangle_array_bytes));
    }
    // 4. The object itself.
    LUISA_ASSERT(opacity_micromap_detail::storage_offset_legal(0u),
                 "the micromap storage offset must stay 256-byte aligned");
    VkMicromapCreateInfoEXT create_info{};
    // VkMicromapCreateInfoEXT has no sType/pNext: it is a plain input structure,
    // not a chainable one.
    create_info.createFlags = 0u;
    create_info.buffer = micromap->_storage.buffer;
    create_info.offset = 0u;
    create_info.size = sizes.micromapSize;
    create_info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    create_info.deviceAddress = 0u;
    VK_CHECK_RESULT(vkCreateMicromapEXT(
        device->logic_device(), &create_info, Device::alloc_callbacks(),
        &micromap->_micromap));
    return micromap;
}

OpacityMicromap::~OpacityMicromap() {
    if (_micromap != VK_NULL_HANDLE) {
        vkDestroyMicromapEXT(_device->logic_device(), _micromap,
                            Device::alloc_callbacks());
        _micromap = VK_NULL_HANDLE;
    }
    destroy_mapped_buffer(_scratch);
    destroy_mapped_buffer(_triangles);
    destroy_mapped_buffer(_data);
    destroy_mapped_buffer(_storage);
}

void OpacityMicromap::fill_build_info(
    VkMicromapBuildInfoEXT &info, detail::MicromapBuildMode mode) const noexcept {
    info = VkMicromapBuildInfoEXT{};
    info.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
    info.pNext = nullptr;
    info.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    info.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    info.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    info.dstMicromap = _micromap;
    info.usageCountsCount = _plan.usage_count;
    info.pUsageCounts = &_usage;
    info.ppUsageCounts = nullptr;
    info.triangleArrayStride = _plan.triangle_array_stride;
    if (mode == detail::MicromapBuildMode::HOST) {
        info.data.hostAddress = _data.mapped;
        info.scratchData.hostAddress = _scratch.mapped;
        info.triangleArray.hostAddress = _triangles.mapped;
    } else {
        info.data.deviceAddress = _data.address;
        info.scratchData.deviceAddress = _scratch.address;
        info.triangleArray.deviceAddress = _triangles.address;
    }
}

void OpacityMicromap::build_device(CommandBuffer &cmdbuffer) {
    if (_micromap == VK_NULL_HANDLE) [[unlikely]] {
        LUISA_ERROR("vkCmdBuildMicromapsEXT was requested for a micromap that "
                    "was never created.");
    }
    if (_plan.mode != detail::MicromapBuildMode::DEVICE) [[unlikely]] {
        LUISA_ERROR("vkCmdBuildMicromapsEXT requires the device build plan "
                    "(the micromap was sized for a host build).");
    }
    // The micromap build stage/access pair exists only in
    // synchronization2 (VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT /
    // VK_ACCESS_2_MICROMAP_{READ,WRITE}_BIT_EXT have no legacy twin), so a
    // device without sync2 cannot express this dependency at all.
    LUISA_ASSERT(_device->sync2_capable(),
                 "building an opacity micromap needs the micromap build stage of "
                 "synchronization2");
    // Host writes (step 3 of create(), and any later content update) have to be
    // made available to the micromap build; the scratch/dstMicromap accesses are
    // covered by the same memory barrier.
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.pNext = nullptr;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
    barrier.dstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT |
                            VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.pNext = nullptr;
    dependency.dependencyFlags = 0u;
    dependency.memoryBarrierCount = 1u;
    dependency.pMemoryBarriers = &barrier;
    detail::cmd_pipeline_barrier(cmdbuffer.cmdbuffer(), _device, &dependency);
    VkMicromapBuildInfoEXT build_info{};
    fill_build_info(build_info, detail::MicromapBuildMode::DEVICE);
    // VkMicromapBuildInfoEXT is only read while the command is recorded, so the
    // stack copy is valid.
    vkCmdBuildMicromapsEXT(cmdbuffer.cmdbuffer(), 1, &build_info);
    _built = true;
}

bool OpacityMicromap::build_host() noexcept {
    if (_plan.mode != detail::MicromapBuildMode::HOST) {
        // Includes the "no micromapHostCommands" case: the entry point must not
        // be called at all.
        return false;
    }
    VkMicromapBuildInfoEXT build_info{};
    fill_build_info(build_info, detail::MicromapBuildMode::HOST);
    // A null deferredOperation means "must complete before returning"; a driver
    // cannot answer VK_OPERATION_DEFERRED_KHR for it.
    VK_CHECK_RESULT(vkBuildMicromapsEXT(_device->logic_device(),
                                        VK_NULL_HANDLE, 1u, &build_info));
    _built = true;
    return true;
}

bool OpacityMicromap::fill_attach_info(
    VkAccelerationStructureTrianglesOpacityMicromapEXT &info,
    detail::OpacityMicromapAttachRequest const &request,
    uint64_t index_buffer_device_address) const noexcept {
    auto plan = detail::plan_opacity_micromap_attach(request);
    if (!plan.chain) { return false; }
    if (_micromap == VK_NULL_HANDLE || !_built) { return false; }
    // Geometry-level micromap inputs must be 128-byte aligned; validate before
    // writing anything so a rejected plan leaves `info` untouched.
    if (plan.index_type != static_cast<uint8_t>(detail::OpacityMicromapIndexType::NONE) &&
        index_buffer_device_address % detail::opacity_micromap_attach_input_alignment != 0u) {
        return false;
    }
    info = VkAccelerationStructureTrianglesOpacityMicromapEXT{};
    // VkAccelerationStructureTrianglesOpacityMicromapEXT has no sType: it is a
    // pNext-only extension of VkAccelerationStructureGeometryTrianglesDataKHR.
    info.pNext = nullptr;
    info.indexType = opacity_micromap_detail::to_vk_index_type(plan.index_type);
    info.indexBuffer.deviceAddress =
        info.indexType == VK_INDEX_TYPE_NONE_KHR ? 0u : index_buffer_device_address;
    info.indexStride = plan.index_stride;
    info.baseTriangle = plan.base_triangle;
    info.usageCountsCount = plan.usage_count;
    info.pUsageCounts = plan.usage_count == 0u ? nullptr : &_usage;
    info.ppUsageCounts = nullptr;
    info.micromap = _micromap;
    return true;
}

}// namespace lc::vk
