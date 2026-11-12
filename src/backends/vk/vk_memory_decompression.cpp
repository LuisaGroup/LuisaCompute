#include "vk_memory_decompression.h"
#include "buffer.h"
#include "device.h"
#include "vk_extension_registry.h"
#include "vk_memory_policy.h"
#include <luisa/backends/ext/vk_custom_cmd.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
#include <algorithm>
#include <cstring>

namespace lc::vk::memory_decompression {

namespace vk_memory_decompression_detail {

using namespace luisa;
using namespace luisa::compute;

// The pipeline stage and access types vkCmdDecompressMemoryEXT uses for its
// source and destination data (synchronization.adoc,
// VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT).
constexpr auto decompression_stage = VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT;
constexpr auto decompression_read = VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT;
constexpr auto decompression_write = VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT;

[[nodiscard]] VkPhysicalDeviceMemoryDecompressionPropertiesEXT
query_properties(const Device *device) noexcept {
    VkPhysicalDeviceMemoryDecompressionPropertiesEXT properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties};
    vkGetPhysicalDeviceProperties2(device->physical_device(), &properties2);
    return properties;
}

// Resolves a backend buffer handle to the device address of the requested byte
// range, rejecting handles that are not backend buffers.
// Whether the physical device exposes VK_KHR_maintenance5, the extension that
// defines VkBufferUsageFlags2CreateInfoKHR (promoted to Vulkan 1.4). Vulkan
// exposes no query for a logical device's enabled extension list, so physical
// support is the best available evidence; `Device::_init_device` only enables
// the extension together with the async-copy feature.
[[nodiscard]] bool maintenance5_supported(
    VkPhysicalDevice physical_device) noexcept {
    uint32_t count = 0u;
    if (vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, nullptr) != VK_SUCCESS ||
        count == 0u) {
        return false;
    }
    luisa::vector<VkExtensionProperties> properties(count);
    if (vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, properties.data()) != VK_SUCCESS) {
        return false;
    }
    return std::any_of(
        properties.begin(), properties.end(), [](auto const &property) noexcept {
            return std::strcmp(property.extensionName,
                               VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0;
        });
}

// VK_NV_memory_decompression is an alias of VK_EXT_memory_decompression for the
// *feature and property* queries only: aliased feature structures share their
// VkStructureType. The commands this backend records -
// vkCmdDecompressMemoryEXT and vkCmdDecompressMemoryIndirectCountEXT - and the
// VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT /
// VK_ACCESS_2_MEMORY_DECOMPRESSION_*_EXT bits are defined by the EXT spelling
// alone, so a device that negotiated only the NV spelling must not report them
// as available.
[[nodiscard]] bool ext_spelling_enabled(const Device *device) noexcept {
    for (auto const &state : device->optional_extensions()) {
        if (state.id == detail::OptionalExtension::MEMORY_DECOMPRESSION) {
            return state.name != nullptr &&
                   std::strcmp(state.name,
                               VK_EXT_MEMORY_DECOMPRESSION_EXTENSION_NAME) == 0;
        }
    }
    return false;
}

[[nodiscard]] VkDeviceAddress resolve_address(
    uint64_t buffer_handle, uint64_t offset, uint64_t size,
    const char *role) noexcept {
    LUISA_ASSERT(buffer_handle != luisa::compute::invalid_resource_handle,
                 "Vulkan memory decompression {} buffer is invalid.", role);
    auto *buffer = reinterpret_cast<Buffer const *>(buffer_handle);
    LUISA_ASSERT(buffer->tag() == Resource::Tag::kBuffer,
                 "Vulkan memory decompression {} handle does not name a buffer.",
                 role);
    auto byte_size = static_cast<uint64_t>(buffer->byte_size());
    LUISA_ASSERT(offset <= byte_size && size <= byte_size - offset,
                 "Vulkan memory decompression {} range [{}, {}) exceeds the "
                 "buffer's {} bytes.",
                 role, offset, offset + size, byte_size);
    auto base = buffer->get_device_address();
    LUISA_ASSERT(base != 0u && offset <= UINT64_MAX - base,
                 "Vulkan memory decompression {} buffer has no usable device "
                 "address ({}).",
                 role, base);
    return static_cast<VkDeviceAddress>(base + offset);
}

// Declares the source read and the destination write of one region. `execute`
// records the same native scope, so reordering and barriers agree with the
// encoded command.
void append_region_usages(
    luisa::vector<VKCustomCmd::ResourceUsage> &usages,
    Region const &region) noexcept {
    usages.emplace_back(
        Argument::Buffer{
            .handle = region.source_buffer,
            .offset = static_cast<size_t>(region.source_offset),
            .size = static_cast<size_t>(region.compressed_size)},
        decompression_stage, decompression_read);
    usages.emplace_back(
        Argument::Buffer{
            .handle = region.destination_buffer,
            .offset = static_cast<size_t>(region.destination_offset),
            .size = static_cast<size_t>(region.decompressed_size)},
        decompression_stage, decompression_write);
}

// vkCmdDecompressMemoryEXT with a fully resolved region list.
class DecompressMemoryCommand final : public VKCustomCmd {
    luisa::vector<VkDecompressMemoryRegionEXT> _regions;
    luisa::vector<VKCustomCmd::ResourceUsage> _usages;
    VkMemoryDecompressionMethodFlagsEXT _method{};

public:
    DecompressMemoryCommand(
        luisa::vector<VkDecompressMemoryRegionEXT> &&regions,
        luisa::vector<VKCustomCmd::ResourceUsage> &&usages,
        VkMemoryDecompressionMethodFlagsEXT method) noexcept
        : _regions{std::move(regions)},
          _usages{std::move(usages)},
          _method{method} {}
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] luisa::span<VKCustomCmd::ResourceUsage>
    get_resource_usages() noexcept override { return _usages; }
    void execute(VkPhysicalDevice, VkDevice, VkQueue,
                 VkCommandBuffer cmdbuffer,
                 VkDescriptorPool) const noexcept override {
        LUISA_ASSERT(vkCmdDecompressMemoryEXT != nullptr,
                     "vkCmdDecompressMemoryEXT is unavailable even though "
                     "VK_EXT_memory_decompression is enabled.");
        LUISA_ASSERT(!_regions.empty(),
                     "vkCmdDecompressMemoryEXT requires at least one region.");
        VkDecompressMemoryInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DECOMPRESS_MEMORY_INFO_EXT,
            .pNext = nullptr,
            .decompressionMethod = _method,
            .regionCount = static_cast<uint32_t>(_regions.size()),
            .pRegions = _regions.data()};
        vkCmdDecompressMemoryEXT(cmdbuffer, &info);
    }
};

// vkCmdDecompressMemoryIndirectCountEXT with parameters read from GPU memory.
class DecompressMemoryIndirectCommand final : public VKCustomCmd {
    luisa::vector<VKCustomCmd::ResourceUsage> _usages;
    VkMemoryDecompressionMethodFlagsEXT _method{};
    VkDeviceAddress _commands_address{0u};
    VkDeviceAddress _count_address{0u};
    uint32_t _max_count{0u};
    uint32_t _stride{0u};

public:
    DecompressMemoryIndirectCommand(
        luisa::vector<VKCustomCmd::ResourceUsage> &&usages,
        VkMemoryDecompressionMethodFlagsEXT method,
        VkDeviceAddress commands_address, VkDeviceAddress count_address,
        uint32_t max_count, uint32_t stride) noexcept
        : _usages{std::move(usages)},
          _method{method},
          _commands_address{commands_address},
          _count_address{count_address},
          _max_count{max_count},
          _stride{stride} {}
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] luisa::span<VKCustomCmd::ResourceUsage>
    get_resource_usages() noexcept override { return _usages; }
    void execute(VkPhysicalDevice, VkDevice, VkQueue,
                 VkCommandBuffer cmdbuffer,
                 VkDescriptorPool) const noexcept override {
        LUISA_ASSERT(vkCmdDecompressMemoryIndirectCountEXT != nullptr,
                     "vkCmdDecompressMemoryIndirectCountEXT is unavailable even "
                     "though VK_EXT_memory_decompression is enabled.");
        vkCmdDecompressMemoryIndirectCountEXT(
            cmdbuffer, _method, _commands_address, _count_address,
            _max_count, _stride);
    }
};

// Validates the request against the spec and fills the flattened native region
// list used for the address-based validation below.
void validate_regions(
    const Device *device, luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method,
    luisa::vector<detail::DecompressionRegion> &flat) noexcept {
    LUISA_ASSERT(memory_decompression::available(device),
                 "Vulkan memory decompression is unavailable on this device "
                 "(VK_EXT_memory_decompression was not enabled).");
    LUISA_ASSERT(!regions.empty(),
                 "Vulkan memory decompression needs at least one region.");
    LUISA_ASSERT(regions.size() <= UINT32_MAX,
                 "Vulkan memory decompression supports at most {} regions, {} "
                 "were requested.",
                 UINT32_MAX, regions.size());
    flat.clear();
    flat.reserve(regions.size());
    for (auto const &region : regions) {
        auto source = resolve_address(
            region.source_buffer, region.source_offset,
            region.compressed_size, "source");
        auto destination = resolve_address(
            region.destination_buffer, region.destination_offset,
            region.decompressed_size, "destination");
        flat.emplace_back(detail::DecompressionRegion{
            .src_address = static_cast<uint64_t>(source),
            .dst_address = static_cast<uint64_t>(destination),
            .compressed_size = region.compressed_size,
            .decompressed_size = region.decompressed_size});
    }
    auto plan = detail::plan_decompression(
        true, method, supported_methods(device), flat.data(),
        static_cast<uint32_t>(flat.size()));
    LUISA_ASSERT(static_cast<bool>(plan),
                 "Invalid Vulkan memory decompression request ({} at region "
                 "{}): method {:#x}, {} regions.",
                 detail::decompression_validation_status_name(plan.status),
                 plan.region_index, static_cast<uint64_t>(method),
                 flat.size());
}

// Runs the recorded command on `stream`, optionally waiting for completion.
void submit(
    Device *device, uint64_t stream, CommandList &&list,
    bool synchronize) noexcept {
    device->dispatch(stream, std::move(list));
    if (synchronize) {
        device->synchronize_stream(stream);
    }
}

}// namespace vk_memory_decompression_detail

bool available(const Device *device) noexcept {
    // Synchronization2 is required because the extension's pipeline stage and
    // access bits live above bit 31 and cannot be expressed by the legacy
    // VkPipelineStageFlags/VkAccessFlags barriers.
    return device != nullptr &&
           device->optional_extension_enabled(
               detail::OptionalExtension::MEMORY_DECOMPRESSION) &&
           vk_memory_decompression_detail::ext_spelling_enabled(device) &&
           device->sync2_capable();
}

VkMemoryDecompressionMethodFlagsEXT supported_methods(
    const Device *device) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression properties need a live device.");
    return vk_memory_decompression_detail::query_properties(device)
        .decompressionMethods;
}

uint64_t max_indirect_count(const Device *device) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression properties need a live device.");
    return vk_memory_decompression_detail::query_properties(device)
        .maxDecompressionIndirectCount;
}

luisa::compute::BufferCreationInfo create_buffer(
    Device *device, size_t byte_size) noexcept {
    using namespace luisa::compute;
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression buffers need a live device.");
    LUISA_ASSERT(available(device),
                 "Vulkan memory decompression is unavailable on this device "
                 "(VK_EXT_memory_decompression was not enabled).");
    LUISA_ASSERT(byte_size > 0u,
                 "A Vulkan memory decompression buffer needs a non-zero size.");
    // VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT is bit 32, so it can only
    // travel through VkBufferUsageFlags2CreateInfoKHR. That struct is defined by
    // VK_KHR_maintenance5 and promoted to Vulkan 1.4; without one of the two the
    // request cannot be expressed conformantly, so fail closed instead of
    // creating a buffer whose usage violates
    // VUID-VkDecompressMemoryRegionEXT-srcAddress-11764.
    auto api_version = device->properties().apiVersion;
    // The backend creates its Vulkan instance as 1.3 and an imported instance
    // only attests 1.3+, so a structure promoted to core in 1.4 is reachable
    // through VK_KHR_maintenance5 alone: pass 0 as the core version so the
    // decision rests on the extension. Physical-device support is the only
    // available attestation, because Vulkan cannot report the extension list a
    // logical device was created with, and `Device::_init_device` currently
    // enables VK_KHR_maintenance5 only together with the async-copy feature.
    if (!detail::plan_buffer_usage_flags2(
            0u,
            vk_memory_decompression_detail::maintenance5_supported(
                device->physical_device()))) {
        LUISA_ERROR(
            "Vulkan memory decompression needs VK_BUFFER_USAGE_2_MEMORY_"
            "DECOMPRESSION_BIT_EXT (bit 32), which requires "
            "VkBufferUsageFlags2CreateInfoKHR from VK_KHR_maintenance5 or "
            "Vulkan 1.4. This device reports Vulkan {}.{}.{} without "
            "VK_KHR_maintenance5.",
            VK_API_VERSION_MAJOR(api_version), VK_API_VERSION_MINOR(api_version),
            VK_API_VERSION_PATCH(api_version));
    }
    LUISA_ASSERT(device->enable_device_address(),
                 "Vulkan memory decompression regions are addressed by "
                 "VkDeviceAddress, so the device must enable buffer device "
                 "addresses.");
    auto usage = storage_buffer_usage_flags(
        device, false, static_cast<VkBufferUsageFlagBits>(0u));
    // VkBufferUsageFlags2CreateInfoKHR replaces the 32-bit usage mask, so the
    // full mask is repeated in the 64-bit form plus the decompression bit.
    auto usage2 = static_cast<VkBufferUsageFlags2KHR>(usage) |
                  VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT;
    device->mark_optional_extension_active(
        detail::OptionalExtension::MEMORY_DECOMPRESSION);
    auto *buffer = new PolicyBuffer(
        device, byte_size, usage, usage2, false,
        MemoryInitPolicy::kUndefined);
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->vk_buffer();
    info.element_stride = 1u;
    info.total_size_bytes = buffer->byte_size();
    return info;
}

void append_decompress_memory(
    luisa::compute::CommandList &list, Device *device,
    luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    using namespace luisa::compute;
    luisa::vector<detail::DecompressionRegion> flat;
    vk_memory_decompression_detail::validate_regions(
        device, regions, method, flat);
    // The extension's decompression command is really in use on this device.
    device->mark_optional_extension_active(
        detail::OptionalExtension::MEMORY_DECOMPRESSION);
    luisa::vector<VkDecompressMemoryRegionEXT> native_regions;
    native_regions.reserve(regions.size());
    luisa::vector<VKCustomCmd::ResourceUsage> usages;
    usages.reserve(regions.size() * 2u);
    for (auto const &region : regions) {
        auto source = vk_memory_decompression_detail::resolve_address(
            region.source_buffer, region.source_offset,
            region.compressed_size, "source");
        auto destination = vk_memory_decompression_detail::resolve_address(
            region.destination_buffer, region.destination_offset,
            region.decompressed_size, "destination");
        native_regions.emplace_back(VkDecompressMemoryRegionEXT{
            .srcAddress = source,
            .dstAddress = destination,
            .compressedSize = region.compressed_size,
            .decompressedSize = region.decompressed_size});
        vk_memory_decompression_detail::append_region_usages(usages, region);
    }
    list << luisa::make_unique<
        vk_memory_decompression_detail::DecompressMemoryCommand>(
        std::move(native_regions), std::move(usages), method);
}

void append_decompress_memory_indirect(
    luisa::compute::CommandList &list, Device *device,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    using namespace luisa::compute;
    LUISA_ASSERT(memory_decompression::available(device),
                 "Vulkan memory decompression is unavailable on this device "
                 "(VK_EXT_memory_decompression was not enabled).");
    // VUID-vkCmdDecompressMemoryIndirectCountEXT-indirectCommandsAddress-07694
    // and -indirectCommandsCountAddress-07697 require both addresses to come
    // from buffers created with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT; the backend
    // cannot inspect a buffer's usage mask, so the caller must use
    // `memory_decompression::create_buffer`, which always sets it.
    auto commands_address = vk_memory_decompression_detail::resolve_address(
        indirect_commands_buffer, 0u,
        sizeof(VkDecompressMemoryRegionEXT) * static_cast<size_t>(max_decompression_count),
        "indirect-commands");
    auto count_address = vk_memory_decompression_detail::resolve_address(
        indirect_count_buffer, 0u, sizeof(uint32_t), "indirect-count");
    auto plan = detail::plan_decompression_indirect(
        true, method, supported_methods(device),
        static_cast<uint64_t>(commands_address),
        static_cast<uint64_t>(count_address),
        max_decompression_count, max_indirect_count(device), stride);
    LUISA_ASSERT(static_cast<bool>(plan),
                 "Invalid Vulkan indirect memory decompression request ({}): "
                 "method {:#x}, {} decompressions, stride {}.",
                 detail::decompression_validation_status_name(plan.status),
                 static_cast<uint64_t>(method), max_decompression_count, stride);
    // The extension's indirect decompression command is really in use.
    device->mark_optional_extension_active(
        detail::OptionalExtension::MEMORY_DECOMPRESSION);
    luisa::vector<VKCustomCmd::ResourceUsage> usages;
    usages.emplace_back(
        Argument::Buffer{
            .handle = indirect_commands_buffer,
            .offset = 0u,
            .size = sizeof(VkDecompressMemoryRegionEXT) *
                    static_cast<size_t>(max_decompression_count)},
        vk_memory_decompression_detail::decompression_stage,
        static_cast<VkAccessFlagBits2>(
            vk_memory_decompression_detail::decompression_read |
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT));
    usages.emplace_back(
        Argument::Buffer{
            .handle = indirect_count_buffer,
            .offset = 0u,
            .size = sizeof(uint32_t)},
        vk_memory_decompression_detail::decompression_stage,
        static_cast<VkAccessFlagBits2>(
            vk_memory_decompression_detail::decompression_read |
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT));
    list << luisa::make_unique<
        vk_memory_decompression_detail::DecompressMemoryIndirectCommand>(
        std::move(usages), method, commands_address, count_address,
        max_decompression_count, stride);
}

void dispatch_decompress_memory(
    Device *device, uint64_t stream, luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression needs a live device.");
    auto list = luisa::compute::CommandList::create();
    append_decompress_memory(list, device, regions, method);
    vk_memory_decompression_detail::submit(
        device, stream, std::move(list), false);
}

void dispatch_decompress_memory_indirect(
    Device *device, uint64_t stream,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression needs a live device.");
    auto list = luisa::compute::CommandList::create();
    append_decompress_memory_indirect(
        list, device, indirect_commands_buffer, indirect_count_buffer,
        max_decompression_count, stride, method);
    vk_memory_decompression_detail::submit(
        device, stream, std::move(list), false);
}

void decompress_memory(
    Device *device, uint64_t stream, luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression needs a live device.");
    auto list = luisa::compute::CommandList::create();
    append_decompress_memory(list, device, regions, method);
    vk_memory_decompression_detail::submit(
        device, stream, std::move(list), true);
}

void decompress_memory_indirect(
    Device *device, uint64_t stream,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Vulkan memory decompression needs a live device.");
    auto list = luisa::compute::CommandList::create();
    append_decompress_memory_indirect(
        list, device, indirect_commands_buffer, indirect_count_buffer,
        max_decompression_count, stride, method);
    vk_memory_decompression_detail::submit(
        device, stream, std::move(list), true);
}

}// namespace lc::vk::memory_decompression
