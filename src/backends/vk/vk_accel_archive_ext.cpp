#include "vk_accel_archive_ext.h"

#include <algorithm>
#include <cstring>

#include <volk.h>
#include <luisa/vstl/functional.h>

#include "blas.h"
#include "command_buffer_sync.h"
#include "default_buffer.h"
#include "device.h"
#include "log.h"
#include "primitive_base.h"
#include "stream.h"
#include "vulkan_builtin_spirv_embedded.h"

namespace lc::vk {

namespace {

using namespace luisa::compute;

constexpr uint32_t kVkCompatMagic = 0x4B564341u;// 'ACVK'
constexpr size_t kVkSerializedHeaderBytes = 2u * VK_UUID_SIZE + 3u * sizeof(uint64_t);

struct VkAccelArchiveCompatBlob {
    uint32_t magic;
    uint32_t struct_size;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t driver_version;
    uint32_t api_version;
    uint8_t pipeline_cache_uuid[VK_UUID_SIZE];
    uint8_t vk_version_data[2u * VK_UUID_SIZE];
    uint64_t deserialized_size;
};

static_assert(sizeof(VkAccelArchiveCompatBlob) == 80u);

struct VkSerializedASHeader {
    uint8_t version_data[2u * VK_UUID_SIZE];
    uint64_t serialized_size;
    uint64_t deserialized_size;
    uint64_t bottom_level_count;
};

static_assert(sizeof(VkSerializedASHeader) == kVkSerializedHeaderBytes);
static_assert(sizeof(VkSerializedASHeader) == 56u);

// Non-trivially-destructible lease object for CommandBufferState::dispose_after_flush.
struct VkSlotLease {
    std::shared_ptr<VkAccelArchiveExt::StagingSlot> slot;
    VkSlotLease() noexcept = default;
    explicit VkSlotLease(std::shared_ptr<VkAccelArchiveExt::StagingSlot> s) noexcept
        : slot{std::move(s)} {}
    VkSlotLease(VkSlotLease &&) noexcept = default;
    VkSlotLease &operator=(VkSlotLease &&) noexcept = default;
    ~VkSlotLease() noexcept {
        if (slot) {
            slot->gpu_refs.fetch_sub(1u, std::memory_order_release);
            slot.reset();
        }
    }
};

[[nodiscard]] luisa::string format_uuid_hex(const uint8_t *bytes, size_t count) noexcept {
    static constexpr char kHex[] = "0123456789abcdef";
    luisa::string out;
    out.reserve(count * 2u);
    for (size_t i = 0u; i < count; ++i) {
        auto v = bytes[i];
        out.push_back(kHex[(v >> 4u) & 0xfu]);
        out.push_back(kHex[v & 0xfu]);
    }
    return out;
}

[[nodiscard]] uint32_t find_best_memory_type(
    const VkPhysicalDeviceMemoryProperties &props,
    uint32_t type_bits,
    VkMemoryPropertyFlags required_flags,
    VkMemoryPropertyFlags preferred_flags) noexcept {
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0u; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0u) {
            continue;
        }
        const auto flags = props.memoryTypes[i].propertyFlags;
        if ((flags & required_flags) != required_flags) {
            continue;
        }
        if ((flags & preferred_flags) == preferred_flags) {
            return i;
        }
        if (fallback == UINT32_MAX) {
            fallback = i;
        }
    }
    return fallback;
}

[[nodiscard]] AccelArchiveResult check_stream_and_blas(
    Device *device,
    uint64_t stream_handle,
    uint64_t mesh_handle,
    Stream *&out_stream,
    Blas *&out_blas) noexcept {
    if (device == nullptr || !device->enable_raytracing() || device->use_fallback_rtx()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "Vulkan hardware KHR raytracing is not active on this device.");
    }
    if (stream_handle == invalid_resource_handle || mesh_handle == invalid_resource_handle) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Invalid stream or mesh handle passed to VkAccelArchiveExt.");
    }
    if (device->owns_fallback_blas(mesh_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Software fallback RTX BLAS cannot be archived through Vulkan KHR.");
    }
    auto *stream = reinterpret_cast<Stream *>(stream_handle);
    if (stream->device() != device) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream belongs to a different Vulkan device.");
    }
    if (stream->stream_tag() == StreamTag::COPY) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Vulkan AS archive operations require a compute or graphics stream.");
    }
    auto *prim = reinterpret_cast<PrimitiveBase *>(mesh_handle);
    if (prim->device() != device) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh belongs to a different Vulkan device.");
    }
    if (prim->prim_tag() != PrimitiveBase::PrimTag::BLAS) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports standard static triangle BLAS only.");
    }
    auto *blas = static_cast<Blas *>(prim);
    if (blas->has_motion() || blas->opacity_micromap() != nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 does not support motion-blur or opacity-micromap BLAS.");
    }
    out_stream = stream;
    out_blas = blas;
    return {};
}

[[nodiscard]] VkAccelArchiveCompatBlob make_compat_blob(
    Device const *device,
    const VkSerializedASHeader &hdr) noexcept {
    const auto &props = device->properties();
    VkAccelArchiveCompatBlob blob{};
    blob.magic = kVkCompatMagic;
    blob.struct_size = static_cast<uint32_t>(sizeof(VkAccelArchiveCompatBlob));
    blob.vendor_id = props.vendorID;
    blob.device_id = props.deviceID;
    blob.driver_version = props.driverVersion;
    blob.api_version = props.apiVersion;
    std::memcpy(blob.pipeline_cache_uuid, props.pipelineCacheUUID, VK_UUID_SIZE);
    std::memcpy(blob.vk_version_data, hdr.version_data, sizeof(hdr.version_data));
    blob.deserialized_size = hdr.deserialized_size;
    return blob;
}

[[nodiscard]] AccelArchiveResult check_vk_compat_and_header(
    Device const *device,
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t expected_unpacked_size,
    uint64_t &out_deserialized_size) noexcept {
    if (compat_blob.size() != sizeof(VkAccelArchiveCompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Vulkan AS archive compat blob has unexpected byte size.");
    }
    VkAccelArchiveCompatBlob compat{};
    std::memcpy(&compat, compat_blob.data(), sizeof(compat));
    if (compat.magic != kVkCompatMagic || compat.struct_size != sizeof(VkAccelArchiveCompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Vulkan AS archive compat blob magic or size mismatch.");
    }

    const auto &props = device->properties();
    if (compat.vendor_id != props.vendorID || compat.device_id != props.deviceID) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            luisa::format(
                "Vulkan AS archive targets vendor=0x{:04x} device=0x{:04x}, current is 0x{:04x}:0x{:04x}.",
                compat.vendor_id, compat.device_id, props.vendorID, props.deviceID));
    }
    if (compat.driver_version != props.driverVersion ||
        compat.api_version != props.apiVersion ||
        std::memcmp(compat.pipeline_cache_uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "Vulkan AS archive driverVersion/apiVersion/pipelineCacheUUID does not match active device.");
    }

    if (payload.size() < sizeof(VkSerializedASHeader)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Vulkan serialized BLAS payload is smaller than 56-byte header.");
    }
    VkSerializedASHeader hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.serialized_size != payload.size() ||
        hdr.bottom_level_count != 0u ||
        hdr.deserialized_size == 0u ||
        hdr.deserialized_size != expected_unpacked_size ||
        hdr.deserialized_size != compat.deserialized_size) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Vulkan serialized BLAS header sizes or bottomLevelCount are invalid for static BLAS.");
    }
    if (std::memcmp(hdr.version_data, compat.vk_version_data, sizeof(hdr.version_data)) != 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Vulkan serialized BLAS version header disagrees with archive compat metadata.");
    }

    VkAccelerationStructureVersionInfoKHR ver_info{};
    ver_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
    ver_info.pVersionData = hdr.version_data;
    VkAccelerationStructureCompatibilityKHR vk_compat =
        VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
    vkGetDeviceAccelerationStructureCompatibilityKHR(
        device->logic_device(), &ver_info, &vk_compat);
    if (vk_compat != VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "vkGetDeviceAccelerationStructureCompatibilityKHR reported INCOMPATIBLE_KHR.");
    }

    out_deserialized_size = hdr.deserialized_size;
    return {};
}

}// namespace

VkAccelArchiveExt::VkAccelArchiveExt(Device *device) noexcept
    : _device{device} {}

VkAccelArchiveExt::~VkAccelArchiveExt() noexcept {
    _destroy_all_pools();
}

void VkAccelArchiveExt::_destroy_all_pools() noexcept {
    if (_device == nullptr || _device->logic_device() == VK_NULL_HANDLE) {
        return;
    }
    const auto vk_dev = _device->logic_device();
    vkDeviceWaitIdle(vk_dev);

    auto free_pool = [&](luisa::vector<std::shared_ptr<StagingSlot>> &pool) noexcept {
        for (auto &slot : pool) {
            if (!slot) {
                continue;
            }
            if (slot->mapped != nullptr && slot->memory != VK_NULL_HANDLE) {
                vkUnmapMemory(vk_dev, slot->memory);
                slot->mapped = nullptr;
            }
            if (slot->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(vk_dev, slot->buffer, Device::alloc_callbacks());
                slot->buffer = VK_NULL_HANDLE;
            }
            if (slot->memory != VK_NULL_HANDLE) {
                vkFreeMemory(vk_dev, slot->memory, Device::alloc_callbacks());
                slot->memory = VK_NULL_HANDLE;
            }
        }
        pool.clear();
    };

    std::lock_guard lock{_pool_mtx};
    free_pool(_rebar_pool);
    free_pool(_pinned_pool);
    free_pool(_vram_pool);
    if (_size_query_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(vk_dev, _size_query_pool, Device::alloc_callbacks());
        _size_query_pool = VK_NULL_HANDLE;
    }

    std::lock_guard csum_init_lock{_csum_init_mtx};
    std::lock_guard csum_lock{_csum_mtx};
    if (_csum_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vk_dev, _csum_pipeline, Device::alloc_callbacks());
        _csum_pipeline = VK_NULL_HANDLE;
    }
    if (_csum_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(vk_dev, _csum_shader, Device::alloc_callbacks());
        _csum_shader = VK_NULL_HANDLE;
    }
    if (_csum_pipe_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vk_dev, _csum_pipe_layout, Device::alloc_callbacks());
        _csum_pipe_layout = VK_NULL_HANDLE;
    }
    if (_csum_desc_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vk_dev, _csum_desc_pool, Device::alloc_callbacks());
        _csum_desc_pool = VK_NULL_HANDLE;
        _csum_desc_set = VK_NULL_HANDLE;
    }
    if (_csum_dsl != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vk_dev, _csum_dsl, Device::alloc_callbacks());
        _csum_dsl = VK_NULL_HANDLE;
    }
    if (_csum_rb_mapped != nullptr && _csum_rb_mem != VK_NULL_HANDLE) {
        vkUnmapMemory(vk_dev, _csum_rb_mem);
        _csum_rb_mapped = nullptr;
    }
    if (_csum_rb_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk_dev, _csum_rb_buf, Device::alloc_callbacks());
        _csum_rb_buf = VK_NULL_HANDLE;
    }
    if (_csum_rb_mem != VK_NULL_HANDLE) {
        vkFreeMemory(vk_dev, _csum_rb_mem, Device::alloc_callbacks());
        _csum_rb_mem = VK_NULL_HANDLE;
    }
    if (_csum_dev_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk_dev, _csum_dev_buf, Device::alloc_callbacks());
        _csum_dev_buf = VK_NULL_HANDLE;
    }
    if (_csum_dev_mem != VK_NULL_HANDLE) {
        vkFreeMemory(vk_dev, _csum_dev_mem, Device::alloc_callbacks());
        _csum_dev_mem = VK_NULL_HANDLE;
    }
    _csum_ready = false;
}

bool VkAccelArchiveExt::_ensure_checksum_pipeline() noexcept {
    std::lock_guard init_lock{_csum_init_mtx};
    if (_csum_attempted) {
        return _csum_ready;
    }
    _csum_attempted = true;
    if (_device == nullptr || _device->logic_device() == VK_NULL_HANDLE ||
        luisa_compute_vk_builtin_accel_archive_checksum_spv_size == 0u ||
        (luisa_compute_vk_builtin_accel_archive_checksum_spv_size % 4u) != 0u) {
        return false;
    }
    const auto vk_dev = _device->logic_device();

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0u;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1u;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1u;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1u;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 2u;
    dsl_ci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk_dev, &dsl_ci, Device::alloc_callbacks(), &_csum_dsl) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0u;
    pcr.size = 20u;

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1u;
    pl_ci.pSetLayouts = &_csum_dsl;
    pl_ci.pushConstantRangeCount = 1u;
    pl_ci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(vk_dev, &pl_ci, Device::alloc_callbacks(), &_csum_pipe_layout) != VK_SUCCESS) {
        return false;
    }

    luisa::vector<uint32_t> spv(luisa_compute_vk_builtin_accel_archive_checksum_spv_size / 4u);
    std::memcpy(spv.data(), luisa_compute_vk_builtin_accel_archive_checksum_spv, luisa_compute_vk_builtin_accel_archive_checksum_spv_size);

    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv.size() * sizeof(uint32_t);
    sm_ci.pCode = spv.data();
    if (vkCreateShaderModule(vk_dev, &sm_ci, Device::alloc_callbacks(), &_csum_shader) != VK_SUCCESS) {
        return false;
    }

    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = _csum_shader;
    cp_ci.stage.pName = "main";
    cp_ci.layout = _csum_pipe_layout;
    if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1u, &cp_ci, Device::alloc_callbacks(), &_csum_pipeline) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 2u;
    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 1u;
    dp_ci.poolSizeCount = 1u;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk_dev, &dp_ci, Device::alloc_callbacks(), &_csum_desc_pool) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo ds_ai{};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = _csum_desc_pool;
    ds_ai.descriptorSetCount = 1u;
    ds_ai.pSetLayouts = &_csum_dsl;
    if (vkAllocateDescriptorSets(vk_dev, &ds_ai, &_csum_desc_set) != VK_SUCCESS) {
        return false;
    }

    auto create_small_buf = [&](
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags req_flags,
                                VkMemoryPropertyFlags pref_flags,
                                VkBuffer &out_buf,
                                VkDeviceMemory &out_mem,
                                bool map_it,
                                void **out_mapped,
                                bool &out_coherent) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 256u;
        bci.usage = usage;
        _device->allocator().apply_queue_sharing(bci);
        if (vkCreateBuffer(vk_dev, &bci, Device::alloc_callbacks(), &out_buf) != VK_SUCCESS) {
            return false;
        }
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(vk_dev, out_buf, &mr);
        const uint32_t mtype = find_best_memory_type(
            _device->allocator().memory_properties(),
            mr.memoryTypeBits,
            req_flags,
            pref_flags);
        if (mtype == UINT32_MAX) {
            return false;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = std::max<VkDeviceSize>(mr.size, 256u);
        mai.memoryTypeIndex = mtype;
        if (vkAllocateMemory(vk_dev, &mai, Device::alloc_callbacks(), &out_mem) != VK_SUCCESS) {
            return false;
        }
        if (vkBindBufferMemory(vk_dev, out_buf, out_mem, 0u) != VK_SUCCESS) {
            return false;
        }
        out_coherent = (_device->allocator().memory_properties().memoryTypes[mtype].propertyFlags &
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;
        if (map_it) {
            if (vkMapMemory(vk_dev, out_mem, 0u, 256u, 0u, out_mapped) != VK_SUCCESS) {
                return false;
            }
        }
        return true;
    };

    bool dummy_coh = true;
    if (!create_small_buf(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            _csum_dev_buf,
            _csum_dev_mem,
            false,
            nullptr,
            dummy_coh) ||
        !create_small_buf(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            _csum_rb_buf,
            _csum_rb_mem,
            true,
            &_csum_rb_mapped,
            _csum_rb_host_coherent)) {
        return false;
    }

    _csum_ready = true;
    return true;
}

void VkAccelArchiveExt::_record_checksum_dispatch(
    VkCommandBuffer cmd,
    VkBuffer src_buf,
    uint32_t base_byte_offset,
    uint64_t byte_size,
    uint32_t start_word) noexcept {
    VkDescriptorBufferInfo src_info{src_buf, 0u, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo out_info{_csum_dev_buf, 0u, 256u};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = _csum_desc_set;
    writes[0].dstBinding = 0u;
    writes[0].descriptorCount = 1u;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &src_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = _csum_desc_set;
    writes[1].dstBinding = 1u;
    writes[1].descriptorCount = 1u;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &out_info;
    vkUpdateDescriptorSets(_device->logic_device(), 2u, writes, 0u, nullptr);

    vkCmdFillBuffer(cmd, _csum_dev_buf, 0u, 16u, 0u);

    const uint64_t total_words_u64 = (byte_size + 3ull) / 4ull;
    if (total_words_u64 > start_word) {
        VkBufferMemoryBarrier2 clear_to_comp{};
        clear_to_comp.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        clear_to_comp.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        clear_to_comp.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clear_to_comp.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clear_to_comp.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        clear_to_comp.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_to_comp.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_to_comp.buffer = _csum_dev_buf;
        clear_to_comp.offset = 0u;
        clear_to_comp.size = 16u;
        VkDependencyInfo dep1{};
        dep1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep1.bufferMemoryBarrierCount = 1u;
        dep1.pBufferMemoryBarriers = &clear_to_comp;
        detail::cmd_pipeline_barrier(cmd, _device, &dep1);

        const uint32_t total_words = static_cast<uint32_t>(total_words_u64);
        const uint32_t words_todo = total_words - start_word;
        const uint32_t groups = std::min<uint32_t>(2048u, (words_todo + 255u) / 256u);
        const uint32_t rem = static_cast<uint32_t>(byte_size & 3ull);
        const uint32_t tail_mask = (rem != 0u) ? ((1u << (rem * 8u)) - 1u) : 0xffffffffu;
        const uint32_t pc[5]{
            total_words,
            tail_mask,
            base_byte_offset,
            start_word,
            groups * 256u};

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _csum_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _csum_pipe_layout, 0u, 1u, &_csum_desc_set, 0u, nullptr);
        vkCmdPushConstants(cmd, _csum_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, 20u, pc);
        vkCmdDispatch(cmd, groups, 1u, 1u);

        VkBufferMemoryBarrier2 comp_to_copy{};
        comp_to_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        comp_to_copy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        comp_to_copy.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        comp_to_copy.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        comp_to_copy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        comp_to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        comp_to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        comp_to_copy.buffer = _csum_dev_buf;
        comp_to_copy.offset = 0u;
        comp_to_copy.size = 16u;
        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.bufferMemoryBarrierCount = 1u;
        dep2.pBufferMemoryBarriers = &comp_to_copy;
        detail::cmd_pipeline_barrier(cmd, _device, &dep2);
    } else {
        VkBufferMemoryBarrier2 clear_to_copy{};
        clear_to_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        clear_to_copy.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        clear_to_copy.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clear_to_copy.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        clear_to_copy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        clear_to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_to_copy.buffer = _csum_dev_buf;
        clear_to_copy.offset = 0u;
        clear_to_copy.size = 16u;
        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.bufferMemoryBarrierCount = 1u;
        dep2.pBufferMemoryBarriers = &clear_to_copy;
        detail::cmd_pipeline_barrier(cmd, _device, &dep2);
    }

    VkBufferCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
    region.srcOffset = 0u;
    region.dstOffset = 0u;
    region.size = 16u;
    VkCopyBufferInfo2 copy_buf{};
    copy_buf.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
    copy_buf.srcBuffer = _csum_dev_buf;
    copy_buf.dstBuffer = _csum_rb_buf;
    copy_buf.regionCount = 1u;
    copy_buf.pRegions = &region;
    detail::cmd_copy_buffer(cmd, _device, &copy_buf);

    VkBufferMemoryBarrier2 rb_to_host{};
    rb_to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    rb_to_host.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    rb_to_host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    rb_to_host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    rb_to_host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    rb_to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rb_to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rb_to_host.buffer = _csum_rb_buf;
    rb_to_host.offset = 0u;
    rb_to_host.size = 16u;
    VkDependencyInfo dep_rb{};
    dep_rb.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_rb.bufferMemoryBarrierCount = 1u;
    dep_rb.pBufferMemoryBarriers = &rb_to_host;
    detail::cmd_pipeline_barrier(cmd, _device, &dep_rb);
}

void VkAccelArchiveExt::_read_checksum_lanes(uint32_t (&out_lanes)[4]) noexcept {
    if (!_csum_rb_host_coherent && _csum_rb_mem != VK_NULL_HANDLE) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = _csum_rb_mem;
        range.offset = 0u;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(_device->logic_device(), 1u, &range);
    }
    std::memcpy(out_lanes, _csum_rb_mapped, 16u);
}

std::shared_ptr<VkAccelArchiveExt::StagingSlot> VkAccelArchiveExt::_acquire_pool_slot(
    luisa::vector<std::shared_ptr<StagingSlot>> &pool,
    size_t min_bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required_flags,
    VkMemoryPropertyFlags preferred_flags,
    bool map_host,
    bool need_device_address,
    bool mark_caller_held,
    bool mark_gpu_ref) noexcept {
    if (_device == nullptr || _device->logic_device() == VK_NULL_HANDLE) {
        return nullptr;
    }
    const size_t wanted = std::max<size_t>((min_bytes + 65535u) & (~size_t{65535u}), 65536u);
    std::lock_guard lock{_pool_mtx};

    std::shared_ptr<StagingSlot> best_slot;
    for (auto &slot : pool) {
        if (slot && !slot->caller_held &&
            slot->gpu_refs.load(std::memory_order_acquire) == 0u &&
            slot->capacity >= wanted) {
            if (!best_slot || slot->capacity < best_slot->capacity) {
                best_slot = slot;
            }
        }
    }
    if (best_slot) {
        if (mark_caller_held) {
            best_slot->caller_held = true;
        }
        if (mark_gpu_ref) {
            best_slot->gpu_refs.fetch_add(1u, std::memory_order_relaxed);
        }
        return best_slot;
    }

    // Evict an idle undersized slot before creating a larger one to bound resident pool size.
    for (size_t i = 0u; i < pool.size(); ++i) {
        auto &slot = pool[i];
        if (slot && !slot->caller_held &&
            slot->gpu_refs.load(std::memory_order_acquire) == 0u &&
            slot->capacity < wanted) {
            const auto vk_dev = _device->logic_device();
            if (slot->mapped != nullptr && slot->memory != VK_NULL_HANDLE) {
                vkUnmapMemory(vk_dev, slot->memory);
                slot->mapped = nullptr;
            }
            if (slot->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(vk_dev, slot->buffer, Device::alloc_callbacks());
                slot->buffer = VK_NULL_HANDLE;
            }
            if (slot->memory != VK_NULL_HANDLE) {
                vkFreeMemory(vk_dev, slot->memory, Device::alloc_callbacks());
                slot->memory = VK_NULL_HANDLE;
            }
            pool.erase(pool.begin() + static_cast<ptrdiff_t>(i));
            break;
        }
    }

    const size_t alloc_cap = wanted;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = alloc_cap;
    bci.usage = usage;
    _device->allocator().apply_queue_sharing(bci);

    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(_device->logic_device(), &bci, Device::alloc_callbacks(), &buf) != VK_SUCCESS) {
        return nullptr;
    }

    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(_device->logic_device(), buf, &mem_req);
    const uint32_t mem_idx = find_best_memory_type(
        _device->allocator().memory_properties(),
        mem_req.memoryTypeBits,
        required_flags,
        preferred_flags);
    if (mem_idx == UINT32_MAX) {
        vkDestroyBuffer(_device->logic_device(), buf, Device::alloc_callbacks());
        return nullptr;
    }

    VkMemoryAllocateFlagsInfo flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = need_device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0u;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = need_device_address ? &flags_info : nullptr;
    mai.allocationSize = std::max<VkDeviceSize>(mem_req.size, alloc_cap);
    mai.memoryTypeIndex = mem_idx;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(_device->logic_device(), &mai, Device::alloc_callbacks(), &mem) != VK_SUCCESS) {
        vkDestroyBuffer(_device->logic_device(), buf, Device::alloc_callbacks());
        return nullptr;
    }
    if (vkBindBufferMemory(_device->logic_device(), buf, mem, 0u) != VK_SUCCESS) {
        vkDestroyBuffer(_device->logic_device(), buf, Device::alloc_callbacks());
        vkFreeMemory(_device->logic_device(), mem, Device::alloc_callbacks());
        return nullptr;
    }

    void *mapped = nullptr;
    if (map_host) {
        if (vkMapMemory(_device->logic_device(), mem, 0u, alloc_cap, 0u, &mapped) != VK_SUCCESS) {
            vkDestroyBuffer(_device->logic_device(), buf, Device::alloc_callbacks());
            vkFreeMemory(_device->logic_device(), mem, Device::alloc_callbacks());
            return nullptr;
        }
    }

    VkDeviceAddress dev_addr = 0u;
    if (need_device_address) {
        VkBufferDeviceAddressInfoKHR dai{};
        dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        dai.buffer = buf;
        dev_addr = vkGetBufferDeviceAddress(_device->logic_device(), &dai);
    }

    auto slot = std::make_shared<StagingSlot>();
    slot->buffer = buf;
    slot->memory = mem;
    slot->mapped = mapped;
    slot->device_address = dev_addr;
    slot->capacity = alloc_cap;
    slot->mem_type_index = mem_idx;
    slot->host_coherent = (_device->allocator().memory_properties().memoryTypes[mem_idx].propertyFlags &
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;
    slot->caller_held = mark_caller_held;
    if (mark_gpu_ref) {
        slot->gpu_refs.store(1u, std::memory_order_relaxed);
    }
    pool.emplace_back(slot);
    return slot;
}

std::shared_ptr<VkAccelArchiveExt::StagingSlot> VkAccelArchiveExt::_find_caller_pinned_slot(
    const void *ptr,
    size_t bytes,
    size_t &out_offset) const noexcept {
    if (ptr == nullptr) {
        return nullptr;
    }
    const auto target = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock{_pool_mtx};
    for (const auto &slot : _pinned_pool) {
        if (!slot || slot->mapped == nullptr || !slot->caller_held) {
            continue;
        }
        const auto base = reinterpret_cast<uintptr_t>(slot->mapped);
        if (target >= base && target - base <= slot->capacity &&
            bytes <= slot->capacity - (target - base)) {
            out_offset = static_cast<size_t>(target - base);
            slot->gpu_refs.fetch_add(1u, std::memory_order_relaxed);
            return slot;
        }
    }
    return nullptr;
}

AccelArchiveCapabilities VkAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = AccelArchiveBackendKind::VULKAN;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    if (_device == nullptr) {
        caps.layout_info = "null_device";
        return caps;
    }
    const auto &props = _device->properties();
    caps.device_fingerprint = luisa::format(
        "{}:0x{:04x}:0x{:04x}",
        props.deviceName, props.vendorID, props.deviceID);
    caps.driver_fingerprint = luisa::format(
        "drv=0x{:08x};api=0x{:08x};uuid={}",
        props.driverVersion, props.apiVersion,
        format_uuid_hex(props.pipelineCacheUUID, VK_UUID_SIZE));

    const bool supported =
        _device->enable_raytracing() &&
        !_device->use_fallback_rtx() &&
        _device->enable_device_address() &&
        vkCmdCopyAccelerationStructureToMemoryKHR != nullptr &&
        vkCmdCopyMemoryToAccelerationStructureKHR != nullptr &&
        vkCmdWriteAccelerationStructuresPropertiesKHR != nullptr &&
        vkGetDeviceAccelerationStructureCompatibilityKHR != nullptr;

    caps.supported = supported;
    caps.static_triangle_blas = supported;
    caps.supports_compacted_export = supported;
    caps.requires_geometry_rebind = false;
    caps.import_is_async_on_stream = supported;
    caps.supports_pinned_staging = supported;
    caps.supports_direct_file_import = supported;
    caps.layout_info = supported ?
                           "vk_khr_acceleration_structure_serialize_v1" :
                           (_device->use_fallback_rtx() ?
                                "vk_fallback_rtx_disabled" :
                                "vk_khr_raytracing_unavailable");
    return caps;
}

AccelArchiveStagingBuffer VkAccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    if (!capabilities().supported) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    auto slot = _acquire_pool_slot(
        _pinned_pool,
        min_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        /*map_host=*/true,
        /*need_device_address=*/false,
        /*mark_caller_held=*/true);
    if (!slot || slot->mapped == nullptr) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    AccelArchiveStagingBuffer out{};
    out.handle = reinterpret_cast<uint64_t>(slot.get());
    out.mapped_data = static_cast<std::byte *>(slot->mapped);
    out.capacity = slot->capacity;
    out.alignment = 65536u;
    out.is_device_local = false;
    out.is_pinned_host = true;
    return out;
}

void VkAccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (buffer.handle == invalid_resource_handle || buffer.handle == 0u) {
        buffer = {};
        return;
    }
    bool found = false;
    {
        std::lock_guard lock{_pool_mtx};
        for (auto &slot : _pinned_pool) {
            if (slot && reinterpret_cast<uint64_t>(slot.get()) == buffer.handle) {
                slot->caller_held = false;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        AccelArchiveExt::release_staging_buffer(std::move(buffer));
    } else {
        buffer = {};
    }
}

AccelArchiveResult VkAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    auto caps = capabilities();
    if (!caps.supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "Vulkan KHR AS archive extension is not supported on this device.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::VULKAN,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t deserialized_size = 0u;
    auto chk = check_vk_compat_and_header(
        _device, view.compat_blob, view.payload, view.meta.unpacked_as_size, deserialized_size);
    if (!chk) {
        return chk;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = deserialized_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult VkAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!capabilities().supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "Vulkan KHR AS archive extension is not supported on this device.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }

    Stream *stream = nullptr;
    Blas *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, stream, blas);
    if (!handle_res) {
        return handle_res;
    }

    const auto &blas_opt = blas->option();
    if (blas_opt.allow_update != desc.option.allow_update ||
        blas_opt.hint != desc.option.hint ||
        blas_opt.allow_compaction != desc.option.allow_compaction ||
        blas_opt.motion.keyframe_count != desc.option.motion.keyframe_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Vulkan BLAS AccelOption does not match AccelArchiveMeshDesc::option.");
    }

    stream->sync();

    if (blas->accel() == VK_NULL_HANDLE || blas->accel_size() == 0u || blas->last_primitive_count() == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "Vulkan BLAS has not been built yet.");
    }
    if (blas->last_primitive_count() != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "Vulkan BLAS primitive count ({}) does not match expected triangle_count ({}).",
                blas->last_primitive_count(), desc.triangle_count));
    }

    VkAccelerationStructureKHR src_accel = blas->accel();
    Buffer const *src_buffer = blas->accel_buffer();
    VkDeviceSize unpacked_size = blas->accel_size();
    bool is_compacted = blas->is_compacted();

    vstd::unique_ptr<DefaultBuffer> temp_compact_buffer;
    VkAccelerationStructureKHR temp_compact_accel = VK_NULL_HANDLE;
    auto cleanup_temp_compact = vstd::scope_exit([&] {
        if (temp_compact_accel != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(
                _device->logic_device(), temp_compact_accel, Device::alloc_callbacks());
            temp_compact_accel = VK_NULL_HANDLE;
        }
        if (temp_compact_buffer) {
            stream->remove_resource_state(temp_compact_buffer.get());
            temp_compact_buffer.reset();
        }
    });

    if (blas_opt.allow_compaction && !is_compacted && blas->compaction_query_written()) {
        if (!blas->has_tlas_refs()) {
            (void)blas->try_compact_in_place(*stream);
            src_accel = blas->accel();
            src_buffer = blas->accel_buffer();
            unpacked_size = blas->accel_size();
            is_compacted = blas->is_compacted();
        } else {
            auto compacted_size = blas->query_compacted_size();
            if (compacted_size > 0u && compacted_size < unpacked_size) {
                const auto alloc_size = std::max<size_t>((compacted_size + 255u) & (~255ull), 256u);
                temp_compact_buffer = vstd::make_unique<DefaultBuffer>(
                    _device,
                    alloc_size,
                    false,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
                VkAccelerationStructureCreateInfoKHR ci{};
                ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                ci.buffer = temp_compact_buffer->vk_buffer();
                ci.offset = 0u;
                ci.size = compacted_size;
                ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(
                    _device->logic_device(), &ci, Device::alloc_callbacks(), &temp_compact_accel));
                stream->execute_immediate([&](CommandBuffer &cb) {
                    cb.resource_barrier->record(
                        blas->accel_buffer(),
                        ResourceBarrier::Usage::kCopyAccelSrc);
                    cb.resource_barrier->record(
                        temp_compact_buffer.get(),
                        ResourceBarrier::Usage::kCopyAccelDst);
                    cb.resource_barrier->update_states(cb.cmdbuffer());
                    VkCopyAccelerationStructureInfoKHR copy_info{};
                    copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                    copy_info.src = blas->accel();
                    copy_info.dst = temp_compact_accel;
                    copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                    vkCmdCopyAccelerationStructureKHR(cb.cmdbuffer(), &copy_info);
                }, true);
                src_accel = temp_compact_accel;
                src_buffer = temp_compact_buffer.get();
                unpacked_size = compacted_size;
                is_compacted = true;
            } else if (compacted_size >= unpacked_size) {
                is_compacted = true;
            }
        }
    }

    uint64_t serialized_size = 0u;
    {
        std::lock_guard qlock{_query_mtx};
        if (_size_query_pool == VK_NULL_HANDLE) {
            VkQueryPoolCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
            qci.queryCount = 1u;
            VK_CHECK_RESULT(vkCreateQueryPool(
                _device->logic_device(), &qci, Device::alloc_callbacks(), &_size_query_pool));
        }

        stream->execute_immediate([&](CommandBuffer &cb) {
            cb.resource_barrier->record(src_buffer, ResourceBarrier::Usage::kCopyAccelSrc);
            cb.resource_barrier->update_states(cb.cmdbuffer());
            vkCmdResetQueryPool(cb.cmdbuffer(), _size_query_pool, 0u, 1u);
            vkCmdWriteAccelerationStructuresPropertiesKHR(
                cb.cmdbuffer(),
                1u,
                &src_accel,
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
                _size_query_pool,
                0u);
        }, true);

        VK_CHECK_RESULT(vkGetQueryPoolResults(
            _device->logic_device(),
            _size_query_pool,
            0u,
            1u,
            sizeof(uint64_t),
            &serialized_size,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    }

    if (serialized_size < kVkSerializedHeaderBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format(
                "Vulkan reported invalid serialized BLAS size {} (< 56 bytes).",
                serialized_size));
    }

    auto vram_ser_slot = _acquire_pool_slot(
        _vram_pool,
        static_cast<size_t>(serialized_size + 256u),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        /*map_host=*/false,
        /*need_device_address=*/true,
        /*mark_caller_held=*/false,
        /*mark_gpu_ref=*/true);
    VkSlotLease vram_lease{vram_ser_slot};

    auto pinned_rb_slot = _acquire_pool_slot(
        _pinned_pool,
        static_cast<size_t>(serialized_size),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        /*map_host=*/true,
        /*need_device_address=*/false,
        /*mark_caller_held=*/false,
        /*mark_gpu_ref=*/true);
    VkSlotLease rb_lease{pinned_rb_slot};

    if (!vram_ser_slot || !pinned_rb_slot || pinned_rb_slot->mapped == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to allocate Vulkan export staging pool buffers.");
    }

    const uint64_t base_addr = vram_ser_slot->device_address;
    const uint64_t aligned_offset = (256u - (base_addr & 255u)) & 255u;
    const uint64_t aligned_addr = base_addr + aligned_offset;
    const bool use_gpu_checksum =
        serialized_size <= 0xffff0000ull &&
        _ensure_checksum_pipeline();
    uint8_t gpu_payload_csum[16]{};
    bool gpu_csum_valid = false;

    {
        std::unique_lock csum_lock{_csum_mtx, std::defer_lock};
        if (use_gpu_checksum) {
            csum_lock.lock();
        }
        stream->execute_immediate([&](CommandBuffer &cb) {
            cb.resource_barrier->record(src_buffer, ResourceBarrier::Usage::kCopyAccelSrc);
            cb.resource_barrier->update_states(cb.cmdbuffer());

            VkBufferMemoryBarrier2 ser_prep{};
            ser_prep.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            ser_prep.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            ser_prep.srcAccessMask = 0u;
            ser_prep.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            ser_prep.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            ser_prep.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ser_prep.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ser_prep.buffer = vram_ser_slot->buffer;
            ser_prep.offset = aligned_offset;
            ser_prep.size = serialized_size;
            VkDependencyInfo dep1{};
            dep1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep1.bufferMemoryBarrierCount = 1u;
            dep1.pBufferMemoryBarriers = &ser_prep;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep1);

            VkCopyAccelerationStructureToMemoryInfoKHR copy_info{};
            copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
            copy_info.src = src_accel;
            copy_info.dst.deviceAddress = aligned_addr;
            copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
            vkCmdCopyAccelerationStructureToMemoryKHR(cb.cmdbuffer(), &copy_info);

            VkBufferMemoryBarrier2 ser_to_dma{};
            ser_to_dma.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            ser_to_dma.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            ser_to_dma.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            ser_to_dma.dstStageMask = use_gpu_checksum ?
                                          (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) :
                                          VK_PIPELINE_STAGE_2_COPY_BIT;
            ser_to_dma.dstAccessMask = use_gpu_checksum ?
                                           (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT) :
                                           VK_ACCESS_2_TRANSFER_READ_BIT;
            ser_to_dma.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ser_to_dma.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ser_to_dma.buffer = vram_ser_slot->buffer;
            ser_to_dma.offset = aligned_offset;
            ser_to_dma.size = serialized_size;
            VkDependencyInfo dep2{};
            dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep2.bufferMemoryBarrierCount = 1u;
            dep2.pBufferMemoryBarriers = &ser_to_dma;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep2);

            VkBufferCopy2 region{};
            region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
            region.srcOffset = aligned_offset;
            region.dstOffset = 0u;
            region.size = serialized_size;
            VkCopyBufferInfo2 copy_buf{};
            copy_buf.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
            copy_buf.srcBuffer = vram_ser_slot->buffer;
            copy_buf.dstBuffer = pinned_rb_slot->buffer;
            copy_buf.regionCount = 1u;
            copy_buf.pRegions = &region;
            detail::cmd_copy_buffer(cb.cmdbuffer(), _device, &copy_buf);

            if (use_gpu_checksum) {
                _record_checksum_dispatch(
                    cb.cmdbuffer(),
                    vram_ser_slot->buffer,
                    static_cast<uint32_t>(aligned_offset),
                    serialized_size,
                    0u);
            }

            // wait_after=true frees dispose_after_flush before execute_immediate returns;
            // keep rb_lease on the stack across encode_envelope below.
            cb.states()->dispose_after_flush(std::move(vram_lease));
        }, true);

        if (use_gpu_checksum) {
            uint32_t lanes[4]{};
            _read_checksum_lanes(lanes);
            accel_archive::finalize_payload_checksum(lanes, serialized_size, gpu_payload_csum);
            gpu_csum_valid = true;
        }
    }

    if (!pinned_rb_slot->host_coherent) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = pinned_rb_slot->memory;
        range.offset = 0u;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(_device->logic_device(), 1u, &range);
    }

    const luisa::span<const std::byte> payload{
        static_cast<const std::byte *>(pinned_rb_slot->mapped),
        static_cast<size_t>(serialized_size)};

    VkSerializedASHeader hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.serialized_size != serialized_size ||
        hdr.bottom_level_count != 0u ||
        hdr.deserialized_size == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Vulkan serialized BLAS header failed post-readback sanity checks.");
    }

    const uint64_t final_unpacked = hdr.deserialized_size;
    auto compat_blob = make_compat_blob(_device, hdr);
    auto encode_res = gpu_csum_valid ?
                          accel_archive::encode_envelope_with_payload_checksum(
                              AccelArchiveBackendKind::VULKAN,
                              desc,
                              is_compacted,
                              final_unpacked,
                              luisa::span<const std::byte>{
                                  reinterpret_cast<const std::byte *>(&compat_blob),
                                  sizeof(compat_blob)},
                              payload,
                              gpu_payload_csum,
                              out_archive) :
                          accel_archive::encode_envelope(
                              AccelArchiveBackendKind::VULKAN,
                              desc,
                              is_compacted,
                              final_unpacked,
                              luisa::span<const std::byte>{
                                  reinterpret_cast<const std::byte *>(&compat_blob),
                                  sizeof(compat_blob)},
                              payload,
                              out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = final_unpacked;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult VkAccelArchiveExt::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveImportOptions opts{};
    return import_mesh_archive_with_options(
        stream_handle, mesh_handle, expected_desc, geometry_binding, archive_bytes, opts);
}

AccelArchiveResult VkAccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &options) noexcept {
    if (!capabilities().supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "Vulkan KHR AS archive extension is not supported on this device.");
    }

    Stream *stream = nullptr;
    Blas *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, stream, blas);
    if (!handle_res) {
        return handle_res;
    }

    const auto &blas_opt = blas->option();
    if (blas_opt.allow_update != expected_desc.option.allow_update ||
        blas_opt.hint != expected_desc.option.hint ||
        blas_opt.allow_compaction != expected_desc.option.allow_compaction ||
        blas_opt.motion.keyframe_count != expected_desc.option.motion.keyframe_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target Vulkan Mesh AccelOption does not match archive AccelOption.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    uint8_t expected_payload_checksum[16]{};
    auto env_res = accel_archive::decode_envelope_header_and_compat(
        archive_bytes,
        AccelArchiveBackendKind::VULKAN,
        expected_desc,
        view,
        expected_payload_checksum);
    if (!env_res) {
        return env_res;
    }

    uint64_t deserialized_size = 0u;
    auto chk = check_vk_compat_and_header(
        _device, view.compat_blob, view.payload, view.meta.unpacked_as_size, deserialized_size);
    if (!chk) {
        return chk;
    }

    const size_t payload_size = view.payload.size();
    size_t pinned_slot_offset = 0u;
    auto pinned_src_slot = _find_caller_pinned_slot(view.payload.data(), payload_size, pinned_slot_offset);
    VkSlotLease pinned_lease{pinned_src_slot};
    VkSlotLease dma_lease{};
    VkSlotLease direct_lease{};

    std::shared_ptr<StagingSlot> direct_upload_slot;
    std::shared_ptr<StagingSlot> dma_stage_slot;
    uint64_t aligned_addr = 0u;
    uint64_t aligned_offset = 0u;
    bool used_dma_copy = false;
    bool dma_already_copied_and_verified = false;

    if (pinned_src_slot) {
        const bool use_gpu_verify =
            options.verify_payload_checksum &&
            payload_size <= 0xffff0000ull &&
            _ensure_checksum_pipeline();
        if (options.verify_payload_checksum && !use_gpu_verify) {
            if (auto p = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum); !p) {
                return p;
            }
        }
        dma_stage_slot = _acquire_pool_slot(
            _vram_pool,
            payload_size + 256u,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            /*map_host=*/false,
            /*need_device_address=*/true,
            /*mark_caller_held=*/false,
            /*mark_gpu_ref=*/true);
        dma_lease = VkSlotLease{dma_stage_slot};
        if (!dma_stage_slot) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to acquire Vulkan VRAM DMA target slot.");
        }
        const uint64_t base_addr = dma_stage_slot->device_address;
        aligned_offset = (256u - (base_addr & 255u)) & 255u;
        aligned_addr = base_addr + aligned_offset;
        used_dma_copy = true;

        if (use_gpu_verify) {
            if (!pinned_src_slot->host_coherent && pinned_src_slot->memory != VK_NULL_HANDLE) {
                VkMappedMemoryRange range{};
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                range.memory = pinned_src_slot->memory;
                range.offset = 0u;
                range.size = VK_WHOLE_SIZE;
                vkFlushMappedMemoryRanges(_device->logic_device(), 1u, &range);
            }
            bool csum_ok = false;
            {
                std::lock_guard csum_lock{_csum_mtx};
                stream->execute_immediate([&](CommandBuffer &cb) {
                    VkMemoryBarrier2 host_to_transfer{};
                    host_to_transfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    host_to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                    host_to_transfer.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
                    host_to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    host_to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    VkDependencyInfo dep_h2t{};
                    dep_h2t.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_h2t.memoryBarrierCount = 1u;
                    dep_h2t.pMemoryBarriers = &host_to_transfer;
                    detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep_h2t);

                    VkBufferCopy2 region{};
                    region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
                    region.srcOffset = pinned_slot_offset;
                    region.dstOffset = aligned_offset;
                    region.size = payload_size;
                    VkCopyBufferInfo2 copy_buf{};
                    copy_buf.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
                    copy_buf.srcBuffer = pinned_src_slot->buffer;
                    copy_buf.dstBuffer = dma_stage_slot->buffer;
                    copy_buf.regionCount = 1u;
                    copy_buf.pRegions = &region;
                    detail::cmd_copy_buffer(cb.cmdbuffer(), _device, &copy_buf);

                    VkBufferMemoryBarrier2 xfer_to_comp{};
                    xfer_to_comp.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    xfer_to_comp.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    xfer_to_comp.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    xfer_to_comp.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    xfer_to_comp.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    xfer_to_comp.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    xfer_to_comp.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    xfer_to_comp.buffer = dma_stage_slot->buffer;
                    xfer_to_comp.offset = aligned_offset;
                    xfer_to_comp.size = payload_size;
                    VkDependencyInfo dep_x2c{};
                    dep_x2c.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_x2c.bufferMemoryBarrierCount = 1u;
                    dep_x2c.pBufferMemoryBarriers = &xfer_to_comp;
                    detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep_x2c);

                    _record_checksum_dispatch(
                        cb.cmdbuffer(),
                        dma_stage_slot->buffer,
                        static_cast<uint32_t>(aligned_offset),
                        payload_size,
                        0u);
                }, /*wait_after=*/true);

                uint32_t lanes[4]{};
                _read_checksum_lanes(lanes);
                uint8_t computed[16]{};
                accel_archive::finalize_payload_checksum(lanes, payload_size, computed);
                csum_ok = std::memcmp(computed, expected_payload_checksum, 16u) == 0;
            }
            pinned_lease = VkSlotLease{};
            if (!csum_ok) {
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::CORRUPT_PAYLOAD,
                    "Vulkan GPU verification kernel detected corrupt native AS payload.");
            }
            dma_already_copied_and_verified = true;
        }
    } else {
        // Prefer ReBAR (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT) for direct CPU parallel write + GPU VRAM deserialize.
        direct_upload_slot = _acquire_pool_slot(
            _rebar_pool,
            payload_size + 256u,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            /*map_host=*/true,
            /*need_device_address=*/true,
            /*mark_caller_held=*/false,
            /*mark_gpu_ref=*/true);
        direct_lease = VkSlotLease{direct_upload_slot};

        if (direct_upload_slot && direct_upload_slot->mapped != nullptr) {
            const uint64_t base_addr = direct_upload_slot->device_address;
            aligned_offset = (256u - (base_addr & 255u)) & 255u;
            aligned_addr = base_addr + aligned_offset;
            if (!accel_archive::copy_payload_optimized(
                    static_cast<std::byte *>(direct_upload_slot->mapped) + aligned_offset,
                    view.payload,
                    options.verify_payload_checksum,
                    expected_payload_checksum)) {
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::CORRUPT_PAYLOAD,
                    "Native payload checksum mismatch.");
            }
            if (!direct_upload_slot->host_coherent) {
                VkMappedMemoryRange range{};
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                range.memory = direct_upload_slot->memory;
                range.offset = 0u;
                range.size = VK_WHOLE_SIZE;
                vkFlushMappedMemoryRanges(_device->logic_device(), 1u, &range);
            }
        } else {
            direct_lease = VkSlotLease{};
            direct_upload_slot.reset();
            // Non-ReBAR fallback: parallel copy+hash into pinned host -> SDMA copy into VRAM -> VRAM deserialize.
            pinned_src_slot = _acquire_pool_slot(
                _pinned_pool,
                payload_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                /*map_host=*/true,
                /*need_device_address=*/false,
                /*mark_caller_held=*/false,
                /*mark_gpu_ref=*/true);
            pinned_lease = VkSlotLease{pinned_src_slot};
            dma_stage_slot = _acquire_pool_slot(
                _vram_pool,
                payload_size + 256u,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                /*map_host=*/false,
                /*need_device_address=*/true,
                /*mark_caller_held=*/false,
                /*mark_gpu_ref=*/true);
            dma_lease = VkSlotLease{dma_stage_slot};
            if (!pinned_src_slot || pinned_src_slot->mapped == nullptr || !dma_stage_slot) {
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "Failed to allocate Vulkan fallback staging pool slots.");
            }
            if (!accel_archive::copy_payload_optimized(
                    pinned_src_slot->mapped,
                    view.payload,
                    options.verify_payload_checksum,
                    expected_payload_checksum)) {
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::CORRUPT_PAYLOAD,
                    "Native payload checksum mismatch.");
            }
            pinned_slot_offset = 0u;
            const uint64_t base_addr = dma_stage_slot->device_address;
            aligned_offset = (256u - (base_addr & 255u)) & 255u;
            aligned_addr = base_addr + aligned_offset;
            used_dma_copy = true;
        }
    }

    const VkDeviceSize dst_size = deserialized_size;
    const size_t dst_buf_bytes = std::max<size_t>((dst_size + 255u) & (~255ull), 256u);
    auto new_accel_buffer = vstd::make_unique<DefaultBuffer>(
        _device,
        dst_buf_bytes,
        false,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = new_accel_buffer->vk_buffer();
    ci.offset = 0u;
    ci.size = dst_size;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR new_accel = VK_NULL_HANDLE;
    VK_CHECK_RESULT(vkCreateAccelerationStructureKHR(
        _device->logic_device(), &ci, Device::alloc_callbacks(), &new_accel));

    stream->execute_immediate([&](CommandBuffer &cb) {
        cb.resource_barrier->record(
            new_accel_buffer.get(),
            ResourceBarrier::Usage::kCopyAccelDst);
        cb.resource_barrier->update_states(cb.cmdbuffer());

        if (used_dma_copy && dma_already_copied_and_verified) {
            VkBufferMemoryBarrier2 comp_to_as{};
            comp_to_as.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            comp_to_as.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            comp_to_as.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            comp_to_as.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            comp_to_as.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                       VK_ACCESS_2_TRANSFER_READ_BIT;
            comp_to_as.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            comp_to_as.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            comp_to_as.buffer = dma_stage_slot->buffer;
            comp_to_as.offset = aligned_offset;
            comp_to_as.size = payload_size;
            VkDependencyInfo dep_c2a{};
            dep_c2a.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep_c2a.bufferMemoryBarrierCount = 1u;
            dep_c2a.pBufferMemoryBarriers = &comp_to_as;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep_c2a);

            cb.states()->dispose_after_flush(std::move(dma_lease));
        } else if (used_dma_copy) {
            if (!pinned_src_slot->host_coherent && pinned_src_slot->memory != VK_NULL_HANDLE) {
                VkMappedMemoryRange range{};
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                range.memory = pinned_src_slot->memory;
                range.offset = 0u;
                range.size = VK_WHOLE_SIZE;
                vkFlushMappedMemoryRanges(_device->logic_device(), 1u, &range);
            }
            VkMemoryBarrier2 host_to_transfer{};
            host_to_transfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            host_to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            host_to_transfer.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
            host_to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            host_to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dep_h2t{};
            dep_h2t.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep_h2t.memoryBarrierCount = 1u;
            dep_h2t.pMemoryBarriers = &host_to_transfer;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep_h2t);

            VkBufferCopy2 region{};
            region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
            region.srcOffset = pinned_slot_offset;
            region.dstOffset = aligned_offset;
            region.size = payload_size;
            VkCopyBufferInfo2 copy_buf{};
            copy_buf.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
            copy_buf.srcBuffer = pinned_src_slot->buffer;
            copy_buf.dstBuffer = dma_stage_slot->buffer;
            copy_buf.regionCount = 1u;
            copy_buf.pRegions = &region;
            detail::cmd_copy_buffer(cb.cmdbuffer(), _device, &copy_buf);

            VkBufferMemoryBarrier2 xfer_to_as{};
            xfer_to_as.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            xfer_to_as.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            xfer_to_as.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            xfer_to_as.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            xfer_to_as.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                       VK_ACCESS_2_TRANSFER_READ_BIT;
            xfer_to_as.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            xfer_to_as.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            xfer_to_as.buffer = dma_stage_slot->buffer;
            xfer_to_as.offset = aligned_offset;
            xfer_to_as.size = payload_size;
            VkDependencyInfo dep_x2a{};
            dep_x2a.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep_x2a.bufferMemoryBarrierCount = 1u;
            dep_x2a.pBufferMemoryBarriers = &xfer_to_as;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep_x2a);

            cb.states()->dispose_after_flush(std::move(pinned_lease));
            cb.states()->dispose_after_flush(std::move(dma_lease));
        } else {
            VkMemoryBarrier2 host_to_copy{};
            host_to_copy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            host_to_copy.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            host_to_copy.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
            host_to_copy.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            host_to_copy.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                         VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1u;
            dep.pMemoryBarriers = &host_to_copy;
            detail::cmd_pipeline_barrier(cb.cmdbuffer(), _device, &dep);

            cb.states()->dispose_after_flush(std::move(direct_lease));
        }

        VkCopyMemoryToAccelerationStructureInfoKHR copy_info{};
        copy_info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
        copy_info.src.deviceAddress = aligned_addr;
        copy_info.dst = new_accel;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
        vkCmdCopyMemoryToAccelerationStructureKHR(cb.cmdbuffer(), &copy_info);

        blas->replace_from_archive(
            cb,
            new_accel,
            std::move(new_accel_buffer),
            dst_size,
            view.meta.is_compacted,
            static_cast<uint32_t>(expected_desc.triangle_count));
    }, /*wait_after=*/false);

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = dst_size;
    res.serialized_payload_size = payload_size;
    res.is_compacted = view.meta.is_compacted;
    res.message = "imported_on_stream";
    return res;
}

}// namespace lc::vk
