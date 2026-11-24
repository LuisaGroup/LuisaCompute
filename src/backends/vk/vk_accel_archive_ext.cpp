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
#include "readback_buffer.h"
#include "stream.h"
#include "upload_buffer.h"

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

}// namespace

VkAccelArchiveExt::VkAccelArchiveExt(Device *device) noexcept
    : _device{device} {}

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
    caps.layout_info = supported ?
                           "vk_khr_acceleration_structure_serialize_v1" :
                           (_device->use_fallback_rtx() ?
                                "vk_fallback_rtx_disabled" :
                                "vk_khr_raytracing_unavailable");
    return caps;
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

    if (view.compat_blob.size() != sizeof(VkAccelArchiveCompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Vulkan AS archive compat blob has unexpected byte size.");
    }
    VkAccelArchiveCompatBlob compat{};
    std::memcpy(&compat, view.compat_blob.data(), sizeof(compat));
    if (compat.magic != kVkCompatMagic || compat.struct_size != sizeof(VkAccelArchiveCompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Vulkan AS archive compat blob magic or size mismatch.");
    }

    const auto &props = _device->properties();
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

    if (view.payload.size() < sizeof(VkSerializedASHeader)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Vulkan serialized BLAS payload is smaller than 56-byte header.");
    }
    VkSerializedASHeader hdr{};
    std::memcpy(&hdr, view.payload.data(), sizeof(hdr));
    if (hdr.serialized_size != view.payload.size() ||
        hdr.bottom_level_count != 0u ||
        hdr.deserialized_size == 0u ||
        hdr.deserialized_size != view.meta.unpacked_as_size ||
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
        _device->logic_device(), &ver_info, &vk_compat);
    if (vk_compat != VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "vkGetDeviceAccelerationStructureCompatibilityKHR reported INCOMPATIBLE_KHR.");
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = hdr.deserialized_size;
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

    // Drain any in-flight build and trigger pending in-place compaction when safe.
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

    VkQueryPool size_query_pool = VK_NULL_HANDLE;
    auto cleanup_query_pool = vstd::scope_exit([&] {
        if (size_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(_device->logic_device(), size_query_pool, Device::alloc_callbacks());
            size_query_pool = VK_NULL_HANDLE;
        }
    });
    {
        VkQueryPoolCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
        qci.queryCount = 1u;
        VK_CHECK_RESULT(vkCreateQueryPool(
            _device->logic_device(), &qci, Device::alloc_callbacks(), &size_query_pool));
    }

    stream->execute_immediate([&](CommandBuffer &cb) {
        cb.resource_barrier->record(src_buffer, ResourceBarrier::Usage::kCopyAccelSrc);
        cb.resource_barrier->update_states(cb.cmdbuffer());
        vkCmdResetQueryPool(cb.cmdbuffer(), size_query_pool, 0u, 1u);
        vkCmdWriteAccelerationStructuresPropertiesKHR(
            cb.cmdbuffer(),
            1u,
            &src_accel,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
            size_query_pool,
            0u);
    }, true);

    uint64_t serialized_size = 0u;
    VK_CHECK_RESULT(vkGetQueryPoolResults(
        _device->logic_device(),
        size_query_pool,
        0u,
        1u,
        sizeof(uint64_t),
        &serialized_size,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    if (serialized_size < kVkSerializedHeaderBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format(
                "Vulkan reported invalid serialized BLAS size {} (< 56 bytes).",
                serialized_size));
    }

    auto serialize_buffer = vstd::make_unique<DefaultBuffer>(
        _device,
        static_cast<size_t>(serialized_size + 256u));
    auto readback_buffer = vstd::make_unique<ReadbackBuffer>(
        _device,
        static_cast<size_t>(serialized_size));
    auto cleanup_io_buffers = vstd::scope_exit([&] {
        if (serialize_buffer) {
            stream->remove_resource_state(serialize_buffer.get());
            serialize_buffer.reset();
        }
        if (readback_buffer) {
            stream->remove_resource_state(readback_buffer.get());
            readback_buffer.reset();
        }
    });

    const uint64_t base_addr = serialize_buffer->get_device_address();
    const uint64_t aligned_offset = (256u - (base_addr & 255u)) & 255u;
    const uint64_t aligned_addr = base_addr + aligned_offset;

    stream->execute_immediate([&](CommandBuffer &cb) {
        cb.resource_barrier->record(src_buffer, ResourceBarrier::Usage::kCopyAccelSrc);
        cb.resource_barrier->record(
            BufferView{serialize_buffer.get(), aligned_offset, serialized_size},
            ResourceBarrier::Usage::kComputeUAV);
        cb.resource_barrier->update_states(cb.cmdbuffer());

        VkCopyAccelerationStructureToMemoryInfoKHR copy_info{};
        copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
        copy_info.src = src_accel;
        copy_info.dst.deviceAddress = aligned_addr;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
        vkCmdCopyAccelerationStructureToMemoryKHR(cb.cmdbuffer(), &copy_info);

        cb.resource_barrier->record(
            BufferView{serialize_buffer.get(), aligned_offset, serialized_size},
            ResourceBarrier::Usage::kCopySource);
        cb.resource_barrier->record(
            BufferView{readback_buffer.get(), 0u, serialized_size},
            ResourceBarrier::Usage::kCopyDest);
        cb.resource_barrier->update_states(cb.cmdbuffer());

        VkBufferCopy2 region{};
        region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
        region.srcOffset = aligned_offset;
        region.dstOffset = 0u;
        region.size = serialized_size;
        VkCopyBufferInfo2 copy_buf{};
        copy_buf.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
        copy_buf.srcBuffer = serialize_buffer->vk_buffer();
        copy_buf.dstBuffer = readback_buffer->vk_buffer();
        copy_buf.regionCount = 1u;
        copy_buf.pRegions = &region;
        detail::cmd_copy_buffer(cb.cmdbuffer(), _device, &copy_buf);
    }, true);

    luisa::vector<std::byte> payload(static_cast<size_t>(serialized_size));
    readback_buffer->copy_to(payload.data(), 0u, static_cast<size_t>(serialized_size));

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
    auto encode_res = accel_archive::encode_envelope(
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
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveMetadata meta{};
    auto val_res = validate_mesh_archive(archive_bytes, expected_desc, &meta);
    if (!val_res) {
        return val_res;
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
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::VULKAN,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    const size_t payload_size = view.payload.size();
    auto upload_buffer = vstd::make_unique<UploadBuffer>(
        _device,
        payload_size + 256u);
    const uint64_t base_addr = upload_buffer->get_device_address();
    const uint64_t aligned_offset = (256u - (base_addr & 255u)) & 255u;
    const uint64_t aligned_addr = base_addr + aligned_offset;
    upload_buffer->copy_from(view.payload.data(), static_cast<size_t>(aligned_offset), payload_size);
    upload_buffer->flush_host();

    const VkDeviceSize dst_size = meta.unpacked_as_size;
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

        VkCopyMemoryToAccelerationStructureInfoKHR copy_info{};
        copy_info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
        copy_info.src.deviceAddress = aligned_addr;
        copy_info.dst = new_accel;
        copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
        vkCmdCopyMemoryToAccelerationStructureKHR(cb.cmdbuffer(), &copy_info);

        cb.states()->dispose_after_flush(std::move(upload_buffer));
        blas->replace_from_archive(
            cb,
            new_accel,
            std::move(new_accel_buffer),
            dst_size,
            meta.is_compacted,
            static_cast<uint32_t>(expected_desc.triangle_count));
    }, /*wait_after=*/false);

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = dst_size;
    res.serialized_payload_size = payload_size;
    res.is_compacted = meta.is_compacted;
    res.message = "imported_on_stream";
    return res;
}

}// namespace lc::vk
