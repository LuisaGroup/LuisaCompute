#include <DXApi/dx_accel_archive_ext.h>

#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCDevice.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/Device.h>
#include <Resource/BottomAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <Resource/UploadBuffer.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/vstl/common.h>

#include <cstring>

namespace lc::dx {

namespace {

using namespace luisa::compute;

constexpr uint32_t kDxCompatMagic = 0x41415844u;// "DXAA"
constexpr uint32_t kDxCompatVersion = 1u;
constexpr luisa::string_view kDxLayoutTag = "dxr_serialized_raytracing_as_v1";

static_assert(sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) == 56u,
              "Unexpected D3D12 serialized AS header size");

void emit_staging_barrier(
    CommandBufferBuilder &builder,
    DefaultBuffer const *buffer,
    D3D12_BARRIER_SYNC sync_before,
    D3D12_BARRIER_SYNC sync_after,
    D3D12_BARRIER_ACCESS access_before,
    D3D12_BARRIER_ACCESS access_after,
    D3D12_RESOURCE_STATES legacy_before,
    D3D12_RESOURCE_STATES legacy_after) {
    auto *dev = buffer->GetDevice();
    auto *cmdlist = builder.get_cb()->cmd_list();
    if (dev->feature_check.enhanced_barriers_supported()) {
        ComPtr<ID3D12GraphicsCommandList7> cmd7;
        ThrowIfFailed(cmdlist->QueryInterface(IID_PPV_ARGS(&cmd7)));
        D3D12_BUFFER_BARRIER barrier{
            .SyncBefore = sync_before,
            .SyncAfter = sync_after,
            .AccessBefore = access_before,
            .AccessAfter = access_after,
            .pResource = buffer->GetResource(),
            .Offset = 0,
            .Size = UINT64_MAX};
        D3D12_BARRIER_GROUP group{
            .Type = D3D12_BARRIER_TYPE_BUFFER,
            .NumBarriers = 1,
            .pBufferBarriers = &barrier};
        cmd7->Barrier(1, &group);
    } else if (legacy_before != legacy_after) {
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            buffer->GetResource(),
            legacy_before,
            legacy_after);
        cmdlist->ResourceBarrier(1, &barrier);
    }
}

[[nodiscard]] luisa::string wide_to_ascii(WCHAR const *wstr) {
    if (!wstr) return {};
    luisa::string out;
    for (size_t i = 0; wstr[i] != L'\0'; ++i) {
        wchar_t wc = wstr[i];
        out.push_back((wc >= 0x20 && wc < 0x7f) ? static_cast<char>(wc) : '?');
    }
    return out;
}

[[nodiscard]] AccelArchiveResult check_stream_and_blas(
    LCDevice *device,
    uint64_t stream_handle,
    uint64_t mesh_handle,
    LCCmdBuffer *&out_cmd_buf,
    BottomAccel *&out_blas) noexcept {
    if (device == nullptr || device->use_fallback_rtx() || !device->hardware_raytracing_available()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 hardware DXR acceleration structure archiving is unavailable on this device.");
    }
    if (stream_handle == invalid_resource_handle || mesh_handle == invalid_resource_handle) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Invalid stream or mesh handle passed to DxAccelArchiveExt.");
    }
    if (device->owns_fallback_blas(mesh_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "DX12 software fallback RTX BLAS cannot be archived via DXR serialization.");
    }
    auto *base = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (base->tag() != CmdQueueTag::MainCmd) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "stream_handle is not a main DX12 compute/graphics stream.");
    }
    auto *cmd_buf = static_cast<LCCmdBuffer *>(base);
    if (cmd_buf->queue.Type() == D3D12_COMMAND_LIST_TYPE_COPY) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "DXR AS serialize/deserialize requires a compute or direct queue.");
    }
    auto *blas = reinterpret_cast<BottomAccel *>(mesh_handle);
    if (blas->Option().allow_update) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static non-updateable triangle BLAS only.");
    }
    out_cmd_buf = cmd_buf;
    out_blas = blas;
    return {};
}

[[nodiscard]] bool options_match(AccelOption const &a, AccelOption const &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           a.motion.keyframe_count == b.motion.keyframe_count;
}

}// namespace

DxAccelArchiveExt::DxAccelArchiveExt(LCDevice *device) noexcept
    : _device{device} {
    _supported = _device != nullptr &&
                 !_device->use_fallback_rtx() &&
                 _device->hardware_raytracing_available() &&
                 _device->native_device.adapter != nullptr &&
                 _device->native_device.device != nullptr;
    if (_supported) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(_device->native_device.adapter->GetDesc1(&desc))) {
            LARGE_INTEGER umd_version{};
            bool const has_umd = SUCCEEDED(_device->native_device.adapter->CheckInterfaceSupport(
                __uuidof(IDXGIDevice), &umd_version));
            auto adapter_name = wide_to_ascii(desc.Description);
            auto const rt_tier = static_cast<uint32_t>(_device->native_device.feature_check.raytracing_tier());
            auto const eb = _device->native_device.feature_check.enhanced_barriers_supported() ? 1u : 0u;
            _compat = CompatInfo{
                .vendor_id = desc.VendorId,
                .device_id = desc.DeviceId,
                .subsys_id = desc.SubSysId,
                .revision = desc.Revision,
                .magic = kDxCompatMagic,
                .version = kDxCompatVersion,
                .umd_version = has_umd ? static_cast<uint64_t>(umd_version.QuadPart) : 0ull,
                .raytracing_tier = rt_tier,
                .enhanced_barriers = eb,
                .deserialized_size = 0ull};
            _device_fingerprint = luisa::format("{}:0x{:04x}:0x{:04x}:rev{}",
                                                adapter_name, desc.VendorId, desc.DeviceId, desc.Revision);
            uint32_t const hi0 = static_cast<uint32_t>((_compat.umd_version >> 48u) & 0xffffu);
            uint32_t const hi1 = static_cast<uint32_t>((_compat.umd_version >> 32u) & 0xffffu);
            uint32_t const lo0 = static_cast<uint32_t>((_compat.umd_version >> 16u) & 0xffffu);
            uint32_t const lo1 = static_cast<uint32_t>(_compat.umd_version & 0xffffu);
            _driver_fingerprint = luisa::format("umd={}.{}.{}.{};rt_tier=0x{:x};eb={}",
                                                hi0, hi1, lo0, lo1, rt_tier, eb);
        } else {
            _supported = false;
        }
    } else if (_device != nullptr) {
        _device_fingerprint = "dx-fallback-or-no-dxr";
        _driver_fingerprint = "unsupported";
    }
}

AccelArchiveCapabilities DxAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = AccelArchiveBackendKind::DX;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    caps.device_fingerprint = _device_fingerprint;
    caps.driver_fingerprint = _driver_fingerprint;
    caps.supported = _supported;
    caps.static_triangle_blas = _supported;
    caps.supports_compacted_export = _supported;
    caps.requires_geometry_rebind = false;
    caps.import_is_async_on_stream = _supported;
    caps.layout_info = _supported ?
                           luisa::string{kDxLayoutTag} :
                           (_device != nullptr && _device->use_fallback_rtx() ?
                                "dx_fallback_rtx_disabled" :
                                "dxr_unavailable");
    return caps;
}

AccelArchiveResult DxAccelArchiveExt::_check_compat_and_header(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t meta_unpacked_size,
    uint64_t &out_deserialized_size) const noexcept {
    if (compat_blob.size() != sizeof(CompatInfo)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "DX12 AS archive compat blob has unexpected byte size.");
    }
    CompatInfo info{};
    std::memcpy(&info, compat_blob.data(), sizeof(CompatInfo));
    if (info.vendor_id != _compat.vendor_id ||
        info.device_id != _compat.device_id ||
        info.subsys_id != _compat.subsys_id ||
        info.revision != _compat.revision ||
        info.magic != kDxCompatMagic ||
        info.version != kDxCompatVersion ||
        info.raytracing_tier != _compat.raytracing_tier ||
        info.enhanced_barriers != _compat.enhanced_barriers) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            luisa::format(
                "DX12 AS archive targets vendor=0x{:04x}:0x{:04x}, current is 0x{:04x}:0x{:04x}.",
                info.vendor_id, info.device_id, _compat.vendor_id, _compat.device_id));
    }
    if (info.umd_version != _compat.umd_version) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "DX12 display driver UMD version does not match active device.");
    }
    if (payload.size() < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "DXR serialized BLAS payload is smaller than 56-byte header.");
    }
    D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.SerializedSizeInBytesIncludingHeader != payload.size() ||
        hdr.DeserializedSizeInBytes == 0u ||
        hdr.DeserializedSizeInBytes != meta_unpacked_size ||
        hdr.DeserializedSizeInBytes != info.deserialized_size ||
        hdr.NumBottomLevelAccelerationStructurePointersAfterHeader != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "DXR serialized BLAS header size or pointer count mismatch.");
    }
    auto const match_status = _device->native_device.device->CheckDriverMatchingIdentifier(
        D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE,
        &hdr.DriverMatchingIdentifier);
    if (match_status != D3D12_DRIVER_MATCHING_IDENTIFIER_COMPATIBLE_WITH_DEVICE) {
        auto const err_status =
            (match_status == D3D12_DRIVER_MATCHING_IDENTIFIER_INCOMPATIBLE_VERSION) ?
                AccelArchiveStatus::DRIVER_INCOMPATIBLE :
                AccelArchiveStatus::DEVICE_INCOMPATIBLE;
        return accel_archive::make_archive_error(
            err_status,
            luisa::format(
                "D3D12 CheckDriverMatchingIdentifier failed (status={}).",
                static_cast<uint32_t>(match_status)));
    }
    out_deserialized_size = hdr.DeserializedSizeInBytes;
    return {};
}

AccelArchiveResult DxAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!capabilities().supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::DX,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t deserialized_size = 0u;
    auto check_res = _check_compat_and_header(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        deserialized_size);
    if (!check_res) {
        return check_res;
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

AccelArchiveResult DxAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!capabilities().supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }

    LCCmdBuffer *cmd_buf = nullptr;
    BottomAccel *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, cmd_buf, blas);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(blas->Option(), desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "DX12 BLAS AccelOption does not match AccelArchiveMeshDesc::option.");
    }

    // Drain any in-flight build and CheckAccel compaction on the caller stream first.
    cmd_buf->Sync();

    if (!blas->IsBuilt() || !blas->HasAccelBuffer()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "DX12 BLAS has not been built yet.");
    }
    if (!blas->IsTriangleMesh()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static triangle BLAS only.");
    }
    if (blas->LastTriangleCount() != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "DX12 BLAS primitive count ({}) does not match expected triangle_count ({}).",
                blas->LastTriangleCount(), desc.triangle_count));
    }

    auto &native_dev = _device->native_device;
    bool const enhanced = native_dev.feature_check.enhanced_barriers_supported();
    auto const blas_va = blas->GetAccelBuffer()->GetAddress();
    auto const init_uav_state = enhanced ?
                                    D3D12_RESOURCE_STATE_COMMON :
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    // Pass 1: Query serialized byte size via EmitRaytracingAccelerationStructurePostbuildInfo.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC ser_desc{};
    {
        DefaultBuffer info_uav(&native_dev, 256u, nullptr, init_uav_state, false, "accel-archive-ser-info-uav");
        ReadbackBuffer info_rb(&native_dev, 256u, nullptr);
        {
            auto allocator = cmd_buf->queue.create_allocator(0);
            auto *cmd_buffer = allocator->get_buffer();
            {
                auto builder = cmd_buffer->build();
                emit_staging_barrier(
                    builder, &info_uav,
                    D3D12_BARRIER_SYNC_NONE,
                    D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC post_info{
                    .DestBuffer = info_uav.GetAddress(),
                    .InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION};
                builder.get_cb()->cmd_list()->EmitRaytracingAccelerationStructurePostbuildInfo(
                    &post_info, 1u, &blas_va);
                emit_staging_barrier(
                    builder, &info_uav,
                    D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                builder.copy_buffer(&info_uav, &info_rb, 0u, 0u, sizeof(ser_desc));
            }
            cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
            cmd_buf->queue.complete();
        }
        info_rb.CopyData(0u, {reinterpret_cast<uint8_t *>(&ser_desc), sizeof(ser_desc)});
    }

    if (ser_desc.SerializedSizeInBytes < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) ||
        ser_desc.NumBottomLevelAccelerationStructurePointers != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format(
                "Invalid DXR serialized size query (size={}, blas_ptrs={}).",
                ser_desc.SerializedSizeInBytes, ser_desc.NumBottomLevelAccelerationStructurePointers));
    }

    // Pass 2: Serialize BLAS into GPU UAV buffer and read back to host.
    luisa::vector<std::byte> payload(ser_desc.SerializedSizeInBytes);
    auto const aligned_ser_size = CalcAlign(ser_desc.SerializedSizeInBytes, 256u);
    {
        DefaultBuffer ser_uav(&native_dev, aligned_ser_size, nullptr, init_uav_state, false, "accel-archive-ser-uav");
        ReadbackBuffer ser_rb(&native_dev, ser_desc.SerializedSizeInBytes, nullptr);
        {
            auto allocator = cmd_buf->queue.create_allocator(0);
            auto *cmd_buffer = allocator->get_buffer();
            {
                auto builder = cmd_buffer->build();
                emit_staging_barrier(
                    builder, &ser_uav,
                    D3D12_BARRIER_SYNC_NONE,
                    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                builder.get_cb()->cmd_list()->CopyRaytracingAccelerationStructure(
                    ser_uav.GetAddress(),
                    blas_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE);
                emit_staging_barrier(
                    builder, &ser_uav,
                    D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                builder.copy_buffer(&ser_uav, &ser_rb, 0u, 0u, ser_desc.SerializedSizeInBytes);
            }
            cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
            cmd_buf->queue.complete();
        }
        ser_rb.CopyData(0u, {reinterpret_cast<uint8_t *>(payload.data()), payload.size()});
    }

    D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.SerializedSizeInBytesIncludingHeader != ser_desc.SerializedSizeInBytes ||
        hdr.DeserializedSizeInBytes == 0u ||
        hdr.NumBottomLevelAccelerationStructurePointersAfterHeader != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "DXR serialized BLAS header failed post-readback sanity checks.");
    }

    CompatInfo compat_to_write = _compat;
    compat_to_write.deserialized_size = hdr.DeserializedSizeInBytes;

    bool const is_compacted = blas->IsCompacted();
    auto encode_res = accel_archive::encode_envelope(
        AccelArchiveBackendKind::DX,
        desc,
        is_compacted,
        hdr.DeserializedSizeInBytes,
        luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(&compat_to_write),
            sizeof(compat_to_write)},
        payload,
        out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = hdr.DeserializedSizeInBytes;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult DxAccelArchiveExt::import_mesh_archive(
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

    LCCmdBuffer *cmd_buf = nullptr;
    BottomAccel *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, cmd_buf, blas);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(blas->Option(), expected_desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target DX12 Mesh AccelOption does not match archive AccelOption.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::DX,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    auto &native_dev = _device->native_device;
    auto allocator = cmd_buf->queue.create_allocator(0);

    auto upload_buf = vstd::create_unique(new UploadBuffer(
        &native_dev,
        CalcAlign(view.payload.size(), 256u),
        native_dev.default_allocator.get()));
    upload_buf->CopyData(
        0u,
        {reinterpret_cast<uint8_t const *>(view.payload.data()), view.payload.size()});

    const uint64_t dst_size = meta.unpacked_as_size;
    auto new_accel_buf = vstd::create_unique(new DefaultBuffer(
        &native_dev,
        CalcAlign(dst_size, 65536u),
        native_dev.default_allocator.get(),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        false,
        "blas-accel-buffer"));

    {
        auto *cmd_buffer = allocator->get_buffer();
        auto builder = cmd_buffer->build();
        builder.get_cb()->cmd_list()->CopyRaytracingAccelerationStructure(
            new_accel_buf->GetAddress(),
            upload_buf->GetAddress(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
        BottomAccel::EmitPostCopyBarrier(builder, new_accel_buf.get());
    }

    blas->ReplaceFromArchive(
        allocator.get(),
        std::move(new_accel_buf),
        dst_size,
        meta.is_compacted,
        expected_desc.triangle_count);

    allocator->dispose_after_complete(std::move(upload_buf));
    cmd_buf->queue.execute(std::move(allocator), {}, {}, false);

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = dst_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = meta.is_compacted;
    res.message = "imported_on_stream";
    return res;
}

}// namespace lc::dx
