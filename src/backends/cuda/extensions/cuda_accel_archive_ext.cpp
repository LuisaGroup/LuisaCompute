#include "cuda_accel_archive_ext.h"

#include <cstring>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#include "../cuda_device.h"
#include "../cuda_mesh.h"
#include "../cuda_primitive.h"
#include "../cuda_stream.h"

namespace luisa::compute::cuda {

namespace {

constexpr luisa::string_view kCudaOptixLayoutTag = "optix_relocatable_gas_v1";
constexpr uint64_t kMaxOptixGasArchiveBytes = 4ull << 30u;

[[nodiscard]] luisa::string format_cuda_uuid(const CUuuid &uuid) noexcept {
    luisa::string out;
    out.reserve(36u);
    uint32_t index = 0u;
    for (auto byte : uuid.bytes) {
        if (index == 4u || index == 6u || index == 8u || index == 10u) {
            out.append("-");
        }
        out.append(luisa::format("{:02x}", static_cast<uint32_t>(static_cast<unsigned char>(byte))));
        ++index;
    }
    return out;
}

[[nodiscard]] AccelArchiveResult make_cuda_error(
    CUresult ec,
    std::string_view context_msg) noexcept {
    const char *err_name = nullptr;
    const char *err_desc = nullptr;
    cuGetErrorName(ec, &err_name);
    cuGetErrorString(ec, &err_desc);
    return accel_archive::make_archive_error(
        AccelArchiveStatus::BACKEND_ERROR,
        luisa::format("{}: {} ({})",
                      context_msg,
                      err_name ? err_name : "CUDA_ERROR",
                      err_desc ? err_desc : "unknown"));
}

[[nodiscard]] AccelArchiveResult make_optix_error(
    AccelArchiveStatus status,
    OptixResult err,
    std::string_view context_msg) noexcept {
    const char *err_name = optix::api().optixGetErrorName(err);
    const char *err_desc = optix::api().optixGetErrorString(err);
    return accel_archive::make_archive_error(
        status,
        luisa::format("{}: {} ({})",
                      context_msg,
                      err_name ? err_name : "OPTIX_ERROR",
                      err_desc ? err_desc : "unknown"));
}

[[nodiscard]] bool options_match(const AccelOption &a, const AccelOption &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           !static_cast<bool>(a.motion) &&
           !static_cast<bool>(b.motion);
}

[[nodiscard]] AccelArchiveResult check_stream_and_mesh(
    CUDADevice *device,
    uint64_t stream_handle,
    uint64_t mesh_handle,
    CUDAStream *&out_stream,
    CUDAMesh *&out_mesh) noexcept {
    out_stream = nullptr;
    out_mesh = nullptr;
    if (device == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "CUDA device pointer is null.");
    }
    if (stream_handle == invalid_resource_handle || stream_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle must be a valid CUDAStream handle.");
    }
    if (mesh_handle == invalid_resource_handle || mesh_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh handle must be a valid CUDAMesh handle.");
    }
    auto *stream = reinterpret_cast<CUDAStream *>(stream_handle);
    if (stream->device() != device) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle does not belong to the active CUDADevice.");
    }
    auto *prim_base = reinterpret_cast<CUDAPrimitiveBase *>(mesh_handle);
    if (prim_base->tag() != CUDAPrimitiveBase::Tag::MESH) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Resource handle does not denote a CUDAMesh.");
    }
    out_stream = stream;
    out_mesh = static_cast<CUDAMesh *>(prim_base);
    return {};
}

}// namespace

CUDAAccelArchiveExt::CUDAAccelArchiveExt(CUDADevice *device) noexcept
    : _device{device} {
    if (_device == nullptr) {
        _device_fingerprint = "cuda_device_null";
        _driver_fingerprint = "unsupported";
        return;
    }

    const auto &h = _device->handle();
    _compute_capability = h.compute_capability();
    _driver_version = h.driver_version();
    _optix_abi_version = OPTIX_ABI_VERSION;

    const uint32_t cc_major = _compute_capability / 10u;
    const uint32_t cc_minor = _compute_capability % 10u;
    _device_fingerprint = luisa::format(
        "{}:sm_{}{}:uuid={}",
        h.name(),
        cc_major,
        cc_minor,
        format_cuda_uuid(h.handle_uuid()));

    if (_device->use_fallback_rtx() || !optix::available()) {
        _supported = false;
        _driver_fingerprint = luisa::format(
            "cuda_drv={};optix_abi=unavailable_or_fallback_rtx",
            _driver_version);
        return;
    }

    _driver_fingerprint = luisa::format(
        "cuda_drv={};optix_abi={}",
        _driver_version,
        _optix_abi_version);
    _supported = true;
}

AccelArchiveCapabilities CUDAAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = AccelArchiveBackendKind::CUDA;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    caps.device_fingerprint = _device_fingerprint;
    caps.driver_fingerprint = _driver_fingerprint;
    caps.supported = _supported;
    caps.static_triangle_blas = _supported;
    caps.supports_compacted_export = _supported;
    caps.requires_geometry_rebind = false;
    caps.import_is_async_on_stream = _supported;
    caps.layout_info = _supported ?
                           luisa::string{kCudaOptixLayoutTag} :
                           "optix_unavailable_or_fallback_rtx";
    return caps;
}

AccelArchiveResult CUDAAccelArchiveExt::_validate_compat_and_relocation(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t meta_unpacked_size,
    OptixRelocationInfo &out_reloc_info,
    uint64_t &out_active_bvh_bytes) const noexcept {
    if (payload.empty() || payload.size() > kMaxOptixGasArchiveBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "CUDA/OptiX GAS archive payload has invalid byte size.");
    }

    CudaOptixAccelArchiveCompatBlob decoded_compat{};
    auto compat_res = decode_and_check_cuda_optix_compat_blob(
        compat_blob,
        payload,
        _optix_abi_version,
        _compute_capability,
        _driver_version,
        decoded_compat);
    if (!compat_res) {
        return compat_res;
    }

    if (decoded_compat.active_bvh_bytes == 0u ||
        decoded_compat.active_bvh_bytes != meta_unpacked_size) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "CUDA/OptiX compatibility active_bvh_bytes does not match envelope unpacked_as_size.");
    }

    static_assert(sizeof(OptixRelocationInfo) == sizeof(decoded_compat.relocation_info));
    std::memcpy(&out_reloc_info, decoded_compat.relocation_info, sizeof(OptixRelocationInfo));

    int compatible = 0;
    const OptixResult compat_err = _device->with_handle([&] {
        return optix::api().optixCheckRelocationCompatibility(
            _device->handle().optix_context(),
            &out_reloc_info,
            &compatible);
    });
    if (compat_err != OPTIX_SUCCESS) {
        return make_optix_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            compat_err,
            "optixCheckRelocationCompatibility failed");
    }
    if (compatible == 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            "OptiX reported GAS relocation is incompatible with this CUDA/OptiX device context.");
    }

    out_active_bvh_bytes = decoded_compat.active_bvh_bytes;
    return {};
}

AccelArchiveResult CUDAAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "CUDA AccelArchiveExt requires hardware OptiX (unavailable when fallback RTX is enabled or OptiX cannot load).");
    }
    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::CUDA,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    OptixRelocationInfo reloc_info{};
    uint64_t active_bvh_bytes = 0u;
    auto check_res = _validate_compat_and_relocation(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        reloc_info,
        active_bvh_bytes);
    if (!check_res) {
        return check_res;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = active_bvh_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult CUDAAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "CUDA AccelArchiveExt requires hardware OptiX (unavailable when fallback RTX is enabled or OptiX cannot load).");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }

    CUDAStream *stream = nullptr;
    CUDAMesh *mesh = nullptr;
    auto handle_res = check_stream_and_mesh(_device, stream_handle, mesh_handle, stream, mesh);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(mesh->option(), desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "CUDAMesh AccelOption does not match AccelArchiveMeshDesc::option.");
    }

    // Drain queued build/compaction on the stream before reading back the GAS
    // and querying its OptiX relocation metadata.
    stream->synchronize();

    const auto snapshot = mesh->archive_snapshot();
    if (!snapshot.built ||
        snapshot.handle == 0ull ||
        snapshot.bvh_buffer_handle == 0ull ||
        snapshot.bvh_active_size == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "CUDAMesh has not been built yet.");
    }
    if (snapshot.last_primitive_count != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "CUDAMesh triangle count ({}) does not match expected ({}).",
                snapshot.last_primitive_count,
                desc.triangle_count));
    }
    if (snapshot.bvh_active_size > kMaxOptixGasArchiveBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "CUDAMesh GAS size exceeds maximum supported archive size.");
    }

    OptixRelocationInfo reloc_info{};
    luisa::vector<std::byte> payload(snapshot.bvh_active_size);
    const auto cuda_or_optix_res = _device->with_handle([&]() -> AccelArchiveResult {
        const OptixResult info_err = optix::api().optixAccelGetRelocationInfo(
            _device->handle().optix_context(),
            snapshot.handle,
            &reloc_info);
        if (info_err != OPTIX_SUCCESS) {
            return make_optix_error(
                AccelArchiveStatus::BACKEND_ERROR,
                info_err,
                "optixAccelGetRelocationInfo failed");
        }
        const CUresult copy_err = cuMemcpyDtoH(
            payload.data(),
            snapshot.bvh_buffer_handle,
            snapshot.bvh_active_size);
        if (copy_err != CUDA_SUCCESS) {
            return make_cuda_error(copy_err, "cuMemcpyDtoH failed reading CUDA/OptiX GAS");
        }
        return {};
    });
    if (!cuda_or_optix_res) {
        return cuda_or_optix_res;
    }

    CudaOptixAccelArchiveCompatBlob compat_blob{};
    compat_blob.magic = kCudaOptixAccelArchiveCompatMagic;
    compat_blob.version = kCudaOptixAccelArchiveCompatVersion;
    compat_blob.optix_abi_version = _optix_abi_version;
    compat_blob.cuda_compute_capability = _compute_capability;
    compat_blob.cuda_driver_version = _driver_version;
    compat_blob.accel_alignment = static_cast<uint32_t>(OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    compat_blob.active_bvh_bytes = static_cast<uint64_t>(snapshot.bvh_active_size);
    static_assert(sizeof(reloc_info.info) == sizeof(compat_blob.relocation_info));
    std::memcpy(compat_blob.relocation_info, reloc_info.info, sizeof(compat_blob.relocation_info));

    const auto encoded_compat = encode_cuda_optix_compat_blob(compat_blob);
    auto encode_res = accel_archive::encode_envelope(
        AccelArchiveBackendKind::CUDA,
        desc,
        snapshot.is_compacted,
        compat_blob.active_bvh_bytes,
        encoded_compat,
        payload,
        out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = compat_blob.active_bvh_bytes;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = snapshot.is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult CUDAAccelArchiveExt::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes) noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "CUDA AccelArchiveExt requires hardware OptiX (unavailable when fallback RTX is enabled or OptiX cannot load).");
    }

    CUDAStream *stream = nullptr;
    CUDAMesh *mesh = nullptr;
    auto handle_res = check_stream_and_mesh(_device, stream_handle, mesh_handle, stream, mesh);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(mesh->option(), expected_desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target CUDAMesh AccelOption does not match archive AccelOption.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::CUDA,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    OptixRelocationInfo reloc_info{};
    uint64_t active_bvh_bytes = 0u;
    auto check_res = _validate_compat_and_relocation(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        reloc_info,
        active_bvh_bytes);
    if (!check_res) {
        return check_res;
    }

    CUdeviceptr target_bvh = 0ull;
    OptixTraversableHandle relocated_handle = 0ull;
    const auto cuda_stream = stream->handle();
    const size_t gas_bytes = static_cast<size_t>(active_bvh_bytes);

    const auto reloc_res = _device->with_handle([&]() -> AccelArchiveResult {
        CUresult alloc_err = cuMemAllocAsync(&target_bvh, gas_bytes, cuda_stream);
        if (alloc_err != CUDA_SUCCESS) {
            target_bvh = 0ull;
            return make_cuda_error(alloc_err, "cuMemAllocAsync failed for relocated OptiX GAS");
        }
        auto free_on_failure = [&]() noexcept {
            if (target_bvh != 0ull) {
                cuMemFreeAsync(target_bvh, cuda_stream);
                target_bvh = 0ull;
            }
        };

        if ((target_bvh % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT) != 0ull) {
            free_on_failure();
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Allocated GAS buffer is not aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.");
        }

        const CUresult htod_err = cuMemcpyHtoDAsync(
            target_bvh,
            view.payload.data(),
            gas_bytes,
            cuda_stream);
        if (htod_err != CUDA_SUCCESS) {
            free_on_failure();
            return make_cuda_error(htod_err, "cuMemcpyHtoDAsync failed uploading OptiX GAS payload");
        }

        // Static triangle BLAS in Luisa always uses a single SBT record and
        // no opacity micromaps (v1). optixAccelRelocate fixes internal device
        // pointers in-place on the stream before TLAS build consumes the GAS.
        OptixRelocateInput reloc_input{};
        reloc_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        reloc_input.triangleArray.numSbtRecords = 1u;
        reloc_input.triangleArray.opacityMicromap.opacityMicromapArray = 0ull;

        const OptixResult optix_err = optix::api().optixAccelRelocate(
            _device->handle().optix_context(),
            cuda_stream,
            &reloc_info,
            &reloc_input,
            1u,
            target_bvh,
            gas_bytes,
            &relocated_handle);
        if (optix_err != OPTIX_SUCCESS || relocated_handle == 0ull) {
            free_on_failure();
            if (optix_err != OPTIX_SUCCESS) {
                return make_optix_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    optix_err,
                    "optixAccelRelocate failed");
            }
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "optixAccelRelocate produced a null OptixTraversableHandle.");
        }

        mesh->replace_from_archive(
            cuda_stream,
            target_bvh,
            gas_bytes,
            relocated_handle,
            expected_desc.triangle_count,
            view.meta.is_compacted);

        return {};
    });
    if (!reloc_res) {
        return reloc_res;
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = active_bvh_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "relocated_on_stream";
    return res;
}

}// namespace luisa::compute::cuda
