#include "cuda_accel_archive_ext.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/vstl/functional.h>

#include "../cuda_device.h"
#include "../cuda_mesh.h"
#include "../cuda_primitive.h"
#include "../cuda_stream.h"

namespace luisa::compute::cuda {

namespace {

constexpr luisa::string_view kCudaOptixLayoutTag = "optix_relocatable_gas_v1";
constexpr uint64_t kMaxOptixGasArchiveBytes = 4ull << 30u;
constexpr size_t kCudaMaxPoolSlots = 16u;
constexpr size_t kCudaMaxPoolBytes = 2ull << 30u;

[[nodiscard]] size_t round_up_cuda_pool_size(size_t needed) noexcept {
    const size_t min_sz = std::max<size_t>(needed, 65536u);
    if (min_sz <= (4ull << 20u)) {
        return ((min_sz + 65535u) / 65536u) * 65536u;
    }
    if (min_sz <= (64ull << 20u)) {
        constexpr size_t k1M = 1ull << 20u;
        return ((min_sz + k1M - 1u) / k1M) * k1M;
    }
    constexpr size_t k4M = 4ull << 20u;
    return ((min_sz + k4M - 1u) / k4M) * k4M;
}

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

struct CUDAAccelArchiveExt::Impl {
    struct PinnedSlot {
        std::byte *ptr{nullptr};
        size_t capacity{0u};
        CUevent event{nullptr};
        CUstream last_stream{nullptr};
        bool in_flight{false};
        bool caller_owned{false};
        uint64_t lru_tick{0u};
        uint64_t token{0u};
    };

    CUDADevice *device{nullptr};
    std::mutex mtx;
    luisa::vector<PinnedSlot> slots;
    size_t total_bytes{0u};
    uint64_t tick{0u};
    uint64_t next_token{0u};

    explicit Impl(CUDADevice *dev) noexcept : device{dev} {}

    ~Impl() noexcept {
        destroy_all();
    }

    void destroy_all() noexcept {
        std::lock_guard lck{mtx};
        if (device == nullptr || slots.empty()) {
            slots.clear();
            total_bytes = 0u;
            return;
        }
        device->with_handle([&] {
            for (auto &s : slots) {
                if (s.in_flight && s.event != nullptr) {
                    (void)cuEventSynchronize(s.event);
                    s.in_flight = false;
                }
                if (s.event != nullptr) {
                    (void)cuEventDestroy(s.event);
                    s.event = nullptr;
                }
                if (s.ptr != nullptr) {
                    (void)cuMemFreeHost(s.ptr);
                    s.ptr = nullptr;
                }
            }
        });
        slots.clear();
        total_bytes = 0u;
    }

    void poll_events_locked() noexcept {
        for (auto &s : slots) {
            if (s.in_flight && s.event != nullptr) {
                if (cuEventQuery(s.event) == CUDA_SUCCESS) {
                    s.in_flight = false;
                }
            }
        }
    }

    [[nodiscard]] bool acquire_slot(
        size_t min_bytes,
        bool for_caller,
        std::byte *&out_ptr,
        size_t &out_cap,
        uint64_t &out_tok) noexcept {
        std::lock_guard lck{mtx};
        if (device == nullptr) return false;
        poll_events_locked();

        size_t best_idx = slots.size();
        for (size_t i = 0u; i < slots.size(); ++i) {
            const auto &s = slots[i];
            if (!s.caller_owned && !s.in_flight && s.capacity >= min_bytes) {
                if (best_idx == slots.size() || s.capacity < slots[best_idx].capacity) {
                    best_idx = i;
                }
            }
        }
        if (best_idx < slots.size()) {
            auto &s = slots[best_idx];
            s.lru_tick = ++tick;
            if (for_caller) s.caller_owned = true;
            else s.in_flight = true;
            out_ptr = s.ptr;
            out_cap = s.capacity;
            out_tok = s.token;
            return true;
        }

        const size_t alloc_bytes = round_up_cuda_pool_size(min_bytes);
        bool ok = false;
        device->with_handle([&] {
            auto evict_oldest_idle = [&]() noexcept -> bool {
                size_t victim = slots.size();
                for (size_t i = 0u; i < slots.size(); ++i) {
                    if (!slots[i].caller_owned && !slots[i].in_flight) {
                        if (victim == slots.size() || slots[i].lru_tick < slots[victim].lru_tick) {
                            victim = i;
                        }
                    }
                }
                if (victim == slots.size()) return false;
                auto &v = slots[victim];
                if (v.event != nullptr) (void)cuEventDestroy(v.event);
                if (v.ptr != nullptr) (void)cuMemFreeHost(v.ptr);
                total_bytes -= v.capacity;
                if (victim + 1u < slots.size()) {
                    slots[victim] = std::move(slots.back());
                }
                slots.pop_back();
                return true;
            };

            while (slots.size() >= kCudaMaxPoolSlots ||
                   (total_bytes + alloc_bytes > kCudaMaxPoolBytes && !slots.empty())) {
                if (!evict_oldest_idle()) break;
            }

            void *raw_ptr = nullptr;
            if (cuMemHostAlloc(&raw_ptr, alloc_bytes, CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS ||
                raw_ptr == nullptr) {
                return;
            }
            CUevent evt = nullptr;
            if (cuEventCreate(&evt, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS || evt == nullptr) {
                (void)cuMemFreeHost(raw_ptr);
                return;
            }

            PinnedSlot s{};
            s.ptr = static_cast<std::byte *>(raw_ptr);
            s.capacity = alloc_bytes;
            s.event = evt;
            s.lru_tick = ++tick;
            s.token = (++next_token) | (1ull << 56u);
            s.caller_owned = for_caller;
            s.in_flight = !for_caller;
            total_bytes += alloc_bytes;
            out_ptr = s.ptr;
            out_cap = s.capacity;
            out_tok = s.token;
            slots.push_back(s);
            ok = true;
        });
        return ok;
    }

    [[nodiscard]] bool match_caller_buffer(
        const std::byte *ptr,
        size_t bytes,
        CUstream current_stream,
        uint64_t &out_tok) noexcept {
        if (ptr == nullptr) return false;
        std::lock_guard lck{mtx};
        poll_events_locked();
        for (auto &s : slots) {
            if (!s.caller_owned || s.ptr == nullptr) continue;
            if (ptr >= s.ptr && ptr <= s.ptr + s.capacity) {
                const size_t off = static_cast<size_t>(ptr - s.ptr);
                if (off <= s.capacity && bytes <= s.capacity - off) {
                    if (s.in_flight && s.event != nullptr && s.last_stream != current_stream) {
                        device->with_handle([&] {
                            (void)cuEventSynchronize(s.event);
                        });
                        s.in_flight = false;
                    }
                    s.in_flight = true;
                    s.last_stream = current_stream;
                    s.lru_tick = ++tick;
                    out_tok = s.token;
                    return true;
                }
            }
        }
        return false;
    }

    void record_stream_event_locked_ctx(uint64_t tok, CUstream stream) noexcept {
        std::lock_guard lck{mtx};
        for (auto &s : slots) {
            if (s.token == tok) {
                s.last_stream = stream;
                s.in_flight = true;
                if (s.event != nullptr) {
                    (void)cuEventRecord(s.event, stream);
                }
                return;
            }
        }
    }

    void release_idle(uint64_t tok) noexcept {
        std::lock_guard lck{mtx};
        for (auto &s : slots) {
            if (s.token == tok) {
                s.in_flight = false;
                s.lru_tick = ++tick;
                return;
            }
        }
    }

    void release_caller(uint64_t tok, const std::byte *mapped) noexcept {
        std::lock_guard lck{mtx};
        poll_events_locked();
        for (auto &s : slots) {
            if ((tok != 0u && s.token == tok) || (mapped != nullptr && s.ptr == mapped)) {
                s.caller_owned = false;
                s.lru_tick = ++tick;
                return;
            }
        }
    }
};

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
    _impl = std::make_unique<Impl>(_device);
    _supported = true;
}

CUDAAccelArchiveExt::~CUDAAccelArchiveExt() noexcept {
    _impl.reset();
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
    caps.supports_pinned_staging = _supported;
    caps.supports_direct_file_import = _supported;
    caps.layout_info = _supported ?
                           luisa::string{kCudaOptixLayoutTag} :
                           "optix_unavailable_or_fallback_rtx";
    return caps;
}

AccelArchiveStagingBuffer CUDAAccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    if (!_supported || !_impl || min_bytes == 0u) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    std::byte *ptr = nullptr;
    size_t cap = 0u;
    uint64_t tok = 0u;
    if (!_impl->acquire_slot(min_bytes, true, ptr, cap, tok)) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    AccelArchiveStagingBuffer out{};
    out.handle = tok;
    out.mapped_data = ptr;
    out.capacity = cap;
    out.alignment = 4096u;
    out.is_device_local = false;
    out.is_pinned_host = true;
    return out;
}

void CUDAAccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (_impl && buffer.handle != invalid_resource_handle && (buffer.handle >> 56u) != 0u) {
        _impl->release_caller(buffer.handle, buffer.mapped_data);
        buffer = {};
        return;
    }
    AccelArchiveExt::release_staging_buffer(std::move(buffer));
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
    std::byte *rb_ptr = nullptr;
    size_t rb_cap = 0u;
    uint64_t rb_tok = 0u;
    luisa::vector<std::byte> pageable_fallback;
    luisa::span<const std::byte> payload;
    const bool use_pooled = _impl && _impl->acquire_slot(snapshot.bvh_active_size, false, rb_ptr, rb_cap, rb_tok);
    auto release_rb = vstd::scope_exit([&]() noexcept {
        if (use_pooled && _impl) {
            _impl->release_idle(rb_tok);
        }
    });
    std::byte *dst_host = nullptr;
    if (use_pooled) {
        dst_host = rb_ptr;
    } else {
        pageable_fallback.resize(snapshot.bvh_active_size);
        dst_host = pageable_fallback.data();
    }

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
            dst_host,
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
    payload = luisa::span<const std::byte>{dst_host, snapshot.bvh_active_size};

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
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveImportOptions default_opts{};
    return import_mesh_archive_with_options(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        archive_bytes,
        default_opts);
}

AccelArchiveResult CUDAAccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &options) noexcept {
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
    uint8_t expected_payload_checksum[16]{};
    auto env_res = accel_archive::decode_envelope_header_and_compat(
        archive_bytes,
        AccelArchiveBackendKind::CUDA,
        expected_desc,
        view,
        expected_payload_checksum);
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

    const auto cuda_stream = stream->handle();
    const size_t gas_bytes = static_cast<size_t>(active_bvh_bytes);

    uint64_t pinned_tok = 0u;
    const bool is_pinned_zero_copy =
        _impl && _impl->match_caller_buffer(view.payload.data(), view.payload.size(), cuda_stream, pinned_tok);

    const std::byte *htod_src = view.payload.data();
    std::byte *pooled_ptr = nullptr;
    size_t pooled_cap = 0u;

    if (is_pinned_zero_copy) {
        if (options.verify_payload_checksum) {
            if (auto p = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum); !p) {
                _impl->release_idle(pinned_tok);
                return p;
            }
        }
    } else if (_impl && _impl->acquire_slot(gas_bytes, false, pooled_ptr, pooled_cap, pinned_tok)) {
        if (!accel_archive::copy_payload_optimized(
                pooled_ptr,
                view.payload,
                options.verify_payload_checksum,
                expected_payload_checksum)) {
            _impl->release_idle(pinned_tok);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                "Native payload checksum mismatch.");
        }
        htod_src = pooled_ptr;
    } else if (options.verify_payload_checksum) {
        if (auto p = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum); !p) {
            return p;
        }
    }

    CUdeviceptr target_bvh = 0ull;
    OptixTraversableHandle relocated_handle = 0ull;

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
            htod_src,
            gas_bytes,
            cuda_stream);
        if (htod_err != CUDA_SUCCESS) {
            free_on_failure();
            return make_cuda_error(htod_err, "cuMemcpyHtoDAsync failed uploading OptiX GAS payload");
        }

        if (pinned_tok != 0u && _impl) {
            _impl->record_stream_event_locked_ctx(pinned_tok, cuda_stream);
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
        if (pinned_tok != 0u && _impl) {
            _impl->release_idle(pinned_tok);
        }
        return reloc_res;
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = active_bvh_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = is_pinned_zero_copy ? "relocated_pinned_on_stream" : "relocated_on_stream";
    return res;
}

}// namespace luisa::compute::cuda
