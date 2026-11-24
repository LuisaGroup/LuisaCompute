#include "hip_accel_archive_ext.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/vstl/functional.h>

#include "hip_accel_archive_builtin_embedded.h"
#include "hip_check.h"
#include "hip_device.h"
#include "hip_geometry.h"
#include "hip_mesh.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

namespace {

constexpr uint32_t kHipCompatMagic = 0x50484341u;// "ACHP"
constexpr luisa::string_view kHipLayoutTag = "hiprt_geom_header_v1";
constexpr uint64_t kMaxGeomArchiveBytes = 4ull << 30u;
constexpr size_t kHipMaxPoolSlots = 16u;
constexpr size_t kHipMaxPoolBytes = 2ull << 30u;

// Binary layout of hiprt::GeomHeader as written by hiprt::Context::saveGeometry
// (src/ext/HIPRT/hiprt/impl/Header.h & Context.cpp). m_boxNodes and m_primNodes
// store byte offsets relative to the beginning of the saved buffer.
struct HiprtGeomHeaderLayout {
    uint64_t box_nodes_offset;
    uint64_t prim_nodes_offset;
    uint64_t total_size;
    uint32_t reference_count;
    uint32_t box_node_count;
    uint32_t prim_node_count;
    uint32_t geom_type;
    uint32_t rtip;
    uint32_t padding;
};

static_assert(sizeof(HiprtGeomHeaderLayout) == 48u);

[[nodiscard]] size_t round_up_hip_pool_size(size_t needed) noexcept {
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

[[nodiscard]] uint32_t parse_gfx_arch_number(std::string_view arch) noexcept {
    // Handles strings like "gfx1201" or "gfx1201:sramecc-:xnack-"
    if (arch.size() < 4u || arch.substr(0u, 3u) != "gfx") {
        return 0u;
    }
    uint32_t val = 0u;
    for (size_t i = 3u; i < arch.size(); ++i) {
        auto ch = static_cast<unsigned char>(arch[i]);
        if (!std::isdigit(ch)) {
            break;
        }
        val = val * 10u + static_cast<uint32_t>(ch - '0');
    }
    return val;
}

[[nodiscard]] uint32_t compute_expected_hiprt_rtip(
    std::string_view device_name,
    std::string_view gcn_arch,
    int driver_version) noexcept {
    if (device_name.find("NVIDIA") != std::string_view::npos) {
        return 0u;
    }
    const uint32_t arch_num = parse_gfx_arch_number(gcn_arch);
    uint32_t rtip = 0u;
    if (arch_num >= 1030u) rtip = 11u;
    if (arch_num >= 1100u) rtip = 20u;
    if (arch_num >= 1200u) {
#ifdef _WIN32
        const bool driver_rtip31 = driver_version >= 60400000;
#else
        const bool driver_rtip31 = driver_version >= 70000000;
#endif
        rtip = driver_rtip31 ? 31u : 20u;
    }
    return rtip;
}

[[nodiscard]] bool options_match(AccelOption const &a, AccelOption const &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           a.motion.keyframe_count == b.motion.keyframe_count;
}

class ScopedHipContext {
    hipCtx_t _ctx{nullptr};

public:
    explicit ScopedHipContext(hipCtx_t ctx) noexcept : _ctx{ctx} {
        if (_ctx != nullptr) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            LUISA_CHECK_HIP(hipCtxPushCurrent(_ctx));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        }
    }
    ~ScopedHipContext() noexcept {
        if (_ctx != nullptr) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            hipCtx_t popped{nullptr};
            LUISA_CHECK_HIP(hipCtxPopCurrent(&popped));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        }
    }
    ScopedHipContext(const ScopedHipContext &) = delete;
    ScopedHipContext &operator=(const ScopedHipContext &) = delete;
};

[[nodiscard]] AccelArchiveResult check_stream_and_mesh(
    HIPDevice *device,
    uint64_t stream_handle,
    uint64_t mesh_handle,
    HIPStream *&out_stream,
    HIPMesh *&out_mesh) noexcept {
    if (device == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "HIP device is null.");
    }
    if (stream_handle == invalid_resource_handle || mesh_handle == invalid_resource_handle) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Invalid stream or mesh handle passed to HIPAccelArchiveExt.");
    }
    auto *stream = reinterpret_cast<HIPStream *>(stream_handle);
    if (stream->device() != device) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream belongs to a different HIP device.");
    }
    auto *prim = reinterpret_cast<HIPPrimitive *>(mesh_handle);
    if (!prim->is_static_triangle_mesh()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static triangle HIPMesh only.");
    }
    auto *mesh = static_cast<HIPMesh *>(prim);
    if (mesh->option().allow_update) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 does not support allow_update meshes.");
    }
    out_stream = stream;
    out_mesh = mesh;
    return {};
}

}// namespace

struct HIPAccelArchiveExt::Impl {
    struct PinnedSlot {
        std::byte *ptr{nullptr};
        size_t capacity{0u};
        hipEvent_t event{nullptr};
        hipStream_t last_stream{nullptr};
        bool in_flight{false};
        bool caller_owned{false};
        uint64_t lru_tick{0u};
        uint64_t token{0u};
    };

    HIPDevice *device{nullptr};
    std::mutex mtx;
    luisa::vector<PinnedSlot> data_slots;
    luisa::vector<PinnedSlot> hdr_slots;
    size_t total_data_bytes{0u};
    uint64_t tick{0u};
    uint64_t next_token{0u};

    std::mutex csum_init_mtx;
    std::mutex csum_mtx;
    bool csum_attempted{false};
    bool csum_ready{false};
    hipModule_t csum_module{nullptr};
    hipFunction_t csum_fn{nullptr};
    hipDeviceptr_t d_hash{nullptr};
    unsigned int *h_hash{nullptr};

    explicit Impl(HIPDevice *dev) noexcept : device{dev} {}

    ~Impl() noexcept {
        destroy_all();
    }

    void destroy_all() noexcept {
        {
            std::lock_guard cinit{csum_init_mtx};
            std::lock_guard clock{csum_mtx};
            if (device != nullptr && device->_hip_context != nullptr &&
                (csum_module != nullptr || d_hash != nullptr || h_hash != nullptr)) {
                ScopedHipContext ctx_guard{device->_hip_context};
                if (csum_module != nullptr) {
                    (void)hipModuleUnload(csum_module);
                    csum_module = nullptr;
                    csum_fn = nullptr;
                }
                if (d_hash != nullptr) {
                    (void)hipFree(d_hash);
                    d_hash = nullptr;
                }
                if (h_hash != nullptr) {
                    (void)hipHostFree(h_hash);
                    h_hash = nullptr;
                }
            }
            csum_ready = false;
        }
        std::lock_guard lck{mtx};
        if (device == nullptr || device->_hip_context == nullptr) {
            data_slots.clear();
            hdr_slots.clear();
            total_data_bytes = 0u;
            return;
        }
        ScopedHipContext ctx_guard{device->_hip_context};
        auto free_vec = [](luisa::vector<PinnedSlot> &vec) noexcept {
            for (auto &s : vec) {
                if (s.in_flight && s.event != nullptr) {
                    (void)hipEventSynchronize(s.event);
                    s.in_flight = false;
                }
                if (s.event != nullptr) {
                    (void)hipEventDestroy(s.event);
                    s.event = nullptr;
                }
                if (s.ptr != nullptr) {
                    (void)hipHostFree(s.ptr);
                    s.ptr = nullptr;
                }
            }
            vec.clear();
        };
        free_vec(data_slots);
        free_vec(hdr_slots);
        total_data_bytes = 0u;
    }

    [[nodiscard]] bool ensure_checksum_kernel() noexcept {
        std::lock_guard cinit{csum_init_mtx};
        if (csum_attempted) {
            return csum_ready;
        }
        csum_attempted = true;
        if (device == nullptr || device->_hip_context == nullptr ||
            luisa_compute_hip_accel_archive_builtin_size == 0u) {
            return false;
        }
        ScopedHipContext ctx_guard{device->_hip_context};
        if (hipModuleLoadData(&csum_module, luisa_compute_hip_accel_archive_builtin) != hipSuccess ||
            csum_module == nullptr) {
            csum_module = nullptr;
            return false;
        }
        if (hipModuleGetFunction(&csum_fn, csum_module, "accel_archive_checksum") != hipSuccess ||
            csum_fn == nullptr) {
            (void)hipModuleUnload(csum_module);
            csum_module = nullptr;
            csum_fn = nullptr;
            return false;
        }
        if (hipMalloc(&d_hash, 16u) != hipSuccess || d_hash == nullptr) {
            (void)hipModuleUnload(csum_module);
            csum_module = nullptr;
            csum_fn = nullptr;
            d_hash = nullptr;
            return false;
        }
        void *h_raw = nullptr;
        if (hipHostMalloc(&h_raw, 16u, hipHostMallocDefault) != hipSuccess || h_raw == nullptr) {
            (void)hipFree(d_hash);
            d_hash = nullptr;
            (void)hipModuleUnload(csum_module);
            csum_module = nullptr;
            csum_fn = nullptr;
            return false;
        }
        h_hash = static_cast<unsigned int *>(h_raw);
        csum_ready = true;
        return true;
    }

    // Caller must hold csum_mtx and have device->_hip_context pushed.
    [[nodiscard]] bool launch_checksum_locked(
        hipStream_t stream,
        hipDeviceptr_t dev_words,
        uint64_t byte_size,
        uint32_t start_word) noexcept {
        if (!csum_ready || byte_size > 0xffff0000ull) {
            return false;
        }
        if (hipMemsetAsync(d_hash, 0, 16u, stream) != hipSuccess) {
            return false;
        }
        const uint64_t total_words_u64 = (byte_size + 3ull) / 4ull;
        if (total_words_u64 > start_word) {
            unsigned int total_words = static_cast<unsigned int>(total_words_u64);
            unsigned int words_todo = total_words - start_word;
            unsigned int groups = std::min<unsigned int>(2048u, (words_todo + 255u) / 256u);
            unsigned int rem = static_cast<unsigned int>(byte_size & 3ull);
            unsigned int tail_mask = (rem != 0u) ? ((1u << (rem * 8u)) - 1u) : 0xffffffffu;
            unsigned int sw = start_word;
            void *args[]{&dev_words, &total_words, &tail_mask, &sw, &d_hash};
            if (hipModuleLaunchKernel(
                    csum_fn,
                    groups, 1u, 1u,
                    256u, 1u, 1u,
                    0u, stream, args, nullptr) != hipSuccess) {
                return false;
            }
        }
        return hipMemcpyDtoHAsync(h_hash, d_hash, 16u, stream) == hipSuccess;
    }

    void poll_events_locked(luisa::vector<PinnedSlot> &vec) noexcept {
        for (auto &s : vec) {
            if (s.in_flight && s.event != nullptr) {
                if (hipEventQuery(s.event) == hipSuccess) {
                    s.in_flight = false;
                }
            }
        }
    }

    [[nodiscard]] bool acquire_slot(
        luisa::vector<PinnedSlot> &vec,
        size_t min_bytes,
        bool is_data_pool,
        bool for_caller,
        size_t &out_idx,
        std::byte *&out_ptr,
        size_t &out_cap,
        uint64_t &out_tok) noexcept {
        std::lock_guard lck{mtx};
        if (device == nullptr || device->_hip_context == nullptr) return false;
        ScopedHipContext ctx_guard{device->_hip_context};
        poll_events_locked(vec);

        size_t best_idx = vec.size();
        for (size_t i = 0u; i < vec.size(); ++i) {
            auto const &s = vec[i];
            if (!s.caller_owned && !s.in_flight && s.capacity >= min_bytes) {
                if (best_idx == vec.size() || s.capacity < vec[best_idx].capacity) {
                    best_idx = i;
                }
            }
        }
        if (best_idx < vec.size()) {
            auto &s = vec[best_idx];
            s.lru_tick = ++tick;
            if (for_caller) s.caller_owned = true;
            else s.in_flight = true;
            out_idx = best_idx;
            out_ptr = s.ptr;
            out_cap = s.capacity;
            out_tok = s.token;
            return true;
        }

        const size_t alloc_bytes = is_data_pool ? round_up_hip_pool_size(min_bytes) : std::max<size_t>(min_bytes, 64u);
        auto evict_oldest_idle = [&]() noexcept -> bool {
            size_t victim = vec.size();
            for (size_t i = 0u; i < vec.size(); ++i) {
                if (!vec[i].caller_owned && !vec[i].in_flight) {
                    if (victim == vec.size() || vec[i].lru_tick < vec[victim].lru_tick) {
                        victim = i;
                    }
                }
            }
            if (victim == vec.size()) return false;
            auto &v = vec[victim];
            if (v.event != nullptr) (void)hipEventDestroy(v.event);
            if (v.ptr != nullptr) (void)hipHostFree(v.ptr);
            if (is_data_pool) total_data_bytes -= v.capacity;
            if (victim + 1u < vec.size()) {
                vec[victim] = std::move(vec.back());
            }
            vec.pop_back();
            return true;
        };

        if (is_data_pool) {
            while (vec.size() >= kHipMaxPoolSlots ||
                   (total_data_bytes + alloc_bytes > kHipMaxPoolBytes && !vec.empty())) {
                if (!evict_oldest_idle()) break;
            }
        } else {
            while (vec.size() >= kHipMaxPoolSlots) {
                if (!evict_oldest_idle()) break;
            }
        }

        void *raw_ptr = nullptr;
        if (hipHostMalloc(&raw_ptr, alloc_bytes, hipHostMallocDefault) != hipSuccess || raw_ptr == nullptr) {
            return false;
        }
        hipEvent_t evt = nullptr;
        if (hipEventCreateWithFlags(&evt, hipEventDisableTiming) != hipSuccess || evt == nullptr) {
            (void)hipHostFree(raw_ptr);
            return false;
        }

        PinnedSlot slot{};
        slot.ptr = static_cast<std::byte *>(raw_ptr);
        slot.capacity = alloc_bytes;
        slot.event = evt;
        slot.lru_tick = ++tick;
        slot.token = (++next_token) | (is_data_pool ? (1ull << 56u) : (2ull << 56u));
        slot.caller_owned = for_caller;
        slot.in_flight = !for_caller;
        if (is_data_pool) total_data_bytes += alloc_bytes;
        out_idx = vec.size();
        out_ptr = slot.ptr;
        out_cap = slot.capacity;
        out_tok = slot.token;
        vec.push_back(slot);
        return true;
    }

    [[nodiscard]] bool match_caller_buffer(
        const std::byte *ptr,
        size_t bytes,
        hipStream_t current_stream,
        uint64_t &out_tok) noexcept {
        if (ptr == nullptr) return false;
        std::lock_guard lck{mtx};
        poll_events_locked(data_slots);
        for (auto &s : data_slots) {
            if (!s.caller_owned || s.ptr == nullptr) continue;
            if (ptr >= s.ptr && ptr <= s.ptr + s.capacity) {
                const size_t off = static_cast<size_t>(ptr - s.ptr);
                if (off <= s.capacity && bytes <= s.capacity - off) {
                    if (s.in_flight && s.event != nullptr && s.last_stream != current_stream) {
                        ScopedHipContext ctx_guard{device->_hip_context};
                        (void)hipEventSynchronize(s.event);
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

    void record_stream_event(luisa::vector<PinnedSlot> &vec, uint64_t tok, hipStream_t stream) noexcept {
        std::lock_guard lck{mtx};
        for (auto &s : vec) {
            if (s.token == tok) {
                s.last_stream = stream;
                s.in_flight = true;
                if (s.event != nullptr) {
                    (void)hipEventRecord(s.event, stream);
                }
                return;
            }
        }
    }

    void release_idle_immediately(luisa::vector<PinnedSlot> &vec, uint64_t tok) noexcept {
        std::lock_guard lck{mtx};
        for (auto &s : vec) {
            if (s.token == tok) {
                s.in_flight = false;
                s.lru_tick = ++tick;
                return;
            }
        }
    }

    void release_caller(uint64_t tok, const std::byte *mapped) noexcept {
        std::lock_guard lck{mtx};
        poll_events_locked(data_slots);
        for (auto &s : data_slots) {
            if ((tok != 0u && s.token == tok) || (mapped != nullptr && s.ptr == mapped)) {
                s.caller_owned = false;
                s.lru_tick = ++tick;
                return;
            }
        }
    }
};

HIPAccelArchiveExt::HIPAccelArchiveExt(HIPDevice *device) noexcept
    : _device{device} {
    if (_device == nullptr) {
        _device_fingerprint = "null_hip_device";
        _driver_fingerprint = "unsupported";
        return;
    }

    hipDeviceProp_t prop{};
    hipError_t prop_err = hipSuccess;
    int rt_ver = 0;
    int drv_ver = 0;
    {
        ScopedHipContext ctx_guard{_device->_hip_context};
        prop_err = hipGetDeviceProperties(&prop, _device->device_id());
        if (hipRuntimeGetVersion(&rt_ver) != hipSuccess) { rt_ver = 0; }
        if (hipDriverGetVersion(&drv_ver) != hipSuccess) { drv_ver = 0; }
    }
    if (prop_err != hipSuccess) {
        _device_fingerprint = "hip_query_failed";
        _driver_fingerprint = "unsupported";
        return;
    }

    const std::string_view dev_name{prop.name};
    const std::string_view gcn_arch{prop.gcnArchName};
    const uint32_t expected_rtip = compute_expected_hiprt_rtip(dev_name, gcn_arch, drv_ver);

    _compat.magic = kHipCompatMagic;
    _compat.struct_size = static_cast<uint32_t>(sizeof(CompatInfo));
    _compat.pci_device_id = static_cast<uint32_t>(prop.pciDeviceID);
    _compat.hip_runtime_version = static_cast<int32_t>(rt_ver);
    _compat.hip_driver_version = static_cast<int32_t>(drv_ver);
    _compat.rtip = expected_rtip;

    const auto arch_copy_len = std::min<size_t>(gcn_arch.size(), sizeof(_compat.gcn_arch) - 1u);
    _compat.gcn_arch_len = static_cast<uint32_t>(arch_copy_len);
    std::memcpy(_compat.gcn_arch, gcn_arch.data(), arch_copy_len);
    const std::string_view hiprt_ver{HIPRT_VERSION_STR};
    const auto hiprt_copy_len = std::min<size_t>(hiprt_ver.size(), sizeof(_compat.hiprt_version) - 1u);
    std::memcpy(_compat.hiprt_version, hiprt_ver.data(), hiprt_copy_len);

    _device_fingerprint = luisa::format(
        "{}:{}:0x{:04x}", dev_name, gcn_arch, static_cast<uint32_t>(prop.pciDeviceID));
    _driver_fingerprint = luisa::format(
        "hip_rt={};hip_drv={};hiprt={};rtip={}",
        rt_ver, drv_ver, hiprt_ver, expected_rtip);
    _impl = std::make_unique<Impl>(_device);
    _supported = true;
}

HIPAccelArchiveExt::~HIPAccelArchiveExt() noexcept {
    _impl.reset();
}

AccelArchiveCapabilities HIPAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = AccelArchiveBackendKind::HIP;
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
                           luisa::string{kHipLayoutTag} :
                           "hiprt_geometry_save_load_unavailable";
    return caps;
}

AccelArchiveStagingBuffer HIPAccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    if (!_supported || !_impl || min_bytes == 0u) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    size_t idx = 0u;
    std::byte *ptr = nullptr;
    size_t cap = 0u;
    uint64_t tok = 0u;
    if (!_impl->acquire_slot(_impl->data_slots, min_bytes, true, true, idx, ptr, cap, tok)) {
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

void HIPAccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (_impl && buffer.handle != invalid_resource_handle && (buffer.handle >> 56u) != 0u) {
        _impl->release_caller(buffer.handle, buffer.mapped_data);
        buffer = {};
        return;
    }
    AccelArchiveExt::release_staging_buffer(std::move(buffer));
}

AccelArchiveResult HIPAccelArchiveExt::_check_compat_and_geom_header(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t meta_unpacked_size,
    uint64_t &out_geom_size) const noexcept {
    if (compat_blob.size() != sizeof(CompatInfo)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "HIP AS archive compat blob has unexpected byte size.");
    }
    CompatInfo info{};
    std::memcpy(&info, compat_blob.data(), sizeof(CompatInfo));
    if (info.magic != kHipCompatMagic || info.struct_size != sizeof(CompatInfo)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "HIP AS archive compat blob magic or size mismatch.");
    }
    if (info.pci_device_id != _compat.pci_device_id ||
        info.gcn_arch_len != _compat.gcn_arch_len ||
        std::memcmp(info.gcn_arch, _compat.gcn_arch, sizeof(_compat.gcn_arch)) != 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            luisa::format(
                "HIP AS archive targets gcnArch '{}' (pci=0x{:04x}), current is '{}' (pci=0x{:04x}).",
                info.gcn_arch, info.pci_device_id, _compat.gcn_arch, _compat.pci_device_id));
    }
    if (info.hip_runtime_version != _compat.hip_runtime_version ||
        info.hip_driver_version != _compat.hip_driver_version ||
        info.rtip != _compat.rtip ||
        std::memcmp(info.hiprt_version, _compat.hiprt_version, sizeof(_compat.hiprt_version)) != 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "HIP runtime/driver, HIPRT version, or RTIP version mismatch.");
    }

    if (payload.size() < sizeof(HiprtGeomHeaderLayout) || payload.size() > kMaxGeomArchiveBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT serialized geometry payload has invalid byte size.");
    }

    HiprtGeomHeaderLayout hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.total_size != payload.size() ||
        hdr.total_size != info.geometry_size ||
        hdr.total_size != meta_unpacked_size) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT GeomHeader m_size disagrees with archive payload/metadata size.");
    }
    if ((hdr.geom_type & 1u) == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT GeomHeader m_geomType does not denote triangle geometry.");
    }
    if (hdr.rtip != _compat.rtip) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            luisa::format(
                "HIPRT GeomHeader m_rtip ({}) does not match device RTIP ({}).",
                hdr.rtip, _compat.rtip));
    }
    if (hdr.box_node_count == 0u || hdr.prim_node_count == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT GeomHeader has zero boxNodeCount or primNodeCount.");
    }

    // Check internal relative offsets & extents strictly before device relocation.
    // Box4Node (rtip < 31) and Box8Node (rtip >= 31) are both 128 B;
    // TrianglePairNode (rtip < 31) is 64 B and TrianglePacketNode (rtip >= 31) is 128 B.
    const uint64_t box_node_size = 128ull;
    const uint64_t prim_node_size = (hdr.rtip >= 31u) ? 128ull : 64ull;
    const uint64_t box_bytes = static_cast<uint64_t>(hdr.box_node_count) * box_node_size;
    const uint64_t prim_bytes = static_cast<uint64_t>(hdr.prim_node_count) * prim_node_size;

    if (hdr.box_nodes_offset < sizeof(HiprtGeomHeaderLayout) ||
        hdr.box_nodes_offset > hdr.total_size ||
        box_bytes > hdr.total_size - hdr.box_nodes_offset) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT GeomHeader m_boxNodes offset/extent exceeds payload bounds.");
    }
    if (hdr.prim_nodes_offset < sizeof(HiprtGeomHeaderLayout) ||
        hdr.prim_nodes_offset > hdr.total_size ||
        prim_bytes > hdr.total_size - hdr.prim_nodes_offset) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "HIPRT GeomHeader m_primNodes offset/extent exceeds payload bounds.");
    }

    out_geom_size = hdr.total_size;
    return {};
}

AccelArchiveResult HIPAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "HIP AccelArchiveExt is unavailable.");
    }
    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::HIP,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t geom_size = 0u;
    auto check_res = _check_compat_and_geom_header(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        geom_size);
    if (!check_res) {
        return check_res;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = geom_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult HIPAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!_supported || !_impl) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "HIP AccelArchiveExt is unavailable.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }

    HIPStream *stream = nullptr;
    HIPMesh *mesh = nullptr;
    auto handle_res = check_stream_and_mesh(_device, stream_handle, mesh_handle, stream, mesh);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(mesh->option(), desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "HIPMesh AccelOption does not match AccelArchiveMeshDesc::option.");
    }

    // Drain queued build / compaction on this stream before async DtoH DMA.
    stream->synchronize();

    if (!mesh->is_built() || mesh->handle() == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "HIPMesh has not been built yet.");
    }
    if (mesh->last_triangle_count() != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "HIPMesh triangle count ({}) does not match expected ({}).",
                mesh->last_triangle_count(), desc.triangle_count));
    }

    const auto dev_geom = reinterpret_cast<hipDeviceptr_t>(mesh->handle());
    const uint64_t base_ptr = reinterpret_cast<uint64_t>(mesh->handle());
    const hipStream_t hip_stream = stream->handle();

    // Pass 1: Read 48-byte GeomHeader from VRAM into pooled pinned 64B slot.
    size_t hdr_idx = 0u;
    std::byte *hdr_ptr = nullptr;
    size_t hdr_cap = 0u;
    uint64_t hdr_tok = 0u;
    if (!_impl->acquire_slot(_impl->hdr_slots, sizeof(HiprtGeomHeaderLayout), false, false, hdr_idx, hdr_ptr, hdr_cap, hdr_tok)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to acquire pinned HIP header slot for export.");
    }
    auto release_hdr = vstd::scope_exit([&]() noexcept {
        _impl->release_idle_immediately(_impl->hdr_slots, hdr_tok);
    });

    {
        ScopedHipContext ctx_guard{_device->_hip_context};
        if (hipMemcpyDtoHAsync(hdr_ptr, dev_geom, sizeof(HiprtGeomHeaderLayout), hip_stream) != hipSuccess ||
            hipStreamSynchronize(hip_stream) != hipSuccess) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed reading HIPRT GeomHeader from device memory.");
        }
    }

    HiprtGeomHeaderLayout hdr{};
    std::memcpy(&hdr, hdr_ptr, sizeof(hdr));
    if (hdr.total_size < sizeof(HiprtGeomHeaderLayout) ||
        hdr.total_size > kMaxGeomArchiveBytes ||
        hdr.box_nodes_offset < base_ptr ||
        hdr.prim_nodes_offset < base_ptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "HIPRT device GeomHeader contains invalid size or node pointers.");
    }

    // Pass 2: Read entire geometry directly into pooled pinned host slot (zero temp files!).
    size_t rb_idx = 0u;
    std::byte *rb_ptr = nullptr;
    size_t rb_cap = 0u;
    uint64_t rb_tok = 0u;
    if (!_impl->acquire_slot(_impl->data_slots, hdr.total_size, true, false, rb_idx, rb_ptr, rb_cap, rb_tok)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to acquire pinned HIP host staging buffer for export.");
    }
    auto release_rb = vstd::scope_exit([&]() noexcept {
        _impl->release_idle_immediately(_impl->data_slots, rb_tok);
    });

    bool const use_gpu_checksum =
        hdr.total_size <= 0xffff0000ull &&
        _impl->ensure_checksum_kernel();
    bool gpu_csum_launched = false;
    {
        std::unique_lock csum_lck{_impl->csum_mtx, std::defer_lock};
        if (use_gpu_checksum) {
            csum_lck.lock();
        }
        ScopedHipContext ctx_guard{_device->_hip_context};
        if (hipMemcpyDtoHAsync(rb_ptr, dev_geom, hdr.total_size, hip_stream) != hipSuccess) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed reading serialized HIPRT geometry from device memory.");
        }
        // Hash device-resident words >= 12 (bytes 48..total_size-1) on the GPU concurrently/on-device;
        // words 0..11 hold device pointers before host relocation below and are folded in 48B CPU check.
        if (use_gpu_checksum) {
            gpu_csum_launched = _impl->launch_checksum_locked(
                hip_stream, dev_geom, hdr.total_size, /*start_word=*/12u);
        }
        if (hipStreamSynchronize(hip_stream) != hipSuccess) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed synchronizing HIPRT geometry export stream.");
        }
    }

    // Relocate absolute device pointers in header back to archive-relative byte offsets.
    std::memcpy(&hdr, rb_ptr, sizeof(hdr));
    if (hdr.box_nodes_offset < base_ptr || hdr.prim_nodes_offset < base_ptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "HIPRT geometry node pointers shifted outside allocation base.");
    }
    hdr.box_nodes_offset -= base_ptr;
    hdr.prim_nodes_offset -= base_ptr;
    std::memcpy(rb_ptr, &hdr, sizeof(hdr));

    uint8_t gpu_payload_csum[16]{};
    bool gpu_csum_valid = false;
    if (gpu_csum_launched) {
        uint32_t lanes[4]{};
        {
            std::lock_guard csum_lck{_impl->csum_mtx};
            for (size_t i = 0u; i < 4u; ++i) {
                lanes[i] = _impl->h_hash[i];
            }
        }
        uint32_t hdr_lanes[4]{};
        const size_t hdr_prefix_bytes = std::min<size_t>(sizeof(HiprtGeomHeaderLayout), static_cast<size_t>(hdr.total_size));
        accel_archive::compute_payload_checksum_partial(
            luisa::span<const std::byte>{rb_ptr, hdr_prefix_bytes},
            0u,
            hdr_lanes);
        lanes[0] += hdr_lanes[0];
        lanes[1] ^= hdr_lanes[1];
        lanes[2] += hdr_lanes[2];
        lanes[3] ^= hdr_lanes[3];
        accel_archive::finalize_payload_checksum(lanes, hdr.total_size, gpu_payload_csum);
        gpu_csum_valid = true;
    }

    CompatInfo compat_to_write = _compat;
    compat_to_write.rtip = hdr.rtip;
    compat_to_write.geometry_size = hdr.total_size;

    const luisa::span<const std::byte> payload{rb_ptr, static_cast<size_t>(hdr.total_size)};
    uint64_t geom_size = 0u;
    if (auto check_res = _check_compat_and_geom_header(
            luisa::span<const std::byte>{
                reinterpret_cast<const std::byte *>(&compat_to_write),
                sizeof(compat_to_write)},
            payload,
            hdr.total_size,
            geom_size);
        !check_res) {
        return check_res;
    }

    const bool is_compacted = mesh->is_compacted();
    auto encode_res = gpu_csum_valid ?
                          accel_archive::encode_envelope_with_payload_checksum(
                              AccelArchiveBackendKind::HIP,
                              desc,
                              is_compacted,
                              hdr.total_size,
                              luisa::span<const std::byte>{
                                  reinterpret_cast<const std::byte *>(&compat_to_write),
                                  sizeof(compat_to_write)},
                              payload,
                              gpu_payload_csum,
                              out_archive) :
                          accel_archive::encode_envelope(
                              AccelArchiveBackendKind::HIP,
                              desc,
                              is_compacted,
                              hdr.total_size,
                              luisa::span<const std::byte>{
                                  reinterpret_cast<const std::byte *>(&compat_to_write),
                                  sizeof(compat_to_write)},
                              payload,
                              out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = hdr.total_size;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult HIPAccelArchiveExt::import_mesh_archive(
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

AccelArchiveResult HIPAccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &options) noexcept {
    if (!_supported || !_impl) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "HIP AccelArchiveExt is unavailable.");
    }

    HIPStream *stream = nullptr;
    HIPMesh *mesh = nullptr;
    auto handle_res = check_stream_and_mesh(_device, stream_handle, mesh_handle, stream, mesh);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(mesh->option(), expected_desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target HIPMesh AccelOption does not match archive AccelOption.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    uint8_t expected_payload_checksum[16]{};
    auto env_res = accel_archive::decode_envelope_header_and_compat(
        archive_bytes,
        AccelArchiveBackendKind::HIP,
        expected_desc,
        view,
        expected_payload_checksum);
    if (!env_res) {
        return env_res;
    }

    uint64_t geom_size = 0u;
    auto check_res = _check_compat_and_geom_header(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        geom_size);
    if (!check_res) {
        return check_res;
    }

    const hipStream_t hip_stream = stream->handle();
    uint64_t caller_pinned_tok = 0u;
    const bool is_pinned_zero_copy = _impl->match_caller_buffer(
        view.payload.data(), view.payload.size(), hip_stream, caller_pinned_tok);

    if (is_pinned_zero_copy) {
        const bool use_gpu_verify =
            options.verify_payload_checksum &&
            geom_size <= 0xffff0000ull &&
            _impl->ensure_checksum_kernel();
        if (options.verify_payload_checksum && !use_gpu_verify) {
            if (auto csum_res = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum);
                !csum_res) {
                _impl->release_idle_immediately(_impl->data_slots, caller_pinned_tok);
                return csum_res;
            }
        }

        size_t hdr_idx = 0u;
        std::byte *hdr_ptr = nullptr;
        size_t hdr_cap = 0u;
        uint64_t hdr_tok = 0u;
        if (!_impl->acquire_slot(_impl->hdr_slots, sizeof(HiprtGeomHeaderLayout), false, false, hdr_idx, hdr_ptr, hdr_cap, hdr_tok)) {
            _impl->release_idle_immediately(_impl->data_slots, caller_pinned_tok);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to acquire pinned HIP header patch slot.");
        }

        void *dev_geom = nullptr;
        {
            ScopedHipContext ctx_guard{_device->_hip_context};
            if (hipMalloc(&dev_geom, geom_size) != hipSuccess || dev_geom == nullptr) {
                _impl->release_idle_immediately(_impl->hdr_slots, hdr_tok);
                _impl->release_idle_immediately(_impl->data_slots, caller_pinned_tok);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "hipMalloc failed allocating restored HIPRT geometry buffer.");
            }
        }

        HiprtGeomHeaderLayout patched_hdr{};
        std::memcpy(&patched_hdr, view.payload.data(), sizeof(patched_hdr));
        const uint64_t base_addr = reinterpret_cast<uint64_t>(dev_geom);
        patched_hdr.box_nodes_offset += base_addr;
        patched_hdr.prim_nodes_offset += base_addr;
        std::memcpy(hdr_ptr, &patched_hdr, sizeof(patched_hdr));

        if (use_gpu_verify) {
            // Stream unpatched payload DMA directly to dev_geom -> verify via GPU compute kernel ->
            // only if checksum matches, patch the 48-byte device header in place.
            bool csum_ok = false;
            {
                ScopedHipContext ctx_guard{_device->_hip_context};
                std::lock_guard csum_lck{_impl->csum_mtx};
                if (hipMemcpyHtoDAsync(
                        reinterpret_cast<hipDeviceptr_t>(dev_geom),
                        view.payload.data(),
                        static_cast<size_t>(geom_size),
                        hip_stream) == hipSuccess &&
                    _impl->launch_checksum_locked(
                        hip_stream, reinterpret_cast<hipDeviceptr_t>(dev_geom), geom_size, 0u) &&
                    hipStreamSynchronize(hip_stream) == hipSuccess) {
                    uint32_t lanes[4]{};
                    for (size_t i = 0u; i < 4u; ++i) {
                        lanes[i] = _impl->h_hash[i];
                    }
                    uint8_t computed[16]{};
                    accel_archive::finalize_payload_checksum(lanes, geom_size, computed);
                    csum_ok = std::memcmp(computed, expected_payload_checksum, 16u) == 0;
                }
            }
            _impl->release_idle_immediately(_impl->data_slots, caller_pinned_tok);
            if (!csum_ok) {
                {
                    ScopedHipContext ctx_guard{_device->_hip_context};
                    (void)hipFree(dev_geom);
                }
                _impl->release_idle_immediately(_impl->hdr_slots, hdr_tok);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::CORRUPT_PAYLOAD,
                    "HIP GPU verification kernel detected corrupt native AS payload.");
            }

            {
                ScopedHipContext ctx_guard{_device->_hip_context};
                if (hipMemcpyHtoDAsync(
                        reinterpret_cast<hipDeviceptr_t>(dev_geom),
                        hdr_ptr,
                        sizeof(HiprtGeomHeaderLayout),
                        hip_stream) != hipSuccess) {
                    (void)hipFree(dev_geom);
                    _impl->release_idle_immediately(_impl->hdr_slots, hdr_tok);
                    return accel_archive::make_archive_error(
                        AccelArchiveStatus::BACKEND_ERROR,
                        "hipMemcpyHtoDAsync failed patching verified HIPRT geometry header.");
                }
                _impl->record_stream_event(_impl->hdr_slots, hdr_tok, hip_stream);
                mesh->replace_geometry_from_archive(
                    _device->hiprt_context(),
                    hip_stream,
                    reinterpret_cast<hiprtGeometry>(dev_geom),
                    expected_desc.triangle_count,
                    view.meta.is_compacted,
                    /*owns_direct_device_geom=*/true);
            }
        } else {
            ScopedHipContext ctx_guard{_device->_hip_context};
            const hipError_t e1 = hipMemcpyHtoDAsync(
                reinterpret_cast<hipDeviceptr_t>(dev_geom),
                hdr_ptr,
                sizeof(HiprtGeomHeaderLayout),
                hip_stream);
            hipError_t e2 = hipSuccess;
            if (e1 == hipSuccess && geom_size > sizeof(HiprtGeomHeaderLayout)) {
                e2 = hipMemcpyHtoDAsync(
                    reinterpret_cast<hipDeviceptr_t>(static_cast<std::byte *>(dev_geom) + sizeof(HiprtGeomHeaderLayout)),
                    view.payload.data() + sizeof(HiprtGeomHeaderLayout),
                    static_cast<size_t>(geom_size - sizeof(HiprtGeomHeaderLayout)),
                    hip_stream);
            }
            if (e1 != hipSuccess || e2 != hipSuccess) {
                (void)hipFree(dev_geom);
                _impl->release_idle_immediately(_impl->hdr_slots, hdr_tok);
                _impl->release_idle_immediately(_impl->data_slots, caller_pinned_tok);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "hipMemcpyHtoDAsync failed streaming caller-pinned HIPRT geometry.");
            }
            _impl->record_stream_event(_impl->hdr_slots, hdr_tok, hip_stream);
            _impl->record_stream_event(_impl->data_slots, caller_pinned_tok, hip_stream);

            mesh->replace_geometry_from_archive(
                _device->hiprt_context(),
                hip_stream,
                reinterpret_cast<hiprtGeometry>(dev_geom),
                expected_desc.triangle_count,
                view.meta.is_compacted,
                /*owns_direct_device_geom=*/true);
        }
    } else {
        // Pageable input: single-pass parallel memcpy+hash into pooled pinned host slot,
        // patch 48-byte GeomHeader in place, then stream async HtoD DMA directly into VRAM.
        size_t st_idx = 0u;
        std::byte *st_ptr = nullptr;
        size_t st_cap = 0u;
        uint64_t st_tok = 0u;
        if (!_impl->acquire_slot(_impl->data_slots, geom_size, true, false, st_idx, st_ptr, st_cap, st_tok)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to acquire pooled HIP pinned staging buffer.");
        }

        if (!accel_archive::copy_payload_optimized(
                st_ptr,
                view.payload,
                options.verify_payload_checksum,
                expected_payload_checksum)) {
            _impl->release_idle_immediately(_impl->data_slots, st_tok);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                "Native payload checksum mismatch.");
        }

        void *dev_geom = nullptr;
        {
            ScopedHipContext ctx_guard{_device->_hip_context};
            if (hipMalloc(&dev_geom, geom_size) != hipSuccess || dev_geom == nullptr) {
                _impl->release_idle_immediately(_impl->data_slots, st_tok);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "hipMalloc failed allocating restored HIPRT geometry buffer.");
            }
        }

        HiprtGeomHeaderLayout patched_hdr{};
        std::memcpy(&patched_hdr, st_ptr, sizeof(patched_hdr));
        const uint64_t base_addr = reinterpret_cast<uint64_t>(dev_geom);
        patched_hdr.box_nodes_offset += base_addr;
        patched_hdr.prim_nodes_offset += base_addr;
        std::memcpy(st_ptr, &patched_hdr, sizeof(patched_hdr));

        {
            ScopedHipContext ctx_guard{_device->_hip_context};
            if (hipMemcpyHtoDAsync(
                    reinterpret_cast<hipDeviceptr_t>(dev_geom),
                    st_ptr,
                    static_cast<size_t>(geom_size),
                    hip_stream) != hipSuccess) {
                (void)hipFree(dev_geom);
                _impl->release_idle_immediately(_impl->data_slots, st_tok);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "hipMemcpyHtoDAsync failed streaming pooled HIPRT geometry.");
            }
            _impl->record_stream_event(_impl->data_slots, st_tok, hip_stream);

            mesh->replace_geometry_from_archive(
                _device->hiprt_context(),
                hip_stream,
                reinterpret_cast<hiprtGeometry>(dev_geom),
                expected_desc.triangle_count,
                view.meta.is_compacted,
                /*owns_direct_device_geom=*/true);
        }
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = geom_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = is_pinned_zero_copy ? "imported_pinned_on_stream" : "imported_on_stream";
    return res;
}

}// namespace luisa::compute::hip
