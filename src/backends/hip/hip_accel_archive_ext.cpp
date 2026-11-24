#include "hip_accel_archive_ext.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/vstl/functional.h>

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

[[nodiscard]] luisa::string make_unique_temp_geom_path() {
    static std::atomic<uint64_t> counter{0u};
    const uint64_t seq = counter.fetch_add(1u, std::memory_order_relaxed);
#ifdef _WIN32
    char dir[MAX_PATH]{};
    DWORD n = GetTempPathA(MAX_PATH, dir);
    std::string base = (n > 0u && n < MAX_PATH) ? std::string(dir, n) : std::string("C:\\Windows\\Temp\\");
    const uint64_t pid = static_cast<uint64_t>(GetCurrentProcessId());
#else
    std::string base = "/tmp/";
    const uint64_t pid = static_cast<uint64_t>(getpid());
#endif
    return luisa::format("{}luisa_hip_aa_{}_{}.bin", base, pid, seq);
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
    _supported = true;
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
    caps.import_is_async_on_stream = false;
    caps.layout_info = _supported ?
                           luisa::string{kHipLayoutTag} :
                           "hiprt_geometry_save_load_unavailable";
    return caps;
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

    // Check internal relative offsets & extents strictly before calling hiprtLoadGeometry.
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
    if (!_supported) {
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

    // Drain any queued build / compaction on this stream before calling
    // synchronous hiprtSaveGeometry.
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

    const auto tmp_path = make_unique_temp_geom_path();
    auto remove_tmp = vstd::scope_exit([&] {
        std::error_code ec;
        std::filesystem::remove(tmp_path.c_str(), ec);
    });

    hiprtError save_err = hiprtSuccess;
    try {
        ScopedHipContext ctx_guard{_device->_hip_context};
        save_err = hiprtSaveGeometry(
            _device->hiprt_context(),
            mesh->handle(),
            tmp_path.c_str());
    } catch (const std::exception &e) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format("hiprtSaveGeometry threw exception: {}", e.what()));
    }
    if (save_err != hiprtSuccess) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format("hiprtSaveGeometry failed with error code {}", static_cast<int>(save_err)));
    }

    luisa::vector<std::byte> payload;
    {
        std::ifstream ifs(tmp_path.c_str(), std::ios::binary | std::ios::ate);
        if (!ifs) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to read temporary serialized HIPRT geometry file.");
        }
        const auto sz = ifs.tellg();
        if (sz <= 0 || static_cast<uint64_t>(sz) < sizeof(HiprtGeomHeaderLayout)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Serialized HIPRT geometry file is empty or truncated.");
        }
        payload.resize(static_cast<size_t>(sz));
        ifs.seekg(0, std::ios::beg);
        if (!ifs.read(reinterpret_cast<char *>(payload.data()), sz)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to read serialized HIPRT geometry bytes.");
        }
    }

    HiprtGeomHeaderLayout hdr{};
    std::memcpy(&hdr, payload.data(), sizeof(hdr));
    if (hdr.total_size != payload.size()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "HIPRT saved geometry m_size does not match file size.");
    }

    CompatInfo compat_to_write = _compat;
    compat_to_write.rtip = hdr.rtip;
    compat_to_write.geometry_size = hdr.total_size;

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
    auto encode_res = accel_archive::encode_envelope(
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
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveMetadata meta{};
    auto val_res = validate_mesh_archive(archive_bytes, expected_desc, &meta);
    if (!val_res) {
        return val_res;
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
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::HIP,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    const auto tmp_path = make_unique_temp_geom_path();
    auto remove_tmp = vstd::scope_exit([&] {
        std::error_code ec;
        std::filesystem::remove(tmp_path.c_str(), ec);
    });

    {
        std::ofstream ofs(tmp_path.c_str(), std::ios::binary | std::ios::trunc);
        if (!ofs || !ofs.write(reinterpret_cast<const char *>(view.payload.data()),
                               static_cast<std::streamsize>(view.payload.size()))) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to write temporary HIPRT geometry file for loading.");
        }
        ofs.flush();
        if (!ofs.good()) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to flush temporary HIPRT geometry file for loading.");
        }
    }

    hiprtGeometry loaded_geom = nullptr;
    hiprtError load_err = hiprtSuccess;
    try {
        ScopedHipContext ctx_guard{_device->_hip_context};
        load_err = hiprtLoadGeometry(
            _device->hiprt_context(),
            loaded_geom,
            tmp_path.c_str());
    } catch (const std::exception &e) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format("hiprtLoadGeometry threw exception: {}", e.what()));
    }
    if (load_err != hiprtSuccess || loaded_geom == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format("hiprtLoadGeometry failed with error code {}", static_cast<int>(load_err)));
    }

    {
        ScopedHipContext ctx_guard{_device->_hip_context};
        mesh->replace_geometry_from_archive(
            _device->hiprt_context(),
            stream->handle(),
            loaded_geom,
            expected_desc.triangle_count,
            meta.is_compacted);
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    res.unpacked_as_size = meta.unpacked_as_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = meta.is_compacted;
    res.message = "imported_immediately";
    return res;
}

}// namespace luisa::compute::hip
