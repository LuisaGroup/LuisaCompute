#include "embree_accel_archive.h"

#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/rtx/triangle.h>

namespace luisa::compute {

namespace {

constexpr uint32_t kEmbreeCompatMagic = 0x454D4252u;// "EMBR"
constexpr uint32_t kEmbreeCompatVersion = 1u;
constexpr uint32_t kEmbreePayloadMagic = 0x45425648u;// "EBVH"
constexpr uint32_t kExpectedEmbreeVersion = 40401u;
constexpr uint32_t kExpectedArchiveAbiVersion = 1u;
constexpr uint64_t kMaxEmbreePayloadBytes = 4ull << 30u;

[[nodiscard]] bool options_match(const AccelOption &a, const AccelOption &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           !static_cast<bool>(a.motion) &&
           !static_cast<bool>(b.motion);
}

[[nodiscard]] void *lookup_embree_proc(const char *symbol_name) noexcept {
#if defined(_WIN32)
    HMODULE mod = GetModuleHandleW(L"embree4.dll");
    if (mod == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<void *>(GetProcAddress(mod, symbol_name));
#else
    return dlsym(RTLD_DEFAULT, symbol_name);
#endif
}

[[nodiscard]] AccelArchiveResult map_embree_status(
    EmbreeArchiveStatus st,
    std::string_view op_name) noexcept {
    switch (st) {
        case EmbreeArchiveStatus::OK:
            return {};
        case EmbreeArchiveStatus::INVALID_ARGUMENT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                luisa::format("{} rejected invalid Embree arguments.", op_name));
        case EmbreeArchiveStatus::UNSUPPORTED_LAYOUT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::UNSUPPORTED_LAYOUT,
                luisa::format("{} encountered an unsupported Embree BVH layout or primitive type.", op_name));
        case EmbreeArchiveStatus::CORRUPT:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                luisa::format("{} rejected a corrupt Embree BVH payload.", op_name));
        case EmbreeArchiveStatus::INCOMPATIBLE_CONFIG:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::DRIVER_INCOMPATIBLE,
                luisa::format("{} rejected incompatible Embree version, build config, or geometry metadata.", op_name));
        case EmbreeArchiveStatus::INCOMPATIBLE_ISA:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::DEVICE_INCOMPATIBLE,
                luisa::format("{} rejected an Embree BVH requiring unsupported CPU ISA features.", op_name));
        case EmbreeArchiveStatus::BUFFER_TOO_SMALL:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                luisa::format("{} buffer capacity was insufficient.", op_name));
        case EmbreeArchiveStatus::INTERNAL_ERROR:
        default:
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                luisa::format("{} failed with internal Embree error code {}.", op_name, static_cast<uint32_t>(st)));
    }
}

}// namespace

EmbreeAccelArchiveExt::EmbreeAccelArchiveExt(EmbreeAccelArchiveOps ops) noexcept
    : _ops{std::move(ops)} {
    if (_ops.device_fingerprint.empty()) {
        _ops.device_fingerprint = _ops.backend_kind == AccelArchiveBackendKind::SIMD ?
                                      "embree4_simd_cpu" :
                                      "embree4_fallback_cpu";
    }

    if (const char *env = std::getenv("LUISA_EMBREE_ACCEL_ARCHIVE");
        env != nullptr && std::string_view{env} == "0") {
        _driver_fingerprint = "embree4_archive_disabled_by_env";
        return;
    }
    if (_ops.rtc_device == nullptr) {
        _driver_fingerprint = "embree4_rtc_device_null";
        return;
    }

    _query_fn = reinterpret_cast<QueryConfigFn>(
        lookup_embree_proc("rtcExQueryStaticTriangleBVHArchiveConfig"));
    _export_fn = reinterpret_cast<ExportBvhFn>(
        lookup_embree_proc("rtcExExportStaticTriangleBVH"));
    _import_fn = reinterpret_cast<EmbreeArchiveImportFn>(
        lookup_embree_proc("rtcExImportStaticTriangleBVH"));

    if (_query_fn == nullptr || _export_fn == nullptr || _import_fn == nullptr) {
        _driver_fingerprint = "embree4_stock_without_archive_api";
        return;
    }

    const auto qstatus = _query_fn(_ops.rtc_device, &_cfg);
    if (qstatus != EmbreeArchiveStatus::OK ||
        _cfg.struct_size < sizeof(EmbreeConfigInfo) ||
        _cfg.archive_abi_version != kExpectedArchiveAbiVersion ||
        _cfg.embree_version_number != kExpectedEmbreeVersion ||
        _cfg.supports_static_triangle_bvh == 0u) {
        _driver_fingerprint = luisa::format(
            "embree4_incompatible_ext(status={},ver={},abi={})",
            static_cast<uint32_t>(qstatus),
            _cfg.embree_version_number,
            _cfg.archive_abi_version);
        return;
    }

    _driver_fingerprint = luisa::format(
        "embree={};cfg=0x{:08x};isa=0x{:08x};bvh4={};bvh8={}",
        _cfg.embree_version_number,
        _cfg.build_config_flags,
        _cfg.enabled_cpu_features,
        _cfg.supports_bvh4,
        _cfg.supports_bvh8);
    _supported = true;
}

AccelArchiveCapabilities EmbreeAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = _ops.backend_kind;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    caps.device_fingerprint = _ops.device_fingerprint;
    caps.driver_fingerprint = _driver_fingerprint;
    caps.supported = _supported;
    caps.static_triangle_blas = _supported;
    caps.supports_compacted_export = _supported;
    caps.requires_geometry_rebind = true;
    caps.import_is_async_on_stream = false;
    caps.layout_info = _supported ?
                           "embree_4_4_1_optin_bvh4_bvh8_static_tri_v1" :
                           "embree4_optin_static_triangle_bvh_unavailable";
    return caps;
}

AccelArchiveResult EmbreeAccelArchiveExt::_check_compat_and_header(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    uint64_t meta_unpacked_size,
    uint64_t &out_unpacked_bytes) const noexcept {
    if (compat_blob.size() != sizeof(CompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree AS archive compat blob has unexpected size.");
    }
    CompatBlob compat{};
    std::memcpy(&compat, compat_blob.data(), sizeof(CompatBlob));
    if (compat.magic != kEmbreeCompatMagic ||
        compat.version != kEmbreeCompatVersion ||
        compat.payload_magic != kEmbreePayloadMagic) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree AS archive compat blob magic or version mismatch.");
    }
    if (compat.embree_version_number != _cfg.embree_version_number ||
        compat.build_config_flags != _cfg.build_config_flags) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            luisa::format(
                "Embree version/config mismatch (archive ver={} cfg=0x{:08x}, active ver={} cfg=0x{:08x}).",
                compat.embree_version_number,
                compat.build_config_flags,
                _cfg.embree_version_number,
                _cfg.build_config_flags));
    }
    if ((_cfg.enabled_cpu_features & compat.enabled_cpu_features) != compat.enabled_cpu_features) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            "Embree AS archive requires CPU ISA features not enabled on active device.");
    }
    if (compat.unpacked_as_bytes != meta_unpacked_size) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "Embree compat unpacked_as_bytes disagrees with envelope unpacked_as_size.");
    }
    if (payload.size() < 136u || payload.size() > kMaxEmbreePayloadBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Embree BVH payload has invalid byte size.");
    }
    uint32_t payload_magic = 0u;
    std::memcpy(&payload_magic, payload.data(), sizeof(uint32_t));
    if (payload_magic != kEmbreePayloadMagic) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "Embree BVH payload header magic mismatch.");
    }
    out_unpacked_bytes = compat.unpacked_as_bytes;
    return {};
}

AccelArchiveResult EmbreeAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree static triangle BVH archive requires opt-in Embree 4.4.1 build; stock embree4.dll falls back to rebuild.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        _ops.backend_kind,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t unpacked_bytes = 0u;
    auto check_res = _check_compat_and_header(
        view.compat_blob,
        view.payload,
        view.meta.unpacked_as_size,
        unpacked_bytes);
    if (!check_res) {
        return check_res;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = unpacked_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult EmbreeAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree static triangle BVH archive requires opt-in Embree 4.4.1 build; stock embree4.dll falls back to rebuild.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }
    if (_ops.validate_stream && !_ops.validate_stream(_ops.user_ctx, stream_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle must be a valid stream on the active device.");
    }
    if (mesh_handle == invalid_resource_handle || mesh_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh handle must be valid.");
    }

    // Drain queued CPU stream builds before inspecting and serializing Embree's BVH.
    _ops.synchronize_stream(_ops.user_ctx, stream_handle);

    EmbreeMeshExportState state{};
    auto mesh_res = _ops.get_mesh_export_state(_ops.user_ctx, mesh_handle, state);
    if (!mesh_res) {
        return mesh_res;
    }
    if (!options_match(state.option, desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Mesh AccelOption does not match AccelArchiveMeshDesc::option.");
    }
    if (!state.is_built || state.committed_rtc_scene == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "Mesh has not been built yet.");
    }
    if (state.triangle_count != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "Mesh triangle count ({}) does not match expected ({}).",
                state.triangle_count,
                desc.triangle_count));
    }

    size_t required_bytes = 0u;
    size_t unpacked_bytes = 0u;
    auto size_status = _export_fn(
        state.committed_rtc_scene, nullptr, 0u, &required_bytes, &unpacked_bytes);
    if (size_status != EmbreeArchiveStatus::OK) {
        return map_embree_status(size_status, "rtcExExportStaticTriangleBVH(size)");
    }
    if (required_bytes == 0u || required_bytes > kMaxEmbreePayloadBytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Embree serialized BVH size is outside valid bounds.");
    }

    luisa::vector<std::byte> payload(required_bytes);
    size_t written_bytes = 0u;
    size_t unpacked_confirm = 0u;
    auto export_status = _export_fn(
        state.committed_rtc_scene,
        payload.data(),
        payload.size(),
        &written_bytes,
        &unpacked_confirm);
    if (export_status != EmbreeArchiveStatus::OK) {
        return map_embree_status(export_status, "rtcExExportStaticTriangleBVH");
    }
    if (written_bytes != payload.size()) {
        payload.resize(written_bytes);
    }

    CompatBlob compat{};
    compat.magic = kEmbreeCompatMagic;
    compat.version = kEmbreeCompatVersion;
    compat.embree_version_number = _cfg.embree_version_number;
    compat.build_config_flags = _cfg.build_config_flags;
    compat.enabled_cpu_features = 0u;
    compat.payload_magic = kEmbreePayloadMagic;
    compat.unpacked_as_bytes = static_cast<uint64_t>(unpacked_confirm);

    if (payload.size() >= 20u) {
        // Record the exact payload required_isa_mask in compat blob for pre-import checks.
        std::memcpy(&compat.enabled_cpu_features, payload.data() + 16u, sizeof(uint32_t));
    }

    const luisa::span<const std::byte> compat_span{
        reinterpret_cast<const std::byte *>(&compat),
        sizeof(CompatBlob)};
    uint64_t checked_unpacked = 0u;
    if (auto self_check = _check_compat_and_header(
            compat_span, payload, compat.unpacked_as_bytes, checked_unpacked);
        !self_check) {
        return self_check;
    }

    const bool is_compacted = desc.option.allow_compaction;
    auto encode_res = accel_archive::encode_envelope(
        _ops.backend_kind,
        desc,
        is_compacted,
        compat.unpacked_as_bytes,
        compat_span,
        payload,
        out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = compat.unpacked_as_bytes;
    encode_res.serialized_payload_size = payload.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult EmbreeAccelArchiveExt::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    if (!_supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "Embree static triangle BVH archive requires opt-in Embree 4.4.1 build; stock embree4.dll falls back to rebuild.");
    }
    if (_ops.validate_stream && !_ops.validate_stream(_ops.user_ctx, stream_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Stream handle must be a valid stream on the active device.");
    }
    if (mesh_handle == invalid_resource_handle || mesh_handle == 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Mesh handle must be valid.");
    }

    AccelArchiveMetadata meta{};
    auto val_res = validate_mesh_archive(archive_bytes, expected_desc, &meta);
    if (!val_res) {
        return val_res;
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        _ops.backend_kind,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    EmbreeResolvedTriangleGeometry resolved_geom{};
    auto geom_res = _ops.resolve_geometry_binding(
        _ops.user_ctx,
        mesh_handle,
        expected_desc,
        geometry_binding,
        resolved_geom);
    if (!geom_res) {
        return geom_res;
    }

    // Drain any prior stream work before rebinding buffers and populating the mesh's RTCScene.
    _ops.synchronize_stream(_ops.user_ctx, stream_handle);

    size_t unpacked_bytes = 0u;
    EmbreeArchiveStatus embree_st = EmbreeArchiveStatus::OK;
    auto import_res = _ops.import_mesh_bvh(
        _ops.user_ctx,
        mesh_handle,
        resolved_geom,
        view.payload,
        _import_fn,
        unpacked_bytes,
        embree_st);
    if (!import_res) {
        return import_res;
    }
    if (embree_st != EmbreeArchiveStatus::OK) {
        return map_embree_status(embree_st, "rtcExImportStaticTriangleBVH");
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    res.unpacked_as_size = unpacked_bytes;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = meta.is_compacted;
    res.message = "imported_immediately";
    return res;
}

}// namespace luisa::compute
