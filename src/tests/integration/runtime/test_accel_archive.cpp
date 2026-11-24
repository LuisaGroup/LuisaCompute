#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/accel_archive_ext.hpp>

using namespace luisa;
using namespace luisa::compute;

namespace {

#define REQUIRE_COND(cond, msg)                                                      \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " (" << #cond   \
                      << "): " << (msg) << std::endl;                                \
            return false;                                                            \
        }                                                                            \
    } while (false)

struct TestOptions {
    std::string backend{"vk"};
    std::string mode{"orchestrate"};
    std::filesystem::path archive_dir;
    size_t device_index{0u};
    bool compact{false};
    bool validation{false};
    bool expect_unsupported{false};
    bool has_explicit_device{false};
};

[[nodiscard]] bool parse_args(int argc, char **argv, TestOptions &opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            if (auto v = next_val("--backend")) { opts.backend = v; } else { return false; }
        } else if (arg == "--mode") {
            if (auto v = next_val("--mode")) { opts.mode = v; } else { return false; }
        } else if (arg == "--archive-dir") {
            if (auto v = next_val("--archive-dir")) { opts.archive_dir = v; } else { return false; }
        } else if (arg == "--device-index") {
            if (auto v = next_val("--device-index")) {
                opts.device_index = static_cast<size_t>(std::stoul(v));
                opts.has_explicit_device = true;
            } else {
                return false;
            }
        } else if (arg == "--compact") {
            opts.compact = true;
        } else if (arg == "--validation") {
            opts.validation = true;
        } else if (arg == "--expect-unsupported") {
            opts.expect_unsupported = true;
        } else if (!arg.empty() && arg[0] != '-') {
            opts.backend = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
bool write_binary_file(const std::filesystem::path &path, luisa::span<const T> data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::cerr << "Failed to open file for write: " << path << std::endl;
        return false;
    }
    if (!data.empty()) {
        f.write(reinterpret_cast<const char *>(data.data()),
                static_cast<std::streamsize>(data.size_bytes()));
    }
    return f.good();
}

template<typename T>
bool read_binary_file(const std::filesystem::path &path, luisa::vector<T> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Failed to open file for read: " << path << std::endl;
        return false;
    }
    auto bytes = static_cast<size_t>(f.tellg());
    if (bytes % sizeof(T) != 0u) {
        std::cerr << "Unexpected file byte size in " << path << ": " << bytes << std::endl;
        return false;
    }
    out.resize(bytes / sizeof(T));
    f.seekg(0);
    if (bytes > 0u) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
    }
    return f.good();
}

[[nodiscard]] bool write_file_bytes(const std::filesystem::path &path, luisa::span<const std::byte> bytes) {
    return write_binary_file(path, bytes);
}

[[nodiscard]] bool read_file_bytes(const std::filesystem::path &path, luisa::vector<std::byte> &out) {
    return read_binary_file(path, out);
}

struct GeneratedMesh {
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
};

[[nodiscard]] GeneratedMesh make_displaced_grid(uint32_t cells_per_side) {
    GeneratedMesh m;
    const uint32_t row_verts = cells_per_side + 1u;
    m.vertices.reserve(static_cast<size_t>(row_verts) * row_verts);
    m.triangles.reserve(static_cast<size_t>(cells_per_side) * cells_per_side * 2u);
    for (uint32_t y = 0u; y <= cells_per_side; ++y) {
        const float fy = (static_cast<float>(y) / static_cast<float>(cells_per_side)) * 2.0f - 1.0f;
        for (uint32_t x = 0u; x <= cells_per_side; ++x) {
            const float fx = (static_cast<float>(x) / static_cast<float>(cells_per_side)) * 2.0f - 1.0f;
            const float fz = 0.18f * std::sin(fx * 3.1f) * std::cos(fy * 2.7f);
            m.vertices.emplace_back(make_float3(fx, fy, fz));
        }
    }
    for (uint32_t y = 0u; y < cells_per_side; ++y) {
        for (uint32_t x = 0u; x < cells_per_side; ++x) {
            const uint32_t i00 = y * row_verts + x;
            const uint32_t i10 = i00 + 1u;
            const uint32_t i01 = (y + 1u) * row_verts + x;
            const uint32_t i11 = i01 + 1u;
            m.triangles.emplace_back(Triangle{i00, i10, i11});
            m.triangles.emplace_back(Triangle{i00, i11, i01});
        }
    }
    return m;
}

[[nodiscard]] luisa::vector<AccelArchiveInstanceDesc> make_four_instance_descriptors() {
    luisa::vector<AccelArchiveInstanceDesc> descs;
    descs.push_back(AccelArchiveInstanceDesc{
        .blas_index = 0u,
        .transform = translation(make_float3(-0.62f, 0.52f, -0.35f)) *
                     rotation(make_float3(0.0f, 0.0f, 1.0f), 0.23f) *
                     scaling(0.55f),
        .visibility_mask = 0xffu,
        .opaque = true,
        .user_id = 101u,
    });
    descs.push_back(AccelArchiveInstanceDesc{
        .blas_index = 0u,
        .transform = translation(make_float3(0.62f, 0.48f, -0.75f)) *
                     rotation(make_float3(0.0f, 1.0f, 0.0f), 0.42f) *
                     scaling(0.65f),
        .visibility_mask = 0xffu,
        .opaque = true,
        .user_id = 202u,
    });
    descs.push_back(AccelArchiveInstanceDesc{
        .blas_index = 0u,
        .transform = translation(make_float3(-0.55f, -0.52f, -1.05f)) *
                     rotation(make_float3(1.0f, 0.0f, 0.0f), -0.35f) *
                     scaling(0.60f),
        .visibility_mask = 0x02u,
        .opaque = true,
        .user_id = 303u,
    });
    descs.push_back(AccelArchiveInstanceDesc{
        .blas_index = 0u,
        .transform = translation(make_float3(0.55f, -0.45f, -1.35f)) *
                     rotation(make_float3(0.0f, 0.0f, 1.0f), -0.55f) *
                     rotation(make_float3(0.0f, 1.0f, 0.0f), 0.25f) *
                     scaling(0.72f),
        .visibility_mask = 0xffu,
        .opaque = true,
        .user_id = 404u,
    });
    return descs;
}

static constexpr uint32_t kRayGridW = 64u;
static constexpr uint32_t kRayGridH = 64u;
static constexpr uint32_t kRaysPerPass = kRayGridW * kRayGridH;
static constexpr uint32_t kTotalRays = kRaysPerPass * 2u;

struct TraceOutput {
    luisa::vector<uint4> ids;
    luisa::vector<float4> geom;
};

[[nodiscard]] TraceOutput run_trace_suite(Device &device, Stream &stream, const Accel &accel) {
    Kernel1D trace_kernel = [](AccelVar accel_var,
                               BufferUInt4 out_ids,
                               BufferFloat4 out_geom,
                               UInt ray_offset,
                               UInt vis_mask) noexcept {
        UInt idx = dispatch_x();
        UInt px = idx % kRayGridW;
        UInt py = idx / kRayGridW;
        Float u = (cast<float>(px) + 0.5f) / static_cast<float>(kRayGridW) * 2.4f - 1.2f;
        Float v = (cast<float>(py) + 0.5f) / static_cast<float>(kRayGridH) * 2.4f - 1.2f;
        Float3 origin = make_float3(u * 0.25f, v * 0.25f, 1.5f);
        Float3 target = make_float3(u, v, -1.0f);
        Float3 dir = normalize(target - origin);
        Var<Ray> ray = make_ray(origin, dir, 1e-4f, 100.0f);
        AccelTraceOptions trace_opts{};
        trace_opts.visibility_mask = vis_mask;
        Var<SurfaceHit> hit = accel_var.intersect(ray, trace_opts);
        Var<bool> occ = accel_var.intersect_any(ray, trace_opts);
        Bool is_hit = !hit->miss();
        UInt uid = 0u;
        if_(is_hit, [&] {
            uid = accel_var.instance_user_id(hit.inst);
        });
        out_ids.write(ray_offset + idx,
                      make_uint4(select(0u, 1u, is_hit), hit.inst, hit.prim, uid));
        out_geom.write(ray_offset + idx,
                       make_float4(hit.committed_ray_t, hit.bary.x, hit.bary.y, select(0.f, 1.f, occ)));
    };
    auto shader = device.compile(trace_kernel);
    auto ids_buf = device.create_buffer<uint4>(kTotalRays);
    auto geom_buf = device.create_buffer<float4>(kTotalRays);
    TraceOutput out;
    out.ids.resize(kTotalRays);
    out.geom.resize(kTotalRays);
    stream << shader(accel, ids_buf, geom_buf, 0u, 0xffu).dispatch(kRaysPerPass)
           << shader(accel, ids_buf, geom_buf, kRaysPerPass, 0x01u).dispatch(kRaysPerPass)
           << ids_buf.copy_to(out.ids.data())
           << geom_buf.copy_to(out.geom.data())
           << synchronize();
    return out;
}

[[nodiscard]] bool verify_trace_parity(
    const TraceOutput &lhs,
    const TraceOutput &rhs,
    const char *label,
    bool check_all_instances_hit) {
    REQUIRE_COND(lhs.ids.size() == kTotalRays && rhs.ids.size() == kTotalRays, "Ray count mismatch");
    REQUIRE_COND(lhs.geom.size() == kTotalRays && rhs.geom.size() == kTotalRays, "Geom count mismatch");
    uint32_t hit_counts[4]{0u, 0u, 0u, 0u};
    uint32_t miss_count = 0u;
    for (uint32_t i = 0u; i < kTotalRays; ++i) {
        const auto &a_id = lhs.ids[i];
        const auto &b_id = rhs.ids[i];
        const auto &a_gm = lhs.geom[i];
        const auto &b_gm = rhs.geom[i];
        if (a_id.x != b_id.x || a_gm.w != b_gm.w) {
            std::cerr << "[PARITY FAIL: " << label << "] ray " << i
                      << " hit=(" << a_id.x << " vs " << b_id.x << ") occ=("
                      << a_gm.w << " vs " << b_gm.w << ")" << std::endl;
            return false;
        }
        if (a_id.x == 1u) {
            if (a_id.y != b_id.y || a_id.z != b_id.z || a_id.w != b_id.w) {
                std::cerr << "[PARITY FAIL: " << label << "] ray " << i
                          << " inst/prim/uid=(" << a_id.y << "," << a_id.z << "," << a_id.w
                          << ") vs (" << b_id.y << "," << b_id.z << "," << b_id.w << ")"
                          << std::endl;
                return false;
            }
            const float dt = std::abs(a_gm.x - b_gm.x);
            const float du = std::abs(a_gm.y - b_gm.y);
            const float dv = std::abs(a_gm.z - b_gm.z);
            if (dt > 1e-4f || du > 1e-3f || dv > 1e-3f) {
                std::cerr << "[PARITY FAIL: " << label << "] ray " << i
                          << " t/bary diff=(" << dt << "," << du << "," << dv << ")" << std::endl;
                return false;
            }
            if (i < kRaysPerPass && a_id.y < 4u) {
                ++hit_counts[a_id.y];
            }
        } else {
            ++miss_count;
        }
    }
    if (check_all_instances_hit) {
        for (uint32_t inst = 0u; inst < 4u; ++inst) {
            REQUIRE_COND(hit_counts[inst] > 0u, "Expected every transformed instance to be hit");
        }
        REQUIRE_COND(miss_count > 0u, "Expected background miss rays");
        // Verify mask 0x01 excluded instance 2 (which has visibility_mask 0x02)
        for (uint32_t i = kRaysPerPass; i < kTotalRays; ++i) {
            REQUIRE_COND(lhs.ids[i].x == 0u || lhs.ids[i].y != 2u,
                         "Visibility mask 0x01 should exclude instance 2");
        }
    }
    std::cout << "[PARITY OK: " << label << "] total_rays=" << kTotalRays
              << " hits_per_inst=[" << hit_counts[0] << "," << hit_counts[1] << ","
              << hit_counts[2] << "," << hit_counts[3] << "] misses=" << miss_count
              << std::endl;
    return true;
}

struct EnvelopeSlices {
    AccelArchiveBackendKind backend_kind{AccelArchiveBackendKind::UNKNOWN};
    bool is_compacted{false};
    uint64_t unpacked_as_size{0u};
    luisa::span<const std::byte> compat_blob;
    luisa::span<const std::byte> payload;
};

[[nodiscard]] bool slice_valid_envelope(
    luisa::span<const std::byte> archive_bytes,
    EnvelopeSlices &out) {
    if (archive_bytes.size() < sizeof(accel_archive::AccelArchiveEnvelopeHeader)) {
        return false;
    }
    accel_archive::AccelArchiveEnvelopeHeader hdr{};
    std::memcpy(&hdr, archive_bytes.data(), sizeof(hdr));
    const size_t compat_off = sizeof(hdr);
    const size_t payload_off = compat_off + hdr.compat_blob_size;
    if (payload_off + hdr.payload_size != archive_bytes.size()) {
        return false;
    }
    out.backend_kind = static_cast<AccelArchiveBackendKind>(hdr.backend_kind);
    out.is_compacted = (hdr.option_flags & accel_archive::kOptionIsCompacted) != 0u;
    out.unpacked_as_size = hdr.unpacked_as_size;
    out.compat_blob = archive_bytes.subspan(compat_off, hdr.compat_blob_size);
    out.payload = archive_bytes.subspan(payload_off, static_cast<size_t>(hdr.payload_size));
    return true;
}

[[nodiscard]] bool run_export_mode(const char *program_path, const TestOptions &opts) {
    Context ctx{program_path};
    DeviceConfig cfg{};
    if (opts.has_explicit_device) {
        cfg.device_index = opts.device_index;
    }
    Device device = ctx.create_device(opts.backend, &cfg, opts.validation);
    Stream stream = device.create_stream();
    auto *ext = device.extension<AccelArchiveExt>();
    REQUIRE_COND(ext != nullptr, "AccelArchiveExt must exist on target backend");

    auto caps = ext->capabilities();
    std::cout << "[EXPORT] backend=" << opts.backend
              << " supported=" << caps.supported
              << " layout=" << caps.layout_info
              << " device=" << caps.device_fingerprint
              << " driver=" << caps.driver_fingerprint << std::endl;

    auto mesh_data = make_displaced_grid(12u);
    const auto geom_hash = accel_archive::compute_static_triangle_geometry_hash(
        mesh_data.vertices.data(), mesh_data.vertices.size(), sizeof(float3),
        mesh_data.triangles.data(), mesh_data.triangles.size());

    AccelOption blas_opt{};
    blas_opt.hint = AccelOption::UsageHint::FAST_TRACE;
    blas_opt.allow_compaction = opts.compact;
    blas_opt.allow_update = false;

    auto vb = device.create_buffer<float3>(mesh_data.vertices.size());
    auto tb = device.create_buffer<Triangle>(mesh_data.triangles.size());
    Mesh mesh = device.create_mesh(vb, tb, blas_opt);
    stream << vb.copy_from(mesh_data.vertices.data())
           << tb.copy_from(mesh_data.triangles.data())
           << synchronize();

    // Unbuilt mesh check (unless extension itself is unsupported)
    if (caps.supported) {
        luisa::vector<std::byte> tmp_archive;
        auto unbuilt_res = export_mesh_archive(ext, stream, mesh, blas_opt, geom_hash, tmp_archive);
        REQUIRE_COND(unbuilt_res.status == AccelArchiveStatus::MESH_NOT_BUILT,
                     "Export before mesh.build() must return MESH_NOT_BUILT");
    }

    stream << mesh.build() << synchronize();

    luisa::vector<std::byte> archive_bytes;
    auto exp_res = export_mesh_archive(ext, stream, mesh, blas_opt, geom_hash, archive_bytes);
    if (opts.expect_unsupported) {
        REQUIRE_COND(exp_res.status == AccelArchiveStatus::UNSUPPORTED_FEATURE || !caps.supported,
                     "Expected UNSUPPORTED_FEATURE when --expect-unsupported is set");
        REQUIRE_COND(write_file_bytes(opts.archive_dir / "unsupported_marker.bin", {}),
                     "Failed to write marker");
        std::cout << "[EXPORT] Verified clean UNSUPPORTED rejection: " << exp_res.message << std::endl;
        return true;
    }

    REQUIRE_COND(static_cast<bool>(exp_res), exp_res.message.c_str());
    REQUIRE_COND(exp_res.readiness == AccelArchiveReadiness::READY_IMMEDIATELY,
                 "Export readiness should be READY_IMMEDIATELY");
    if (opts.compact) {
        REQUIRE_COND(exp_res.is_compacted, "Compacted export should set is_compacted=true");
    }

    // Build 4-instance TLAS and trace reference rays
    auto inst_descs = make_four_instance_descriptors();
    const Mesh *blas_ptrs[1] = {&mesh};
    Accel accel = device.create_accel();
    populate_tlas_from_descriptors(accel, blas_ptrs, inst_descs);
    stream << accel.build() << synchronize();

    auto ref_trace = run_trace_suite(device, stream, accel);
    REQUIRE_COND(write_file_bytes(opts.archive_dir / "archive.bin", archive_bytes),
                 "Failed to write archive.bin");
    REQUIRE_COND(write_binary_file(opts.archive_dir / "ref_ids.bin", luisa::span<const uint4>{ref_trace.ids}),
                 "Failed to write ref_ids.bin");
    REQUIRE_COND(write_binary_file(opts.archive_dir / "ref_geom.bin", luisa::span<const float4>{ref_trace.geom}),
                 "Failed to write ref_geom.bin");
    std::cout << "[EXPORT] Wrote archive (" << archive_bytes.size()
              << " B, payload=" << exp_res.serialized_payload_size
              << " B, unpacked=" << exp_res.unpacked_as_size << " B)" << std::endl;
    return true;
}

[[nodiscard]] bool run_import_mode(const char *program_path, const TestOptions &opts) {
    Context ctx{program_path};
    DeviceConfig cfg{};
    if (opts.has_explicit_device) {
        cfg.device_index = opts.device_index;
    }
    Device device = ctx.create_device(opts.backend, &cfg, opts.validation);
    Stream stream = device.create_stream();
    auto *ext = device.extension<AccelArchiveExt>();
    REQUIRE_COND(ext != nullptr, "AccelArchiveExt must exist in import process");
    auto caps = ext->capabilities();

    auto mesh_data = make_displaced_grid(12u);
    const auto geom_hash = accel_archive::compute_static_triangle_geometry_hash(
        mesh_data.vertices.data(), mesh_data.vertices.size(), sizeof(float3),
        mesh_data.triangles.data(), mesh_data.triangles.size());

    AccelOption blas_opt{};
    blas_opt.hint = AccelOption::UsageHint::FAST_TRACE;
    blas_opt.allow_compaction = opts.compact;
    blas_opt.allow_update = false;

    if (opts.expect_unsupported) {
        auto vb = device.create_buffer<float3>(mesh_data.vertices.size());
        auto tb = device.create_buffer<Triangle>(mesh_data.triangles.size());
        Mesh mesh = device.create_mesh(vb, tb, blas_opt);
        stream << vb.copy_from(mesh_data.vertices.data())
               << tb.copy_from(mesh_data.triangles.data());
        std::array<std::byte, 4> empty_cache{};
        auto dummy_desc = make_accel_archive_mesh_desc(mesh, blas_opt, geom_hash);
        auto vr = ext->validate_mesh_archive(empty_cache, dummy_desc);
        REQUIRE_COND(!caps.supported || !static_cast<bool>(vr), "Cache miss expected cleanly");
        stream << mesh.build() << synchronize();
        luisa::vector<std::byte> dummy_out;
        auto exp_r = export_mesh_archive(ext, stream, mesh, blas_opt, geom_hash, dummy_out);
        REQUIRE_COND(exp_r.status == AccelArchiveStatus::UNSUPPORTED_FEATURE || !caps.supported,
                     "Expected UNSUPPORTED_FEATURE on export");
        std::cout << "[IMPORT] Stock unsupported + cache-miss fallback verified." << std::endl;
        return true;
    }

    luisa::vector<std::byte> archive_bytes;
    TraceOutput ref_trace;
    REQUIRE_COND(read_file_bytes(opts.archive_dir / "archive.bin", archive_bytes),
                 "Failed to read archive.bin");
    REQUIRE_COND(read_binary_file(opts.archive_dir / "ref_ids.bin", ref_trace.ids),
                 "Failed to read ref_ids.bin");
    REQUIRE_COND(read_binary_file(opts.archive_dir / "ref_geom.bin", ref_trace.geom),
                 "Failed to read ref_geom.bin");

    // Fresh Mesh allocation without copying reference geometry unless requires_geometry_rebind is true.
    auto vb = device.create_buffer<float3>(mesh_data.vertices.size());
    auto tb = device.create_buffer<Triangle>(mesh_data.triangles.size());
    if (caps.requires_geometry_rebind) {
        stream << vb.copy_from(mesh_data.vertices.data())
               << tb.copy_from(mesh_data.triangles.data())
               << synchronize();
    } else {
        // Deliberately zero buffers on self-contained hardware BLAS backends to prove
        // tracing uses deserialized BLAS contents rather than geometry buffers.
        luisa::vector<float3> zeros_v(mesh_data.vertices.size(), make_float3(0.f));
        luisa::vector<Triangle> zeros_t(mesh_data.triangles.size(), Triangle{0u, 0u, 0u});
        stream << vb.copy_from(zeros_v.data())
               << tb.copy_from(zeros_t.data())
               << synchronize();
    }
    Mesh mesh = device.create_mesh(vb, tb, blas_opt);

    AccelArchiveMetadata meta{};
    auto val_res = validate_mesh_archive(ext, archive_bytes, mesh, blas_opt, geom_hash, &meta);
    REQUIRE_COND(static_cast<bool>(val_res), val_res.message.c_str());

    auto imp_res = import_mesh_archive(ext, stream, mesh, blas_opt, geom_hash, archive_bytes);
    REQUIRE_COND(static_cast<bool>(imp_res), imp_res.message.c_str());
    stream.synchronize();

    // Build fresh TLAS from restored BLAS + stable caller-owned instance descriptors
    auto inst_descs = make_four_instance_descriptors();
    const Mesh *blas_ptrs[1] = {&mesh};
    Accel accel = device.create_accel();
    populate_tlas_from_descriptors(accel, blas_ptrs, inst_descs);
    stream << accel.build() << synchronize();

    auto imp_trace = run_trace_suite(device, stream, accel);
    REQUIRE_COND(verify_trace_parity(ref_trace, imp_trace, "cross_process_import", true),
                 "Cross-process trace parity check failed");

    // Verify acquire_staging_buffer / release_staging_buffer + import_mesh_archive_with_options
    // (both verify_payload_checksum = true and false, releasing staging immediately after submit)
    for (bool verify_csum : {true, false}) {
        Mesh staged_mesh = device.create_mesh(vb, tb, blas_opt);
        auto st = ext->acquire_staging_buffer(archive_bytes.size());
        REQUIRE_COND(st.mapped_data != nullptr && st.capacity >= archive_bytes.size(),
                     "acquire_staging_buffer returned invalid buffer");
        std::memcpy(st.mapped_data, archive_bytes.data(), archive_bytes.size());
        AccelArchiveImportOptions iopts{};
        iopts.verify_payload_checksum = verify_csum;
        auto st_res = import_mesh_archive_with_options(
            ext, stream, staged_mesh, blas_opt, geom_hash,
            luisa::span<const std::byte>{st.mapped_data, archive_bytes.size()}, iopts);
        ext->release_staging_buffer(std::move(st));
        REQUIRE_COND(static_cast<bool>(st_res), st_res.message.c_str());
        stream.synchronize();

        const Mesh *st_ptrs[1] = {&staged_mesh};
        Accel st_accel = device.create_accel();
        populate_tlas_from_descriptors(st_accel, st_ptrs, inst_descs);
        stream << st_accel.build() << synchronize();
        auto st_trace = run_trace_suite(device, stream, st_accel);
        REQUIRE_COND(verify_trace_parity(ref_trace, st_trace,
                                         verify_csum ? "pinned_staging_verified" : "pinned_staging_unverified", true),
                     "Pinned staging trace parity check failed");
    }

    // Verify direct file import path (verified cached read and unverified unbuffered/DirectStorage)
    const std::string arc_file_str = (opts.archive_dir / "archive.bin").string();
    for (bool verify_csum : {true, false}) {
        Mesh file_mesh = device.create_mesh(vb, tb, blas_opt);
        AccelArchiveImportOptions fopts{};
        fopts.verify_payload_checksum = verify_csum;
        fopts.bypass_os_file_cache = !verify_csum;
        auto fl_res = import_mesh_archive_file(
            ext, stream, file_mesh, blas_opt, geom_hash, arc_file_str, fopts);
        REQUIRE_COND(static_cast<bool>(fl_res), fl_res.message.c_str());
        stream.synchronize();

        const Mesh *fl_ptrs[1] = {&file_mesh};
        Accel fl_accel = device.create_accel();
        populate_tlas_from_descriptors(fl_accel, fl_ptrs, inst_descs);
        stream << fl_accel.build() << synchronize();
        auto fl_trace = run_trace_suite(device, stream, fl_accel);
        REQUIRE_COND(verify_trace_parity(ref_trace, fl_trace,
                                         verify_csum ? "direct_file_verified" : "direct_file_fast", true),
                     "Direct file import trace parity check failed");
    }

    // Release temporary vb/tb on hardware backends and confirm restored BLAS still traces cleanly.
    if (!caps.requires_geometry_rebind) {
        vb = Buffer<float3>{};
        tb = Buffer<Triangle>{};
        auto after_release_trace = run_trace_suite(device, stream, accel);
        REQUIRE_COND(verify_trace_parity(ref_trace, after_release_trace, "after_geometry_release", true),
                     "Trace parity failed after releasing empty geometry buffers");
        // Reallocate fresh valid buffers for negative/rebuild tests below
        vb = device.create_buffer<float3>(mesh_data.vertices.size());
        tb = device.create_buffer<Triangle>(mesh_data.triangles.size());
        mesh = device.create_mesh(vb, tb, blas_opt);
    }

    // ---- Negative & native compatibility rejection tests ----
    auto desc = make_accel_archive_mesh_desc(mesh, blas_opt, geom_hash);
    auto binding = make_accel_archive_geometry_binding(mesh);

    // 1. Truncated archives
    {
        auto trunc = luisa::span<const std::byte>{archive_bytes}.subspan(0u, archive_bytes.size() - 16u);
        REQUIRE_COND(ext->validate_mesh_archive(trunc, desc).status == AccelArchiveStatus::TRUNCATED,
                     "Truncated archive must fail validate with TRUNCATED");
        REQUIRE_COND(ext->import_mesh_archive(stream.handle(), mesh.handle(), desc, binding, trunc).status ==
                         AccelArchiveStatus::TRUNCATED,
                     "Truncated archive must fail import with TRUNCATED");
    }

    // 2. Corrupt header byte
    {
        auto bad_hdr = archive_bytes;
        bad_hdr[24u] = static_cast<std::byte>(std::to_integer<uint8_t>(bad_hdr[24u]) ^ 0x5au);
        REQUIRE_COND(ext->validate_mesh_archive(bad_hdr, desc).status == AccelArchiveStatus::CORRUPT_HEADER,
                     "Corrupted header must return CORRUPT_HEADER");
        REQUIRE_COND(ext->import_mesh_archive(stream.handle(), mesh.handle(), desc, binding, bad_hdr).status ==
                         AccelArchiveStatus::CORRUPT_HEADER,
                     "Corrupted header import must return CORRUPT_HEADER");
    }

    // 3. Corrupt payload byte
    {
        auto bad_pay = archive_bytes;
        bad_pay.back() = static_cast<std::byte>(std::to_integer<uint8_t>(bad_pay.back()) ^ 0xffu);
        REQUIRE_COND(ext->validate_mesh_archive(bad_pay, desc).status == AccelArchiveStatus::CORRUPT_PAYLOAD,
                     "Corrupted payload must return CORRUPT_PAYLOAD");
        REQUIRE_COND(ext->import_mesh_archive(stream.handle(), mesh.handle(), desc, binding, bad_pay).status ==
                         AccelArchiveStatus::CORRUPT_PAYLOAD,
                     "Corrupted payload import must return CORRUPT_PAYLOAD");

        // 3b. Pinned staging + verify_payload_checksum=true exercises GPU verification kernel rejection
        auto bad_st = ext->acquire_staging_buffer(bad_pay.size());
        REQUIRE_COND(bad_st.mapped_data != nullptr && bad_st.capacity >= bad_pay.size(),
                     "acquire_staging_buffer returned invalid buffer for negative payload test");
        std::memcpy(bad_st.mapped_data, bad_pay.data(), bad_pay.size());
        AccelArchiveImportOptions verify_opts{};
        verify_opts.verify_payload_checksum = true;
        auto pin_bad_res = ext->import_mesh_archive_with_options(
            stream.handle(), mesh.handle(), desc, binding,
            luisa::span<const std::byte>{bad_st.mapped_data, bad_pay.size()}, verify_opts);
        ext->release_staging_buffer(std::move(bad_st));
        REQUIRE_COND(pin_bad_res.status == AccelArchiveStatus::CORRUPT_PAYLOAD,
                     "Pinned verified import must reject corrupt payload with CORRUPT_PAYLOAD");

        // 3c. Direct file import (including DirectStorage unbuffered path) + verify=true rejects corrupt payload
        const std::filesystem::path bad_file_path = opts.archive_dir / "corrupt_payload.bin";
        REQUIRE_COND(write_binary_file(bad_file_path, luisa::span<const std::byte>{bad_pay}),
                     "Failed to write corrupt_payload.bin for verified file-import negative test");
        const std::string bad_file_str = bad_file_path.string();
        for (bool bypass_cache : {true, false}) {
            Mesh bad_file_mesh = device.create_mesh(vb, tb, blas_opt);
            AccelArchiveImportOptions fbad_opts{};
            fbad_opts.verify_payload_checksum = true;
            fbad_opts.bypass_os_file_cache = bypass_cache;
            auto fbad_res = import_mesh_archive_file(
                ext, stream, bad_file_mesh, blas_opt, geom_hash, bad_file_str, fbad_opts);
            REQUIRE_COND(fbad_res.status == AccelArchiveStatus::CORRUPT_PAYLOAD,
                         "Verified direct file import must reject corrupt payload with CORRUPT_PAYLOAD");
        }
        std::error_code ec;
        std::filesystem::remove(bad_file_path, ec);
    }

    // 4. Native compat blob mismatch inside valid outer envelope checksums
    {
        EnvelopeSlices slices{};
        REQUIRE_COND(slice_valid_envelope(archive_bytes, slices), "Failed to slice valid envelope");
        if (!slices.compat_blob.empty()) {
            luisa::vector<std::byte> mutated_compat(slices.compat_blob.begin(), slices.compat_blob.end());
            // Flip middle compat bytes (device/driver IDs or header inside compat blob)
            const size_t flip_idx = std::min<size_t>(10u, mutated_compat.size() - 1u);
            mutated_compat[flip_idx] =
                static_cast<std::byte>(std::to_integer<uint8_t>(mutated_compat[flip_idx]) ^ 0x3cu);
            luisa::vector<std::byte> forged_archive;
            auto enc = accel_archive::encode_envelope(
                slices.backend_kind, desc, slices.is_compacted, slices.unpacked_as_size,
                mutated_compat, slices.payload, forged_archive);
            REQUIRE_COND(static_cast<bool>(enc), "Failed to encode mutated-compat envelope");
            auto compat_res = ext->import_mesh_archive(
                stream.handle(), mesh.handle(), desc, binding, forged_archive);
            REQUIRE_COND(!static_cast<bool>(compat_res),
                         "Native compat mismatch must reject before loading opaque bytes");
            REQUIRE_COND(compat_res.status == AccelArchiveStatus::DEVICE_INCOMPATIBLE ||
                             compat_res.status == AccelArchiveStatus::DRIVER_INCOMPATIBLE ||
                             compat_res.status == AccelArchiveStatus::UNSUPPORTED_LAYOUT ||
                             compat_res.status == AccelArchiveStatus::CORRUPT_HEADER ||
                             compat_res.status == AccelArchiveStatus::CORRUPT_PAYLOAD,
                         "Unexpected status for native compat mismatch");
        }
    }

    // 5. Options mismatch
    {
        AccelOption wrong_opt = blas_opt;
        wrong_opt.hint = AccelOption::UsageHint::FAST_BUILD;
        auto r = validate_mesh_archive(ext, archive_bytes, mesh, wrong_opt, geom_hash);
        REQUIRE_COND(r.status == AccelArchiveStatus::OPTIONS_MISMATCH,
                     "Changed AccelOption must fail with OPTIONS_MISMATCH");
    }

    // 6. Unsupported feature (allow_update = true)
    {
        AccelOption upd_opt = blas_opt;
        upd_opt.allow_update = true;
        auto r = validate_mesh_archive(ext, archive_bytes, mesh, upd_opt, geom_hash);
        REQUIRE_COND(r.status == AccelArchiveStatus::UNSUPPORTED_FEATURE,
                     "allow_update=true must be rejected as UNSUPPORTED_FEATURE");
    }

    // 7. Changed geometry + cache-miss fallback rebuild produces exact parity
    {
        auto mutated = mesh_data;
        mutated.vertices[mutated.vertices.size() / 2u].z += 0.35f;
        const auto mutated_hash = accel_archive::compute_static_triangle_geometry_hash(
            mutated.vertices.data(), mutated.vertices.size(), sizeof(float3),
            mutated.triangles.data(), mutated.triangles.size());
        auto miss_res = import_mesh_archive(ext, stream, mesh, blas_opt, mutated_hash, archive_bytes);
        REQUIRE_COND(miss_res.status == AccelArchiveStatus::KEY_MISMATCH,
                     "Changed geometry must trigger cache miss (KEY_MISMATCH)");

        // Fallback rebuild on same Mesh after cache miss
        stream << vb.copy_from(mutated.vertices.data())
               << tb.copy_from(mutated.triangles.data())
               << mesh.build();
        populate_tlas_from_descriptors(accel, blas_ptrs, inst_descs);
        stream << accel.build() << synchronize();
        auto rebuilt_trace = run_trace_suite(device, stream, accel);

        // Compare against completely fresh reference Mesh+Accel built from mutated geometry
        auto vb2 = device.create_buffer<float3>(mutated.vertices.size());
        auto tb2 = device.create_buffer<Triangle>(mutated.triangles.size());
        Mesh ref_mesh2 = device.create_mesh(vb2, tb2, blas_opt);
        const Mesh *ref_ptrs[1] = {&ref_mesh2};
        Accel ref_accel2 = device.create_accel();
        populate_tlas_from_descriptors(ref_accel2, ref_ptrs, inst_descs);
        stream << vb2.copy_from(mutated.vertices.data())
               << tb2.copy_from(mutated.triangles.data())
               << ref_mesh2.build()
               << ref_accel2.build()
               << synchronize();
        auto ref_trace2 = run_trace_suite(device, stream, ref_accel2);
        REQUIRE_COND(verify_trace_parity(ref_trace2, rebuilt_trace, "cache_miss_rebuild", false),
                     "Fallback rebuild after cache miss parity failed");
    }

    std::cout << "[IMPORT] All parity, corruption, compat, and cache-miss recovery checks PASSED." << std::endl;
    return true;
}

[[nodiscard]] int spawn_child(const std::vector<std::string> &args) {
#if defined(_WIN32)
    std::string cmdline;
    for (size_t i = 0u; i < args.size(); ++i) {
        if (i > 0u) { cmdline.push_back(' '); }
        cmdline.push_back('"');
        for (char c : args[i]) {
            if (c == '"') { cmdline.push_back('\\'); }
            cmdline.push_back(c);
        }
        cmdline.push_back('"');
    }
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    luisa::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
    mutable_cmd.push_back('\0');
    if (!CreateProcessA(
            nullptr, mutable_cmd.data(), nullptr, nullptr, FALSE,
            0u, nullptr, nullptr, &si, &pi)) {
        std::cerr << "CreateProcessA failed (" << GetLastError() << ") for: " << cmdline << std::endl;
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1u;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(exit_code);
#else
    return 1;
#endif
}

[[nodiscard]] bool run_orchestrator(const char *program_path, const TestOptions &opts) {
    namespace fs = std::filesystem;
    fs::path temp_dir = opts.archive_dir;
    bool owns_temp = temp_dir.empty();
    if (owns_temp) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        temp_dir = fs::temp_directory_path() / ("luisa_accel_archive_test_" + std::to_string(stamp));
    }
    std::error_code ec;
    fs::create_directories(temp_dir, ec);
    if (ec) {
        std::cerr << "Failed to create temp directory " << temp_dir << ": " << ec.message() << std::endl;
        return false;
    }

    auto make_child_args = [&](const char *mode) {
        std::vector<std::string> args{
            program_path,
            "--backend", opts.backend,
            "--mode", mode,
            "--archive-dir", temp_dir.string()};
        if (opts.compact) { args.emplace_back("--compact"); }
        if (opts.validation) { args.emplace_back("--validation"); }
        if (opts.expect_unsupported) { args.emplace_back("--expect-unsupported"); }
        if (opts.has_explicit_device) {
            args.emplace_back("--device-index");
            args.emplace_back(std::to_string(opts.device_index));
        }
        return args;
    };

    const int exp_rc = spawn_child(make_child_args("export"));
    if (exp_rc != 0) {
        if (owns_temp) { fs::remove_all(temp_dir, ec); }
        std::cerr << "Export child failed with exit code " << exp_rc << std::endl;
        return false;
    }

    const int imp_rc = spawn_child(make_child_args("import"));
    if (owns_temp) { fs::remove_all(temp_dir, ec); }
    if (imp_rc != 0) {
        std::cerr << "Import child failed with exit code " << imp_rc << std::endl;
        return false;
    }

    std::cout << "[PASS] Cross-process AccelArchiveExt suite succeeded (backend="
              << opts.backend << ", compact=" << opts.compact
              << ", validation=" << opts.validation << ")." << std::endl;
    return true;
}

}// namespace

int main(int argc, char **argv) {
    TestOptions opts{};
    if (!parse_args(argc, argv, opts)) {
        return 2;
    }
    bool ok = false;
    if (opts.mode == "orchestrate") {
        ok = run_orchestrator(argv[0], opts);
    } else if (opts.mode == "export") {
        REQUIRE_COND(!opts.archive_dir.empty(), "--archive-dir is required for --mode export");
        ok = run_export_mode(argv[0], opts);
    } else if (opts.mode == "import") {
        REQUIRE_COND(!opts.archive_dir.empty(), "--archive-dir is required for --mode import");
        ok = run_import_mode(argv[0], opts);
    } else {
        std::cerr << "Invalid --mode: " << opts.mode << std::endl;
        return 2;
    }
    return ok ? 0 : 1;
}
