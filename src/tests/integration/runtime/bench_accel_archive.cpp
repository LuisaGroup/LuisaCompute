#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/accel_archive_ext.hpp>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../../common/tiny_obj_loader.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#endif

using namespace luisa;
using namespace luisa::compute;

namespace {

struct BenchConfig {
    std::string backend{"vk"};
    std::string mode{"both"};// both | uncompacted | compacted | rebuild_only
    std::vector<std::string> scales{"10k", "100k", "500k", "2m"};
    std::string obj_path;
    uint32_t repeats{5u};
    uint32_t warmup{1u};
    uint32_t ray_res{512u};
    uint32_t trace_iters{8u};
    std::filesystem::path out_dir;
    std::filesystem::path json_path;
};

struct MeshData {
    std::string name;
    std::string source;
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    luisa::Hash128 content_hash{};
};

struct Stats {
    double median{0.0};
    double mean{0.0};
    double min{0.0};
    double max{0.0};
    double p95{0.0};
    std::vector<double> samples;
};

struct MemorySnapshot {
    uint64_t host_working_set{0ull};
    uint64_t host_peak_working_set{0ull};
    uint64_t device_total_bytes{0ull};
    uint64_t device_free_bytes{0ull};
};

struct ModeBenchResult {
    std::string mesh_name;
    std::string mesh_source;
    uint64_t vertex_count{0ull};
    uint64_t triangle_count{0ull};
    bool allow_compaction{false};
    bool archive_attempted{false};
    bool archive_succeeded{false};
    bool is_compacted{false};
    std::string readiness;
    std::string status;
    std::string message;
    uint64_t archive_bytes{0ull};
    uint64_t payload_bytes{0ull};
    uint64_t unpacked_as_bytes{0ull};

    Stats blas_build_ms;
    Stats tlas_build_ms;
    Stats export_ms;
    Stats relocate_ms;
    Stats relocate_pinned_ms;
    Stats relocate_fast_ms;
    Stats warm_read_ms;
    Stats warm_import_ms;
    Stats warm_tlas_ms;
    Stats warm_e2e_ms;
    Stats warm_direct_e2e_ms;
    Stats cold_read_ms;
    Stats cold_import_ms;
    Stats cold_tlas_ms;
    Stats cold_e2e_ms;
    Stats cold_direct_e2e_ms;

    Stats built_closest_mrays_s;
    Stats imported_closest_mrays_s;
    Stats built_any_mrays_s;
    Stats imported_any_mrays_s;

    uint64_t parity_total_rays{0ull};
    uint64_t parity_built_hits{0ull};
    uint64_t parity_built_any_hits{0ull};
    uint64_t parity_miss_mismatches{0ull};
    uint64_t parity_inst_mismatches{0ull};
    uint64_t parity_prim_mismatches{0ull};
    uint64_t parity_t_mismatches{0ull};
    uint64_t parity_bary_mismatches{0ull};
    uint64_t parity_occ_mismatches{0ull};

    MemorySnapshot mem_baseline{};
    MemorySnapshot mem_after_built{};
    MemorySnapshot mem_after_imported{};
    uint64_t host_peak_delta_bytes{0ull};
    uint64_t vram_built_delta_bytes{0ull};
    uint64_t vram_imported_delta_bytes{0ull};
};

[[nodiscard]] std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8u);
    for (const char ch : s) {
        switch (ch) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(ch)));
                    out += buf;
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    return out;
}

[[nodiscard]] Stats compute_stats(const std::vector<double> &values) {
    Stats st{};
    if (values.empty()) {
        return st;
    }
    st.samples = values;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    st.min = sorted.front();
    st.max = sorted.back();
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    st.mean = sum / static_cast<double>(sorted.size());
    const size_t n = sorted.size();
    if ((n % 2u) == 1u) {
        st.median = sorted[n / 2u];
    } else {
        st.median = 0.5 * (sorted[n / 2u - 1u] + sorted[n / 2u]);
    }
    const double rank = 0.95 * static_cast<double>(n - 1u);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    const double frac = rank - static_cast<double>(lo);
    st.p95 = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    return st;
}

[[nodiscard]] uint64_t parse_u64_query(Device &device, luisa::string_view key) noexcept {
    if (!device) {
        return 0ull;
    }
    const luisa::string raw = device.impl()->query(key);
    if (raw.empty()) {
        return 0ull;
    }
    uint64_t val = 0ull;
    const char *begin = raw.data();
    const char *end = begin + raw.size();
    while (begin < end && (*begin == ' ' || *begin == '\t')) {
        ++begin;
    }
    auto res = std::from_chars(begin, end, val);
    return (res.ec == std::errc{}) ? val : 0ull;
}

[[nodiscard]] MemorySnapshot capture_memory(Device &device) noexcept {
    MemorySnapshot snap{};
#ifdef _WIN32
    struct Pmc {
        DWORD cb;
        DWORD PageFaultCount;
        SIZE_T PeakWorkingSetSize;
        SIZE_T WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage;
        SIZE_T QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage;
        SIZE_T QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage;
        SIZE_T PeakPagefileUsage;
    };
    using GetPmcFn = BOOL(WINAPI *)(HANDLE, Pmc *, DWORD);
    static GetPmcFn fn = []() noexcept -> GetPmcFn {
        if (HMODULE k32 = GetModuleHandleW(L"kernel32.dll")) {
            if (auto p = reinterpret_cast<GetPmcFn>(GetProcAddress(k32, "K32GetProcessMemoryInfo"))) {
                return p;
            }
        }
        if (HMODULE psapi = LoadLibraryW(L"psapi.dll")) {
            return reinterpret_cast<GetPmcFn>(GetProcAddress(psapi, "GetProcessMemoryInfo"));
        }
        return nullptr;
    }();
    if (fn != nullptr) {
        Pmc pmc{};
        pmc.cb = sizeof(pmc);
        if (fn(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            snap.host_working_set = static_cast<uint64_t>(pmc.WorkingSetSize);
            snap.host_peak_working_set = static_cast<uint64_t>(pmc.PeakWorkingSetSize);
        }
    }
#endif
    snap.device_total_bytes = parse_u64_query(device, "total_memory");
    snap.device_free_bytes = parse_u64_query(device, "free_memory");
    return snap;
}

void write_file_flushed(const std::filesystem::path &path, luisa::span<const std::byte> bytes) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
#ifdef _WIN32
    HANDLE h = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        0u,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    LUISA_ASSERT(h != INVALID_HANDLE_VALUE, "Failed to open archive file '{}' for writing (err={}).", path.string(), GetLastError());
    size_t offset = 0u;
    while (offset < bytes.size()) {
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes.size() - offset, 64ull * 1024ull * 1024ull));
        DWORD written = 0u;
        const BOOL ok = WriteFile(h, bytes.data() + offset, chunk, &written, nullptr);
        LUISA_ASSERT(ok && written == chunk, "WriteFile failed for '{}' (err={}).", path.string(), GetLastError());
        offset += written;
    }
    const BOOL flushed = FlushFileBuffers(h);
    CloseHandle(h);
    LUISA_ASSERT(flushed, "FlushFileBuffers failed for '{}'.", path.string());
#else
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    LUISA_ASSERT(ofs.is_open(), "Failed to open '{}'.", path.string());
    ofs.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ofs.flush();
    LUISA_ASSERT(ofs.good(), "Failed writing '{}'.", path.string());
#endif
}

[[nodiscard]] luisa::vector<std::byte> read_file_warm(const std::filesystem::path &path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    LUISA_ASSERT(ifs.is_open(), "Failed to open '{}' for warm read.", path.string());
    const auto sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    luisa::vector<std::byte> buf(sz);
    if (sz > 0u) {
        ifs.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
        LUISA_ASSERT(ifs.good(), "Warm read failed on '{}'.", path.string());
    }
    return buf;
}

#ifdef _WIN32
struct AlignedFreeDeleter {
    void operator()(std::byte *p) const noexcept {
        if (p != nullptr) {
            _aligned_free(p);
        }
    }
};
using AlignedReadBuffer = std::unique_ptr<std::byte, AlignedFreeDeleter>;

[[nodiscard]] AlignedReadBuffer read_file_cold(const std::filesystem::path &path, size_t &out_size) {
    HANDLE h = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);
    LUISA_ASSERT(h != INVALID_HANDLE_VALUE, "CreateFileW cold read failed for '{}' (err={}).", path.string(), GetLastError());

    LARGE_INTEGER li{};
    const BOOL got_size = GetFileSizeEx(h, &li);
    if (!got_size) {
        const DWORD err = GetLastError();
        CloseHandle(h);
        LUISA_ERROR_WITH_LOCATION("GetFileSizeEx failed for '{}' (err={}).", path.string(), err);
    }
    out_size = static_cast<size_t>(li.QuadPart);
    constexpr size_t kAlign = 65536u;
    const size_t padded = std::max<size_t>(kAlign, ((out_size + kAlign - 1u) / kAlign) * kAlign);
    auto *raw = static_cast<std::byte *>(_aligned_malloc(padded, kAlign));
    LUISA_ASSERT(raw != nullptr, "_aligned_malloc({}) failed.", padded);
    AlignedReadBuffer buf(raw);

    size_t total = 0u;
    while (total < out_size) {
        const size_t rem = out_size - total;
        const size_t max_chunk = 32ull * 1024ull * 1024ull;
        const DWORD req = static_cast<DWORD>(std::min<size_t>(
            max_chunk,
            ((rem + kAlign - 1u) / kAlign) * kAlign));
        DWORD got = 0u;
        if (!ReadFile(h, buf.get() + total, req, &got, nullptr)) {
            const DWORD err = GetLastError();
            CloseHandle(h);
            LUISA_ERROR_WITH_LOCATION("ReadFile cold read failed for '{}' (err={}).", path.string(), err);
        }
        if (got == 0u) {
            break;
        }
        total += got;
    }
    CloseHandle(h);
    LUISA_ASSERT(total == out_size, "Cold read size mismatch on '{}': expected {}, got {}.", path.string(), out_size, total);
    return buf;
}
#endif

[[nodiscard]] MeshData make_grid_mesh(std::string name, uint32_t side) {
    MeshData mesh;
    mesh.name = std::move(name);
    mesh.source = luisa::format("synthetic_grid_{}x{}", side, side);
    mesh.vertices.reserve(static_cast<size_t>(side) * side);
    mesh.triangles.reserve(static_cast<size_t>(side - 1u) * (side - 1u) * 2u);

    for (uint32_t j = 0u; j < side; ++j) {
        const float v = static_cast<float>(j) / static_cast<float>(side - 1u) * 2.0f - 1.0f;
        for (uint32_t i = 0u; i < side; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(side - 1u) * 2.0f - 1.0f;
            const float z = 0.15f * std::sin(u * 12.0f) * std::cos(v * 10.0f) +
                            0.05f * std::sin((u + v) * 23.0f);
            mesh.vertices.push_back(make_float3(u, v, z));
        }
    }
    for (uint32_t j = 0u; j < side - 1u; ++j) {
        for (uint32_t i = 0u; i < side - 1u; ++i) {
            const uint32_t i0 = j * side + i;
            const uint32_t i1 = i0 + 1u;
            const uint32_t i2 = i0 + side;
            const uint32_t i3 = i2 + 1u;
            mesh.triangles.push_back(Triangle{i0, i1, i3});
            mesh.triangles.push_back(Triangle{i0, i3, i2});
        }
    }
    mesh.content_hash = accel_archive::compute_static_triangle_geometry_hash(
        mesh.vertices.data(),
        mesh.vertices.size(),
        sizeof(float3),
        mesh.triangles.data(),
        mesh.triangles.size());
    return mesh;
}

[[nodiscard]] MeshData load_obj_mesh(const std::filesystem::path &path) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;
    tinyobj::ObjReader reader;
    LUISA_ASSERT(reader.ParseFromFile(path.string(), config),
                 "Failed to parse OBJ '{}': {}", path.string(), reader.Error());
    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();
    LUISA_ASSERT(!attrib.vertices.empty(), "OBJ '{}' has no vertices.", path.string());

    const size_t raw_vcount = attrib.vertices.size() / 3u;
    MeshData mesh;
    mesh.name = "pavilion_obj";
    mesh.source = path.string();
    mesh.vertices.resize(raw_vcount);

    float3 bmin = make_float3(1e30f);
    float3 bmax = make_float3(-1e30f);
    for (size_t i = 0u; i < raw_vcount; ++i) {
        const float3 p = make_float3(
            static_cast<float>(attrib.vertices[3u * i + 0u]),
            static_cast<float>(attrib.vertices[3u * i + 1u]),
            static_cast<float>(attrib.vertices[3u * i + 2u]));
        mesh.vertices[i] = p;
        bmin = min(bmin, p);
        bmax = max(bmax, p);
    }
    const float3 center = 0.5f * (bmin + bmax);
    const float3 extent = bmax - bmin;
    const float max_ext = std::max({extent.x, extent.y, extent.z, 1e-6f});
    const float inv_scale = 2.0f / max_ext;
    for (auto &p : mesh.vertices) {
        p = (p - center) * inv_scale;
    }

    for (const auto &shape : shapes) {
        size_t idx_offset = 0u;
        for (const auto fv : shape.mesh.num_face_vertices) {
            if (fv == 3u) {
                const int a = shape.mesh.indices[idx_offset + 0u].vertex_index;
                const int b = shape.mesh.indices[idx_offset + 1u].vertex_index;
                const int c = shape.mesh.indices[idx_offset + 2u].vertex_index;
                if (a >= 0 && b >= 0 && c >= 0 &&
                    static_cast<size_t>(a) < raw_vcount &&
                    static_cast<size_t>(b) < raw_vcount &&
                    static_cast<size_t>(c) < raw_vcount &&
                    a != b && b != c && a != c) {
                    mesh.triangles.push_back(Triangle{
                        static_cast<uint32_t>(a),
                        static_cast<uint32_t>(b),
                        static_cast<uint32_t>(c)});
                }
            }
            idx_offset += fv;
        }
    }
    LUISA_ASSERT(!mesh.triangles.empty(), "OBJ '{}' yielded 0 valid triangles.", path.string());
    mesh.content_hash = accel_archive::compute_static_triangle_geometry_hash(
        mesh.vertices.data(),
        mesh.vertices.size(),
        sizeof(float3),
        mesh.triangles.data(),
        mesh.triangles.size());
    return mesh;
}

[[nodiscard]] std::array<AccelArchiveInstanceDesc, 4u> make_four_instances() noexcept {
    std::array<AccelArchiveInstanceDesc, 4u> instances{};
    instances[0].blas_index = 0u;
    instances[0].transform = translation(make_float3(-1.45f, -1.45f, 0.0f)) *
                             rotation(make_float3(0.0f, 0.0f, 1.0f), 0.15f) *
                             scaling(1.0f);
    instances[0].user_id = 100u;

    instances[1].blas_index = 0u;
    instances[1].transform = translation(make_float3(1.45f, -1.45f, 0.25f)) *
                             rotation(make_float3(0.0f, 1.0f, 0.0f), 0.25f) *
                             scaling(0.95f);
    instances[1].user_id = 101u;

    instances[2].blas_index = 0u;
    instances[2].transform = translation(make_float3(-1.45f, 1.45f, -0.25f)) *
                             rotation(make_float3(1.0f, 0.0f, 0.0f), -0.20f) *
                             scaling(0.85f);
    instances[2].user_id = 102u;

    instances[3].blas_index = 0u;
    instances[3].transform = translation(make_float3(1.45f, 1.45f, 0.10f)) *
                             rotation(normalize(make_float3(0.3f, 0.7f, 0.6f)), 0.30f) *
                             scaling(1.05f);
    instances[3].user_id = 103u;
    return instances;
}

[[nodiscard]] std::string make_ready_string(AccelArchiveReadiness r) {
    switch (r) {
        case AccelArchiveReadiness::READY_IMMEDIATELY: return "READY_IMMEDIATELY";
        case AccelArchiveReadiness::READY_ON_STREAM: return "READY_ON_STREAM";
        default: return "NONE";
    }
}

void write_stats_json(std::ostream &os, const Stats &s, int indent) {
    std::string pad(static_cast<size_t>(indent), ' ');
    os << std::fixed << std::setprecision(6);
    os << "{\n"
       << pad << "  \"median\": " << s.median << ",\n"
       << pad << "  \"mean\": " << s.mean << ",\n"
       << pad << "  \"min\": " << s.min << ",\n"
       << pad << "  \"max\": " << s.max << ",\n"
       << pad << "  \"p95\": " << s.p95 << ",\n"
       << pad << "  \"samples\": [";
    for (size_t i = 0u; i < s.samples.size(); ++i) {
        if (i > 0u) os << ", ";
        os << s.samples[i];
    }
    os << "]\n"
       << pad << "}";
}

void write_mem_json(std::ostream &os, const MemorySnapshot &m, int indent) {
    std::string pad(static_cast<size_t>(indent), ' ');
    os << "{\n"
       << pad << "  \"host_working_set_bytes\": " << m.host_working_set << ",\n"
       << pad << "  \"host_peak_working_set_bytes\": " << m.host_peak_working_set << ",\n"
       << pad << "  \"device_total_bytes\": " << m.device_total_bytes << ",\n"
       << pad << "  \"device_free_bytes\": " << m.device_free_bytes << "\n"
       << pad << "}";
}

void write_full_json(
    const std::filesystem::path &path,
    const BenchConfig &cfg,
    const std::string &device_name,
    const AccelArchiveCapabilities &caps,
    bool ext_present,
    const std::vector<ModeBenchResult> &results) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    LUISA_ASSERT(ofs.is_open(), "Failed to open JSON output '{}'.", path.string());
    ofs << "{\n"
        << "  \"backend\": \"" << json_escape(cfg.backend) << "\",\n"
        << "  \"device_name\": \"" << json_escape(device_name) << "\",\n"
        << "  \"mode\": \"" << json_escape(cfg.mode) << "\",\n"
        << "  \"repeats\": " << cfg.repeats << ",\n"
        << "  \"warmup\": " << cfg.warmup << ",\n"
        << "  \"ray_res\": " << cfg.ray_res << ",\n"
        << "  \"trace_iters\": " << cfg.trace_iters << ",\n"
        << "  \"extension_present\": " << (ext_present ? "true" : "false") << ",\n"
        << "  \"capabilities\": {\n"
        << "    \"supported\": " << (caps.supported ? "true" : "false") << ",\n"
        << "    \"static_triangle_blas\": " << (caps.static_triangle_blas ? "true" : "false") << ",\n"
        << "    \"supports_compacted_export\": " << (caps.supports_compacted_export ? "true" : "false") << ",\n"
        << "    \"requires_geometry_rebind\": " << (caps.requires_geometry_rebind ? "true" : "false") << ",\n"
        << "    \"import_is_async_on_stream\": " << (caps.import_is_async_on_stream ? "true" : "false") << ",\n"
        << "    \"supports_pinned_staging\": " << (caps.supports_pinned_staging ? "true" : "false") << ",\n"
        << "    \"supports_direct_file_import\": " << (caps.supports_direct_file_import ? "true" : "false") << ",\n"
        << "    \"backend_kind\": \"" << json_escape(to_string(caps.backend_kind)) << "\",\n"
        << "    \"envelope_version\": " << caps.envelope_version << ",\n"
        << "    \"device_fingerprint\": \"" << json_escape(caps.device_fingerprint) << "\",\n"
        << "    \"driver_fingerprint\": \"" << json_escape(caps.driver_fingerprint) << "\",\n"
        << "    \"layout_info\": \"" << json_escape(caps.layout_info) << "\"\n"
        << "  },\n"
        << "  \"cases\": [\n";
    for (size_t i = 0u; i < results.size(); ++i) {
        const auto &r = results[i];
        ofs << "    {\n"
            << "      \"mesh_name\": \"" << json_escape(r.mesh_name) << "\",\n"
            << "      \"mesh_source\": \"" << json_escape(r.mesh_source) << "\",\n"
            << "      \"vertex_count\": " << r.vertex_count << ",\n"
            << "      \"triangle_count\": " << r.triangle_count << ",\n"
            << "      \"allow_compaction\": " << (r.allow_compaction ? "true" : "false") << ",\n"
            << "      \"archive_attempted\": " << (r.archive_attempted ? "true" : "false") << ",\n"
            << "      \"archive_succeeded\": " << (r.archive_succeeded ? "true" : "false") << ",\n"
            << "      \"is_compacted\": " << (r.is_compacted ? "true" : "false") << ",\n"
            << "      \"readiness\": \"" << json_escape(r.readiness) << "\",\n"
            << "      \"status\": \"" << json_escape(r.status) << "\",\n"
            << "      \"message\": \"" << json_escape(r.message) << "\",\n"
            << "      \"archive_bytes\": " << r.archive_bytes << ",\n"
            << "      \"payload_bytes\": " << r.payload_bytes << ",\n"
            << "      \"unpacked_as_bytes\": " << r.unpacked_as_bytes << ",\n"
            << "      \"blas_build_ms\": ";
        write_stats_json(ofs, r.blas_build_ms, 6);
        ofs << ",\n      \"tlas_build_ms\": ";
        write_stats_json(ofs, r.tlas_build_ms, 6);
        ofs << ",\n      \"export_ms\": ";
        write_stats_json(ofs, r.export_ms, 6);
        ofs << ",\n      \"relocate_ms\": ";
        write_stats_json(ofs, r.relocate_ms, 6);
        ofs << ",\n      \"relocate_pinned_ms\": ";
        write_stats_json(ofs, r.relocate_pinned_ms, 6);
        ofs << ",\n      \"relocate_fast_ms\": ";
        write_stats_json(ofs, r.relocate_fast_ms, 6);
        ofs << ",\n      \"warm_read_ms\": ";
        write_stats_json(ofs, r.warm_read_ms, 6);
        ofs << ",\n      \"warm_import_ms\": ";
        write_stats_json(ofs, r.warm_import_ms, 6);
        ofs << ",\n      \"warm_tlas_ms\": ";
        write_stats_json(ofs, r.warm_tlas_ms, 6);
        ofs << ",\n      \"warm_e2e_ms\": ";
        write_stats_json(ofs, r.warm_e2e_ms, 6);
        ofs << ",\n      \"warm_direct_e2e_ms\": ";
        write_stats_json(ofs, r.warm_direct_e2e_ms, 6);
        ofs << ",\n      \"cold_read_ms\": ";
        write_stats_json(ofs, r.cold_read_ms, 6);
        ofs << ",\n      \"cold_import_ms\": ";
        write_stats_json(ofs, r.cold_import_ms, 6);
        ofs << ",\n      \"cold_tlas_ms\": ";
        write_stats_json(ofs, r.cold_tlas_ms, 6);
        ofs << ",\n      \"cold_e2e_ms\": ";
        write_stats_json(ofs, r.cold_e2e_ms, 6);
        ofs << ",\n      \"cold_direct_e2e_ms\": ";
        write_stats_json(ofs, r.cold_direct_e2e_ms, 6);
        ofs << ",\n      \"built_closest_mrays_s\": ";
        write_stats_json(ofs, r.built_closest_mrays_s, 6);
        ofs << ",\n      \"imported_closest_mrays_s\": ";
        write_stats_json(ofs, r.imported_closest_mrays_s, 6);
        ofs << ",\n      \"built_any_mrays_s\": ";
        write_stats_json(ofs, r.built_any_mrays_s, 6);
        ofs << ",\n      \"imported_any_mrays_s\": ";
        write_stats_json(ofs, r.imported_any_mrays_s, 6);
        ofs << ",\n      \"parity\": {\n"
            << "        \"total_rays\": " << r.parity_total_rays << ",\n"
            << "        \"built_hits\": " << r.parity_built_hits << ",\n"
            << "        \"built_any_hits\": " << r.parity_built_any_hits << ",\n"
            << "        \"miss_mismatches\": " << r.parity_miss_mismatches << ",\n"
            << "        \"inst_mismatches\": " << r.parity_inst_mismatches << ",\n"
            << "        \"prim_mismatches\": " << r.parity_prim_mismatches << ",\n"
            << "        \"t_mismatches\": " << r.parity_t_mismatches << ",\n"
            << "        \"bary_mismatches\": " << r.parity_bary_mismatches << ",\n"
            << "        \"occ_mismatches\": " << r.parity_occ_mismatches << "\n"
            << "      },\n"
            << "      \"memory\": {\n"
            << "        \"baseline\": ";
        write_mem_json(ofs, r.mem_baseline, 8);
        ofs << ",\n        \"after_built\": ";
        write_mem_json(ofs, r.mem_after_built, 8);
        ofs << ",\n        \"after_imported\": ";
        write_mem_json(ofs, r.mem_after_imported, 8);
        ofs << ",\n        \"host_peak_delta_bytes\": " << r.host_peak_delta_bytes << ",\n"
            << "        \"vram_built_delta_bytes\": " << r.vram_built_delta_bytes << ",\n"
            << "        \"vram_imported_delta_bytes\": " << r.vram_imported_delta_bytes << "\n"
            << "      }\n"
            << "    }" << (i + 1u < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
    ofs.flush();
    LUISA_ASSERT(ofs.good(), "Failed writing JSON output '{}'.", path.string());
}

[[nodiscard]] BenchConfig parse_args(int argc, char *argv[]) {
    BenchConfig cfg;
    cfg.out_dir = std::filesystem::temp_directory_path() / "luisa_accel_archive_bench";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto need_val = [&](std::string_view flag) -> std::string_view {
            LUISA_ASSERT(i + 1 < argc, "Missing value for '{}'.", flag);
            return argv[++i];
        };
        if (arg == "--backend") {
            cfg.backend = std::string(need_val(arg));
        } else if (arg == "--mode") {
            cfg.mode = std::string(need_val(arg));
        } else if (arg == "--scales") {
            cfg.scales.clear();
            std::string raw(need_val(arg));
            std::stringstream ss(raw);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    cfg.scales.push_back(item);
                }
            }
        } else if (arg == "--obj") {
            cfg.obj_path = std::string(need_val(arg));
        } else if (arg == "--repeats") {
            cfg.repeats = static_cast<uint32_t>(std::stoul(std::string(need_val(arg))));
        } else if (arg == "--warmup") {
            cfg.warmup = static_cast<uint32_t>(std::stoul(std::string(need_val(arg))));
        } else if (arg == "--ray-res") {
            cfg.ray_res = static_cast<uint32_t>(std::stoul(std::string(need_val(arg))));
        } else if (arg == "--trace-iters") {
            cfg.trace_iters = static_cast<uint32_t>(std::stoul(std::string(need_val(arg))));
        } else if (arg == "--out-dir") {
            cfg.out_dir = std::string(need_val(arg));
        } else if (arg == "--json") {
            cfg.json_path = std::string(need_val(arg));
        } else {
            LUISA_ERROR_WITH_LOCATION("Unknown argument '{}'.", arg);
        }
    }
    LUISA_ASSERT(cfg.repeats >= 1u && cfg.ray_res >= 1u && cfg.trace_iters >= 1u, "Invalid repeat/ray parameters.");
    return cfg;
}

}// namespace

int main(int argc, char *argv[]) {
    const BenchConfig cfg = parse_args(argc, argv);

    Context context{argv[0]};
    Device device = context.create_device(cfg.backend);
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    const luisa::string device_name_l = device.impl()->query("device_name");
    const std::string device_name(device_name_l.data(), device_name_l.size());

    auto *ext = device.extension<AccelArchiveExt>();
    AccelArchiveCapabilities caps{};
    if (ext != nullptr) {
        caps = ext->capabilities();
    }
    const bool archive_usable = (ext != nullptr) && caps.supported && (cfg.mode != "rebuild_only");

    std::cout << "[BENCH_INIT] backend=" << cfg.backend
              << " device=\"" << device_name << "\""
              << " ext_present=" << (ext != nullptr ? 1 : 0)
              << " supported=" << (caps.supported ? 1 : 0)
              << " layout=" << caps.layout_info
              << " driver=" << caps.driver_fingerprint
              << " repeats=" << cfg.repeats
              << " warmup=" << cfg.warmup
              << " ray_res=" << cfg.ray_res
              << " trace_iters=" << cfg.trace_iters << "\n";

    // Compile benchmark & parity shaders once.
    Kernel2D closest_parity_kernel = [](AccelVar accel,
                                        BufferVar<float4> out_hits,
                                        BufferVar<uint2> out_ids,
                                        UInt res) noexcept {
        set_block_size(16u, 16u, 1u);
        auto xy = dispatch_id().xy();
        auto idx = xy.y * res + xy.x;
        auto fx = (cast<float>(xy.x) + 0.5f) / cast<float>(res) * 2.0f - 1.0f;
        auto fy = (cast<float>(xy.y) + 0.5f) / cast<float>(res) * 2.0f - 1.0f;
        auto origin = make_float3(fx * 0.5f, fy * 0.5f - 0.25f, 7.5f);
        auto dir = normalize(make_float3(fx * 3.8f, fy * 3.8f, -6.5f));
        auto ray = make_ray(origin, dir, 1e-4f, 1e4f);
        auto hit = accel.intersect(ray, {});
        out_hits.write(idx, make_float4(hit.bary.x, hit.bary.y, hit.committed_ray_t, ite(hit->miss(), 0.0f, 1.0f)));
        out_ids.write(idx, make_uint2(hit.inst, hit.prim));
    };

    Kernel2D any_parity_kernel = [](AccelVar accel,
                                    BufferUInt out_occ,
                                    UInt res) noexcept {
        set_block_size(16u, 16u, 1u);
        auto xy = dispatch_id().xy();
        auto idx = xy.y * res + xy.x;
        auto fx = (cast<float>(xy.x) + 0.5f) / cast<float>(res) * 2.0f - 1.0f;
        auto fy = (cast<float>(xy.y) + 0.5f) / cast<float>(res) * 2.0f - 1.0f;
        auto origin = make_float3(fx * 0.5f, fy * 0.5f - 0.25f, 7.5f);
        auto dir = normalize(make_float3(fx * 3.8f, fy * 3.8f, -6.5f));
        auto ray = make_ray(origin, dir, 1e-4f, 1e4f);
        auto hit = accel.intersect_any(ray, {});
        out_occ.write(idx, ite(hit, 1u, 0u));
    };

    Kernel2D closest_bench_kernel = [](AccelVar accel,
                                       BufferUInt out_accum,
                                       UInt res,
                                       UInt iters) noexcept {
        set_block_size(16u, 16u, 1u);
        auto xy = dispatch_id().xy();
        auto idx = xy.y * res + xy.x;
        Var<uint> acc = 0u;
        $for (it, iters) {
            auto jitter = (cast<float>(it) + 0.5f) / cast<float>(iters);
            auto fx = (cast<float>(xy.x) + jitter) / cast<float>(res) * 2.0f - 1.0f;
            auto fy = (cast<float>(xy.y) + jitter) / cast<float>(res) * 2.0f - 1.0f;
            auto origin = make_float3(fx * 0.5f, fy * 0.5f - 0.25f, 7.5f);
            auto dir = normalize(make_float3(fx * 3.8f, fy * 3.8f, -6.5f));
            auto ray = make_ray(origin, dir, 1e-4f, 1e4f);
            auto hit = accel.intersect(ray, {});
            acc += ite(hit->miss(), 0u, (hit.inst + 1u) ^ (hit.prim & 0xffffu));
        };
        out_accum.write(idx, acc);
    };

    Kernel2D any_bench_kernel = [](AccelVar accel,
                                   BufferUInt out_accum,
                                   UInt res,
                                   UInt iters) noexcept {
        set_block_size(16u, 16u, 1u);
        auto xy = dispatch_id().xy();
        auto idx = xy.y * res + xy.x;
        Var<uint> acc = 0u;
        $for (it, iters) {
            auto jitter = (cast<float>(it) + 0.5f) / cast<float>(iters);
            auto fx = (cast<float>(xy.x) + jitter) / cast<float>(res) * 2.0f - 1.0f;
            auto fy = (cast<float>(xy.y) + jitter) / cast<float>(res) * 2.0f - 1.0f;
            auto origin = make_float3(fx * 0.5f, fy * 0.5f - 0.25f, 7.5f);
            auto dir = normalize(make_float3(fx * 3.8f, fy * 3.8f, -6.5f));
            auto ray = make_ray(origin, dir, 1e-4f, 1e4f);
            auto hit = accel.intersect_any(ray, {});
            acc += ite(hit, 1u, 0u);
        };
        out_accum.write(idx, acc);
    };

    auto closest_parity_shader = device.compile(closest_parity_kernel);
    auto any_parity_shader = device.compile(any_parity_kernel);
    auto closest_bench_shader = device.compile(closest_bench_kernel);
    auto any_bench_shader = device.compile(any_bench_kernel);

    const size_t pixel_count = static_cast<size_t>(cfg.ray_res) * cfg.ray_res;
    auto hit_f_buf = device.create_buffer<float4>(pixel_count);
    auto hit_id_buf = device.create_buffer<uint2>(pixel_count);
    auto occ_buf = device.create_buffer<uint>(pixel_count);
    auto bench_accum_buf = device.create_buffer<uint>(pixel_count);

    std::vector<float4> built_hit_f(pixel_count);
    std::vector<uint2> built_hit_id(pixel_count);
    std::vector<uint> built_occ(pixel_count);
    std::vector<float4> imp_hit_f(pixel_count);
    std::vector<uint2> imp_hit_id(pixel_count);
    std::vector<uint> imp_occ(pixel_count);

    std::vector<MeshData> meshes;
    for (const auto &sc : cfg.scales) {
        if (sc == "10k") {
            meshes.push_back(make_grid_mesh("10k", 71u));
        } else if (sc == "100k") {
            meshes.push_back(make_grid_mesh("100k", 224u));
        } else if (sc == "500k") {
            meshes.push_back(make_grid_mesh("500k", 500u));
        } else if (sc == "2m") {
            meshes.push_back(make_grid_mesh("2m", 1000u));
        } else {
            LUISA_ERROR_WITH_LOCATION("Unknown synthetic scale '{}'.", sc);
        }
    }
    if (!cfg.obj_path.empty()) {
        meshes.push_back(load_obj_mesh(cfg.obj_path));
    }

    std::vector<bool> compaction_modes;
    if (cfg.mode == "both") {
        compaction_modes = {false, true};
    } else if (cfg.mode == "uncompacted" || cfg.mode == "rebuild_only") {
        compaction_modes = {false};
    } else if (cfg.mode == "compacted") {
        compaction_modes = {true};
    } else {
        LUISA_ERROR_WITH_LOCATION("Unknown --mode '{}'.", cfg.mode);
    }

    const auto instances = make_four_instances();
    const double total_bench_mrays = static_cast<double>(pixel_count) * cfg.trace_iters / 1e6;

    std::vector<ModeBenchResult> all_results;

    for (const auto &m : meshes) {
        auto vb = device.create_buffer<float3>(m.vertices.size());
        auto tb = device.create_buffer<Triangle>(m.triangles.size());
        stream << vb.copy_from(m.vertices.data())
               << tb.copy_from(m.triangles.data())
               << synchronize();

        for (const bool compact : compaction_modes) {
            ModeBenchResult rec{};
            rec.mesh_name = m.name;
            rec.mesh_source = m.source;
            rec.vertex_count = m.vertices.size();
            rec.triangle_count = m.triangles.size();
            rec.allow_compaction = compact;
            rec.archive_attempted = archive_usable;
            rec.mem_baseline = capture_memory(device);

            AccelOption option{};
            option.allow_compaction = compact;
            option.allow_update = false;

            const AccelOption tlas_option{};
            const uint32_t total_iters = cfg.warmup + cfg.repeats;

            std::vector<double> blas_ms;
            std::vector<double> tlas_ms;
            std::vector<double> export_ms;
            std::vector<double> relocate_ms;
            std::vector<double> relocate_pinned_ms;
            std::vector<double> relocate_fast_ms;
            std::vector<double> warm_read_ms;
            std::vector<double> warm_imp_ms;
            std::vector<double> warm_tlas_ms;
            std::vector<double> warm_e2e_ms;
            std::vector<double> warm_direct_e2e_ms;
            std::vector<double> cold_read_ms;
            std::vector<double> cold_imp_ms;
            std::vector<double> cold_tlas_ms;
            std::vector<double> cold_e2e_ms;
            std::vector<double> cold_direct_e2e_ms;
            std::vector<double> built_closest_mrays;
            std::vector<double> imp_closest_mrays;
            std::vector<double> built_any_mrays;
            std::vector<double> imp_any_mrays;

            luisa::vector<std::byte> ref_archive;
            const std::filesystem::path arc_path =
                cfg.out_dir / luisa::format("{}_{}_c{}.lcas", cfg.backend, m.name, compact ? 1 : 0);

            for (uint32_t it = 0u; it < total_iters; ++it) {
                const bool record = (it >= cfg.warmup);
                const bool is_last = (it + 1u == total_iters);

                Mesh built_mesh = device.create_mesh(vb, tb, option);
                Clock c_blas;
                stream << built_mesh.build() << synchronize();
                const double t_b = c_blas.toc();

                const Mesh *blas_ptr = &built_mesh;
                Accel built_tlas = device.create_accel(tlas_option);
                populate_tlas_from_descriptors(built_tlas, luisa::span{&blas_ptr, 1u}, instances);
                Clock c_tlas;
                stream << built_tlas.build(Accel::BuildRequest::FORCE_BUILD) << synchronize();
                const double t_t = c_tlas.toc();

                if (record) {
                    blas_ms.push_back(t_b);
                    tlas_ms.push_back(t_t);
                }

                if (archive_usable) {
                    luisa::vector<std::byte> tmp_archive;
                    Clock c_exp;
                    auto exp_res = export_mesh_archive(
                        ext, stream, built_mesh, option, m.content_hash, tmp_archive);
                    stream.synchronize();
                    const double t_e = c_exp.toc();
                    LUISA_ASSERT(static_cast<bool>(exp_res),
                                 "export_mesh_archive failed for {} c={}: status={} msg={}",
                                 m.name, compact ? 1 : 0, to_string(exp_res.status), exp_res.message);
                    if (record) {
                        export_ms.push_back(t_e);
                    }
                    if (is_last) {
                        ref_archive = std::move(tmp_archive);
                        rec.is_compacted = exp_res.is_compacted;
                        rec.payload_bytes = exp_res.serialized_payload_size;
                        rec.unpacked_as_bytes = exp_res.unpacked_as_size;
                        rec.archive_bytes = ref_archive.size();
                        write_file_flushed(arc_path, ref_archive);
                    }
                }

                if (is_last) {
                    rec.mem_after_built = capture_memory(device);
                    if (archive_usable) {
                        stream << closest_parity_shader(built_tlas, hit_f_buf, hit_id_buf, cfg.ray_res).dispatch(cfg.ray_res, cfg.ray_res)
                               << any_parity_shader(built_tlas, occ_buf, cfg.ray_res).dispatch(cfg.ray_res, cfg.ray_res)
                               << hit_f_buf.copy_to(built_hit_f.data())
                               << hit_id_buf.copy_to(built_hit_id.data())
                               << occ_buf.copy_to(built_occ.data())
                               << synchronize();
                        rec.parity_total_rays = pixel_count;
                        for (size_t p = 0u; p < pixel_count; ++p) {
                            if (built_hit_f[p].w > 0.5f) {
                                ++rec.parity_built_hits;
                            }
                            if (built_occ[p] != 0u) {
                                ++rec.parity_built_any_hits;
                            }
                        }
                    }

                    for (uint32_t r = 0u; r < total_iters; ++r) {
                        Clock c_rc;
                        stream << closest_bench_shader(built_tlas, bench_accum_buf, cfg.ray_res, cfg.trace_iters)
                                      .dispatch(cfg.ray_res, cfg.ray_res)
                               << synchronize();
                        const double ms_c = c_rc.toc();

                        Clock c_ra;
                        stream << any_bench_shader(built_tlas, bench_accum_buf, cfg.ray_res, cfg.trace_iters)
                                      .dispatch(cfg.ray_res, cfg.ray_res)
                               << synchronize();
                        const double ms_a = c_ra.toc();

                        if (r >= cfg.warmup) {
                            built_closest_mrays.push_back(total_bench_mrays / (ms_c * 1e-3));
                            built_any_mrays.push_back(total_bench_mrays / (ms_a * 1e-3));
                        }
                    }
                }

                built_tlas = Accel{};
                built_mesh = Mesh{};
                stream.synchronize();
            }

            if (archive_usable) {
                rec.archive_succeeded = true;
                rec.status = "OK";
                AccelArchiveMetadata vmeta{};
                AccelArchiveMeshDesc expected_desc{};
                expected_desc.vertex_count = m.vertices.size();
                expected_desc.triangle_count = m.triangles.size();
                expected_desc.vertex_stride = sizeof(float3);
                expected_desc.option = option;
                expected_desc.geometry_content_hash = m.content_hash;
                auto vres = ext->validate_mesh_archive(ref_archive, expected_desc, &vmeta);
                LUISA_ASSERT(static_cast<bool>(vres), "validate_mesh_archive failed: {}", vres.message);
                rec.payload_bytes = vmeta.payload_size;
                rec.unpacked_as_bytes = vmeta.unpacked_as_size;
                rec.is_compacted = vmeta.is_compacted;

                // 1. In-memory relocate / deserialize
                for (uint32_t it = 0u; it < total_iters; ++it) {
                    Mesh imp_mesh = device.create_mesh(vb, tb, option);
                    Clock c_imp;
                    auto ires = import_mesh_archive(
                        ext, stream, imp_mesh, option, m.content_hash, ref_archive);
                    if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                        stream.synchronize();
                    }
                    const double t_imp = c_imp.toc();
                    LUISA_ASSERT(static_cast<bool>(ires), "import_mesh_archive failed: {}", ires.message);
                    if (it == cfg.warmup) {
                        rec.readiness = make_ready_string(ires.readiness);
                        rec.message = std::string(ires.message.data(), ires.message.size());
                    }
                    if (it >= cfg.warmup) {
                        relocate_ms.push_back(t_imp);
                    }
                    imp_mesh = Mesh{};
                    stream.synchronize();
                }

                // 1b. Persistent staging zero-copy relocate (verified & unverified)
                if (auto staging = ext->acquire_staging_buffer(ref_archive.size());
                    staging.mapped_data != nullptr && staging.capacity >= ref_archive.size()) {
                    std::memcpy(staging.mapped_data, ref_archive.data(), ref_archive.size());
                    const luisa::span<const std::byte> staged_span{staging.mapped_data, ref_archive.size()};
                    for (uint32_t it = 0u; it < total_iters; ++it) {
                        Mesh imp_mesh = device.create_mesh(vb, tb, option);
                        Clock c_imp;
                        auto ires = import_mesh_archive_with_options(
                            ext, stream, imp_mesh, option, m.content_hash, staged_span,
                            AccelArchiveImportOptions{.verify_payload_checksum = true});
                        if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                            stream.synchronize();
                        }
                        const double t_imp = c_imp.toc();
                        LUISA_ASSERT(static_cast<bool>(ires), "pinned relocate failed: {}", ires.message);
                        if (it >= cfg.warmup) {
                            relocate_pinned_ms.push_back(t_imp);
                        }
                        imp_mesh = Mesh{};
                        stream.synchronize();
                    }
                    for (uint32_t it = 0u; it < total_iters; ++it) {
                        Mesh imp_mesh = device.create_mesh(vb, tb, option);
                        Clock c_imp;
                        auto ires = import_mesh_archive_with_options(
                            ext, stream, imp_mesh, option, m.content_hash, staged_span,
                            AccelArchiveImportOptions{.verify_payload_checksum = false});
                        if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                            stream.synchronize();
                        }
                        const double t_imp = c_imp.toc();
                        LUISA_ASSERT(static_cast<bool>(ires), "fast relocate failed: {}", ires.message);
                        if (it >= cfg.warmup) {
                            relocate_fast_ms.push_back(t_imp);
                        }
                        imp_mesh = Mesh{};
                        stream.synchronize();
                    }
                    ext->release_staging_buffer(std::move(staging));
                }

                // 2. Warm disk load -> import -> TLAS
                for (uint32_t it = 0u; it < total_iters; ++it) {
                    Clock c_e2e;
                    Clock c_rd;
                    auto disk_bytes = read_file_warm(arc_path);
                    const double t_rd = c_rd.toc();

                    Mesh imp_mesh = device.create_mesh(vb, tb, option);
                    Clock c_imp;
                    auto ires = import_mesh_archive(
                        ext, stream, imp_mesh, option, m.content_hash, disk_bytes);
                    if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                        stream.synchronize();
                    }
                    const double t_imp = c_imp.toc();
                    LUISA_ASSERT(static_cast<bool>(ires), "warm import failed: {}", ires.message);

                    const Mesh *imp_ptr = &imp_mesh;
                    Accel imp_tlas = device.create_accel(tlas_option);
                    populate_tlas_from_descriptors(imp_tlas, luisa::span{&imp_ptr, 1u}, instances);
                    Clock c_tl;
                    stream << imp_tlas.build(Accel::BuildRequest::FORCE_BUILD) << synchronize();
                    const double t_tl = c_tl.toc();
                    const double t_tot = c_e2e.toc();

                    if (it >= cfg.warmup) {
                        warm_read_ms.push_back(t_rd);
                        warm_imp_ms.push_back(t_imp);
                        warm_tlas_ms.push_back(t_tl);
                        warm_e2e_ms.push_back(t_tot);
                    }

                    imp_tlas = Accel{};
                    imp_mesh = Mesh{};
                    stream.synchronize();
                }

                // 3. Cold unbuffered disk load -> import -> TLAS + parity + throughput
                for (uint32_t it = 0u; it < total_iters; ++it) {
                    const bool is_last = (it + 1u == total_iters);
                    Clock c_e2e;
                    Clock c_rd;
                    size_t cold_size = 0u;
#ifdef _WIN32
                    auto cold_buf = read_file_cold(arc_path, cold_size);
                    luisa::span<const std::byte> cold_span{cold_buf.get(), cold_size};
#else
                    auto cold_buf = read_file_warm(arc_path);
                    cold_size = cold_buf.size();
                    luisa::span<const std::byte> cold_span{cold_buf};
#endif
                    const double t_rd = c_rd.toc();

                    Mesh imp_mesh = device.create_mesh(vb, tb, option);
                    Clock c_imp;
                    auto ires = import_mesh_archive(
                        ext, stream, imp_mesh, option, m.content_hash, cold_span);
                    if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                        stream.synchronize();
                    }
                    const double t_imp = c_imp.toc();
                    LUISA_ASSERT(static_cast<bool>(ires), "cold import failed: {}", ires.message);

                    const Mesh *imp_ptr = &imp_mesh;
                    Accel imp_tlas = device.create_accel(tlas_option);
                    populate_tlas_from_descriptors(imp_tlas, luisa::span{&imp_ptr, 1u}, instances);
                    Clock c_tl;
                    stream << imp_tlas.build(Accel::BuildRequest::FORCE_BUILD) << synchronize();
                    const double t_tl = c_tl.toc();
                    const double t_tot = c_e2e.toc();

                    if (it >= cfg.warmup) {
                        cold_read_ms.push_back(t_rd);
                        cold_imp_ms.push_back(t_imp);
                        cold_tlas_ms.push_back(t_tl);
                        cold_e2e_ms.push_back(t_tot);
                    }

                    if (is_last) {
                        rec.mem_after_imported = capture_memory(device);
                        stream << closest_parity_shader(imp_tlas, hit_f_buf, hit_id_buf, cfg.ray_res).dispatch(cfg.ray_res, cfg.ray_res)
                               << any_parity_shader(imp_tlas, occ_buf, cfg.ray_res).dispatch(cfg.ray_res, cfg.ray_res)
                               << hit_f_buf.copy_to(imp_hit_f.data())
                               << hit_id_buf.copy_to(imp_hit_id.data())
                               << occ_buf.copy_to(imp_occ.data())
                               << synchronize();

                        uint32_t logged = 0u;
                        for (size_t p = 0u; p < pixel_count; ++p) {
                            const bool bm = (built_hit_f[p].w < 0.5f);
                            const bool im = (imp_hit_f[p].w < 0.5f);
                            if (bm != im) {
                                ++rec.parity_miss_mismatches;
                                if (logged++ < 5u) {
                                    LUISA_WARNING("Miss mismatch at pixel {}: built={} imp={}", p, !bm, !im);
                                }
                            } else if (!bm) {
                                if (built_hit_id[p].x != imp_hit_id[p].x) {
                                    ++rec.parity_inst_mismatches;
                                }
                                if (built_hit_id[p].y != imp_hit_id[p].y) {
                                    ++rec.parity_prim_mismatches;
                                }
                                const float dt = std::abs(built_hit_f[p].z - imp_hit_f[p].z);
                                const float du = std::abs(built_hit_f[p].x - imp_hit_f[p].x);
                                const float dv = std::abs(built_hit_f[p].y - imp_hit_f[p].y);
                                if (dt > 1e-3f) {
                                    ++rec.parity_t_mismatches;
                                }
                                if (du > 1e-4f || dv > 1e-4f) {
                                    ++rec.parity_bary_mismatches;
                                }
                            }
                            if (built_occ[p] != imp_occ[p]) {
                                ++rec.parity_occ_mismatches;
                            }
                        }
                        const uint64_t total_mismatch =
                            rec.parity_miss_mismatches +
                            rec.parity_inst_mismatches +
                            rec.parity_prim_mismatches +
                            rec.parity_t_mismatches +
                            rec.parity_bary_mismatches +
                            rec.parity_occ_mismatches;
                        LUISA_ASSERT(total_mismatch == 0ull,
                                     "Parity mismatch for backend={} mesh={} c={}: miss={} inst={} prim={} t={} bary={} occ={}",
                                     cfg.backend, m.name, compact ? 1 : 0,
                                     rec.parity_miss_mismatches, rec.parity_inst_mismatches,
                                     rec.parity_prim_mismatches, rec.parity_t_mismatches,
                                     rec.parity_bary_mismatches, rec.parity_occ_mismatches);

                        for (uint32_t r = 0u; r < total_iters; ++r) {
                            Clock c_rc;
                            stream << closest_bench_shader(imp_tlas, bench_accum_buf, cfg.ray_res, cfg.trace_iters)
                                          .dispatch(cfg.ray_res, cfg.ray_res)
                                   << synchronize();
                            const double ms_c = c_rc.toc();

                            Clock c_ra;
                            stream << any_bench_shader(imp_tlas, bench_accum_buf, cfg.ray_res, cfg.trace_iters)
                                          .dispatch(cfg.ray_res, cfg.ray_res)
                                   << synchronize();
                            const double ms_a = c_ra.toc();

                            if (r >= cfg.warmup) {
                                imp_closest_mrays.push_back(total_bench_mrays / (ms_c * 1e-3));
                                imp_any_mrays.push_back(total_bench_mrays / (ms_a * 1e-3));
                            }
                        }
                    }

                    imp_tlas = Accel{};
                    imp_mesh = Mesh{};
                    stream.synchronize();
                }

                // 2b. Warm direct file import -> TLAS (skipping extra host copy/hash)
                const std::string arc_path_str = arc_path.string();
                for (uint32_t it = 0u; it < total_iters; ++it) {
                    Clock c_e2e;
                    Mesh imp_mesh = device.create_mesh(vb, tb, option);
                    auto ires = import_mesh_archive_file(
                        ext, stream, imp_mesh, option, m.content_hash, arc_path_str,
                        AccelArchiveImportOptions{.verify_payload_checksum = false, .bypass_os_file_cache = false});
                    if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                        stream.synchronize();
                    }
                    LUISA_ASSERT(static_cast<bool>(ires), "warm direct import failed: {}", ires.message);
                    const Mesh *imp_ptr = &imp_mesh;
                    Accel imp_tlas = device.create_accel(tlas_option);
                    populate_tlas_from_descriptors(imp_tlas, luisa::span{&imp_ptr, 1u}, instances);
                    stream << imp_tlas.build(Accel::BuildRequest::FORCE_BUILD) << synchronize();
                    const double t_tot = c_e2e.toc();
                    if (it >= cfg.warmup) {
                        warm_direct_e2e_ms.push_back(t_tot);
                    }
                    imp_tlas = Accel{};
                    imp_mesh = Mesh{};
                    stream.synchronize();
                }

                // 3b. Cold direct file import -> TLAS (DirectStorage / unbuffered)
                for (uint32_t it = 0u; it < total_iters; ++it) {
                    Clock c_e2e;
                    Mesh imp_mesh = device.create_mesh(vb, tb, option);
                    auto ires = import_mesh_archive_file(
                        ext, stream, imp_mesh, option, m.content_hash, arc_path_str,
                        AccelArchiveImportOptions{.verify_payload_checksum = false, .bypass_os_file_cache = true});
                    if (ires.readiness == AccelArchiveReadiness::READY_ON_STREAM) {
                        stream.synchronize();
                    }
                    LUISA_ASSERT(static_cast<bool>(ires), "cold direct import failed: {}", ires.message);
                    const Mesh *imp_ptr = &imp_mesh;
                    Accel imp_tlas = device.create_accel(tlas_option);
                    populate_tlas_from_descriptors(imp_tlas, luisa::span{&imp_ptr, 1u}, instances);
                    stream << imp_tlas.build(Accel::BuildRequest::FORCE_BUILD) << synchronize();
                    const double t_tot = c_e2e.toc();
                    if (it >= cfg.warmup) {
                        cold_direct_e2e_ms.push_back(t_tot);
                    }
                    imp_tlas = Accel{};
                    imp_mesh = Mesh{};
                    stream.synchronize();
                }

                std::error_code ec;
                std::filesystem::remove(arc_path, ec);
            } else {
                rec.status = (cfg.mode == "rebuild_only") ? "REBUILD_ONLY" : "UNSUPPORTED_BACKEND";
                rec.mem_after_imported = rec.mem_after_built;
            }

            rec.blas_build_ms = compute_stats(blas_ms);
            rec.tlas_build_ms = compute_stats(tlas_ms);
            rec.export_ms = compute_stats(export_ms);
            rec.relocate_ms = compute_stats(relocate_ms);
            rec.relocate_pinned_ms = compute_stats(relocate_pinned_ms);
            rec.relocate_fast_ms = compute_stats(relocate_fast_ms);
            rec.warm_read_ms = compute_stats(warm_read_ms);
            rec.warm_import_ms = compute_stats(warm_imp_ms);
            rec.warm_tlas_ms = compute_stats(warm_tlas_ms);
            rec.warm_e2e_ms = compute_stats(warm_e2e_ms);
            rec.warm_direct_e2e_ms = compute_stats(warm_direct_e2e_ms);
            rec.cold_read_ms = compute_stats(cold_read_ms);
            rec.cold_import_ms = compute_stats(cold_imp_ms);
            rec.cold_tlas_ms = compute_stats(cold_tlas_ms);
            rec.cold_e2e_ms = compute_stats(cold_e2e_ms);
            rec.cold_direct_e2e_ms = compute_stats(cold_direct_e2e_ms);
            rec.built_closest_mrays_s = compute_stats(built_closest_mrays);
            rec.imported_closest_mrays_s = compute_stats(imp_closest_mrays);
            rec.built_any_mrays_s = compute_stats(built_any_mrays);
            rec.imported_any_mrays_s = compute_stats(imp_any_mrays);

            const uint64_t peak_max = std::max(rec.mem_after_built.host_peak_working_set,
                                               rec.mem_after_imported.host_peak_working_set);
            if (peak_max > rec.mem_baseline.host_working_set) {
                rec.host_peak_delta_bytes = peak_max - rec.mem_baseline.host_working_set;
            }
            if (rec.mem_baseline.device_free_bytes > rec.mem_after_built.device_free_bytes) {
                rec.vram_built_delta_bytes = rec.mem_baseline.device_free_bytes - rec.mem_after_built.device_free_bytes;
            }
            if (rec.mem_baseline.device_free_bytes > rec.mem_after_imported.device_free_bytes) {
                rec.vram_imported_delta_bytes = rec.mem_baseline.device_free_bytes - rec.mem_after_imported.device_free_bytes;
            }

            std::cout << std::fixed << std::setprecision(3)
                      << "[BENCH] backend=" << cfg.backend
                      << " mesh=" << rec.mesh_name
                      << " tris=" << rec.triangle_count
                      << " compact=" << (compact ? 1 : 0)
                      << " status=" << rec.status
                      << " blas_med=" << rec.blas_build_ms.median << "ms"
                      << " reloc_med=" << rec.relocate_ms.median << "ms"
                      << " reloc_pin=" << rec.relocate_pinned_ms.median << "ms"
                      << " reloc_fast=" << rec.relocate_fast_ms.median << "ms"
                      << " warm_e2e=" << rec.warm_e2e_ms.median << "ms"
                      << " warm_dir=" << rec.warm_direct_e2e_ms.median << "ms"
                      << " cold_e2e=" << rec.cold_e2e_ms.median << "ms"
                      << " cold_dir=" << rec.cold_direct_e2e_ms.median << "ms"
                      << " tlas_med=" << rec.tlas_build_ms.median << "ms"
                      << " bytes=" << rec.archive_bytes
                      << " unpacked=" << rec.unpacked_as_bytes
                      << " built_mrays=" << rec.built_closest_mrays_s.median
                      << " imp_mrays=" << rec.imported_closest_mrays_s.median << "\n";

            all_results.push_back(std::move(rec));
        }
    }

    if (!cfg.json_path.empty()) {
        write_full_json(cfg.json_path, cfg, device_name, caps, ext != nullptr, all_results);
        std::cout << "[BENCH_JSON] Wrote " << cfg.json_path.string() << "\n";
    }
    return 0;
}
