// A self-check of the `luisa-fallback-rtx` library.
//
// It creates a real device through the ordinary `Context` API, hands
// `device.impl()` to a `FallbackRtxDevice`, builds a two-level LBVH through the
// library's public API only - a few meshes (a cube, a quad, and a degenerate
// single triangle, one of them with a tightly packed 12-byte vertex stride) and
// a TLAS that is first built with one instance, then rebuilt with three, then
// updated in place - and checks the result twice:
//
//   * `FallbackRtxDevice::validate_blas()` / `validate_accel()` - the library's
//     own host-side structural check (reachability, AABB unions, handle ranges,
//     table resolution, world-AABB containment);
//   * the *contents* of the downloaded acceleration buffer, read directly
//     through the ABI of fallback_rtx_layout.h: the geometry the build copied
//     into every BLAS region, the leaf permutation of every tree, the blas-table
//     rows, and the instance records (transform rows and property lanes), all
//     compared against what this program uploaded.
//
// Usage: fallback_rtx_selfcheck [backend]   (default: cuda)
// Exit code 0 = every check passed.

#include "fallback_rtx.h"
#include "fallback_rtx_layout.h"

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

using namespace luisa;
using namespace luisa::compute;
using namespace lc::fallback_rtx;

namespace {

// ---------------------------------------------------------------------------
// A tiny failure counter: the self-check keeps going after a failed check, so
// one broken build prints everything it broke instead of only the first thing.
// ---------------------------------------------------------------------------
class Checker {

public:
    void check(bool condition, const char *what) noexcept {
        if (condition) {
            _passed++;
        } else {
            _failed++;
            std::printf("  FAIL: %s\n", what);
        }
    }

    [[nodiscard]] size_t failed() const noexcept { return _failed; }
    [[nodiscard]] size_t passed() const noexcept { return _passed; }

private:
    size_t _passed{0u};
    size_t _failed{0u};
};

// ---------------------------------------------------------------------------
// The scene: three meshes with their host geometry, and the instances of the
// TLAS.
// ---------------------------------------------------------------------------

struct MeshDesc {
    const char *name{};
    luisa::vector<float3> vertices;
    luisa::vector<uint> indices;// three per triangle
    // 16 (the DSL's `float3`) or 12 (three tightly packed floats): the build has
    // to read both, and the second one is the shape a tightly packed vertex
    // buffer of a real mesh has.
    size_t vertex_stride{sizeof(float3)};
};

struct UploadedMesh {
    Buffer<float3> vertices_float3;
    Buffer<float> vertices_float;
    Buffer<uint> indices;
    uint64_t blas{};
    size_t triangle_count{};
    size_t vertex_count{};
    size_t vertex_stride{};
    size_t vertex_buffer_size{};
};

struct InstanceDesc {
    uint mesh{};
    float4x4 to_world{};
    uint visibility{};
    uint user_id{};
};

[[nodiscard]] float3 transform_point(const float4x4 &m, float3 p) noexcept {
    auto v = m * make_float4(p, 1.0f);
    return v.xyz();
}

[[nodiscard]] MeshDesc make_cube() noexcept {
    MeshDesc mesh;
    mesh.name = "cube";
    mesh.vertices = {
        make_float3(-1.0f, -1.0f, -1.0f), make_float3(1.0f, -1.0f, -1.0f),
        make_float3(1.0f, 1.0f, -1.0f), make_float3(-1.0f, 1.0f, -1.0f),
        make_float3(-1.0f, -1.0f, 1.0f), make_float3(1.0f, -1.0f, 1.0f),
        make_float3(1.0f, 1.0f, 1.0f), make_float3(-1.0f, 1.0f, 1.0f)};
    const uint faces[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};
    for (auto &&f : faces) {
        mesh.indices.emplace_back(f[0]);
        mesh.indices.emplace_back(f[1]);
        mesh.indices.emplace_back(f[2]);
    }
    return mesh;
}

// Two triangles that share an edge, with a *tightly packed* vertex buffer: the
// shape that makes the build's word-addressed vertex view necessary.
[[nodiscard]] MeshDesc make_quad() noexcept {
    MeshDesc mesh;
    mesh.name = "quad(tight)";
    mesh.vertex_stride = 3u * sizeof(float);
    mesh.vertices = {make_float3(-2.0f, 0.0f, 3.0f), make_float3(2.0f, 0.0f, 3.0f),
                     make_float3(2.0f, 0.0f, 7.0f), make_float3(-2.0f, 0.0f, 7.0f)};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

// The degenerate case: one triangle, i.e. one leaf and no internal node.
[[nodiscard]] MeshDesc make_single_triangle() noexcept {
    MeshDesc mesh;
    mesh.name = "single-triangle";
    mesh.vertices = {make_float3(0.0f, 4.0f, -1.0f), make_float3(1.0f, 4.0f, -1.0f),
                     make_float3(0.5f, 4.0f, 1.0f)};
    mesh.indices = {0u, 1u, 2u};
    return mesh;
}

// `luisa::Matrix<T, 4>`'s default constructor is the identity and stores its
// *columns*, so `m[3] = (t, 1)` is the translation and scaling the three
// diagonal elements is the scale.
[[nodiscard]] float4x4 translation_matrix(float3 t) noexcept {
    float4x4 m;
    m[3] = make_float4(t, 1.0f);
    return m;
}

[[nodiscard]] float4x4 scaling_matrix(float s) noexcept {
    float4x4 m;
    m[0].x = s;
    m[1].y = s;
    m[2].z = s;
    return m;
}

[[nodiscard]] float max_component(float3 v) noexcept {
    return max(max(v.x, v.y), v.z);
}

// ---------------------------------------------------------------------------
// Upload one mesh and build its BLAS, which is the only way the library's
// geometry path runs: a stride-16 mesh goes through `Buffer<float3>`, a
// stride-12 one through a raw `Buffer<float>` with three floats per vertex.
// ---------------------------------------------------------------------------
[[nodiscard]] UploadedMesh upload_mesh(Device &device, Stream &stream,
                                       const MeshDesc &mesh,
                                       FallbackRtxDevice &fallback) noexcept {
    UploadedMesh uploaded;
    uploaded.vertex_stride = mesh.vertex_stride;
    uploaded.vertex_count = mesh.vertices.size();
    uploaded.triangle_count = mesh.indices.size() / 3u;
    uploaded.vertex_buffer_size = uploaded.vertex_count * mesh.vertex_stride;
    if (mesh.vertex_stride == sizeof(float3)) {
        uploaded.vertices_float3 = device.create_buffer<float3>(uploaded.vertex_count);
        stream << uploaded.vertices_float3.view().copy_from(luisa::span{mesh.vertices});
    } else {
        luisa::vector<float> packed;
        packed.reserve(uploaded.vertex_count * 3u);
        for (auto &&v : mesh.vertices) {
            packed.emplace_back(v.x);
            packed.emplace_back(v.y);
            packed.emplace_back(v.z);
        }
        uploaded.vertices_float = device.create_buffer<float>(packed.size());
        stream << uploaded.vertices_float.view().copy_from(luisa::span{packed});
    }
    uploaded.indices = device.create_buffer<uint>(mesh.indices.size());
    stream << uploaded.indices.view().copy_from(luisa::span{mesh.indices});
    stream << synchronize();

    uploaded.blas = fallback.create_blas(AccelOption{});
    FallbackRtxDevice::MeshGeometry geometry;
    geometry.vertex_buffer = uploaded.vertices_float3.valid() ? uploaded.vertices_float3.handle() : uploaded.vertices_float.handle();
    geometry.vertex_buffer_offset = 0u;
    geometry.vertex_stride = mesh.vertex_stride;
    geometry.vertex_buffer_size = uploaded.vertex_buffer_size;
    geometry.triangle_buffer = uploaded.indices.handle();
    geometry.triangle_buffer_offset = 0u;
    geometry.triangle_buffer_size = mesh.indices.size() * sizeof(uint);
    stream << fallback.build_blas(uploaded.blas, geometry).commit() << synchronize();
    return uploaded;
}

// ---------------------------------------------------------------------------
// The ABI of fallback_rtx_layout.h, restated as host-side accessors, so the
// self-check reads the downloaded buffer exactly the way a traversal would.
// ---------------------------------------------------------------------------
[[nodiscard]] uint u32_at(luisa::span<const uint4> buffer, size_t index) noexcept {
    auto value = buffer[index / 4u];
    switch (index % 4u) {
        case 0u: return value.x;
        case 1u: return value.y;
        case 2u: return value.z;
        default: return value.w;
    }
}

[[nodiscard]] uint header_at(luisa::span<const uint4> accel, uint base, uint slot) noexcept {
    return u32_at(accel, static_cast<size_t>(base) * 4u + slot);
}

[[nodiscard]] float3 xyz_as_float3(const uint4 &v) noexcept {
    return make_float3(luisa::bit_cast<float>(v.x), luisa::bit_cast<float>(v.y),
                       luisa::bit_cast<float>(v.z));
}

[[nodiscard]] float4 as_float4(const uint4 &v) noexcept {
    return make_float4(luisa::bit_cast<float>(v.x), luisa::bit_cast<float>(v.y),
                       luisa::bit_cast<float>(v.z), luisa::bit_cast<float>(v.w));
}

// Bit-exact: `Vector::operator==` is component-wise (it returns a bool4), and
// the point of this check is that the build copies the very bits it was given.
[[nodiscard]] bool same_float4(float4 a, float4 b) noexcept {
    return luisa::bit_cast<uint>(a.x) == luisa::bit_cast<uint>(b.x) &&
           luisa::bit_cast<uint>(a.y) == luisa::bit_cast<uint>(b.y) &&
           luisa::bit_cast<uint>(a.z) == luisa::bit_cast<uint>(b.z) &&
           luisa::bit_cast<uint>(a.w) == luisa::bit_cast<uint>(b.w);
}

// ---------------------------------------------------------------------------
// The content check: every region the build produced, read back through the ABI
// and compared against what this program uploaded.
// ---------------------------------------------------------------------------
void check_contents(luisa::span<const uint4> accel, uint tlas_base, uint instance_count,
                    const luisa::vector<MeshDesc> &meshes,
                    const luisa::vector<UploadedMesh> &uploaded,
                    const luisa::vector<InstanceDesc> &instances, Checker &check) noexcept {
    auto base = header_at(accel, tlas_base, h_base);
    auto node_base = header_at(accel, tlas_base, h_node_base);
    auto node_count = header_at(accel, tlas_base, h_node_count);
    auto prim_count = header_at(accel, tlas_base, h_prim_count);
    auto table_base = header_at(accel, tlas_base, h_blas_table_base);
    auto blas_count = header_at(accel, tlas_base, h_blas_count);
    auto root = header_at(accel, tlas_base, h_root);
    auto flags = header_at(accel, tlas_base, h_flags);
    check.check(base == tlas_base, "the TLAS header carries its own base");
    check.check((flags & region_flag_tlas) != 0u, "the TLAS region is flagged as a TLAS");
    check.check(prim_count == instance_count, "the TLAS has one leaf per instance");
    check.check(blas_count == instance_count, "the TLAS has one blas-table record per instance");
    check.check(node_count == instance_count * 2u - 1u, "the TLAS node count is 2n-1");
    check.check(root == node_base, "the TLAS root is its node array");
    check.check(static_cast<size_t>(table_base) + static_cast<size_t>(blas_record_u4) * blas_count <=
                    accel.size(),
                "the TLAS blas table is inside the downloaded region");
    // The TLAS root AABB has to contain every instance's world AABB; the
    // library's validator checks it too, but here it is checked against the
    // *instance records this program wrote*.
    auto root_lo = xyz_as_float3(accel[root]);
    auto root_hi = xyz_as_float3(accel[root + 1u]);
    for (auto i = 0u; i < instance_count; i++) {
        // ---- the instance record ----
        auto row = static_cast<size_t>(table_base) * 4u + static_cast<size_t>(i) * 8u;
        auto blas_base = u32_at(accel, row + 0u);
        auto blas_node_base = u32_at(accel, row + 1u);
        auto blas_index_base = u32_at(accel, row + 2u);
        auto blas_vertex_base = u32_at(accel, row + 3u);
        auto blas_triangle_count = u32_at(accel, row + 4u);
        auto &&instance = instances[i];
        auto &&mesh = meshes[instance.mesh];
        auto &&built = uploaded[instance.mesh];
        auto blas = blas_base;
        if (blas + header_u4 > accel.size()) {
            check.check(false, "the blas-table row points at a region inside the buffer");
            continue;
        }
        // ---- the referenced BLAS region ----
        check.check(header_at(accel, blas, h_base) == blas, "the BLAS header carries its own base");
        check.check((header_at(accel, blas, h_flags) & region_flag_tlas) == 0u,
                    "the referenced region is a BLAS");
        check.check(header_at(accel, blas, h_root) == blas_node_base,
                    "the blas-table node base is the BLAS root");
        check.check(header_at(accel, blas, h_index_base) == blas_index_base,
                    "the blas-table index base matches the BLAS header");
        check.check(header_at(accel, blas, h_vertex_base) == blas_vertex_base,
                    "the blas-table vertex base matches the BLAS header");
        check.check(blas_triangle_count == built.triangle_count,
                    "the blas-table triangle count matches the mesh");
        check.check(header_at(accel, blas, h_vertex_count) == built.vertex_count,
                    "the BLAS header vertex count matches the mesh");
        check.check(header_at(accel, blas, h_prim_count) == built.triangle_count,
                    "the BLAS header primitive count matches the mesh");

        // ---- the copied geometry, bit for bit ----
        auto index_base = static_cast<size_t>(blas_index_base) * 4u;
        for (auto t = 0u; t < built.triangle_count; t++) {
            auto i0 = u32_at(accel, index_base + static_cast<size_t>(t) * 4u + 0u);
            auto i1 = u32_at(accel, index_base + static_cast<size_t>(t) * 4u + 1u);
            auto i2 = u32_at(accel, index_base + static_cast<size_t>(t) * 4u + 2u);
            check.check(i0 == mesh.indices[t * 3u + 0u] &&
                            i1 == mesh.indices[t * 3u + 1u] &&
                            i2 == mesh.indices[t * 3u + 2u],
                        "the region carries the triangle indices the mesh was built from");
        }
        auto vertex_base = blas_vertex_base;
        for (auto v = 0u; v < built.vertex_count; v++) {
            check.check(same_float4(as_float4(accel[vertex_base + v]),
                                    make_float4(mesh.vertices[v], 0.0f)),
                        "the region carries the vertices the mesh was built from");
        }

        // ---- the leaves: every triangle of this mesh exactly once ----
        auto leaf_base = static_cast<size_t>(blas_node_base) +
                         static_cast<size_t>(built.triangle_count - 1u) * node_u4;
        luisa::vector<uint32_t> seen(built.triangle_count, 0u);
        for (auto leaf = 0u; leaf < built.triangle_count; leaf++) {
            auto plane_lo = accel[leaf_base + static_cast<size_t>(leaf) * node_u4];
            auto plane_hi = accel[leaf_base + static_cast<size_t>(leaf) * node_u4 + 1u];
            check.check(plane_lo.w == invalid_offset, "a leaf is marked by the invalid handle");
            if (plane_hi.w < built.triangle_count) { seen[plane_hi.w]++; }
        }
        auto permutation_ok = true;
        for (auto s : seen) {
            if (s != 1u) { permutation_ok = false; }
        }
        check.check(permutation_ok, "the leaves of a BLAS are a permutation of its triangles");

        // ---- the instance's world AABB, from the transform this program set ----
        // (the instance *records* themselves are checked by
        // `check_instance_records`, from the instance buffer the binding reports)
        auto blas_lo = xyz_as_float3(accel[blas_node_base]);
        auto blas_hi = xyz_as_float3(accel[blas_node_base + 1u]);
        auto world_lo = make_float3(1.0e30f);
        auto world_hi = make_float3(-1.0e30f);
        for (auto c = 0u; c < 8u; c++) {
            auto corner = make_float3((c & 1u) != 0u ? blas_hi.x : blas_lo.x,
                                      (c & 2u) != 0u ? blas_hi.y : blas_lo.y,
                                      (c & 4u) != 0u ? blas_hi.z : blas_lo.z);
            auto p = transform_point(instance.to_world, corner);
            world_lo = min(world_lo, p);
            world_hi = max(world_hi, p);
        }
        auto tolerance = 1.0e-4f * std::max(1.0f, max_component(root_hi - root_lo));
        check.check(world_lo.x >= root_lo.x - tolerance && world_lo.y >= root_lo.y - tolerance &&
                        world_lo.z >= root_lo.z - tolerance &&
                        world_hi.x <= root_hi.x + tolerance && world_hi.y <= root_hi.y + tolerance &&
                        world_hi.z <= root_hi.z + tolerance,
                    "the TLAS root AABB contains the instance's world AABB");
    }
}

// Compare the transform rows and the property lanes of every instance record
// against what this program wrote, through the instance buffer the binding
// reports.
void check_instance_records(luisa::span<const uint4> instances, uint instance_base_u4,
                            const luisa::vector<InstanceDesc> &descs, Checker &check) noexcept {
    for (auto i = 0u; i < descs.size(); i++) {
        auto first = static_cast<size_t>(instance_base_u4) + static_cast<size_t>(i) * instance_u4;
        if (first + instance_u4 > instances.size()) {
            check.check(false, "the instance slice is inside the downloaded buffer");
            return;
        }
        auto &&desc = descs[i];
        auto to_world = desc.to_world;
        // `Modification::set_transform` stores the rows of the matrix
        // (affine[row * 4 + column] = m[column][row]); the record has to carry
        // exactly those bits.
        for (auto r = 0u; r < 3u; r++) {
            auto row = make_float4(to_world[0][r], to_world[1][r], to_world[2][r], to_world[3][r]);
            check.check(same_float4(row, as_float4(instances[first + i_to_world + r])),
                        "the instance record carries the object->world row that was set");
        }
        auto misc = instances[first + i_misc];
        check.check(misc.x == i, "the instance record's blas_index is its own table row");
        check.check(misc.y == desc.visibility, "the instance record carries the visibility mask");
        check.check(misc.z == desc.user_id, "the instance record carries the user id");
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto program = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "fallback_rtx_selfcheck";
    auto backend = argc > 1 && argv[1] != nullptr ? luisa::string{argv[1]} : luisa::string{"cuda"};
    // Unbuffered: this program is a diagnostic, so its own progress has to
    // survive a crash instead of dying in a stdio buffer.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("fallback_rtx_selfcheck: backend=%s\n", backend.c_str());
    std::fflush(stdout);

    Context context{program};
    Device device = context.create_device(backend);
    Stream stream = device.create_stream();

    // The fallback only borrows the interface; the device outlives it.
    FallbackRtxDevice fallback{device.impl()};

    Checker check;
    // ---- meshes -> BLASes ---------------------------------------------------
    luisa::vector<MeshDesc> meshes;
    meshes.emplace_back(make_cube());
    meshes.emplace_back(make_quad());
    meshes.emplace_back(make_single_triangle());
    luisa::vector<UploadedMesh> uploaded;
    uploaded.reserve(meshes.size());
    for (auto &&mesh : meshes) {
        auto built = upload_mesh(device, stream, mesh, fallback);
        auto problems = fallback.validate_blas(built.blas);
        std::printf("  BLAS %-16s triangles=%zu vertices=%zu stride=%zu -> %zu problem(s)\n",
                    mesh.name, built.triangle_count, built.vertex_count, mesh.vertex_stride,
                    problems);
        check.check(problems == 0u, "the structural validation of a fresh BLAS");
        uploaded.emplace_back(std::move(built));
    }
    std::printf("  a 1-triangle mesh (the degenerate case) and a stride-12 mesh were built\n");
    std::printf("  fallback stats: blas=%zu accel=%zu blas_builds=%zu accel_builds=%zu "
                "accel_buffer=%zu B instance_buffer=%zu B\n",
                fallback.stats().blas_count, fallback.stats().accel_count,
                fallback.stats().blas_builds, fallback.stats().accel_builds,
                fallback.stats().accel_buffer_bytes, fallback.stats().instance_buffer_bytes);

    // ---- growth ------------------------------------------------------------
    // The API knows no scene size up front, so the storage starts small and has
    // to grow; keep building until the shared acceleration buffer has grown, and
    // then validate the trees that were built *before* it: the growth appends a
    // larger buffer, copies the live prefix into the same offsets and retires the
    // old one, so a region and every offset inside it have to have survived.
    auto buffer_bytes_before_growth = fallback.stats().accel_buffer_bytes;
    size_t problems = 0u;
    luisa::vector<MeshDesc> fillers;
    luisa::vector<UploadedMesh> extra;
    for (auto i = 0u; i < 16u; i++) {
        fillers.emplace_back(meshes[0]);
        extra.emplace_back(upload_mesh(device, stream, fillers.back(), fallback));
    }
    auto buffer_bytes_after_growth = fallback.stats().accel_buffer_bytes;
    std::printf("  the acceleration buffer grew %zu B -> %zu B after %zu more cube BLASes\n",
                buffer_bytes_before_growth, buffer_bytes_after_growth, extra.size());
    check.check(buffer_bytes_after_growth > buffer_bytes_before_growth,
                "the shared acceleration buffer grew on demand");
    for (auto i = 0u; i < extra.size(); i++) {
        problems = fallback.validate_blas(extra[i].blas);
        check.check(problems == 0u, "the structural validation of a BLAS built after a growth");
    }
    for (auto &&built : uploaded) {
        problems = fallback.validate_blas(built.blas);
        check.check(problems == 0u,
                    "the structural validation of a BLAS built *before* a growth");
    }
    std::printf("  %zu BLASes built before the growth still validate (their regions kept "
                "their offsets)\n",
                uploaded.size());

    // ---- an instance the caller never gave a mesh ---------------------------
    // The other negative control: an instance without a primitive must not turn
    // into a reference to whatever region happens to sit at offset 0.  The build
    // leaves the blas-table row of such an instance clear (a null reference,
    // fallback_rtx_layout.h) and the structural check has to report it.
    {
        auto tlas = fallback.create_accel(AccelOption{});
        luisa::vector<AccelBuildCommand::Modification> incomplete(2u);
        incomplete[0] = AccelBuildCommand::Modification{0u};
        incomplete[0].set_transform(translation_matrix(make_float3(0.0f, 0.0f, -6.0f)));
        incomplete[0].set_primitive(uploaded[0].blas);
        incomplete[1] = AccelBuildCommand::Modification{1u};
        incomplete[1].set_transform(translation_matrix(make_float3(8.0f, 0.0f, 0.0f)));
        // instance 1 never gets a primitive
        stream << fallback.build_accel(tlas, 2u, luisa::span{incomplete}, false).commit()
               << synchronize();
        problems = fallback.validate_accel(tlas);
        std::printf("  a TLAS whose instance 1 has no mesh -> %zu problem(s) "
                    "(the validator has to report it)\n",
                    problems);
        check.check(problems > 0u,
                    "the structural validation reports an instance that has no mesh");
        fallback.destroy_accel(tlas);
    }

    // ---- one-instance TLAS --------------------------------------------------
    {
        auto tlas = fallback.create_accel(AccelOption{});
        luisa::vector<AccelBuildCommand::Modification> modifications(1u);
        modifications[0] = AccelBuildCommand::Modification{0u};
        auto to_world = translation_matrix(make_float3(0.0f, 0.0f, -6.0f));
        modifications[0].set_transform(to_world);
        modifications[0].set_primitive(uploaded[0].blas);
        modifications[0].set_visibility(0xFFu);
        modifications[0].set_user_id(42u);
        stream << fallback.build_accel(tlas, 1u, luisa::span{modifications}, false).commit()
               << synchronize();
        problems = fallback.validate_accel(tlas);
        std::printf("  TLAS with 1 instance -> %zu problem(s)\n", problems);
        check.check(problems == 0u, "the structural validation of a 1-instance TLAS");
        check.check(fallback.binding(tlas).valid(), "binding() is valid after a build");
        fallback.destroy_accel(tlas);
        std::printf("  (the 1-instance TLAS was destroyed; the tree is rebuilt below)\n");

        // ---- three instances, rebuilt from scratch --------------------------
        luisa::vector<InstanceDesc> descs;
        descs.emplace_back(InstanceDesc{0u, translation_matrix(make_float3(0.0f, 0.0f, -6.0f)), 0xFFu, 7u});
        descs.emplace_back(InstanceDesc{1u, scaling_matrix(4.0f), 0x03u, 8u});
        descs.emplace_back(InstanceDesc{2u, translation_matrix(make_float3(5.0f, 0.0f, 2.0f)), 0x01u, 9u});
        auto tlas3 = fallback.create_accel(AccelOption{});
        luisa::vector<AccelBuildCommand::Modification> mods3(descs.size());
        for (auto i = 0u; i < descs.size(); i++) {
            mods3[i] = AccelBuildCommand::Modification{i};
            mods3[i].set_transform(descs[i].to_world);
            mods3[i].set_primitive(uploaded[descs[i].mesh].blas);
            mods3[i].set_visibility(static_cast<uint8_t>(descs[i].visibility));
            mods3[i].set_user_id(descs[i].user_id);
        }
        stream << fallback.build_accel(tlas3, static_cast<uint32_t>(descs.size()),
                                       luisa::span{mods3}, false)
                      .commit()
               << synchronize();
        problems = fallback.validate_accel(tlas3);
        std::printf("  TLAS with %zu instances (3 meshes) -> %zu problem(s)\n", descs.size(),
                    problems);
        check.check(problems == 0u, "the structural validation of a 3-instance TLAS");

        // ---- the downloaded buffer, read through the ABI --------------------
        auto binding = fallback.binding(tlas3);
        check.check(binding.valid(), "binding() of a built TLAS is valid");
        check.check(binding.accel_offset_bytes % 16u == 0u &&
                        binding.instance_offset_bytes % 16u == 0u,
                    "both descriptors start at a uint4 boundary");
        auto tlas_region_bytes = lc::fallback_rtx::tlas_region_bytes(descs.size());
        auto accel_u4 = (binding.accel_offset_bytes + tlas_region_bytes) / 16u;
        luisa::vector<uint4> accel(accel_u4);
        auto accel_words = BufferView<uint4>{nullptr, binding.accel_buffer, 16u, 0u, accel_u4,
                                             accel_u4};
        auto instance_slots = descs.size() * instance_u4;
        luisa::vector<uint4> instance_buffer(instance_slots);
        auto instance_words = BufferView<uint4>{nullptr, binding.instance_buffer, 16u,
                                                binding.instance_offset_bytes, instance_slots,
                                                instance_slots};
        stream << accel_words.copy_to(luisa::span{accel})
               << instance_words.copy_to(luisa::span{instance_buffer})
               << synchronize();
        std::printf("  downloaded %zu u4 of the acceleration buffer and %zu u4 of the instance "
                    "buffer through `binding()`\n",
                    accel.size(), instance_buffer.size());
        auto tlas_base = static_cast<uint>(binding.accel_offset_bytes / 16u);
        check_contents(luisa::span<const uint4>{accel.data(), accel.size()}, tlas_base,
                       static_cast<uint>(descs.size()), meshes, uploaded, descs, check);
        check_instance_records(luisa::span<const uint4>{instance_buffer.data(), instance_buffer.size()},
                               0u, descs, check);

        // ---- an in-place update: only the instance buffer and the table -----
        // `update_instance_buffer_only` leaves the radix tree as it is - the
        // hardware backends use it for exactly this, a property-only edit - so
        // the update touches the user id and the visibility mask and *not* the
        // transform, which would make the tree's AABBs stale.
        luisa::vector<AccelBuildCommand::Modification> update(1u);
        update[0] = AccelBuildCommand::Modification{1u};
        update[0].set_user_id(123u);
        update[0].set_visibility(0x05u);
        descs[1].user_id = 123u;
        descs[1].visibility = 0x05u;
        stream << fallback.build_accel(tlas3, static_cast<uint32_t>(descs.size()),
                                       luisa::span{update}, true)
                      .commit()
               << synchronize();
        problems = fallback.validate_accel(tlas3);
        std::printf("  TLAS update_instance_buffer_only (1 of %zu instances) -> %zu problem(s)\n",
                    descs.size(), problems);
        check.check(problems == 0u, "the structural validation after an in-place instance update");
        // The records of the updated instances must have reached the device.
        luisa::vector<uint4> instances_after(instance_slots);
        stream << instance_words.copy_to(luisa::span{instances_after}) << synchronize();
        check_instance_records(
            luisa::span<const uint4>{instances_after.data(), instances_after.size()}, 0u, descs,
            check);

        // ---- a full rebuild of the same TLAS, moving an instance ------------
        // This is the other half of the update path: the instance buffer is
        // written again *and* the tree is rebuilt, so the moved instance ends up
        // inside the new root AABB.
        auto moved = translation_matrix(make_float3(-9.0f, 0.0f, -6.0f));
        luisa::vector<AccelBuildCommand::Modification> rebuild(1u);
        rebuild[0] = AccelBuildCommand::Modification{1u};
        rebuild[0].set_transform(moved);
        descs[1].to_world = moved;
        stream << fallback.build_accel(tlas3, static_cast<uint32_t>(descs.size()),
                                       luisa::span{rebuild}, false)
                      .commit()
               << synchronize();
        problems = fallback.validate_accel(tlas3);
        std::printf("  TLAS rebuilt with a moved instance -> %zu problem(s)\n", problems);
        check.check(problems == 0u, "the structural validation after a full TLAS rebuild");
        instances_after.resize(instance_slots);
        stream << instance_words.copy_to(luisa::span{instances_after}) << synchronize();
        check_instance_records(
            luisa::span<const uint4>{instances_after.data(), instances_after.size()}, 0u, descs,
            check);

        // ---- the validator has to be able to fail ---------------------------
        // A negative control: break one blas-table row *through the descriptor
        // `binding()` reports* and check that the structural check sees it.  A
        // rebuild then rewrites the whole table and the tree, and the check has to
        // be clean again - which is also the evidence that the 167 positive checks
        // above are not vacuous.
        {
            luisa::vector<uint4> zeroed{make_uint4(0u, 0u, 0u, 0u),
                                        make_uint4(0u, 0u, 0u, 0u)};
            // the header stores *absolute* uint4 offsets (fallback_rtx_layout.h)
            auto table_u4 = static_cast<size_t>(header_at(accel, tlas_base, h_blas_table_base));
            auto row_view = BufferView<uint4>{nullptr, binding.accel_buffer, sizeof(uint4),
                                              table_u4 * sizeof(uint4), 2u, accel_u4};
            stream << row_view.copy_from(luisa::span{zeroed}) << synchronize();
            auto corrupted = fallback.validate_accel(tlas3);
            std::printf("  one blas-table row zeroed on purpose -> %zu problem(s) "
                        "(the validator has to report it)\n",
                        corrupted);
            check.check(corrupted > 0u,
                        "the structural validation reports a deliberately corrupted table row");
        }
        stream << fallback.build_accel(tlas3, static_cast<uint32_t>(descs.size()),
                                       luisa::span{rebuild}, false)
                      .commit()
               << synchronize();
        problems = fallback.validate_accel(tlas3);
        std::printf("  after a rebuild of the restored table -> %zu problem(s)\n", problems);
        check.check(problems == 0u, "the structural validation of a rebuilt TLAS");
        fallback.destroy_accel(tlas3);
    }

    for (auto &&built : extra) { fallback.destroy_blas(built.blas); }
    for (auto &&built : uploaded) { fallback.destroy_blas(built.blas); }
    auto stats = fallback.stats();
    std::printf("  final stats: blas=%zu accel=%zu blas_builds=%zu accel_builds=%zu\n",
                stats.blas_count, stats.accel_count, stats.blas_builds, stats.accel_builds);
    check.check(stats.blas_count == 0u && stats.accel_count == 0u,
                "destroying every handle empties the device's tables");
    std::printf("fallback_rtx_selfcheck: %zu check(s) passed, %zu failed\n", check.passed(),
                check.failed());
    return check.failed() == 0u ? 0 : 1;
}
