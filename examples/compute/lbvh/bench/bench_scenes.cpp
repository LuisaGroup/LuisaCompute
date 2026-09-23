// The adversarial scene catalogue of the software-LBVH benchmark: the
// generators, the OBJ loader and the ray distributions (see bench_scenes.h).
//
// Every generator writes *world-space* triangles; a scene split into K meshes
// keeps the identity transform on every instance, so the K-mesh run contains the
// same triangles as the single-mesh run.  The multi-BLAS path is still fully
// exercised (K trees, K radix sorts) and the TLAS of the chained scenes is the
// tree over K mesh AABBs, which is the two-level worst case in a nutshell.

#include "bench_scenes.h"

#include <cstdio>
#include <fstream>

#include <algorithm>
#include <cmath>

namespace luisa::example::lbvh {

namespace {

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

// splitmix64 seeded per triangle: no <random>, no platform dependence, and a
// triangle's geometry depends only on (seed, triangle index), never on how the
// scene happens to be split into meshes.
struct BenchRng {
    uint64_t state;
    void seed(uint64_t s, size_t index) noexcept {
        state = s ^ (static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
    }
    [[nodiscard]] uint32_t next_u32() noexcept {
        state += 0x9E3779B97F4A7C15ull;
        auto z = state;
        z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
        return static_cast<uint32_t>((z ^ (z >> 31u)) >> 32u);
    }
    [[nodiscard]] float unit() noexcept {// [0, 1)
        return static_cast<float>(next_u32() >> 8u) * (1.0f / 16777216.0f);
    }
    [[nodiscard]] float symmetric() noexcept { return unit() * 2.0f - 1.0f; }
    [[nodiscard]] float3 in_unit_ball() noexcept {
        for (auto attempt = 0u; attempt < 8u; attempt++) {
            auto p = make_float3(symmetric(), symmetric(), symmetric());
            if (dot(p, p) <= 1.0f) { return p; }
        }
        return make_float3(0.0f);// only reachable through rounding
    }
    [[nodiscard]] float3 unit_vector() noexcept {
        auto p = in_unit_ball();
        auto length_squared = dot(p, p);
        return length_squared > 0.0f ? p * (1.0f / std::sqrt(length_squared)) : make_float3(0.0f, 0.0f, 1.0f);
    }
};

[[nodiscard]] BenchRng triangle_rng(uint64_t seed, size_t index) noexcept {
    BenchRng rng;
    rng.seed(seed, index);
    return rng;
}

// A direction perpendicular to `d`, chosen from the least aligned axis so the
// cross product never degenerates.
[[nodiscard]] float3 perpendicular(float3 d) noexcept {
    auto a = abs(d);
    auto axis = a.x <= a.y && a.x <= a.z ? make_float3(1.0f, 0.0f, 0.0f) : a.y <= a.z ? make_float3(0.0f, 1.0f, 0.0f) :
                                                                                        make_float3(0.0f, 0.0f, 1.0f);
    return normalize(cross(d, axis));
}

// ---------------------------------------------------------------------------
// Soup writer
// ---------------------------------------------------------------------------

// Appends the triangles of one mesh to the scene soup and remembers that mesh's
// object-space bounds, i.e. the volume its BLAS normalizes the Morton codes with.
struct SoupWriter {
    BenchScene *scene;
    MeshRange open{};
    float3 lo{1.0e30f};
    float3 hi{-1.0e30f};

    void begin_mesh() noexcept {
        open.triangle_offset = static_cast<uint>(scene->triangles.size());
        open.triangle_count = 0u;
        lo = make_float3(1.0e30f);
        hi = make_float3(-1.0e30f);
    }
    void add_triangle(float3 a, float3 b, float3 c) noexcept {
        auto base = static_cast<uint>(scene->vertices.size());
        scene->vertices.emplace_back(a);
        scene->vertices.emplace_back(b);
        scene->vertices.emplace_back(c);
        scene->triangles.emplace_back(Triangle{base, base + 1u, base + 2u});
        open.triangle_count++;
        lo = min(lo, min(a, min(b, c)));
        hi = max(hi, max(a, max(b, c)));
    }
    void end_mesh() noexcept {
        if (open.triangle_count == 0u) { return; }// never create an empty BLAS
        open.lo = lo;
        open.hi = hi;
        auto index = static_cast<uint>(scene->meshes.size());
        scene->meshes.emplace_back(open);
        scene->instances.emplace_back(InstanceSpec{index, translation(make_float3(0.0f))});
    }
};

// The scene AABB, from the triangles that were just generated.
void update_scene_bounds(BenchScene &scene) noexcept {
    scene.scene_lo = make_float3(1.0e30f);
    scene.scene_hi = make_float3(-1.0e30f);
    for (auto &&v : scene.vertices) {
        scene.scene_lo = min(scene.scene_lo, v);
        scene.scene_hi = max(scene.scene_hi, v);
    }
}

// ---------------------------------------------------------------------------
// Generators
//
// Every generator emits the triangles of `triangles` in triangle-index order and
// splits them into `instances` contiguous meshes (the identity transform per
// mesh keeps the K-mesh scene identical to the one-mesh scene).
// ---------------------------------------------------------------------------

template<typename Emit>
void emit_chunked(BenchScene &scene, size_t triangles, size_t instances,
                  Emit &&emit) noexcept {
    SoupWriter writer{&scene};
    for (auto m = 0u; m < instances; m++) {
        auto begin = triangles * m / instances;
        auto end = triangles * (m + 1u) / instances;
        writer.begin_mesh();
        for (auto i = begin; i < end; i++) { emit(writer, i); }
        writer.end_mesh();
    }
}

// `uniform` - the baseline: tiny isotropic triangles with centroids uniform in a
// cube.  No worst case; every ratio of the report is taken against this scene.
void emit_uniform(SoupWriter &writer, size_t i, uint64_t seed) noexcept {
    auto rng = triangle_rng(seed, i);
    auto center = make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric());
    auto r = 1.0e-3f + 4.0e-3f * rng.unit();
    writer.add_triangle(center + r * make_float3(1.0f, 0.0f, 0.0f),
                        center + r * make_float3(-0.5f, 0.8660254f, 0.0f),
                        center + r * make_float3(-0.5f, -0.8660254f, 0.0f));
}

// `coincident` - every centroid is *exactly* the origin.  The triangle sizes are
// powers of two, so the AABBs are symmetric about the origin in exact float
// arithmetic and all N Morton codes are bit-identical: the radix-sort histogram
// has a single non-empty bin (shared-atomic contention, zero scatter
// parallelism), `delta()` takes its equal-code branch and the tree degenerates.
void emit_coincident(SoupWriter &writer, size_t i) noexcept {
    auto a = std::ldexp(1.0f, -static_cast<int>((i % 8u) + 2u));// 2^-2 .. 2^-9
    auto b = std::ldexp(1.0f, -static_cast<int>((i % 4u) + 3u));// 2^-3 .. 2^-6
    auto sx = (i & 1u) != 0u ? 1.0f : -1.0f;
    writer.add_triangle(make_float3(a, b, 0.0f), make_float3(-a, b, 0.0f),
                        make_float3(sx * a, -b, 0.0f));
}

// `grid-duplicates` - 1024 cells of a 16x8x8 grid, each holding exactly
// coincident centroids.  The cells' centres are multiples of 2^-3, so the
// duplicates are exact again, but now the codes cover 1024 distinct Morton bins
// while the *low* bits of every code are zero: the first radix pass sees a
// single non-empty bin, the later passes see a handful - the duplicate path and
// a maximally clustered histogram in the same scene.
void emit_grid_duplicates(SoupWriter &writer, size_t i) noexcept {
    constexpr auto cells_x = 16u;
    constexpr auto cells_y = 8u;
    constexpr auto cells_z = 8u;
    auto cell = static_cast<uint>(i % (cells_x * cells_y * cells_z));
    auto ix = cell % cells_x;
    auto iy = (cell / cells_x) % cells_y;
    auto iz = cell / (cells_x * cells_y);
    auto center = make_float3(-1.0f + (static_cast<float>(ix) + 0.5f) * 0.125f,
                              -0.5f + (static_cast<float>(iy) + 0.5f) * 0.125f,
                              -0.5f + (static_cast<float>(iz) + 0.5f) * 0.125f);
    auto a = std::ldexp(1.0f, -9);
    auto b = std::ldexp(1.0f, -10);
    auto sx = (i & 1u) != 0u ? 1.0f : -1.0f;
    writer.add_triangle(center + make_float3(a, b, 0.0f),
                        center + make_float3(-a, b, 0.0f),
                        center + make_float3(sx * a, -b, 0.0f));
}

// `exponential` - the Morton caterpillar: 30 chain positions, one per bit of the
// 30-bit code, where position k sits at distance 2^-(k/3) along axis k%3, so
// every step down the sorted key order drops exactly one more bit of common
// prefix.  (A single axis only carries 10 quantized bits, so the three axes are
// needed to reach all 30 bits; the exponents are still the geometric 2^-i the
// worst case asks for.)  This maximizes the `determine_range` search and the
// depth of the radix tree; every 30th triangle shares a position, which adds the
// equal-code branch on top.
void emit_exponential(SoupWriter &writer, size_t i) noexcept {
    constexpr auto chain_length = 30u;
    auto k = static_cast<uint>(i % chain_length);
    auto level = k / 3u;
    auto position = make_float3(0.0f);
    position[k % 3u] = std::ldexp(1.0f, -static_cast<int>(level));
    auto a = std::ldexp(1.0f, -static_cast<int>(level) - 5);
    writer.add_triangle(position + make_float3(a, a, a),
                        position + make_float3(-a, a, a),
                        position + make_float3(0.0f, -a, -a));
}

// `line` - tiny triangles on the main diagonal: a deep, thin chain of heavily
// overlapping AABBs, which is the classic bad case for any BVH and, with K
// instances, for the TLAS of a chained layout.
void emit_line(SoupWriter &writer, size_t i, size_t n) noexcept {
    auto t = (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
    auto center = (2.0f * t - 1.0f) * make_float3(1.0f, 1.0f, 1.0f);
    auto r = 2.0e-3f;
    writer.add_triangle(center + r * make_float3(1.0f, 0.0f, 0.0f),
                        center + r * make_float3(-0.5f, 0.8660254f, 0.0f),
                        center + r * make_float3(-0.5f, -0.8660254f, 0.0f));
}

// `sliver-soup` - long thin slivers fanned out through a small ball: nearly
// every triangle AABB overlaps nearly every other one, so a ray through the ball
// cannot be culled and the walk degenerates to O(N) node visits and O(N)
// triangle tests *per ray* (the default triangle count is deliberately small).
void emit_sliver_soup(SoupWriter &writer, size_t i, uint64_t seed) noexcept {
    auto rng = triangle_rng(seed, i);
    auto d = rng.unit_vector();
    auto u = perpendicular(d);
    constexpr auto half_length = 0.5f;
    constexpr auto thickness = 2.0e-3f;
    // The three vertices:
    //
    //   * the sliver is long (1.0) and thin (2e-3), so its AABB is essentially
    //     the box of a segment through the ball and every pair of slivers
    //     overlaps near the ball: a ray through the ball cannot be culled;
    //   * the centre is offset randomly inside the ball (0.25) instead of being
    //     the origin, so no triangle *edge* passes through the ball centre:
    //     with the edges through the centre, rays through it hit the shared edge
    //     of thousands of slivers at once and the two intersecters of
    //     `--validate` disagree on which side of the edge the hit is, which is a
    //     property of the geometry and not of the traversal.
    auto c = 0.25f * rng.in_unit_ball();
    writer.add_triangle(c - half_length * d, c + half_length * d, c + thickness * u);
}

// `bimodal` - 90% of the primitives in a tiny blob at one corner plus 10% spread
// over a 2000x larger box: an extreme scale disparity ("teapot in a stadium"),
// enormous AABB overlap inside the blob and Morton codes that use a tiny
// fraction of the 30-bit range.
void emit_bimodal(SoupWriter &writer, size_t i, uint64_t seed) noexcept {
    auto rng = triangle_rng(seed, i);
    constexpr auto stadium = 10.0f;
    constexpr auto blob_radius = 0.01f;
    auto in_blob = (i % 10u) != 0u;
    auto center = in_blob ? make_float3(stadium) + blob_radius * rng.in_unit_ball() : stadium * make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric());
    auto r = in_blob ? 1.0e-5f : 1.0e-3f;
    writer.add_triangle(center + r * make_float3(1.0f, 0.0f, 0.0f),
                        center + r * make_float3(-0.5f, 0.8660254f, 0.0f),
                        center + r * make_float3(-0.5f, -0.8660254f, 0.0f));
}

// `instance-chain` - K meshes on a diagonal chain whose spacing decays like
// 2^-i: the TLAS is itself a bad chain of heavily overlapping instance AABBs,
// and the build exercises K separate radix sorts (one work-group each).
void generate_instance_chain(BenchScene &scene, size_t triangles, size_t instances,
                             uint64_t seed) noexcept {
    SoupWriter writer{&scene};
    auto axis = normalize(make_float3(1.0f, 1.0f, 1.0f));
    for (auto m = 0u; m < instances; m++) {
        auto begin = triangles * m / instances;
        auto end = triangles * (m + 1u) / instances;
        auto t = instances > 1u ? static_cast<float>(m) / static_cast<float>(instances - 1u) : 0.0f;
        auto distance = std::ldexp(1.0f, static_cast<int>(-10.0f * t));
        auto center = distance * axis;
        auto radius = 0.05f * distance;
        writer.begin_mesh();
        for (auto i = begin; i < end; i++) {
            auto rng = triangle_rng(seed, i);
            auto c = center + radius * rng.in_unit_ball();
            auto r = 0.1f * radius;
            writer.add_triangle(c + r * make_float3(1.0f, 0.0f, 0.0f),
                                c + r * make_float3(-0.5f, 0.8660254f, 0.0f),
                                c + r * make_float3(-0.5f, -0.8660254f, 0.0f));
        }
        writer.end_mesh();
    }
}

// ---------------------------------------------------------------------------
// Ray distributions
// ---------------------------------------------------------------------------

// Frustum looking at the scene from a fixed direction, like the demo.
[[nodiscard]] BenchRaySetup camera_setup(float3 lo, float3 hi) noexcept {
    auto center = (lo + hi) * 0.5f;
    auto radius = std::max(0.5f * length(hi - lo), 1.0e-6f);
    auto direction = normalize(make_float3(0.55f, 0.35f, 0.75f));
    BenchRaySetup setup;
    setup.mode = BenchRayMode::CAMERA;
    setup.eye = center + direction * (radius * 2.5f);
    setup.forward = normalize(center - setup.eye);
    setup.right = normalize(cross(setup.forward, make_float3(0.0f, 1.0f, 0.0f)));
    setup.up = cross(setup.right, setup.forward);
    auto half_height = static_cast<float>(std::tan(22.5 * 3.14159265358979323846 / 180.0));
    setup.half_w = half_height;
    setup.half_h = half_height;
    setup.axis = setup.forward;
    return setup;
}

// Origins on a shell around a degenerate region, aimed through it.
[[nodiscard]] BenchRaySetup blob_setup(float3 center, float radius, float3 axis) noexcept {
    BenchRaySetup setup;
    setup.mode = BenchRayMode::BLOB;
    setup.blob_center = center;
    setup.blob_radius = radius;
    setup.half_w = radius * 3.0f;// the shell radius
    setup.half_h = radius;
    setup.forward = axis;
    setup.axis = axis;
    return setup;
}

// Rays nearly parallel to a chain axis, spread over a disc perpendicular to it.
[[nodiscard]] BenchRaySetup axis_setup(float3 lo, float3 hi, float3 axis) noexcept {
    auto center = (lo + hi) * 0.5f;
    auto radius = std::max(0.5f * length(hi - lo), 1.0e-6f);
    auto direction = normalize(axis);
    BenchRaySetup setup;
    setup.mode = BenchRayMode::AXIS;
    setup.eye = center - direction * (radius * 4.0f);
    setup.forward = direction;
    setup.axis = direction;
    setup.right = perpendicular(direction);
    setup.up = cross(setup.right, direction);
    setup.half_w = radius;
    setup.half_h = radius;
    return setup;
}

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

constexpr BenchSceneInfo kSceneTable[] = {
    // The `dispatch_note` of every scene is the *measured* cost of the longest
    // single device submission at the default sizes of that scene (release, RTX
    // 4060).  One submission is what the Windows driver resets the device for
    // (TDR, ~2 s), and the traversal cost does not scale linearly with the ray
    // count: a small dispatch cannot hide the memory latency of the walk, so the
    // whole range in one submission is usually the *cheapest* way to trace it
    // (e.g. on `coincident`: 1.1 ms/ray at 1024 rays against 8.2 us/ray at
    // 262144 rays).  The defaults therefore keep the whole traversal of a scene
    // in one submission of a few hundred milliseconds, and the benchmark still
    // pre-flights the build and slices the traversal for anything larger.
    {"uniform", "both", "baseline: no worst case, every ratio is taken against it",
     "262144 rays in one 3.4 ms submission; 1M-triangle build: node stage 159 ms",
     1u << 20u, 1u, 262144u},
    {"coincident", "build", "build: N identical Morton codes (single sort bin, equal-code delta)",
     "262144 triangles (every centroid still identical, so the build keeps its worst case: "
     "one non-empty radix-sort bin and the equal-code `delta` path) + 65536 rays: 255 ms in "
     "one submission.  At 1M triangles the walk enters ~337k nodes per ray and needs "
     "1.1-1.5 s for *any* ray count (measured), i.e. no safe submission exists for it",
     1u << 18u, 1u, 65536u},
    {"grid-duplicates", "build", "build: 1024 clustered codes with exact duplicates inside each cell",
     "262144 rays in one 2.3 ms submission; clustered codes, almost nothing is hit",
     1u << 20u, 1u, 262144u},
    {"exponential", "build", "build: Morton caterpillar, one common-prefix bit lost per step",
     "262144 rays in one 39 ms submission (149 ns/ray, culled_ratio 0.005); build node stage 127 ms",
     1u << 20u, 1u, 262144u},
    {"line", "build", "build: deep thin chain of overlapping AABBs on the diagonal",
     "262144 rays in one 75 ms submission (285 ns/ray); build node stage 137 ms",
     1u << 20u, 1u, 262144u},
    {"sliver-soup", "traversal", "traversal: O(N) node visits and triangle tests per ray",
     "16384 triangles + 65536 rays: 159 ms in one submission (2.4 us/ray); 1M triangles "
     "would need a multi-second submission",
     1u << 14u, 1u, 65536u},
    {"bimodal", "both", "both: teapot-in-a-stadium scale disparity, blob AABBs all overlap",
     "262144 triangles + 8192 rays: 198 ms in one submission (24 us/ray, linear in the "
     "primitive count, zero culling); 1M triangles needs 1.0 s (measured)",
     1u << 18u, 1u, 8192u},
    {"instance-chain", "both", "both: K chained meshes, K radix sorts, chained TLAS",
     "262144 rays in one 2.5 ms submission; 256 trees, the largest single build stage "
     "(one tree) is ~78 ms",
     1u << 20u, 256u, 262144u},
};

[[nodiscard]] const BenchSceneInfo *find_info(const char *name) noexcept {
    for (auto &&info : kSceneTable) {
        if (std::strcmp(info.name, name) == 0) { return &info; }
    }
    return nullptr;
}

void generate(const BenchSceneInfo &info, BenchScene &scene, size_t triangles,
              size_t instances, uint64_t seed) noexcept {
    scene.name = info.name;
    scene.stress = info.stress;
    scene.worst_case = info.worst_case;
    scene.dispatch_note = info.dispatch_note;
    scene.default_triangles = info.default_triangles;
    scene.default_instances = info.default_instances;
    scene.default_rays = info.default_rays;
    auto name = luisa::string_view{info.name};
    if (name == "uniform") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_uniform(w, i, seed); });
    } else if (name == "coincident") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_coincident(w, i); });
    } else if (name == "grid-duplicates") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_grid_duplicates(w, i); });
    } else if (name == "exponential") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_exponential(w, i); });
    } else if (name == "line") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_line(w, i, triangles); });
    } else if (name == "sliver-soup") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_sliver_soup(w, i, seed); });
    } else if (name == "bimodal") {
        emit_chunked(scene, triangles, instances,
                     [&](SoupWriter &w, size_t i) { emit_bimodal(w, i, seed); });
    } else if (name == "instance-chain") {
        generate_instance_chain(scene, triangles, instances, seed);
    }
    update_scene_bounds(scene);
    // The ray distribution is derived from the geometry that was just
    // generated: it is what makes the worst case of the scene *reachable*
    // (rays that never touch the degenerate region would measure nothing).
    if (name == "sliver-soup") {
        scene.rays = blob_setup(make_float3(0.0f), 0.5f, make_float3(0.0f, 0.0f, 1.0f));
    } else if (name == "bimodal") {
        scene.rays = blob_setup(make_float3(10.0f), 0.01f, make_float3(0.0f, 0.0f, 1.0f));
    } else if (name == "line") {
        scene.rays = axis_setup(scene.scene_lo, scene.scene_hi, make_float3(1.0f, 1.0f, 1.0f));
    } else if (name == "instance-chain") {
        scene.rays = axis_setup(scene.scene_lo, scene.scene_hi, make_float3(1.0f, 1.0f, 1.0f));
    } else {
        scene.rays = camera_setup(scene.scene_lo, scene.scene_hi);
    }
}

}// namespace

luisa::span<const BenchSceneInfo> bench_scene_catalogue() noexcept {
    return luisa::span{kSceneTable};
}

bool make_bench_scene(const char *name, size_t triangles, size_t instances, size_t rays,
                      uint64_t seed, BenchScene &scene, luisa::string &error) noexcept {
    auto info = find_info(name);
    if (info == nullptr) {
        error = luisa::format("unknown scene '{}'", name);
        return false;
    }
    auto effective_triangles = triangles == 0u ? info->default_triangles : triangles;
    auto effective_instances = instances == 0u ? info->default_instances : instances;
    if (effective_triangles == 0u) {
        error = luisa::format("scene '{}' needs at least one triangle", name);
        return false;
    }
    // A BLAS cannot be empty, and a mesh cannot hold more triangles than the
    // scene has, so the mesh count is clamped instead of failing.
    effective_instances = std::min(effective_instances, effective_triangles);
    effective_instances = std::max<size_t>(effective_instances, 1u);
    generate(*info, scene, effective_triangles, effective_instances, seed);
    update_scene_bounds(scene);
    if (scene.meshes.empty() || scene.triangles.empty()) {
        error = luisa::format("scene '{}' generated no geometry", name);
        return false;
    }
    (void)rays;// the ray count lives in the options; the setup is scene-side
    return true;
}

// ---------------------------------------------------------------------------
// OBJ scenes
// ---------------------------------------------------------------------------

bool load_obj(const char *path, HostMesh &mesh, luisa::string &error,
              size_t max_triangles) noexcept {
    std::ifstream file{path};
    if (!file.is_open()) {
        error = luisa::format("cannot open '{}'", path);
        return false;
    }
    auto line = luisa::string{};
    auto line_number = size_t{0u};
    while (std::getline(file, line)) {
        line_number++;
        // strip the comment, then the trailing CR of a DOS file
        if (auto comment = line.find('#'); comment != luisa::string::npos) {
            line.resize(comment);
        }
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        auto begin = line.find_first_not_of(" \t");
        if (begin == luisa::string::npos) { continue; }
        auto keyword = luisa::string_view{line}.substr(begin);
        auto head = keyword.substr(0u, 1u);
        // "v <x> <y> <z>" and "f ..." are the only lines that matter; vt/vn/vp,
        // o/g/s/usemtl/mtllib and everything else is skipped.
        auto is_vertex = head == "v" && (keyword.size() == 1u || keyword[1u] == ' ' ||
                                         keyword[1u] == '\t');
        auto is_face = head == "f" && (keyword.size() == 1u || keyword[1u] == ' ' ||
                                       keyword[1u] == '\t');
        if (!is_vertex && !is_face) { continue; }
        // tokenize the rest of the line
        luisa::vector<luisa::string_view> tokens;
        auto cursor = keyword.data() + 1u;
        auto end = keyword.data() + keyword.size();
        while (cursor < end) {
            while (cursor < end && (*cursor == ' ' || *cursor == '\t')) { cursor++; }
            if (cursor >= end) { break; }
            auto token = cursor;
            while (cursor < end && *cursor != ' ' && *cursor != '\t') { cursor++; }
            tokens.emplace_back(token, static_cast<size_t>(cursor - token));
        }
        if (is_vertex) {
            if (tokens.size() < 3u) {
                error = luisa::format("{}:{}: a vertex needs three coordinates", path, line_number);
                return false;
            }
            auto parse = [&](luisa::string_view token, float &value) noexcept {
                auto buffer = luisa::string{token};
                char *parse_end = nullptr;
                value = std::strtof(buffer.c_str(), &parse_end);
                return parse_end != nullptr && parse_end != buffer.c_str();
            };
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (!parse(tokens[0], x) || !parse(tokens[1], y) || !parse(tokens[2], z)) {
                error = luisa::format("{}:{}: malformed vertex", path, line_number);
                return false;
            }
            mesh.vertices.emplace_back(make_float3(x, y, z));
        } else {
            if (max_triangles != 0u && mesh.triangles.size() >= max_triangles) {
                break;// a bounded probe of a possibly huge asset
            }
            if (tokens.size() < 3u) {
                error = luisa::format("{}:{}: a face needs at least three vertices", path, line_number);
                return false;
            }
            luisa::vector<uint> indices;
            indices.reserve(tokens.size());
            for (auto &&token : tokens) {
                // the index is everything before the first '/'; negative indices
                // are relative to the number of vertices read so far
                auto slash = token.find('/');
                auto text = slash == luisa::string_view::npos ? token : token.substr(0u, slash);
                if (text.empty()) {
                    error = luisa::format("{}:{}: malformed face index", path, line_number);
                    return false;
                }
                auto buffer = luisa::string{text};
                char *parse_end = nullptr;
                auto value = std::strtol(buffer.c_str(), &parse_end, 10);
                if (parse_end == nullptr || parse_end == buffer.c_str() || *parse_end != '\0') {
                    error = luisa::format("{}:{}: malformed face index '{}'", path, line_number, text);
                    return false;
                }
                auto count = static_cast<long>(mesh.vertices.size());
                auto index = value > 0 ? value - 1 : count + value;
                if (index < 0 || index >= count) {
                    error = luisa::format("{}:{}: face index {} is out of range", path, line_number, value);
                    return false;
                }
                indices.emplace_back(static_cast<uint>(index));
            }
            // fan triangulation: (0, i, i + 1)
            for (auto i = 1u; i + 1u < indices.size(); i++) {
                mesh.add_triangle(indices[0], indices[i], indices[i + 1u]);
            }
        }
    }
    if (mesh.triangles.empty()) {
        error = luisa::format("'{}' contains no triangle", path);
        return false;
    }
    mesh.update_bounds();
    return true;
}

bool make_obj_scene(const char *path, size_t instances, size_t rays, uint64_t seed,
                    BenchScene &scene, luisa::string &error, size_t max_triangles) noexcept {
    (void)seed;
    HostMesh mesh;
    if (!load_obj(path, mesh, error, max_triangles)) { return false; }
    scene.name = "mesh";
    scene.stress = "both";
    scene.worst_case = "real asset: whatever its mesh hierarchy and density stress";
    scene.dispatch_note = "an asset has no default size: the pre-flight measures it "
                          "as loaded and refuses a size it cannot build safely";
    scene.default_triangles = mesh.triangles.size();
    scene.default_instances = instances == 0u ? 1u : instances;
    scene.default_rays = rays == 0u ? 262144u : rays;
    scene.vertices = mesh.vertices;
    scene.triangles = mesh.triangles;
    auto triangle_count = static_cast<uint>(scene.triangles.size());
    scene.meshes.emplace_back(MeshRange{0u, triangle_count, mesh.lo, mesh.hi});
    // The copies are laid out on a square grid with a half-extent gap, so the
    // instance AABBs never overlap: the TLAS stays cheap and the numbers stay
    // attributable to the BLAS the asset describes.
    auto count = std::max<size_t>(scene.default_instances, 1u);
    auto side = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count))));
    auto extent = mesh.hi - mesh.lo;
    auto pitch = std::max(extent.x, std::max(extent.y, extent.z)) * 1.5f;
    auto offset_lo = make_float3(1.0e30f);
    auto offset_hi = make_float3(-1.0e30f);
    for (auto m = 0u; m < count; m++) {
        auto cx = static_cast<float>(m % side);
        auto cy = static_cast<float>(m / side);
        auto offset = pitch * make_float3(cx - 0.5f * static_cast<float>(side - 1u),
                                          cy - 0.5f * static_cast<float>(side - 1u), 0.0f);
        offset_lo = min(offset_lo, offset);
        offset_hi = max(offset_hi, offset);
        scene.instances.emplace_back(InstanceSpec{0u, translation(offset)});
    }
    if (mesh.triangles.empty() || mesh.vertices.empty()) {
        error = luisa::format("'{}' has vertices but no usable geometry", path);
        return false;
    }
    // world-space AABB of the *instanced* scene: the camera is aimed at that
    scene.scene_lo = mesh.lo + offset_lo;
    scene.scene_hi = mesh.hi + offset_hi;
    scene.rays = camera_setup(scene.scene_lo, scene.scene_hi);
    return true;
}

}// namespace luisa::example::lbvh
