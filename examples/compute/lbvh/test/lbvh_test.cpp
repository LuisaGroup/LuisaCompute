// Independent correctness tests of the software two-level LBVH.
//
// `example_software_lbvh` (the demo) validates one hand-picked scene against the
// hardware RTX reference.  This target is the safety net for the cases the
// demo's single scene cannot reach:
//
//   * the boundary sizes of every build dispatch (the tail guards and the
//     `count == 1` special case of `build_tree`),
//   * degenerate geometry (coincident triangles, one shared centroid with every
//     vertex order, zero-area/sliver triangles, axis-aligned quads, a scene of
//     coordinates from 1e-6 to 1e6, a flat scene whose rays have a zero
//     direction component),
//   * multi-BLAS / multi-instance scenes with identity, translation, rotation,
//     uniform/non-uniform scale, a mirrored and a 1e-4 scale transform,
//   * a seeded randomized property loop,
//   * strided-slice invariance and build/trace determinism (including the node
//     buffers of two independent storages).
//
// The reference is *independent* of the library: every ray is intersected on the
// host against every triangle of every instance with the same two-sided
// Moller-Trumbore formula and the same t_min/t_max semantics the library uses
// (`lbvh_common.h`), and the closest hit is taken.  The hardware RTX traversal
// is cross-checked against that same host reference as well, so a bug shared by
// "software LBVH + RTX" cannot hide.  Disagreements are classified with the
// benchmark's rules (see `benchmark_lbvh.cpp`): hit/miss, closest distance beyond
// the tolerance, or a different primitive at a different distance are fatal;
// grazing rays (< ~1.7 degrees to the triangle plane), hits within the
// barycentric tolerance of a triangle edge and exact same-distance ties between
// different triangles are counted and reported but are not fatal.
//
// Two scenes this test has to cover (a 1e-4-scale instance, and sliver /
// zero-area triangles) leave the regime in which the intersection *point* is
// comparable at all between a float Moller-Trumbore, a double one and the
// hardware's watertight fixed-point test: the barycentrics of a hit are off by
// `eps * (rounded ray origin) / (triangle's minimum altitude)`, which the
// `barycentric_error_bound` measurement below turns into an estimated error that
// widens the benchmark's boundary rule and the barycentric tolerance (and, for a
// determinant at the `|det| > 1e-12` guard the software test has and the hardware
// test does not, excludes the hit entirely).  Both measurements are derived from
// the scene's own geometry with the benchmark's own tolerance constants; every
// ray they affect is counted per check (`conditioned=` / `guard=`) and in the
// summary, never dropped silently.
//
// The library is driven only through its public API.  In particular no node
// member is ever read on the host: the tree checks use `validate_tree()` plus the
// node/leaf counts the API reports, and the determinism check compares the raw
// bytes of the node buffers (skipping the padding lanes of the `float3` members,
// which the build stores with an undefined value - the mask comes from the
// library's own `LUISA_STRUCT` reflection, so it names no member either).
//
// Usage: example_software_lbvh_test <backend> [--verbose] [--seed n] [--quick]
// Exit code 0 = every check passed, non-zero = at least one check failed.

#include "../lbvh_common.h"
#include "../software_lbvh.h"

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::lbvh;

namespace {

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Options {
    luisa::string backend;
    bool verbose{false};
    bool quick{false};
    uint64_t seed{0u};
};

void print_usage(const char *executable) noexcept {
    std::printf("usage: %s <backend> [--verbose] [--seed n] [--quick]\n", executable);
}

[[nodiscard]] bool parse_options(int argc, char *const *argv, Options &options) noexcept {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') { return false; }
    options.backend = argv[1];
    for (auto i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { continue; }
        if (std::strcmp(argv[i], "--verbose") == 0) {
            options.verbose = true;
        } else if (std::strcmp(argv[i], "--quick") == 0) {
            options.quick = true;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc && argv[i + 1] != nullptr) {
            options.seed = std::strtoull(argv[++i], nullptr, 10);
        } else {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host-side plumbing: reproducible RNG, double-precision vector math
// ---------------------------------------------------------------------------

// splitmix64: deterministic, portable, and independent of the C++ standard
// library's distribution implementations (so every backend sees the same rays).
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) noexcept
        : state{seed + 0x9E3779B97F4A7C15ull} {}
    [[nodiscard]] uint64_t next_u64() noexcept {
        state += 0x9E3779B97F4A7C15ull;
        auto z = state;
        z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31u);
    }
    [[nodiscard]] uint32_t next_u32() noexcept { return static_cast<uint32_t>(next_u64() >> 32u); }
    [[nodiscard]] float next_float() noexcept {
        return static_cast<float>(next_u32() >> 8u) * (1.0f / 16777216.0f);
    }
    [[nodiscard]] float symmetric() noexcept { return next_float() * 2.0f - 1.0f; }
    [[nodiscard]] uint32_t below(uint32_t n) noexcept { return n == 0u ? 0u : next_u32() % n; }
};

[[nodiscard]] inline double3 as_double(float3 v) noexcept {
    return double3{static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

// One row of a transform applied to a (point, w) pair; the library stores the
// transforms as explicit rows (`LbvhInstance`), and this is the same evaluation.
[[nodiscard]] inline double row_dot(float4 row, double3 p, double w) noexcept {
    return static_cast<double>(row.x) * p.x + static_cast<double>(row.y) * p.y +
           static_cast<double>(row.z) * p.z + static_cast<double>(row.w) * w;
}

[[nodiscard]] float3 random_unit(Rng &rng) noexcept {
    auto z = rng.symmetric();
    auto phi = rng.next_float() * 6.283185307179586f;
    auto r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return make_float3(r * std::cos(phi), r * std::sin(phi), z);
}

// ---------------------------------------------------------------------------
// Host scene: exactly the geometry that is uploaded to the device.
// ---------------------------------------------------------------------------

struct TestMesh {
    uint triangle_offset{};
    uint triangle_count{};
    float3 lo{};
    float3 hi{};
};

struct TestInstance {
    uint mesh{};
    float4x4 to_world{};
};

struct TestScene {
    luisa::string label;
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    luisa::vector<TestMesh> meshes;
    luisa::vector<TestInstance> instances;
    luisa::vector<LbvhRay> rays;
};

// Small builder so a scene's BLAS bounds are computed from the triangles the
// mesh actually uses (which is what `create_blas` requires).
struct SceneBuilder {
    TestScene scene;

    uint vertex(float3 p) noexcept {
        scene.vertices.emplace_back(p);
        return static_cast<uint>(scene.vertices.size() - 1u);
    }
    void triangle(uint a, uint b, uint c) noexcept {
        scene.triangles.emplace_back(Triangle{a, b, c});
    }
    [[nodiscard]] uint begin_mesh() const noexcept {
        return static_cast<uint>(scene.triangles.size());
    }
    [[nodiscard]] uint end_mesh(uint first_triangle) noexcept {
        TestMesh mesh;
        mesh.triangle_offset = first_triangle;
        mesh.triangle_count = static_cast<uint>(scene.triangles.size()) - first_triangle;
        if (mesh.triangle_count == 0u) { return invalid_node; }
        auto lo = make_float3(1.0e30f);
        auto hi = make_float3(-1.0e30f);
        for (auto i = first_triangle; i < static_cast<uint>(scene.triangles.size()); i++) {
            auto t = scene.triangles[i];
            for (auto index : {t.i0, t.i1, t.i2}) {
                auto v = scene.vertices[index];
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
        mesh.lo = lo;
        mesh.hi = hi;
        scene.meshes.emplace_back(mesh);
        return static_cast<uint>(scene.meshes.size() - 1u);
    }
    uint instance(uint mesh, const float4x4 &to_world) noexcept {
        scene.instances.emplace_back(TestInstance{mesh, to_world});
        return static_cast<uint>(scene.instances.size() - 1u);
    }
};

[[nodiscard]] uint emit_triangle(SceneBuilder &builder, float3 p0, float3 p1, float3 p2) noexcept {
    auto a = builder.vertex(p0);
    auto b = builder.vertex(p1);
    auto c = builder.vertex(p2);
    builder.triangle(a, b, c);
    return a;
}

// A small, deliberately non-degenerate triangle around `center`.
void emit_random_triangle(SceneBuilder &builder, Rng &rng, float3 center, float size) noexcept {
    auto normal = random_unit(rng);
    auto helper = std::abs(normal.x) < 0.9f ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    auto tangent = normalize(cross(normal, helper));
    auto bitangent = cross(normal, tangent);
    auto p0 = center + (tangent * rng.symmetric() + bitangent * rng.symmetric()) * (0.5f * size);
    auto p1 = center + (tangent * rng.symmetric() + bitangent * rng.symmetric()) * (0.5f * size);
    auto p2 = center + (tangent * rng.symmetric() + bitangent * rng.symmetric()) * (0.5f * size);
    // never let a "random" triangle become (nearly) degenerate: the boundary
    // sizes are about triangles a ray can actually hit.
    if (length(cross(p1 - p0, p2 - p0)) < 0.05f * size * size) {
        p1 = p0 + tangent * size;
        p2 = p0 + bitangent * size;
    }
    emit_triangle(builder, p0, p1, p2);
}

void scene_bounds(const TestScene &scene, float3 &lo, float3 &hi) noexcept {
    lo = make_float3(1.0e30f);
    hi = make_float3(-1.0e30f);
    for (auto &&instance : scene.instances) {
        auto mesh = scene.meshes[instance.mesh];
        for (auto i = 0u; i < mesh.triangle_count; i++) {
            auto t = scene.triangles[mesh.triangle_offset + i];
            for (auto index : {t.i0, t.i1, t.i2}) {
                auto p = (instance.to_world * make_float4(scene.vertices[index], 1.0f)).xyz();
                lo = min(lo, p);
                hi = max(hi, p);
            }
        }
    }
}

[[nodiscard]] float3 world_vertex(const TestScene &scene, uint instance, uint index) noexcept {
    return (scene.instances[instance].to_world * make_float4(scene.vertices[index], 1.0f)).xyz();
}

// ---------------------------------------------------------------------------
// Ray generation (host side, so every backend sees bit-identical rays)
// ---------------------------------------------------------------------------

// The three ways a targeted ray can be cast at the interior of a triangle.  The
// two "must miss" modes are what pins down the t_min / t_max semantics: the same
// ray has to miss in the library and in the host reference.
enum struct TargetMode : uint {
    HIT = 0u,       // the ray hits the triangle
    BELOW_T_MIN = 1,// t_min above the distance: must miss
    ABOVE_T_MAX = 2,// t_max below the distance: must miss
};

// A ray aimed at the interior of the triangle `prim` of `instance`: the origin
// sits on the triangle's normal at a distance proportional to the triangle's own
// size (so it works for a 1e-4-scale instance and for a 1e6-sized one alike) and
// the aim point is kept well away from the edges.
void add_targeted_ray(TestScene &scene, uint instance, uint prim, Rng &rng,
                      TargetMode mode) noexcept {
    auto &&inst = scene.instances[instance];
    auto mesh = scene.meshes[inst.mesh];
    auto t = scene.triangles[mesh.triangle_offset + prim];
    auto v0 = world_vertex(scene, instance, t.i0);
    auto v1 = world_vertex(scene, instance, t.i1);
    auto v2 = world_vertex(scene, instance, t.i2);
    auto center = (v0 + v1 + v2) * (1.0f / 3.0f);
    auto normal = cross(v1 - v0, v2 - v0);
    auto twice_area = length(normal);
    auto diagonal = length(max(max(v0, v1), v2) - min(min(v0, v1), v2));
    auto scale = std::max(diagonal, 1.0e-12f);
    auto n = twice_area > 0.0f ? normal * (1.0f / twice_area) : make_float3(0.0f, 0.0f, 1.0f);
    // a barycentric point with every weight in [0.2, 0.8] / sum
    auto w0 = 0.2f + 0.6f * rng.next_float();
    auto w1 = 0.2f + 0.6f * rng.next_float();
    auto w2 = 0.2f + 0.6f * rng.next_float();
    auto aim = (v0 * w0 + v1 * w1 + v2 * w2) * (1.0f / (w0 + w1 + w2));
    auto distance = 2.0f * scale;
    LbvhRay ray;
    ray.origin = aim + n * distance;
    ray.direction = normalize(aim - ray.origin);
    ray.t_max = 1.0e30f;
    switch (mode) {
        case TargetMode::HIT:
            ray.t_min = 0.25f * distance;
            break;
        case TargetMode::BELOW_T_MIN:
            ray.t_min = 1.5f * distance;
            break;
        default:
            ray.t_min = 0.0f;
            ray.t_max = 0.5f * distance;
            break;
    }
    scene.rays.emplace_back(ray);
}

// Binary-coded target modes, so every mode is exercised on every scene.
[[nodiscard]] TargetMode target_mode_for(size_t index) noexcept {
    switch (index % 4u) {
        case 0u: return TargetMode::HIT;
        case 1u: return TargetMode::HIT;
        case 2u: return TargetMode::BELOW_T_MIN;
        default: return TargetMode::ABOVE_T_MAX;
    }
}

void add_targeted_rays(TestScene &scene, size_t count, uint64_t seed) noexcept {
    if (count == 0u || scene.instances.empty()) { return; }
    Rng rng(seed ^ 0x0BADC0DEu);
    for (auto i = 0u; i < count; i++) {
        auto instance = rng.below(static_cast<uint32_t>(scene.instances.size()));
        auto mesh = scene.meshes[scene.instances[instance].mesh];
        auto prim = rng.below(mesh.triangle_count);
        add_targeted_ray(scene, instance, prim, rng, target_mode_for(i));
    }
}

// General rays: a shell around the scene, origins in a box, rays exactly along a
// coordinate axis and rays with the direction exactly (0, 0, 1).
void add_general_rays(TestScene &scene, size_t count, uint64_t seed, bool axis_aligned,
                      float t_min, float t_max) noexcept {
    float3 lo;
    float3 hi;
    scene_bounds(scene, lo, hi);
    auto center = (lo + hi) * 0.5f;
    auto extent = max(hi - lo, make_float3(1.0e-3f));
    auto radius = 0.5f * length(extent);
    Rng rng(seed ^ 0xA5A5A5A5u);
    for (auto i = 0u; i < count; i++) {
        LbvhRay ray;
        ray.t_min = t_min;
        ray.t_max = t_max;
        auto mode = static_cast<uint>(i % 8u);
        if (mode < 4u) {
            // a shell around the scene, aimed at a point inside it
            auto offset = random_unit(rng) * (radius * (1.3f + rng.next_float()));
            auto target = center + make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                                       (0.5f * extent);
            ray.origin = center + offset;
            ray.direction = normalize(target - ray.origin);
        } else if (mode < 6u) {
            auto origin = center + make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                                       (1.6f * extent);
            auto target = center + make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                                       (0.5f * extent);
            ray.origin = origin;
            ray.direction = normalize(target - origin);
        } else if (mode == 6u && axis_aligned) {
            // exactly axis-aligned: one direction component is exactly zero
            auto axis = static_cast<uint>(rng.below(6u));
            auto sign = (axis % 2u) == 0u ? 1.0f : -1.0f;
            ray.origin = center + make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                                      (1.5f * extent);
            ray.direction = axis / 2u == 0u ? make_float3(sign, 0.0f, 0.0f) : axis / 2u == 1u ? make_float3(0.0f, sign, 0.0f) :
                                                                                                make_float3(0.0f, 0.0f, sign);
        } else {
            // exactly (0, 0, 1): the direction is a unit axis vector, which is
            // what makes the reciprocal of a component and a shared coordinate
            // (z == const) enormous in the slab test
            ray.origin = center + make_float3(rng.symmetric() * extent.x,
                                              rng.symmetric() * extent.y,
                                              -(1.5f * radius + 1.0f));
            ray.direction = make_float3(0.0f, 0.0f, 1.0f);
        }
        scene.rays.emplace_back(ray);
    }
}

// ---------------------------------------------------------------------------
// The independent host reference: brute force over every triangle
// ---------------------------------------------------------------------------

struct RefInstance {
    uint mesh{};
    float4x4 to_world{};
    float4x4 to_object{};
};

struct RefScene {
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    luisa::vector<TestMesh> meshes;
    luisa::vector<RefInstance> instances;
};

// The two-sided Moller-Trumbore test of `triangle_test` (lbvh_common.h) with the
// same acceptance rules, in double precision.
[[nodiscard]] bool ref_triangle_test(const RefScene &scene, uint triangle_index,
                                     double3 origin, double3 direction, double t_min,
                                     double t_max, double &t_out, double &u_out,
                                     double &v_out) noexcept {
    auto tri = scene.triangles[triangle_index];
    auto v0 = as_double(scene.vertices[tri.i0]);
    auto v1 = as_double(scene.vertices[tri.i1]);
    auto v2 = as_double(scene.vertices[tri.i2]);
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pv = cross(direction, e2);
    auto det = dot(e1, pv);
    if (!(std::abs(det) > 1.0e-12)) { return false; }// same guard as triangle_test
    auto inv_det = 1.0 / det;
    auto tv = origin - v0;
    auto u = dot(tv, pv) * inv_det;
    if (!(u >= 0.0)) { return false; }
    auto qv = cross(tv, e1);
    auto v = dot(direction, qv) * inv_det;
    if (!(v >= 0.0) || !(u + v <= 1.0)) { return false; }
    auto t = dot(e2, qv) * inv_det;
    if (!(t >= t_min) || !(t <= t_max)) { return false; }
    t_out = t;
    u_out = u;
    v_out = v;
    return true;
}

[[nodiscard]] RefScene make_ref_scene(const TestScene &scene) noexcept {
    RefScene ref;
    ref.vertices = scene.vertices;
    ref.triangles = scene.triangles;
    ref.meshes = scene.meshes;
    ref.instances.reserve(scene.instances.size());
    for (auto &&instance : scene.instances) {
        // the same object<->world pair the library uploads for the instance
        ref.instances.emplace_back(RefInstance{instance.mesh, instance.to_world,
                                               inverse(instance.to_world)});
    }
    return ref;
}

struct RefHit {
    bool hit{false};
    uint inst{invalid_node};
    uint prim{invalid_node};
    double t{0.0};
    double u{0.0};
    double v{0.0};
};

// Everything the double reference can say about one triangle of one ray: the
// diagnostics a failing check prints, so a disagreement can be judged without a
// debugger (a determinant at the guard, an ill-conditioned intersection point or
// a real miss all look different here).
struct RefProbe {
    bool valid{false};
    double det{0.0};
    double det_scale{0.0};
    double min_altitude{0.0};
    bool accepted{false};
    double t{0.0};
    double u{0.0};
    double v{0.0};
    double3 object_origin{};
    double3 object_direction{};
    double3 v0{}, v1{}, v2{};
};

[[nodiscard]] RefProbe probe_triangle(const RefScene &scene, const LbvhRay &ray,
                                      uint instance, uint primitive) noexcept {
    RefProbe probe;
    if (instance >= scene.instances.size()) { return probe; }
    auto mesh_index = scene.instances[instance].mesh;
    if (mesh_index >= scene.meshes.size()) { return probe; }
    auto range = scene.meshes[mesh_index];
    if (primitive >= range.triangle_count) { return probe; }
    auto tri = scene.triangles[range.triangle_offset + primitive];
    auto to_object = scene.instances[instance].to_object;
    auto origin = as_double(ray.origin);
    auto direction = as_double(ray.direction);
    auto r0 = matrix_row(to_object, 0u);
    auto r1 = matrix_row(to_object, 1u);
    auto r2 = matrix_row(to_object, 2u);
    auto object_origin = double3{row_dot(r0, origin, 1.0), row_dot(r1, origin, 1.0),
                                 row_dot(r2, origin, 1.0)};
    auto object_direction = double3{row_dot(r0, direction, 0.0), row_dot(r1, direction, 0.0),
                                    row_dot(r2, direction, 0.0)};
    auto v0 = as_double(scene.vertices[tri.i0]);
    auto v1 = as_double(scene.vertices[tri.i1]);
    auto v2 = as_double(scene.vertices[tri.i2]);
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto twice_area = std::sqrt(dot(cross(e1, e2), cross(e1, e2)));
    auto longest = std::max({std::sqrt(dot(e1, e1)), std::sqrt(dot(e2, e2)),
                             std::sqrt(dot(e2 - e1, e2 - e1))});
    probe.valid = true;
    probe.det = dot(e1, cross(object_direction, e2));
    probe.det_scale = std::sqrt(dot(e1, e1)) * std::sqrt(dot(e2, e2)) *
                      std::sqrt(dot(object_direction, object_direction));
    probe.min_altitude = longest > 0.0 ? twice_area / longest : 0.0;
    probe.object_origin = object_origin;
    probe.object_direction = object_direction;
    probe.v0 = v0;
    probe.v1 = v1;
    probe.v2 = v2;
    double t, u, v;
    probe.accepted = ref_triangle_test(scene, range.triangle_offset + primitive, object_origin,
                                       object_direction, static_cast<double>(ray.t_min),
                                       static_cast<double>(ray.t_max), t, u, v);
    probe.t = t;
    probe.u = u;
    probe.v = v;
    return probe;
}

// Closest hit of one ray against every triangle of every instance, with the ray
// transformed into the object space of the instance exactly as the two-level
// traversal does it (the direction is not renormalized, so `t` stays in the
// world-space ray parameter).
[[nodiscard]] RefHit ref_trace(const RefScene &scene, const LbvhRay &ray) noexcept {
    RefHit best;
    best.t = static_cast<double>(ray.t_max);
    auto origin = as_double(ray.origin);
    auto direction = as_double(ray.direction);
    auto t_min = static_cast<double>(ray.t_min);
    auto t_max = static_cast<double>(ray.t_max);
    for (auto i = 0u; i < scene.instances.size(); i++) {
        auto &&instance = scene.instances[i];
        auto r0 = matrix_row(instance.to_object, 0u);
        auto r1 = matrix_row(instance.to_object, 1u);
        auto r2 = matrix_row(instance.to_object, 2u);
        auto object_origin = double3{row_dot(r0, origin, 1.0), row_dot(r1, origin, 1.0),
                                     row_dot(r2, origin, 1.0)};
        auto object_direction = double3{row_dot(r0, direction, 0.0), row_dot(r1, direction, 0.0),
                                        row_dot(r2, direction, 0.0)};
        auto mesh = scene.meshes[instance.mesh];
        for (auto p = 0u; p < mesh.triangle_count; p++) {
            double t, u, v;
            if (!ref_triangle_test(scene, mesh.triangle_offset + p, object_origin,
                                   object_direction, t_min, t_max, t, u, v)) {
                continue;
            }
            if (!best.hit || t < best.t) {
                best.hit = true;
                best.inst = i;
                best.prim = p;
                best.t = t;
                best.u = u;
                best.v = v;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Classification of a disagreement (the benchmark's rules, reused verbatim)
// ---------------------------------------------------------------------------

// A tolerance-comparable view of an `LbvhHit` and of the host reference hit.
struct HitView {
    bool hit{false};
    uint inst{invalid_node};
    uint prim{invalid_node};
    double t{0.0};
    double u{0.0};
    double v{0.0};
};

[[nodiscard]] HitView view_of(const LbvhHit &hit) noexcept {
    return HitView{hit.inst != invalid_node, hit.inst, hit.prim,
                   static_cast<double>(hit.t), static_cast<double>(hit.bary.x),
                   static_cast<double>(hit.bary.y)};
}

[[nodiscard]] HitView view_of(const RefHit &hit) noexcept {
    return HitView{hit.hit, hit.inst, hit.prim, hit.t, hit.u, hit.v};
}

// The demo's tolerances: the hardware traversal has its own triangle intersecter,
// and the software one is a float evaluation of a formula the reference evaluates
// in double, so the distances agree only up to rounding.
constexpr double distance_tolerance = 1.0e-3;
constexpr double barycentric_tolerance = 5.0e-3;
// A hit within ~1.7 degrees of the plane of its triangle is not comparable
// between two intersecters (the condition number of the intersection distance is
// 1/sin(angle)); measured from the geometry, never assumed.
constexpr float grazing_sin_angle = 3.0e-2f;

// Sine of the angle between the ray and the plane of the triangle it hit: 0 for a
// ray parallel to the plane, 1 for a perpendicular one.  A degenerate triangle
// (zero area) has no plane, so it counts as grazing: neither traversal can be
// asked for a meaningful intersection point there.
[[nodiscard]] float grazing_sine(const RefScene &scene, const LbvhRay &ray, uint instance,
                                 uint primitive) noexcept {
    if (instance >= scene.instances.size()) { return 1.0f; }
    auto mesh_index = scene.instances[instance].mesh;
    if (mesh_index >= scene.meshes.size()) { return 1.0f; }
    auto range = scene.meshes[mesh_index];
    if (primitive >= range.triangle_count) { return 1.0f; }
    auto t = scene.triangles[range.triangle_offset + primitive];
    if (t.i0 >= scene.vertices.size() || t.i1 >= scene.vertices.size() ||
        t.i2 >= scene.vertices.size()) {
        return 1.0f;
    }
    auto to_world = scene.instances[instance].to_world;
    auto transform = [&to_world](float3 p) noexcept {
        return (to_world * make_float4(p, 1.0f)).xyz();
    };
    auto v0 = transform(scene.vertices[t.i0]);
    auto v1 = transform(scene.vertices[t.i1]);
    auto v2 = transform(scene.vertices[t.i2]);
    auto normal = cross(v1 - v0, v2 - v0);
    auto twice_area = length(normal);
    if (twice_area <= 0.0f) { return 0.0f; }
    return std::abs(dot(normal * (1.0f / twice_area), normalize(ray.direction)));
}

[[nodiscard]] bool is_grazing(const RefScene &scene, const LbvhRay &ray,
                              const HitView &hit) noexcept {
    return hit.hit && grazing_sine(scene, ray, hit.inst, hit.prim) < grazing_sin_angle;
}

// ---------------------------------------------------------------------------
// Comparability, measured from the geometry.
//
// The benchmark's rules (grazing rays, hits on a triangle edge, same-distance
// ties) come from the *triangle intersecter*: the software one is a float
// Moller-Trumbore, the hardware one a watertight fixed-point test, and the host
// reference a double Moller-Trumbore, so the intersection point itself is only
// comparable while the geometry keeps it well conditioned.  Two of the scenes
// this test has to cover (a 1e-4-scale instance and sliver / zero-area triangles)
// leave that regime, and the two measurements below are what say so: they are
// computed from the scene's own geometry, with the benchmark's own tolerances as
// the thresholds, and every ray they exclude is counted and reported (see
// `conditioned` / `guard_decided`), never silently dropped.
// ---------------------------------------------------------------------------

constexpr double float32_epsilon = 1.1920928955078125e-7;
// The guard of `triangle_test`: a determinant below this is rejected by the
// software test (the hardware watertight test has no such guard).
constexpr double determinant_guard = 1.0e-12;

// Object-space triangle data of one (instance, primitive) pair.
struct ObjectTriangle {
    bool valid{false};
    double3 v0{}, v1{}, v2{};
    double3 direction{};// the object-space ray direction
    double transform_rounding{0.0};
    double origin_magnitude{0.0};
    float4x4 to_object{};
    float3 world_origin{};
};

[[nodiscard]] ObjectTriangle object_triangle(const RefScene &scene, const LbvhRay &ray,
                                             uint instance, uint primitive) noexcept {
    ObjectTriangle out;
    if (instance >= scene.instances.size()) { return out; }
    auto mesh_index = scene.instances[instance].mesh;
    if (mesh_index >= scene.meshes.size()) { return out; }
    auto range = scene.meshes[mesh_index];
    if (primitive >= range.triangle_count) { return out; }
    auto tri = scene.triangles[range.triangle_offset + primitive];
    if (tri.i0 >= scene.vertices.size() || tri.i1 >= scene.vertices.size() ||
        tri.i2 >= scene.vertices.size()) {
        return out;
    }
    out.to_object = scene.instances[instance].to_object;
    out.world_origin = ray.origin;
    out.v0 = as_double(scene.vertices[tri.i0]);
    out.v1 = as_double(scene.vertices[tri.i1]);
    out.v2 = as_double(scene.vertices[tri.i2]);
    auto origin = as_double(ray.origin);
    auto direction = as_double(ray.direction);
    auto r0 = matrix_row(out.to_object, 0u);
    auto r1 = matrix_row(out.to_object, 1u);
    auto r2 = matrix_row(out.to_object, 2u);
    auto object_origin = double3{row_dot(r0, origin, 1.0), row_dot(r1, origin, 1.0),
                                 row_dot(r2, origin, 1.0)};
    out.direction = double3{row_dot(r0, direction, 0.0), row_dot(r1, direction, 0.0),
                            row_dot(r2, direction, 0.0)};
    // The float evaluation of the transform rounds the products; the largest row
    // sum of magnitudes bounds its absolute error, and the same order of
    // magnitude bounds the rounding of the `origin - v0` subtraction.
    for (auto r : {r0, r1, r2}) {
        auto term = std::abs(static_cast<double>(r.x) * origin.x) +
                    std::abs(static_cast<double>(r.y) * origin.y) +
                    std::abs(static_cast<double>(r.z) * origin.z) + std::abs(static_cast<double>(r.w));
        out.transform_rounding = std::max(out.transform_rounding, term);
    }
    out.origin_magnitude = std::max({std::abs(object_origin.x), std::abs(object_origin.y),
                                     std::abs(object_origin.z)});
    out.valid = true;
    return out;
}

// Bound of the absolute error of the *barycentric* coordinates of one float
// evaluation of a hit: the object-space ray origin is off by the rounding of the
// world->object transform (which a 1e-4-scale instance amplifies by 1e4) plus
// its own rounding, and a barycentric coordinate moves by that error divided by
// the triangle's minimum altitude.  Returns infinity for a triangle with no
// altitude at all (collinear), where the barycentrics are not defined.
//
// The safety factor covers the fact that the two implementations under
// comparison each transform the ray in their own arithmetic (the hardware
// intersecter uses its own, lower-precision internal transform), so the two
// object-space rays differ by more than the software's own rounding.
constexpr double conditioning_safety_factor = 8.0;

[[nodiscard]] double barycentric_error_bound(const RefScene &scene, const LbvhRay &ray,
                                             uint instance, uint primitive) noexcept {
    auto tri = object_triangle(scene, ray, instance, primitive);
    if (!tri.valid) { return 0.0; }
    auto e1 = tri.v1 - tri.v0;
    auto e2 = tri.v2 - tri.v0;
    auto twice_area = std::sqrt(dot(cross(e1, e2), cross(e1, e2)));
    auto longest = std::max({std::sqrt(dot(e1, e1)),
                             std::sqrt(dot(e2, e2)),
                             std::sqrt(dot(e2 - e1, e2 - e1))});
    if (!(twice_area > 0.0) || !(longest > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    auto min_altitude = twice_area / longest;
    auto error = float32_epsilon * (tri.transform_rounding + tri.origin_magnitude);
    return conditioning_safety_factor * error / min_altitude;
}

// True when the `|det| > 1e-12` guard of the software test decides this hit
// instead of the geometry, so that the two sides of a comparison legitimately
// disagree.  Measured from the geometry: the float evaluation of the determinant
// carries the rounding of a few products (`|e1| * |e2| * |dir|`), and
// `guard_is_shared` says whether the implementation on the other side of the
// comparison applies the same guard - the software LBVH and the host reference
// both do, the hardware watertight intersecter does not, and for it *every* hit
// on a determinant the guard rejects is a disagreement the traversal cannot help.
enum struct GuardSemantics : uint {
    SHARED = 0u,       // both sides apply `|det| > 1e-12`
    SOFTWARE_ONLY = 1u,// only the software / reference side does
};

[[nodiscard]] bool determinant_at_guard(const RefScene &scene, const LbvhRay &ray,
                                        uint instance, uint primitive,
                                        GuardSemantics semantics) noexcept {
    auto tri = object_triangle(scene, ray, instance, primitive);
    if (!tri.valid) { return false; }
    auto e1 = tri.v1 - tri.v0;
    auto e2 = tri.v2 - tri.v0;
    auto det = dot(e1, cross(tri.direction, e2));
    auto det_scale = std::sqrt(dot(e1, e1)) * std::sqrt(dot(e2, e2)) *
                     std::sqrt(dot(tri.direction, tri.direction));
    auto evaluation_error = 8.0 * float32_epsilon * det_scale;
    return semantics == GuardSemantics::SHARED ? std::abs(std::abs(det) - determinant_guard) <= evaluation_error : std::abs(det) <= determinant_guard + evaluation_error;
}

// Why one hit cannot be compared (see the two functions above); `NONE` means it
// can.
enum struct Exclusion : uint {
    NONE = 0u,
    GRAZING = 1u,
    BOUNDARY = 2u,
    CONDITIONED = 3u,// the barycentrics are not determined to the tolerance
    GUARD = 4u,      // the 1e-12 determinant guard decides the hit
};

// A hit is within `tolerance` of the boundary of its triangle: Moller-Trumbore
// and the watertight hardware intersecter use different edge rules there, so the
// two legitimately disagree on such a hit.
[[nodiscard]] bool near_boundary(const HitView &hit, double tolerance) noexcept {
    if (!hit.hit) { return false; }
    return std::min({hit.u, hit.v, 1.0 - hit.u - hit.v}) < tolerance;
}

[[nodiscard]] Exclusion classify(const RefScene &scene, const LbvhRay &ray,
                                 const HitView &hit, double error_bound,
                                 GuardSemantics guard) noexcept {
    if (!hit.hit) { return Exclusion::NONE; }
    if (is_grazing(scene, ray, hit)) { return Exclusion::GRAZING; }
    if (near_boundary(hit, barycentric_tolerance)) { return Exclusion::BOUNDARY; }
    if (near_boundary(hit, error_bound)) { return Exclusion::CONDITIONED; }
    if (determinant_at_guard(scene, ray, hit.inst, hit.prim, guard)) { return Exclusion::GUARD; }
    return Exclusion::NONE;
}

[[nodiscard]] Exclusion merge(Exclusion a, Exclusion b) noexcept {
    // the benchmark reports a grazing ray as grazing and an edge hit as boundary,
    // so those two keep their priority; the measured exclusions come last
    if (a == Exclusion::GRAZING || b == Exclusion::GRAZING) { return Exclusion::GRAZING; }
    if (a == Exclusion::BOUNDARY || b == Exclusion::BOUNDARY) { return Exclusion::BOUNDARY; }
    if (a == Exclusion::CONDITIONED || b == Exclusion::CONDITIONED) { return Exclusion::CONDITIONED; }
    if (a == Exclusion::GUARD || b == Exclusion::GUARD) { return Exclusion::GUARD; }
    return Exclusion::NONE;
}

struct MismatchExample {
    size_t ray{};
    const char *kind{};
    HitView a{};
    HitView b{};
};

struct HitComparison {
    size_t compared{0u};
    size_t miss_mismatch{0u};
    size_t distance_mismatch{0u};
    size_t id_mismatch{0u};
    size_t ties{0u};
    size_t grazing{0u};
    size_t boundary{0u};
    size_t conditioned{0u};
    size_t guard_decided{0u};
    size_t bary_mismatch{0u};
    double max_distance_error{0.0};
    double max_barycentric_error{0.0};
    double max_excluded_distance_error{0.0};
    double max_excluded_barycentric_error{0.0};
    luisa::vector<MismatchExample> examples;

    [[nodiscard]] size_t fatal() const noexcept {
        return miss_mismatch + distance_mismatch + id_mismatch + bary_mismatch;
    }
    // excluded from the comparison: the benchmark's rules plus the two measured ones
    [[nodiscard]] size_t excluded() const noexcept {
        return grazing + boundary + conditioned + guard_decided;
    }
};

constexpr size_t max_reported_examples = 6u;

// Compares two hit arrays for the same rays, classifying every disagreement with
// the benchmark's rules: `a` is the traversal under test, `b` the host reference.
void compare_hits(const RefScene &scene, luisa::span<const LbvhRay> rays,
                  luisa::span<const HitView> a, luisa::span<const HitView> b,
                  HitComparison &counts, GuardSemantics guard) noexcept {
    for (auto i = 0u; i < rays.size(); i++) {
        auto x = a[i];
        auto y = b[i];
        auto ray = rays[i];
        auto record = [&](const char *kind) noexcept {
            if (counts.examples.size() < max_reported_examples) {
                counts.examples.emplace_back(MismatchExample{i, kind, x, y});
            }
        };
        // How well conditioned the barycentrics of this pair are: the tolerance
        // they are comparable with is never tighter than the benchmark's, and it
        // grows with the measured float error (see `barycentric_error_bound`).
        auto error_a = barycentric_error_bound(scene, ray, x.inst, x.prim);
        auto error_b = barycentric_error_bound(scene, ray, y.inst, y.prim);
        auto pair_tolerance = std::max(barycentric_tolerance, std::max(error_a, error_b));
        auto exclusion = merge(classify(scene, ray, x, error_a, guard),
                               classify(scene, ray, y, error_b, guard));
        auto count_exclusion = [&counts](Exclusion why) noexcept {
            switch (why) {
                case Exclusion::GRAZING: counts.grazing++; break;
                case Exclusion::BOUNDARY: counts.boundary++; break;
                case Exclusion::CONDITIONED: counts.conditioned++; break;
                case Exclusion::GUARD: counts.guard_decided++; break;
                default: break;
            }
        };
        if (x.hit != y.hit) {
            // no second hit to compare with: the ray either grazes the triangle
            // the traversal hit, or hits it where the intersecters legitimately
            // disagree (its boundary, a determinant the guard decides, a
            // barycentric coordinate the float evaluation cannot resolve)
            if (exclusion == Exclusion::NONE) {
                counts.miss_mismatch++;
                record(x.hit ? "reference missed, traversal hit" : "traversal missed, reference hit");
            } else {
                count_exclusion(exclusion);
            }
            continue;
        }
        if (!x.hit) { continue; }
        auto distance_error = std::abs(x.t - y.t) / std::max(1.0, std::abs(y.t));
        auto barycentric_error = std::max(std::abs(x.u - y.u), std::abs(x.v - y.v));
        auto distance_ok = distance_error <= distance_tolerance;
        auto barycentric_ok = barycentric_error <= pair_tolerance;
        auto same_hit = x.inst == y.inst && x.prim == y.prim;
        if (exclusion != Exclusion::NONE) {
            // not comparable, reported with its own maxima and never fatal
            count_exclusion(exclusion);
            counts.max_excluded_distance_error =
                std::max(counts.max_excluded_distance_error, distance_error);
            counts.max_excluded_barycentric_error =
                std::max(counts.max_excluded_barycentric_error, barycentric_error);
            continue;
        }
        if (!same_hit) {
            // a different primitive: only comparable if the distance says so (the
            // scenes overlap their own primitives on purpose)
            if (distance_ok) {
                counts.ties++;
            } else {
                counts.id_mismatch++;
                counts.compared++;
                counts.max_distance_error = std::max(counts.max_distance_error, distance_error);
                record("different primitive at a different distance");
            }
            continue;
        }
        counts.compared++;
        counts.max_distance_error = std::max(counts.max_distance_error, distance_error);
        counts.max_barycentric_error = std::max(counts.max_barycentric_error, barycentric_error);
        if (!distance_ok) {
            counts.distance_mismatch++;
            record("distance beyond the tolerance");
        }
        if (!barycentric_ok) {
            counts.bary_mismatch++;
            record("barycentrics beyond the tolerance");
        }
    }
}

// ---------------------------------------------------------------------------
// Scene catalogue
// ---------------------------------------------------------------------------

[[nodiscard]] float4x4 instance_transform(uint kind, float3 position) noexcept {
    switch (kind % 8u) {
        case 0u: return translation(position);
        case 1u: return translation(position + make_float3(0.5f, -0.3f, 0.8f));
        case 2u:
            return translation(position) *
                   rotation(make_float3(0.0f, 1.0f, 0.0f), radians(35.0f));
        case 3u:
            return translation(position) *
                   rotation(make_float3(1.0f, 0.2f, 0.0f), radians(20.0f)) * scaling(0.6f);
        case 4u:
            return translation(position + make_float3(0.2f, 0.4f, -0.2f)) *
                   rotation(make_float3(0.0f, 0.0f, 1.0f), radians(15.0f)) *
                   scaling(make_float3(1.7f, 0.4f, 1.1f));
        case 5u:
            // a mirrored (negative determinant) transform
            return translation(position) *
                   rotation(make_float3(0.0f, 1.0f, 0.0f), radians(30.0f)) *
                   scaling(make_float3(-1.0f, 1.0f, 1.0f));
        case 6u: return translation(position) * scaling(1.0e-4f);
        default:
            return translation(position) *
                   rotation(make_float3(1.0f, 0.0f, 0.0f), radians(90.0f)) *
                   scaling(make_float3(0.3f, 2.0f, 0.5f));
    }
}

[[nodiscard]] float3 grid_position(uint index) noexcept {
    return make_float3(0.9f * static_cast<float>(index % 3u),
                       0.9f * static_cast<float>((index / 3u) % 3u),
                       0.9f * static_cast<float>((index / 9u) % 3u));
}

// 1. boundary sizes: one BLAS with `triangle_count` triangles, one instance.
[[nodiscard]] TestScene make_boundary_scene(uint triangle_count, uint64_t seed,
                                            size_t ray_budget) noexcept {
    SceneBuilder builder;
    builder.scene.label = luisa::format("n={}", triangle_count);
    Rng rng(seed);
    auto first = builder.begin_mesh();
    for (auto i = 0u; i < triangle_count; i++) {
        auto center = make_float3(rng.symmetric() * 0.9f, rng.symmetric() * 0.9f,
                                  rng.symmetric() * 0.9f);
        emit_random_triangle(builder, rng, center, 0.2f + 0.2f * rng.next_float());
    }
    auto mesh = builder.end_mesh(first);
    builder.instance(mesh, translation(make_float3(0.0f)));
    add_general_rays(builder.scene, ray_budget, seed, true, 1.0e-3f, 1.0e30f);
    add_targeted_rays(builder.scene, std::min<size_t>(triangle_count, 24u), seed);
    return std::move(builder.scene);
}

// 2. degenerate geometry: the cases the demo's sphere/box/torus scene cannot
// reach.  Every one of them is still expected to produce hits (checked by the
// scalar gate below), otherwise the check would be vacuous.

// 64 copies of the same triangle: one shared centroid, one shared AABB and one
// shared Morton code, so the tree order is the only thing that can decide which
// of the coincident leaves a traversal reports.
[[nodiscard]] TestScene make_coincident_scene(uint64_t seed, size_t ray_budget,
                                              const char *label) noexcept {
    SceneBuilder builder;
    builder.scene.label = label;
    auto first = builder.begin_mesh();
    auto a = builder.vertex(make_float3(-0.4f, -0.3f, 0.1f));
    auto b = builder.vertex(make_float3(0.5f, -0.2f, -0.1f));
    auto c = builder.vertex(make_float3(0.0f, 0.6f, 0.2f));
    for (auto i = 0u; i < 64u; i++) { builder.triangle(a, b, c); }
    auto mesh = builder.end_mesh(first);
    builder.instance(mesh, translation(make_float3(0.0f)));
    add_general_rays(builder.scene, ray_budget, seed + 1u, true, 1.0e-3f, 1.0e30f);
    add_targeted_rays(builder.scene, 12u, seed + 1u);
    return std::move(builder.scene);
}

[[nodiscard]] luisa::vector<TestScene> make_degenerate_scenes(uint64_t seed,
                                                              size_t ray_budget) noexcept {
    luisa::vector<TestScene> scenes;
    auto add = [&scenes](TestScene &&scene) noexcept { scenes.emplace_back(std::move(scene)); };

    add(make_coincident_scene(seed, ray_budget, "identical"));
    {
        // all centroids identical, every vertex order (both windings)
        SceneBuilder builder;
        builder.scene.label = "same-centroid";
        auto first = builder.begin_mesh();
        auto a = builder.vertex(make_float3(-0.35f, -0.25f, 0.05f));
        auto b = builder.vertex(make_float3(0.45f, -0.25f, 0.05f));
        auto c = builder.vertex(make_float3(-0.1f, 0.5f, 0.05f));
        const uint order[6][3] = {{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b}, {c, b, a}};
        for (auto r = 0u; r < 16u; r++) {
            for (auto o = 0u; o < 6u; o++) {
                builder.triangle(order[o][0], order[o][1], order[o][2]);
            }
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        add_general_rays(builder.scene, ray_budget, seed + 2u, true, 1.0e-3f, 1.0e30f);
        add_targeted_rays(builder.scene, 12u, seed + 2u);
        add(std::move(builder.scene));
    }
    {
        // zero-area (collinear) triangles next to a few real ones
        SceneBuilder builder;
        builder.scene.label = "zero-area";
        auto first = builder.begin_mesh();
        for (auto i = 0u; i < 32u; i++) {
            auto base = make_float3(0.4f * static_cast<float>(i % 4u) - 0.6f,
                                    0.3f * static_cast<float>((i / 4u) % 4u) - 0.45f,
                                    0.1f * static_cast<float>(i / 16u) - 0.05f);
            auto d = normalize(make_float3(1.0f, 0.3f * static_cast<float>(i % 3u), 0.7f));
            emit_triangle(builder, base, base + d * 0.3f, base + d * 0.6f);
        }
        for (auto i = 0u; i < 4u; i++) {
            Rng rng(seed + 100u + i);
            auto center = make_float3(0.5f * static_cast<float>(i) - 0.75f, 0.4f, 0.0f);
            emit_random_triangle(builder, rng, center, 0.3f);
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        Rng rng(seed + 3u);
        add_general_rays(builder.scene, ray_budget, seed + 3u, true, 1.0e-3f, 1.0e30f);
        // aim at the collinear segment: the same "triangle" is a knife edge for
        // both implementations
        for (auto i = 0u; i < 12u; i++) {
            auto p = make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) * 0.4f;
            auto d = normalize(make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()));
            LbvhRay ray;
            ray.origin = p - d * 2.0f;
            ray.direction = d;
            ray.t_min = 1.0e-3f;
            ray.t_max = 1.0e30f;
            builder.scene.rays.emplace_back(ray);
        }
        add_targeted_rays(builder.scene, 8u, seed + 3u);
        add(std::move(builder.scene));
    }
    {
        // slivers: area 1e-7 over a length of 1
        SceneBuilder builder;
        builder.scene.label = "sliver";
        auto first = builder.begin_mesh();
        for (auto i = 0u; i < 32u; i++) {
            auto base = make_float3(0.35f * static_cast<float>(i % 4u) - 0.5f,
                                    0.35f * static_cast<float>((i / 4u) % 4u) - 0.5f,
                                    0.1f * static_cast<float>(i / 16u) - 0.05f);
            auto tangent = normalize(make_float3(1.0f, 0.2f * static_cast<float>(i % 5u), -0.3f));
            auto helper = std::abs(tangent.x) < 0.9f ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
            auto thin = normalize(cross(tangent, helper));
            emit_triangle(builder, base, base + tangent, base + thin * 1.0e-7f);
        }
        for (auto i = 0u; i < 4u; i++) {
            Rng rng(seed + 200u + i);
            emit_random_triangle(builder, rng, make_float3(0.6f * static_cast<float>(i) - 0.9f, -0.6f, 0.2f), 0.3f);
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        add_general_rays(builder.scene, ray_budget, seed + 4u, true, 1.0e-3f, 1.0e30f);
        add_targeted_rays(builder.scene, 12u, seed + 4u);
        add(std::move(builder.scene));
    }
    {
        // one triangle repeated, plus a single far outlier
        SceneBuilder builder;
        builder.scene.label = "far-outlier";
        auto first = builder.begin_mesh();
        auto a = builder.vertex(make_float3(-0.3f, -0.2f, 0.0f));
        auto b = builder.vertex(make_float3(0.4f, -0.2f, 0.0f));
        auto c = builder.vertex(make_float3(0.0f, 0.5f, 0.0f));
        for (auto i = 0u; i < 31u; i++) { builder.triangle(a, b, c); }
        emit_triangle(builder,
                      make_float3(9.0e3f, 1.0e4f, 1.0e4f),
                      make_float3(9.4e3f, 1.0e4f, 1.0e4f),
                      make_float3(9.0e3f, 1.04e4f, 1.0e4f));
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        add_general_rays(builder.scene, ray_budget, seed + 5u, true, 1.0e-3f, 1.0e30f);
        add_targeted_rays(builder.scene, 12u, seed + 5u);
        add(std::move(builder.scene));
    }
    {
        // axis-aligned quads: in each coordinate plane, plus rays exactly along
        // the axes (which are parallel to two of the three quad families)
        SceneBuilder builder;
        builder.scene.label = "axis-quads";
        auto first = builder.begin_mesh();
        for (auto plane = 0u; plane < 3u; plane++) {
            for (auto q = 0u; q < 4u; q++) {
                auto u = 0.9f * static_cast<float>(q % 2u) - 0.45f;
                auto v = 0.9f * static_cast<float>(q / 2u) - 0.45f;
                auto make = [&](float x, float y) noexcept {
                    return plane == 0u ? make_float3(0.0f, x, y) : plane == 1u ? make_float3(x, 0.0f, y) :
                                                                                 make_float3(x, y, 0.0f);
                };
                auto p00 = make(u, v);
                auto p10 = make(u + 0.4f, v);
                auto p01 = make(u, v + 0.4f);
                auto p11 = make(u + 0.4f, v + 0.4f);
                emit_triangle(builder, p00, p10, p01);
                emit_triangle(builder, p10, p11, p01);
            }
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        add_general_rays(builder.scene, ray_budget, seed + 6u, true, 1.0e-3f, 1.0e30f);
        Rng rng(seed + 6u);
        // exactly along the axes, exactly through the quads
        for (auto axis = 0u; axis < 3u; axis++) {
            for (auto k = 0u; k < 6u; k++) {
                auto p = make_float3(rng.symmetric() * 0.9f, rng.symmetric() * 0.9f,
                                     rng.symmetric() * 0.9f);
                LbvhRay ray;
                ray.origin = p;
                ray.direction = axis == 0u ? make_float3(1.0f, 0.0f, 0.0f) : axis == 1u ? make_float3(0.0f, 1.0f, 0.0f) :
                                                                                          make_float3(0.0f, 0.0f, 1.0f);
                ray.t_min = 1.0e-3f;
                ray.t_max = 1.0e30f;
                builder.scene.rays.emplace_back(ray);
            }
        }
        add_targeted_rays(builder.scene, 12u, seed + 6u);
        add(std::move(builder.scene));
    }
    {
        // a huge extent: coordinates from 1e-6 to 1e6
        SceneBuilder builder;
        builder.scene.label = "huge-extent";
        auto first = builder.begin_mesh();
        Rng rng(seed + 7u);
        for (auto i = 0u; i < 3u; i++) {
            auto center = make_float3(1.0e-6f * (0.5f + static_cast<float>(i)), 1.0e-6f,
                                      1.0e-6f * static_cast<float>(i));
            emit_random_triangle(builder, rng, center, 1.0e-6f);
        }
        for (auto i = 0u; i < 3u; i++) {
            auto center = make_float3(1.0e6f * (0.5f + 0.1f * static_cast<float>(i)),
                                      1.0e6f, -1.0e6f * static_cast<float>(i));
            emit_random_triangle(builder, rng, center, 1.0e5f);
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        add_general_rays(builder.scene, ray_budget / 2u, seed + 7u, true, 1.0e-9f, 1.0e30f);
        add_targeted_rays(builder.scene, 12u, seed + 7u);
        add(std::move(builder.scene));
    }
    {
        // every primitive shares the coordinate z == 0
        SceneBuilder builder;
        builder.scene.label = "flat-z0";
        auto first = builder.begin_mesh();
        Rng rng(seed + 8u);
        for (auto i = 0u; i < 16u; i++) {
            auto center = make_float3(0.2f * static_cast<float>(i % 4u) - 0.3f,
                                      0.2f * static_cast<float>(i / 4u) - 0.3f, 0.0f);
            emit_random_triangle(builder, rng, make_float3(center.x, center.y, 0.0f), 0.15f);
        }
        auto mesh = builder.end_mesh(first);
        builder.instance(mesh, translation(make_float3(0.0f)));
        // the shared coordinate makes one reciprocal of the slab test enormous;
        // the rays either have a zero z direction or a z direction that is still
        // tiny, which is where the traversal has to stay consistent with the
        // reference.
        for (auto i = 0u; i < ray_budget; i++) {
            auto p = make_float3(rng.symmetric() * 2.0f, rng.symmetric() * 2.0f, 0.0f);
            LbvhRay ray;
            ray.origin = p;
            if (i % 4u == 0u) {
                ray.direction = make_float3(1.0f, 0.0f, 0.0f);// direction.z == 0 exactly
            } else if (i % 4u == 1u) {
                ray.direction = make_float3(0.0f, 1.0f, 0.0f);
            } else if (i % 4u == 2u) {
                ray.direction = make_float3(0.0f, 0.0f, 1.0e-20f);// a reciprocal of 1e20
            } else {
                ray.direction = normalize(make_float3(rng.symmetric(), rng.symmetric(),
                                                      0.05f));
            }
            ray.t_min = 1.0e-3f;
            ray.t_max = 1.0e30f;
            builder.scene.rays.emplace_back(ray);
        }
        add_targeted_rays(builder.scene, 12u, seed + 8u);
        add(std::move(builder.scene));
    }
    return scenes;
}

// 3. multi-BLAS / TLAS: `blas_count` meshes of different sizes and every kind of
// instance transform.
[[nodiscard]] TestScene make_multi_blas_scene(uint blas_count, uint64_t seed,
                                              size_t ray_budget,
                                              const char *label) noexcept {
    SceneBuilder builder;
    builder.scene.label = label;
    Rng rng(seed);
    luisa::vector<uint> meshes;
    meshes.reserve(blas_count);
    for (auto j = 0u; j < blas_count; j++) {
        auto count = blas_count == 1u ? 33u : 1u + ((j * 7u) % 17u);
        auto mesh_scale = 0.15f + 0.05f * static_cast<float>(j % 5u);
        auto first = builder.begin_mesh();
        for (auto i = 0u; i < count; i++) {
            auto center = make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) *
                          mesh_scale;
            emit_random_triangle(builder, rng, center, 0.4f * mesh_scale);
        }
        meshes.emplace_back(builder.end_mesh(first));
    }
    auto instance_count = std::max(blas_count, 8u);
    for (auto i = 0u; i < instance_count; i++) {
        builder.instance(meshes[i % blas_count], instance_transform(i, grid_position(i)));
    }
    add_general_rays(builder.scene, ray_budget, seed, true, 1.0e-3f, 1.0e30f);
    add_targeted_rays(builder.scene, std::min<size_t>(4u + 2u * blas_count, 48u), seed);
    return std::move(builder.scene);
}

// 4. randomised property loop: random triangle count, placement, instances and
// rays, reproducible from `--seed`.
[[nodiscard]] TestScene make_random_scene(uint64_t seed, bool quick,
                                          size_t ray_budget) noexcept {
    SceneBuilder builder;
    builder.scene.label = luisa::format("seed={}", seed);
    Rng rng(seed);
    auto triangle_count = 1u + rng.below(quick ? 256u : 2000u);
    auto mesh_scale = 0.1f + 1.5f * rng.next_float();
    auto mesh_center = make_float3(rng.symmetric() * 2.0f, rng.symmetric() * 2.0f,
                                   rng.symmetric() * 2.0f);
    auto first = builder.begin_mesh();
    for (auto i = 0u; i < triangle_count; i++) {
        auto center = mesh_center +
                      make_float3(rng.symmetric(), rng.symmetric(), rng.symmetric()) * mesh_scale;
        emit_random_triangle(builder, rng, center, 0.05f + 0.35f * rng.next_float());
    }
    auto mesh = builder.end_mesh(first);
    auto instance_count = 1u + rng.below(3u);
    for (auto i = 0u; i < instance_count; i++) {
        builder.instance(mesh, instance_transform(rng.below(8u), grid_position(i) + mesh_center));
    }
    add_general_rays(builder.scene, ray_budget, seed, true, 1.0e-3f, 1.0e30f);
    add_targeted_rays(builder.scene, 24u, seed);
    return std::move(builder.scene);
}

// ---------------------------------------------------------------------------
// One scene: build, structural check, trace, slices, host reference, RTX
// ---------------------------------------------------------------------------

using RtxShader = Shader1D<Accel, Buffer<LbvhRay>, Buffer<LbvhHit>, uint>;

// The demo's RTX reference: the same rays through the Luisa acceleration
// structure, mapped into the same hit record.  Compiled once.
[[nodiscard]] auto make_rtx_trace_kernel() noexcept {
    return Kernel1D{[](AccelVar accel, BufferVar<LbvhRay> rays, BufferVar<LbvhHit> hits,
                       UInt count) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < count) {
            auto r = rays.read(index);
            auto ray = make_ray(r.origin, r.direction, r.t_min, r.t_max);
            auto hit = accel.intersect(ray, {});
            Var<LbvhHit> result;
            result.inst = invalid_node;
            result.prim = invalid_node;
            result.bary = make_float2(0.0f);
            result.t = r.t_max;
            $if (!hit->miss()) {
                result.inst = hit.inst;
                result.prim = hit.prim;
                result.bary = hit.bary;
                result.t = hit->distance();
            };
            hits.write(index, result);
        };
    }};
}

struct SceneRun {
    size_t tree_problems{0u};
    size_t range_problems{0u};
    size_t heap_problems{0u};
    size_t contract_problems{0u};
    size_t slice_problems{0u};
    size_t repeat_problems{0u};
    size_t repeat_hit_problems{0u};
    size_t repeat_node_problems{0u};
    size_t reference_hit_problems{0u};
    size_t compaction_problems{0u};
    size_t software_hits{0u};
    size_t reference_hits{0u};
    size_t rtx_hits{0u};
    bool rtx_ran{false};
    HitComparison software;
    HitComparison rtx;

    [[nodiscard]] size_t problems() const noexcept {
        return tree_problems + range_problems + heap_problems + contract_problems +
               slice_problems + repeat_problems + reference_hit_problems +
               compaction_problems;
    }
    [[nodiscard]] size_t fatal() const noexcept {
        return problems() + software.fatal() + rtx.fatal();
    }
    // the benchmark's not-comparable buckets, over both cross-checks
    [[nodiscard]] size_t grazing() const noexcept { return software.grazing + rtx.grazing; }
    [[nodiscard]] size_t boundary() const noexcept { return software.boundary + rtx.boundary; }
    // the two measured ones (see `barycentric_error_bound` / `determinant_at_guard`)
    [[nodiscard]] size_t conditioned() const noexcept { return software.conditioned + rtx.conditioned; }
    [[nodiscard]] size_t guard_decided() const noexcept {
        return software.guard_decided + rtx.guard_decided;
    }
};

[[nodiscard]] bool exact_same(const LbvhHit &a, const LbvhHit &b) noexcept {
    return a.inst == b.inst && a.prim == b.prim && a.t == b.t &&
           a.bary.x == b.bary.x && a.bary.y == b.bary.y;
}

[[nodiscard]] size_t count_hit_mismatches(luisa::span<const LbvhHit> a,
                                          luisa::span<const LbvhHit> b) noexcept {
    auto n = std::min(a.size(), b.size());
    size_t mismatches = a.size() == b.size() ? 0u : std::max(a.size(), b.size()) - n;
    for (auto i = 0u; i < n; i++) { mismatches += exact_same(a[i], b[i]) ? 0u : 1u; }
    return mismatches;
}

// ---------------------------------------------------------------------------
// Which bytes of one `LbvhNode` record hold a value.
//
// The build stores a 16-byte `float3` member with a 16-byte vector store whose
// fourth lane is undefined, so the padding lanes of a node record contain
// garbage that differs between two builds of the same scene - they are not part
// of the tree and the library never reads them.  The mask of the *defined* bytes
// is derived from the library's own reflection (`LUISA_STRUCT` exports every
// member's offset and its DSL type, whose size is the device-side one), so this
// test never names a member nor assumes an offset: a layout change under it is
// picked up automatically.
// ---------------------------------------------------------------------------
// The bytes of one member that hold a value.  A 3-lane vector reserves 16 bytes
// of layout (the device-side `float3` is a 16-byte element, exactly like the host
// one) but only 12 of them carry the value, and the build stores the fourth lane
// with an undefined value - so it is a padding lane that must not enter the
// comparison.  Every other type occupies its layout size.
[[nodiscard]] size_t member_value_bytes(const Type *type) noexcept {
    if (type->is_vector()) {
        return static_cast<size_t>(type->dimension()) * member_value_bytes(type->element());
    }
    return type->size();
}

// The mask is one *byte* per byte of the record (not `bool`): `luisa::span` can
// only view a contiguous container, and a bit-packed `vector<bool>` - the
// `std::vector<bool>` specialization some STL configurations pick - cannot be
// viewed at all, so a byte mask keeps this check portable across the STL the
// project is configured with (it also makes the mask printable verbatim).
template<typename S, size_t... K>
[[nodiscard]] luisa::vector<uint8_t> defined_bytes_from(const size_t *member_offsets,
                                                        std::index_sequence<K...>) noexcept {
    using members = typename luisa::compute::struct_member_tuple<S>::type;
    static_assert(sizeof...(K) == std::tuple_size_v<members>,
                  "the member reflection of the struct and its offset sequence disagree");
    std::array<size_t, sizeof...(K)> member_sizes{
        member_value_bytes(luisa::compute::Type::of<std::tuple_element_t<K, members>>())...};
    luisa::vector<uint8_t> defined(luisa::compute::Type::of<S>()->size(), uint8_t{0u});
    for (auto k = 0u; k < sizeof...(K); k++) {
        for (auto b = member_offsets[k]; b < member_offsets[k] + member_sizes[k]; b++) {
            if (b < defined.size()) { defined[b] = uint8_t{1u}; }
        }
    }
    return defined;
}

template<typename S, size_t... I>
[[nodiscard]] luisa::vector<uint8_t> defined_bytes_impl(std::integer_sequence<size_t, I...>) noexcept {
    constexpr size_t member_offsets[] = {I...};
    return defined_bytes_from<S>(member_offsets, std::make_index_sequence<sizeof...(I)>{});
}

template<typename S>
[[nodiscard]] luisa::vector<uint8_t> defined_bytes() noexcept {
    return defined_bytes_impl<S>(typename luisa::compute::struct_member_tuple<S>::offset{});
}

struct NodeDifference {
    size_t records{0u};// node records with a differing defined byte
    size_t bytes{0u};  // differing defined bytes
    ptrdiff_t first{-1};
};

// Compares the *defined* bytes of every node record of two node buffers.
[[nodiscard]] NodeDifference compare_node_buffers(luisa::span<const std::byte> a,
                                                  luisa::span<const std::byte> b,
                                                  luisa::span<const uint8_t> defined,
                                                  size_t stride) noexcept {
    NodeDifference difference;
    if (stride == 0u || defined.size() != stride) { return difference; }
    auto records = std::min(a.size(), b.size()) / stride;
    if (a.size() != b.size()) { difference.records = 1u; }
    for (auto r = 0u; r < records; r++) {
        auto base = static_cast<size_t>(r) * stride;
        auto record_differs = false;
        for (auto k = 0u; k < stride; k++) {
            if (!defined[k] || a[base + k] == b[base + k]) { continue; }
            difference.bytes++;
            if (!record_differs) {
                record_differs = true;
                difference.records++;
                if (difference.first < 0) { difference.first = static_cast<ptrdiff_t>(base + k); }
            }
        }
    }
    return difference;
}

// The documented hit contract: a miss has `inst == prim == invalid_node` and
// `t == t_max`; a hit has both ids valid.
[[nodiscard]] size_t check_hit_contract(luisa::span<const LbvhRay> rays,
                                        luisa::span<const LbvhHit> hits) noexcept {
    size_t problems = 0u;
    for (auto i = 0u; i < std::min(rays.size(), hits.size()); i++) {
        auto &&hit = hits[i];
        if (hit.inst == invalid_node) {
            problems += (hit.prim != invalid_node || hit.t != rays[i].t_max) ? 1u : 0u;
        } else {
            problems += hit.prim == invalid_node ? 1u : 0u;
        }
    }
    return problems;
}

[[nodiscard]] SceneRun run_scene(Device &device, Stream &stream, SoftwareLbvh &lbvh,
                                 const TestScene &scene, const RefScene &ref_scene,
                                 const RtxShader &rtx_shader, bool with_rtx, bool verbose,
                                 size_t &running_nodes, size_t &running_primitives,
                                 size_t &running_blases) noexcept {
    SceneRun run;
    auto ray_count = static_cast<uint>(scene.rays.size());
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(scene.vertices.size());
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(scene.triangles.size());
    Buffer<LbvhRay> ray_buffer = device.create_buffer<LbvhRay>(ray_count);
    Buffer<LbvhHit> hit_buffer = device.create_buffer<LbvhHit>(ray_count);
    Buffer<LbvhHit> slice_buffer = device.create_buffer<LbvhHit>(ray_count);
    Buffer<LbvhHit> rtx_buffer;// only allocated when the RTX check runs
    if (with_rtx) { rtx_buffer = device.create_buffer<LbvhHit>(ray_count); }
    stream << vertex_buffer.copy_from(luisa::span{scene.vertices})
           << triangle_buffer.copy_from(luisa::span{scene.triangles})
           << ray_buffer.copy_from(luisa::span{scene.rays})
           << synchronize();

    // ---- estimate -> create -> pre_build -> build (the backend order) ----
    luisa::vector<Blas> blases;
    blases.reserve(scene.meshes.size());
    auto node_total = running_nodes;
    // The plan records (`_plan_kernel`) are handed out exactly like the nodes,
    // one per internal node, so the offset a BLAS reports must follow the previous
    // one by its own internal-node count.  A build that loses the field would
    // instead give every tree the same (or a garbage) base, which is a *silently*
    // correct tree - the plan pass and the reduction pass would agree on the wrong
    // place - so it has to be checked here and not by the structural self-check.
    auto plan_total = 0u;
    auto plan_base = 0u;
    auto have_plan_base = false;
    for (auto &&mesh : scene.meshes) {
        auto blas = lbvh.create_blas(AccelOption{}, mesh.triangle_offset, mesh.triangle_count,
                                     mesh.lo, mesh.hi);
        lbvh.pre_build_blas(blas);
        if (!have_plan_base) {
            plan_base = blas.plan_offset();
            have_plan_base = true;
        }
        run.range_problems += blas.node_count() != 2u * mesh.triangle_count - 1u ? 1u : 0u;
        run.range_problems += blas.node_offset() != node_total ? 1u : 0u;
        run.range_problems += blas.plan_offset() != plan_base + plan_total ? 1u : 0u;
        node_total += blas.node_count();
        plan_total += blas.triangle_count() - 1u;
        blases.emplace_back(blas);
    }
    luisa::vector<InstanceDesc> descriptions;
    descriptions.reserve(scene.instances.size());
    for (auto &&instance : scene.instances) {
        descriptions.emplace_back(InstanceDesc{instance.to_world, instance.mesh});
    }
    auto tlas = lbvh.create_accel(AccelOption{}, static_cast<uint>(descriptions.size()));
    lbvh.pre_build_accel(stream, tlas, luisa::span{blases}, luisa::span{descriptions});
    if (!have_plan_base) {
        plan_base = tlas.plan_offset();
        have_plan_base = true;
    }
    run.range_problems += tlas.node_count() != 2u * descriptions.size() - 1u ? 1u : 0u;
    run.range_problems += tlas.node_offset() != node_total ? 1u : 0u;
    run.range_problems += tlas.plan_offset() != plan_base + plan_total ? 1u : 0u;
    run.range_problems += lbvh.sizes().plan_capacity <
                                  plan_base + plan_total + descriptions.size() - 1u
                              ? 1u
                              : 0u;
    node_total += tlas.node_count();
    plan_total += descriptions.size() - 1u;
    for (auto &&blas : blases) {
        lbvh.build_blas(stream, blas, vertex_buffer, triangle_buffer);
    }
    lbvh.build_accel(stream, tlas);
    stream << synchronize();

    // ---- the storage bookkeeping the public API reports ----
    running_primitives += scene.triangles.size() + scene.instances.size();
    running_nodes = node_total;
    running_blases += blases.size();
    run.range_problems += lbvh.primitive_count() != running_primitives ? 1u : 0u;
    run.range_problems += lbvh.node_count() != running_nodes ? 1u : 0u;
    run.range_problems += lbvh.blas_count() != running_blases ? 1u : 0u;
    run.range_problems += lbvh.sizes().node_capacity < running_nodes ? 1u : 0u;
    run.range_problems += lbvh.sizes().primitive_capacity < running_primitives ? 1u : 0u;

    // ---- the bindless heap of the TLAS (lbvh_common.h) ----
    // It must exist and hold one slot per BLAS plus the two reserved ones
    // (null + the TLAS' own region); the device-side check then verifies that
    // every record resolves through its slot to the same root node the shared
    // node buffer holds.
    run.heap_problems += tlas.has_heap() &&
                                 tlas.heap_size() >= blases.size() + heap_first_blas_slot
                             ? 0u
                             : 1u;

    // ---- structural self-check of every tree that was just built ----
    for (auto &&blas : blases) {
        run.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                blas.triangle_count());
    }
    run.tree_problems += lbvh.validate_tree(stream, tlas.node_offset(), tlas.instance_count());
    if (tlas.has_heap()) {
        run.heap_problems += lbvh.validate_heap(stream, tlas,
                                                static_cast<uint>(blases.size()));
    }

    // ---- traversal ----
    lbvh.trace_software(stream, vertex_buffer, triangle_buffer, ray_buffer, hit_buffer,
                        tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> host_software(ray_count);
    luisa::vector<HitView> software_view(ray_count);
    luisa::vector<HitView> reference_view(ray_count);
    stream << hit_buffer.copy_to(luisa::span{host_software}) << synchronize();
    run.contract_problems += check_hit_contract(luisa::span{scene.rays}, luisa::span{host_software});
    for (auto i = 0u; i < ray_count; i++) {
        software_view[i] = view_of(host_software[i]);
        reference_view[i] = view_of(ref_trace(ref_scene, scene.rays[i]));
    }
    compare_hits(ref_scene, luisa::span{scene.rays}, luisa::span{software_view},
                 luisa::span{reference_view}, run.software, GuardSemantics::SHARED);
    for (auto i = 0u; i < ray_count; i++) {
        if (software_view[i].hit) { run.software_hits++; }
        if (reference_view[i].hit) { run.reference_hits++; }
    }
    // A scene whose host reference finds nothing would make the comparison
    // vacuous, so an empty reference is a check of the test itself.
    run.reference_hit_problems += run.reference_hits == 0u ? 1u : 0u;

    // ---- slice invariance: strided slices must be bit-identical ----
    if (ray_count > 0u) {
        // four strided slices of stride 4 (which together cover every ray)
        for (auto s = 0u; s < 4u && s < ray_count; s++) {
            auto slice_count = (ray_count - s + 3u) / 4u;
            lbvh.trace_software(stream, vertex_buffer, triangle_buffer, ray_buffer,
                                slice_buffer, tlas, slice_count, s, 4u);
        }
        // one slice of stride 3, preceded by a poison value: exactly the covered
        // indices may be written
        luisa::vector<LbvhHit> poison(ray_count);
        for (auto &&hit : poison) {
            hit.inst = 0xDEADBEEFu;
            hit.prim = 0xDEADBEEFu;
            hit.bary = make_float2(-1234.5f);
            hit.t = -1.0f;
        }
        stream << slice_buffer.copy_from(luisa::span{poison}) << synchronize();
        lbvh.trace_software(stream, vertex_buffer, triangle_buffer, ray_buffer, slice_buffer,
                            tlas, (ray_count + 2u) / 3u, 0u, 3u);
        stream << synchronize();
        luisa::vector<LbvhHit> host_slices(ray_count);
        stream << slice_buffer.copy_to(luisa::span{host_slices}) << synchronize();
        for (auto i = 0u; i < ray_count; i++) {
            if (i % 3u != 0u) {
                // not part of the slice: it must still hold the poison value
                run.slice_problems += exact_same(host_slices[i], poison[i]) ? 0u : 1u;
            } else if (!exact_same(host_slices[i], host_software[i])) {
                run.slice_problems++;
            }
        }
        // the four slices of stride 4 must reproduce every contiguous hit
        for (auto s = 0u; s < 4u && s < ray_count; s++) {
            auto slice_count = (ray_count - s + 3u) / 4u;
            lbvh.trace_software(stream, vertex_buffer, triangle_buffer, ray_buffer,
                                slice_buffer, tlas, slice_count, s, 4u);
        }
        stream << slice_buffer.copy_to(luisa::span{host_slices}) << synchronize();
        run.slice_problems += count_hit_mismatches(luisa::span{host_software},
                                                   luisa::span{host_slices});
    }

    // ---- the hardware RTX traversal, against the same host reference ----
    if (with_rtx) {
        luisa::vector<Mesh> meshes;
        meshes.reserve(scene.meshes.size());
        for (auto &&mesh : scene.meshes) {
            meshes.emplace_back(device.create_mesh(
                vertex_buffer, triangle_buffer.view(mesh.triangle_offset, mesh.triangle_count)));
        }
        Accel accel = device.create_accel();
        for (auto &&instance : scene.instances) {
            accel.emplace_back(meshes[instance.mesh], instance.to_world);
        }
        for (auto &&mesh : meshes) { stream << mesh.build(); }
        stream << accel.build() << synchronize();
        stream << rtx_shader(accel, ray_buffer, rtx_buffer, ray_count).dispatch(ray_count)
               << synchronize();
        luisa::vector<LbvhHit> host_rtx(ray_count);
        luisa::vector<HitView> rtx_view(ray_count);
        stream << rtx_buffer.copy_to(luisa::span{host_rtx}) << synchronize();
        for (auto i = 0u; i < ray_count; i++) {
            rtx_view[i] = view_of(host_rtx[i]);
            if (rtx_view[i].hit) { run.rtx_hits++; }
        }
        compare_hits(ref_scene, luisa::span{scene.rays}, luisa::span{rtx_view},
                     luisa::span{reference_view}, run.rtx, GuardSemantics::SOFTWARE_ONLY);
        run.rtx_ran = true;
    }

    if (verbose) {
        std::printf("       scene %-16s rays=%u  build=%zu BLAS/%zu inst  "
                    "software=%zu reference=%zu rtx=%zu  compared=%zu ties=%zu excluded=%zu  "
                    "max_dist_err=%.3e max_bary_err=%.3e\n",
                    scene.label.c_str(), ray_count, blases.size(), scene.instances.size(),
                    run.software_hits, run.reference_hits, run.rtx_hits, run.software.compared,
                    run.software.ties, run.software.excluded(),
                    run.software.max_distance_error, run.software.max_barycentric_error);
    }
    return run;
}

// ---------------------------------------------------------------------------
// (6)/(7) build+trace determinism, node buffers included
// ---------------------------------------------------------------------------

struct BuildOutcome {
    luisa::vector<LbvhHit> hits;
    luisa::vector<std::byte> nodes;
    uint node_base{0u};
    uint node_count{0u};
    size_t tree_problems{0u};
    size_t heap_problems{0u};
};

// Builds and traces the scene into `lbvh` (whose node buffer is zeroed first so
// that bytes the build never writes compare equal everywhere) and returns the
// hits plus the raw bytes of the tree's node array.  The node bytes are never
// interpreted: the determinism check only asks whether two builds produced the
// same bytes.
[[nodiscard]] BuildOutcome build_outcome(Device &device, Stream &stream, SoftwareLbvh &lbvh,
                                         const TestScene &scene) noexcept {
    BuildOutcome outcome;
    auto ray_count = static_cast<uint>(scene.rays.size());
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(scene.vertices.size());
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(scene.triangles.size());
    Buffer<LbvhRay> ray_buffer = device.create_buffer<LbvhRay>(ray_count);
    Buffer<LbvhHit> hit_buffer = device.create_buffer<LbvhHit>(ray_count);
    stream << vertex_buffer.copy_from(luisa::span{scene.vertices})
           << triangle_buffer.copy_from(luisa::span{scene.triangles})
           << ray_buffer.copy_from(luisa::span{scene.rays})
           << synchronize();
    {
        luisa::vector<std::byte> zeros(lbvh.nodes().size_bytes(), std::byte{0});
        stream << lbvh.nodes().copy_from(luisa::span{zeros}) << synchronize();
    }
    luisa::vector<Blas> blases;
    blases.reserve(scene.meshes.size());
    for (auto &&mesh : scene.meshes) {
        auto blas = lbvh.create_blas(AccelOption{}, mesh.triangle_offset, mesh.triangle_count,
                                     mesh.lo, mesh.hi);
        lbvh.pre_build_blas(blas);
        blases.emplace_back(blas);
    }
    luisa::vector<InstanceDesc> descriptions;
    descriptions.reserve(scene.instances.size());
    for (auto &&instance : scene.instances) {
        descriptions.emplace_back(InstanceDesc{instance.to_world, instance.mesh});
    }
    auto tlas = lbvh.create_accel(AccelOption{}, static_cast<uint>(descriptions.size()));
    lbvh.pre_build_accel(stream, tlas, luisa::span{blases}, luisa::span{descriptions});
    for (auto &&blas : blases) { lbvh.build_blas(stream, blas, vertex_buffer, triangle_buffer); }
    lbvh.build_accel(stream, tlas);
    stream << synchronize();
    for (auto &&blas : blases) {
        outcome.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                    blas.triangle_count());
    }
    outcome.tree_problems += lbvh.validate_tree(stream, tlas.node_offset(),
                                                tlas.instance_count());
    if (tlas.has_heap()) {
        outcome.heap_problems += lbvh.validate_heap(stream, tlas,
                                                   static_cast<uint>(blases.size()));
    } else {
        outcome.heap_problems++;
    }
    lbvh.trace_software(stream, vertex_buffer, triangle_buffer, ray_buffer, hit_buffer,
                        tlas, ray_count);
    stream << synchronize();
    outcome.hits.resize(ray_count);
    stream << hit_buffer.copy_to(luisa::span{outcome.hits}) << synchronize();
    // Only the slice of the node buffer this run allocated is downloaded, so two
    // runs that land at different node bases stay comparable.  The bytes are
    // never interpreted, they are only compared with another run's.
    outcome.node_base = blases.empty() ? tlas.node_offset() : blases.front().node_offset();
    auto range_end = tlas.node_offset() + tlas.node_count();
    outcome.node_count = range_end - outcome.node_base;
    outcome.nodes.resize(static_cast<size_t>(outcome.node_count) * lbvh.nodes().stride());
    stream << lbvh.nodes().view(outcome.node_base, outcome.node_count).copy_to(luisa::span{outcome.nodes})
           << synchronize();
    return outcome;
}

// The same scene built twice into two independent storages and once more into the
// first one (at a different node base).  The hits of all three builds must be
// bit-identical, and the *node buffers* of the two independent storages must be
// byte-identical (defined bytes only, see `defined_bytes`).  This is what catches
// a non-deterministic scatter in the build, which the hit comparison alone cannot
// always see (a reordered tree still finds the closest hit).
//
// The node buffer of the third build is not byte-compared: it lands at a
// different node base, and child pointers are absolute node indices (a documented
// property of the layout), so the two node arrays are legitimately different.
[[nodiscard]] SceneRun run_determinism(Device &device, Stream &stream, const TestScene &scene,
                                       const Options &options) noexcept {
    SceneRun run;
    // room for two builds into the first storage (the third build lands at a
    // different node base, which is part of what the check verifies)
    auto sizes = SoftwareLbvh::estimate(2u * scene.triangles.size(),
                                        2u * scene.instances.size(), 2u * scene.meshes.size());
    SoftwareLbvh first{device, sizes};
    SoftwareLbvh second{device, sizes};
    auto a = build_outcome(device, stream, first, scene);
    auto b = build_outcome(device, stream, second, scene);
    auto c = build_outcome(device, stream, first, scene);
    run.tree_problems += a.tree_problems + b.tree_problems + c.tree_problems;
    run.heap_problems += a.heap_problems + b.heap_problems + c.heap_problems;
    run.repeat_hit_problems += count_hit_mismatches(luisa::span{a.hits}, luisa::span{b.hits});
    run.repeat_hit_problems += count_hit_mismatches(luisa::span{a.hits}, luisa::span{c.hits});
    auto stride = first.nodes().stride();
    auto defined = defined_bytes<LbvhNode>();
    if (defined.size() != stride) {
        // the reflected struct layout and the buffer's element stride disagree:
        // the node bytes cannot be compared reliably, so say so instead of
        // silently skipping bytes
        run.repeat_problems++;
        std::printf("       the reflected LbvhNode size (%zu) differs from the buffer stride (%zu)\n",
                    defined.size(), stride);
        defined.resize(stride, uint8_t{1u});
    }
    auto nodes_ab = compare_node_buffers(luisa::span{a.nodes}, luisa::span{b.nodes}, defined, stride);
    run.repeat_node_problems += nodes_ab.records == 0u ? 0u : 1u;
    run.repeat_node_problems += a.node_base == b.node_base ? 0u : 1u;
    run.repeat_problems = run.repeat_hit_problems + run.repeat_node_problems;
    if (nodes_ab.records != 0u) {
        std::printf("       node buffers of two storages: %zu of %u record(s) differ "
                    "(%zu defined byte(s), first at %lld, stride %zu)\n",
                    nodes_ab.records, a.node_count, nodes_ab.bytes,
                    static_cast<long long>(nodes_ab.first), stride);
    }
    if (options.verbose) {
        luisa::string mask;
        for (auto byte : defined) { mask.push_back(byte ? '1' : '0'); }
        std::printf("       determinism %-16s nodes=%u records=%zu bytes=%zu defined=%zu "
                    "hits=%zu bases=%u/%u/%u mask=%s\n",
                    scene.label.c_str(), a.node_count, a.nodes.size() / stride, a.nodes.size(),
                    defined.size(), a.hits.size(), a.node_base, b.node_base, c.node_base,
                    mask.c_str());
    }
    return run;
}

// ---------------------------------------------------------------------------
// Storage compaction (P5/P8): contract, structure, hit identity, no-op,
// determinism and the repeated-compaction lifetime, on the *dense* buffer.
//
// Not covered here (per test/SKILL.md's harness limits): the *failure* mode of
// `compact()` - a storage built without `AccelOption::allow_compaction`, or one
// whose reserved trees were not all built, trips a `LUISA_ASSERT` and aborts the
// process, and this harness has no way to run a crashing case in a child
// process.  The positive side of the same contract is covered: every check below
// builds with `allow_compaction` set, and the no-op / repeat cases pin the
// bookkeeping `compact()` asserts on.
// ---------------------------------------------------------------------------

// Geometry + rays of one scene, plus one build into `lbvh`.  `option` carries the
// caller's `allow_compaction` intent (which `compact()` requires); the returned
// BLAS handles and the TLAS are what a compaction and a traversal address.
struct BuiltScene {
    Buffer<float3> vertices;
    Buffer<Triangle> triangles;
    Buffer<LbvhRay> rays;
    luisa::vector<Blas> blases;
    Tlas tlas;
};

[[nodiscard]] BuiltScene build_scene(Device &device, Stream &stream, SoftwareLbvh &lbvh,
                                     const TestScene &scene,
                                     const AccelOption &option) noexcept {
    BuiltScene built;
    auto ray_count = static_cast<uint>(scene.rays.size());
    built.vertices = device.create_buffer<float3>(scene.vertices.size());
    built.triangles = device.create_buffer<Triangle>(scene.triangles.size());
    built.rays = device.create_buffer<LbvhRay>(ray_count);
    stream << built.vertices.copy_from(luisa::span{scene.vertices})
           << built.triangles.copy_from(luisa::span{scene.triangles})
           << built.rays.copy_from(luisa::span{scene.rays})
           << synchronize();
    built.blases.reserve(scene.meshes.size());
    for (auto &&mesh : scene.meshes) {
        auto blas = lbvh.create_blas(option, mesh.triangle_offset, mesh.triangle_count,
                                     mesh.lo, mesh.hi);
        lbvh.pre_build_blas(blas);
        lbvh.build_blas(stream, blas, built.vertices, built.triangles);
        built.blases.emplace_back(blas);
    }
    luisa::vector<InstanceDesc> descriptions;
    descriptions.reserve(scene.instances.size());
    for (auto &&instance : scene.instances) {
        descriptions.emplace_back(InstanceDesc{instance.to_world, instance.mesh});
    }
    built.tlas = lbvh.create_accel(option, static_cast<uint>(descriptions.size()));
    lbvh.pre_build_accel(stream, built.tlas, luisa::span{built.blases},
                         luisa::span{descriptions});
    lbvh.build_accel(stream, built.tlas);
    stream << synchronize();
    return built;
}

// Number of nodes the trees of `scene` occupy: `2t - 1` per BLAS and `2I - 1`
// for the TLAS.
[[nodiscard]] size_t scene_node_count(const TestScene &scene) noexcept {
    size_t nodes = 0u;
    for (auto &&mesh : scene.meshes) { nodes += 2u * mesh.triangle_count - 1u; }
    nodes += 2u * scene.instances.size() - 1u;
    return nodes;
}

// Everything the compaction contract promises, on one scene built into its own
// storage (sized with headroom, so there is something to reclaim):
//
//   1. `nodes()`/`node_count()`/`nodes_after`/`compacted_bytes` agree, and the
//      device size query equals the host's `2n-1` bookkeeping (P5.1/P5.2);
//   2. `validate_tree` of every BLAS + the TLAS and `validate_heap` still report
//      0 on the *dense* buffer (the heap was re-registered onto it);
//   3. the traversal is bit-identical before and after compaction (P5.3);
//   4. the dense node bytes are exactly the loose ones (`as_built` keeps the
//      indices, so the copy is an identity on the defined bytes) (P5.4);
//   5. a second `compact()` on the already-dense storage reclaims nothing,
//      records no copy and leaves the traversal untouched.
struct CompactionCheck {
    size_t contract_problems{0u};
    size_t tree_problems{0u};
    size_t heap_problems{0u};
    size_t hit_problems{0u};
    size_t node_problems{0u};
    size_t trees{0u};
    size_t nodes_before{0u};
    size_t nodes_after{0u};
    size_t reclaimed_bytes{0u};
    bool compacted{false};
};

[[nodiscard]] CompactionCheck run_compaction_checks(Device &device, Stream &stream,
                                                    const TestScene &scene,
                                                    SoftwareLbvh::CompactionPolicy policy) noexcept {
    CompactionCheck check;
    auto ray_count = static_cast<uint>(scene.rays.size());
    if (ray_count == 0u || scene.meshes.empty() || scene.instances.empty()) { return check; }
    auto expected_nodes = scene_node_count(scene);
    // Deliberate headroom: the storage is larger than the scene, which is the
    // "size it before the scene is known" case and what makes the compaction
    // observable (reclaiming `(capacity - used) * 32` bytes).
    auto sizes = SoftwareLbvh::estimate(2u * scene.triangles.size() + 1u,
                                        2u * scene.instances.size() + 1u,
                                        2u * scene.meshes.size() + 1u);
    SoftwareLbvh lbvh{device, sizes};
    AccelOption option;
    option.allow_compaction = true;
    auto built = build_scene(device, stream, lbvh, scene, option);
    check.contract_problems += lbvh.node_count() == expected_nodes ? 0u : 1u;
    check.contract_problems += lbvh.node_capacity() >= expected_nodes ? 0u : 1u;
    auto nodes_before = lbvh.node_capacity();

    // the loose node bytes, before anything is copied (P5.4, `as_built` only)
    auto stride = lbvh.nodes().stride();
    auto defined = defined_bytes<LbvhNode>();
    if (defined.size() != stride) {
        check.node_problems++;
        defined.resize(stride, uint8_t{1u});
    }
    luisa::vector<std::byte> loose(expected_nodes * stride);
    stream << lbvh.nodes().view(0u, expected_nodes).copy_to(luisa::span{loose})
           << synchronize();

    // traversal before compaction
    Buffer<LbvhHit> hits = device.create_buffer<LbvhHit>(ray_count);
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> host_before(ray_count);
    stream << hits.copy_to(luisa::span{host_before}) << synchronize();

    // ---- the compaction ----
    auto result = lbvh.compact(stream, built.tlas, luisa::span{built.blases}, policy);
    stream << synchronize();
    check.trees = result.trees;
    check.nodes_before = result.nodes_before;
    check.nodes_after = result.nodes_after;
    check.reclaimed_bytes = result.compacted_bytes;
    check.compacted = result.compacted();
    check.contract_problems += result.trees == scene.meshes.size() + 1u ? 0u : 1u;
    check.contract_problems += result.nodes_before == nodes_before ? 0u : 1u;
    check.contract_problems += result.nodes_after == expected_nodes ? 0u : 1u;
    check.contract_problems += result.compacted() ? 0u : 1u;// used < capacity here
    check.contract_problems += result.nodes_after == lbvh.node_count() ? 0u : 1u;
    check.contract_problems += lbvh.nodes().size() == expected_nodes ? 0u : 1u;
    check.contract_problems += result.bytes_after == expected_nodes * sizeof(LbvhNode) ? 0u : 1u;
    check.contract_problems +=
        result.compacted_bytes ==
                (result.nodes_before - result.nodes_after) * sizeof(LbvhNode)
            ? 0u
            : 1u;

    // ---- structural check on the dense buffer ----
    for (auto &&blas : built.blases) {
        check.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                  blas.triangle_count());
    }
    check.tree_problems += lbvh.validate_tree(stream, built.tlas.node_offset(),
                                              built.tlas.instance_count());
    check.heap_problems += lbvh.validate_heap(stream, built.tlas,
                                              static_cast<uint>(built.blases.size()));

    // ---- hit identity ----
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> host_after(ray_count);
    stream << hits.copy_to(luisa::span{host_after}) << synchronize();
    check.hit_problems += count_hit_mismatches(luisa::span{host_before},
                                               luisa::span{host_after});

    // ---- the copy is exact (`as_built` keeps every index) ----
    if (policy == SoftwareLbvh::CompactionPolicy::as_built) {
        luisa::vector<std::byte> dense(expected_nodes * stride);
        stream << lbvh.nodes().view(0u, expected_nodes).copy_to(luisa::span{dense})
               << synchronize();
        auto difference = compare_node_buffers(luisa::span{loose}, luisa::span{dense},
                                               defined, stride);
        check.node_problems += difference.records == 0u ? 0u : 1u;
        if (difference.records != 0u) {
            std::printf("       compacted node bytes differ from the loose ones: "
                        "%zu record(s), %zu byte(s), first at %lld\n",
                        difference.records, difference.bytes,
                        static_cast<long long>(difference.first));
        }
    }

    // ---- a second compact() is a no-op ----
    auto again = lbvh.compact(stream, built.tlas, luisa::span{built.blases}, policy);
    stream << synchronize();
    check.contract_problems += again.compacted() ? 1u : 0u;
    check.contract_problems += again.nodes_after == result.nodes_after ? 0u : 1u;
    check.contract_problems += again.nodes_before == result.nodes_after ? 0u : 1u;
    check.contract_problems += again.compacted_bytes == 0u ? 0u : 1u;
    check.contract_problems += lbvh.nodes().size() == expected_nodes ? 0u : 1u;
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> host_idempotent(ray_count);
    stream << hits.copy_to(luisa::span{host_idempotent}) << synchronize();
    check.hit_problems += count_hit_mismatches(luisa::span{host_before},
                                               luisa::span{host_idempotent});
    // ---- `as_built` keeps a rebuild of the *existing* trees valid: the indices
    // did not move, so re-building into the dense buffer reproduces the same
    // structure and the same hits.  (Reserving *new* trees fails closed instead -
    // a compacted storage has no spare capacity; `subtree_contiguous` would not
    // keep even the in-place rebuild valid, which is why it is rejected.) ----
    if (policy == SoftwareLbvh::CompactionPolicy::as_built) {
        for (auto &&blas : built.blases) {
            lbvh.build_blas(stream, blas, built.vertices, built.triangles);
        }
        lbvh.build_accel(stream, built.tlas);
        stream << synchronize();
        for (auto &&blas : built.blases) {
            check.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                      blas.triangle_count());
        }
        check.tree_problems += lbvh.validate_tree(stream, built.tlas.node_offset(),
                                                  built.tlas.instance_count());
        check.heap_problems += lbvh.validate_heap(stream, built.tlas,
                                                  static_cast<uint>(built.blases.size()));
        lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                            built.tlas, ray_count);
        stream << synchronize();
        luisa::vector<LbvhHit> host_rebuilt(ray_count);
        stream << hits.copy_to(luisa::span{host_rebuilt}) << synchronize();
        check.hit_problems += count_hit_mismatches(luisa::span{host_before},
                                                   luisa::span{host_rebuilt});
    }
    return check;
}

// The no-op boundary (P5.5): a storage whose capacity is *exactly* the scene's
// node count.  `compact()` must reclaim 0 bytes, record no copy (a zero-count
// dispatch must not exist) and leave the traversal untouched.
struct NoOpCheck {
    size_t contract_problems{0u};
    size_t hit_problems{0u};
    size_t tree_problems{0u};
};

[[nodiscard]] NoOpCheck run_compaction_noop_check(Device &device, Stream &stream,
                                                  const TestScene &scene) noexcept {
    NoOpCheck check;
    auto ray_count = static_cast<uint>(scene.rays.size());
    if (ray_count == 0u || scene.meshes.empty() || scene.instances.empty()) { return check; }
    auto expected_nodes = scene_node_count(scene);
    auto sizes = SoftwareLbvh::estimate(scene.triangles.size(), scene.instances.size(),
                                        scene.meshes.size());
    // shrink the node budget to exactly the scene (the no-op case)
    sizes.node_capacity = expected_nodes;
    sizes.node_bytes = expected_nodes * sizeof(LbvhNode);
    sizes.block_capacity = (expected_nodes + node_reduction_block - 1u) / node_reduction_block;
    sizes.block_bytes = sizes.block_capacity * sizeof(LbvhNode);
    SoftwareLbvh lbvh{device, sizes};
    AccelOption option;
    option.allow_compaction = true;
    auto built = build_scene(device, stream, lbvh, scene, option);
    check.contract_problems += lbvh.node_capacity() == expected_nodes ? 0u : 1u;
    check.contract_problems += lbvh.node_count() == expected_nodes ? 0u : 1u;
    Buffer<LbvhHit> hits = device.create_buffer<LbvhHit>(ray_count);
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> before(ray_count);
    stream << hits.copy_to(luisa::span{before}) << synchronize();
    auto result = lbvh.compact(stream, built.tlas, luisa::span{built.blases});
    stream << synchronize();
    check.contract_problems += result.compacted() ? 1u : 0u;
    check.contract_problems += result.nodes_before == expected_nodes ? 0u : 1u;
    check.contract_problems += result.nodes_after == expected_nodes ? 0u : 1u;
    check.contract_problems += result.compacted_bytes == 0u ? 0u : 1u;
    check.contract_problems += lbvh.nodes().size() == expected_nodes ? 0u : 1u;
    check.tree_problems += lbvh.validate_tree(stream, built.tlas.node_offset(),
                                              built.tlas.instance_count());
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> after(ray_count);
    stream << hits.copy_to(luisa::span{after}) << synchronize();
    check.hit_problems += count_hit_mismatches(luisa::span{before}, luisa::span{after});
    return check;
}

// The build-scratch release (P6, opt-in): after `compact(..., release_scratch=true)`
// the storage must no longer be buildable, the returned scratch bytes must be the
// scratch the estimate reports, and the structure must still validate and trace.
struct ScratchCheck {
    size_t contract_problems{0u};
    size_t tree_problems{0u};
    size_t heap_problems{0u};
    size_t hit_problems{0u};
};

[[nodiscard]] ScratchCheck run_scratch_release_check(Device &device, Stream &stream,
                                                     const TestScene &scene) noexcept {
    ScratchCheck check;
    auto ray_count = static_cast<uint>(scene.rays.size());
    if (ray_count == 0u || scene.meshes.empty() || scene.instances.empty()) { return check; }
    auto expected_nodes = scene_node_count(scene);
    auto sizes = SoftwareLbvh::estimate(2u * scene.triangles.size() + 1u,
                                        2u * scene.instances.size() + 1u,
                                        2u * scene.meshes.size() + 1u);
    // the scratch the size query reports and `release_build_scratch` hands back
    auto expected_scratch = sizes.primitive_bytes + 2u * sizes.key_bytes +
                            sizes.block_bytes + sizes.plan_bytes;
    SoftwareLbvh lbvh{device, sizes};
    AccelOption option;
    option.allow_compaction = true;
    auto built = build_scene(device, stream, lbvh, scene, option);
    check.contract_problems += lbvh.buildable() ? 0u : 1u;
    Buffer<LbvhHit> hits = device.create_buffer<LbvhHit>(ray_count);
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> before(ray_count);
    stream << hits.copy_to(luisa::span{before}) << synchronize();
    auto result = lbvh.compact(stream, built.tlas, luisa::span{built.blases},
                               SoftwareLbvh::CompactionPolicy::as_built,
                               /*release_scratch=*/true);
    stream << synchronize();
    check.contract_problems += result.compacted() ? 0u : 1u;
    check.contract_problems += result.nodes_after == expected_nodes ? 0u : 1u;
    check.contract_problems += result.reclaimed_scratch_bytes == expected_scratch ? 0u : 1u;
    check.contract_problems += result.reclaimed_scratch_bytes != 0u ? 0u : 1u;
    check.contract_problems += lbvh.buildable() ? 1u : 0u;
    for (auto &&blas : built.blases) {
        check.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                  blas.triangle_count());
    }
    check.tree_problems += lbvh.validate_tree(stream, built.tlas.node_offset(),
                                              built.tlas.instance_count());
    check.heap_problems += lbvh.validate_heap(stream, built.tlas,
                                              static_cast<uint>(built.blases.size()));
    lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                        built.tlas, ray_count);
    stream << synchronize();
    luisa::vector<LbvhHit> after(ray_count);
    stream << hits.copy_to(luisa::span{after}) << synchronize();
    check.hit_problems += count_hit_mismatches(luisa::span{before}, luisa::span{after});
    return check;
}

// The repeated-compaction lifetime check (P5.6): several storages are built and
// compacted back to back (each `compact()` synchronises for its size query, so
// the lists cannot overlap), then every one is validated and traced.  A wrong
// callback type (`add_dtor_callback` would retire the old buffer at submit time)
// or a dangling `BufferView` in the heap fails here.
struct RepeatCheck {
    size_t contract_problems{0u};
    size_t tree_problems{0u};
    size_t heap_problems{0u};
    size_t hit_problems{0u};
    size_t reclaimed_bytes{0u};
};

[[nodiscard]] RepeatCheck run_compaction_repeat_check(Device &device, Stream &stream,
                                                      const TestScene &scene,
                                                      size_t repeats) noexcept {
    RepeatCheck check;
    auto ray_count = static_cast<uint>(scene.rays.size());
    if (ray_count == 0u || scene.meshes.empty() || scene.instances.empty()) { return check; }
    auto expected_nodes = scene_node_count(scene);
    auto sizes = SoftwareLbvh::estimate(2u * scene.triangles.size() + 1u,
                                        2u * scene.instances.size() + 1u,
                                        2u * scene.meshes.size() + 1u);
    luisa::vector<luisa::unique_ptr<SoftwareLbvh>> storages;
    luisa::vector<BuiltScene> builds;
    AccelOption option;
    option.allow_compaction = true;
    for (auto i = 0u; i < repeats; i++) {
        storages.emplace_back(luisa::make_unique<SoftwareLbvh>(device, sizes));
        builds.emplace_back(build_scene(device, stream, *storages.back(), scene, option));
    }
    // compact all of them (this is the retirement stress: each compaction retires
    // its loose buffer through a completion callback)
    for (auto i = 0u; i < repeats; i++) {
        auto result = storages[i]->compact(stream, builds[i].tlas,
                                           luisa::span{builds[i].blases});
        stream << synchronize();
        check.reclaimed_bytes += result.compacted_bytes;
        check.contract_problems += result.nodes_after == expected_nodes ? 0u : 1u;
        check.contract_problems += storages[i]->nodes().size() == expected_nodes ? 0u : 1u;
    }
    // every storage must still validate and trace correctly, on its dense buffer
    for (auto i = 0u; i < repeats; i++) {
        auto &&lbvh = *storages[i];
        auto &&built = builds[i];
        for (auto &&blas : built.blases) {
            check.tree_problems += lbvh.validate_tree(stream, blas.node_offset(),
                                                      blas.triangle_count());
        }
        check.tree_problems += lbvh.validate_tree(stream, built.tlas.node_offset(),
                                                  built.tlas.instance_count());
        check.heap_problems += lbvh.validate_heap(stream, built.tlas,
                                                  static_cast<uint>(built.blases.size()));
        Buffer<LbvhHit> hits = device.create_buffer<LbvhHit>(ray_count);
        lbvh.trace_software(stream, built.vertices, built.triangles, built.rays, hits,
                            built.tlas, ray_count);
        stream << synchronize();
        luisa::vector<LbvhHit> host_hits(ray_count);
        stream << hits.copy_to(luisa::span{host_hits}) << synchronize();
        // the same scene must give the same hits in every storage
        if (i != 0u) {
            // re-trace storage 0 for the comparison
            Buffer<LbvhHit> reference_hits = device.create_buffer<LbvhHit>(ray_count);
            storages[0]->trace_software(stream, builds[0].vertices, builds[0].triangles,
                                        builds[0].rays, reference_hits, builds[0].tlas,
                                        ray_count);
            stream << synchronize();
            luisa::vector<LbvhHit> host_reference(ray_count);
            stream << reference_hits.copy_to(luisa::span{host_reference}) << synchronize();
            check.hit_problems += count_hit_mismatches(luisa::span{host_reference},
                                                       luisa::span{host_hits});
        }
    }
    return check;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Totals {
    size_t checks{0u};
    size_t failed{0u};
    size_t tree_problems{0u};
    size_t range_problems{0u};
    size_t heap_problems{0u};
    size_t contract_problems{0u};
    size_t slice_problems{0u};
    size_t repeat_problems{0u};
    size_t compaction_problems{0u};
    size_t reference_hit_problems{0u};
    size_t software_mismatches{0u};
    size_t rtx_mismatches{0u};
    size_t compared{0u};
    size_t ties{0u};
    size_t grazing{0u};
    size_t boundary{0u};
    size_t conditioned{0u};
    size_t guard_decided{0u};
    size_t rtx_checks{0u};
};

void print_hit(const char *prefix, const HitView &hit) noexcept {
    if (!hit.hit) {
        std::printf("%smiss (t=%.9g)\n", prefix, hit.t);
        return;
    }
    std::printf("%shit inst=%u prim=%u t=%.9g u=%.6g v=%.6g\n", prefix,
                static_cast<unsigned>(hit.inst), static_cast<unsigned>(hit.prim),
                hit.t, hit.u, hit.v);
}

void report_failures(const char *what, const TestScene &scene, const HitComparison &counts,
                     luisa::span<const LbvhRay> rays, const RefScene *ref_scene) noexcept {
    for (auto &&example : counts.examples) {
        std::printf("       %s mismatch (%s): ray %llu\n", what, example.kind,
                    static_cast<unsigned long long>(example.ray));
        std::printf("         expected (host reference):\n");
        print_hit("           ", example.b);
        std::printf("         got (%s):\n", what);
        print_hit("           ", example.a);
        if (example.ray < rays.size()) {
            auto ray = rays[example.ray];
            std::printf("         ray: origin=(%.9g, %.9g, %.9g) direction=(%.9g, %.9g, %.9g) t=[%.9g, %.9g]\n",
                        static_cast<double>(ray.origin.x), static_cast<double>(ray.origin.y),
                        static_cast<double>(ray.origin.z), static_cast<double>(ray.direction.x),
                        static_cast<double>(ray.direction.y), static_cast<double>(ray.direction.z),
                        static_cast<double>(ray.t_min), static_cast<double>(ray.t_max));
            if (ref_scene != nullptr) {
                // which side the triangle geometry itself can explain
                auto probe_of = [&](const HitView &hit) noexcept {
                    if (!hit.hit) { return RefProbe{}; }
                    return probe_triangle(*ref_scene, ray, hit.inst, hit.prim);
                };
                for (auto pass = 0; pass < 2; pass++) {
                    auto &&hit = pass == 0 ? example.b : example.a;
                    if (!hit.hit) { continue; }
                    auto probe = probe_of(hit);
                    if (!probe.valid) { continue; }
                    auto guard = std::abs(probe.det) <= determinant_guard +
                                                            8.0 * float32_epsilon * probe.det_scale;
                    std::printf("         %s triangle: det=%.9e det_scale=%.9e guard=%s "
                                "min_altitude=%.6e double_mt=%s\n",
                                pass == 0 ? "reference" : "other", probe.det, probe.det_scale,
                                guard ? "yes" : "no", probe.min_altitude,
                                probe.accepted ? "accepted" : "rejected");
                    std::printf("           object ray origin=(%.9g, %.9g, %.9g) direction=(%.9g, %.9g, %.9g)\n"
                                "           object v0=(%.9g, %.9g, %.9g) v1=(%.9g, %.9g, %.9g) v2=(%.9g, %.9g, %.9g)\n",
                                probe.object_origin.x, probe.object_origin.y, probe.object_origin.z,
                                probe.object_direction.x, probe.object_direction.y, probe.object_direction.z,
                                probe.v0.x, probe.v0.y, probe.v0.z, probe.v1.x, probe.v1.y, probe.v1.z,
                                probe.v2.x, probe.v2.y, probe.v2.z);
                }
            }
        }
    }
    (void)scene;
}

void report_check(const char *family, const TestScene &scene, const SceneRun &run,
                  const Options &options, double elapsed_ms,
                  const RefScene *ref_scene) noexcept {
    auto ok = run.fatal() == 0u;
    std::printf("%-6s %-15s %-16s problems=%-3zu mismatches=%-3zu rtx=%-3zu ties=%-4zu "
                "grazing=%-3zu boundary=%-3zu conditioned=%-3zu guard=%-3zu %.0fms\n",
                ok ? "[ok]" : "[fail]", family, scene.label.c_str(), run.problems(),
                run.software.fatal(), run.rtx.fatal(), run.software.ties + run.rtx.ties,
                run.grazing(), run.boundary(), run.conditioned(), run.guard_decided(),
                elapsed_ms);
    if (ok) { return; }
    if (run.tree_problems != 0u) {
        std::printf("       validate_tree reported %zu problem(s)\n", run.tree_problems);
    }
    if (run.range_problems != 0u) {
        std::printf("       node/primitive range bookkeeping: %zu problem(s)\n", run.range_problems);
    }
    if (run.heap_problems != 0u) {
        std::printf("       bindless heap (slot assignment / region resolution): %zu problem(s)\n",
                    run.heap_problems);
    }
    if (run.contract_problems != 0u) {
        std::printf("       hit contract (inst/prim/t on a miss): %zu problem(s)\n",
                    run.contract_problems);
    }
    if (run.slice_problems != 0u) {
        std::printf("       strided slices differ from the contiguous trace: %zu problem(s)\n",
                    run.slice_problems);
    }
    if (run.repeat_problems != 0u) {
        std::printf("       build/trace repeat is not bit-identical: %zu problem(s) "
                    "(%zu hit, %zu node)\n",
                    run.repeat_problems, run.repeat_hit_problems, run.repeat_node_problems);
    }
    if (run.compaction_problems != 0u) {
        std::printf("       storage compaction (contract/structure/heap/hits/copy): "
                    "%zu problem(s)\n",
                    run.compaction_problems);
    }
    if (run.reference_hit_problems != 0u) {
        std::printf("       the host reference found no hit at all for this scene\n");
    }
    report_failures("software LBVH", scene, run.software, luisa::span{scene.rays}, ref_scene);
    if (run.rtx_ran) {
        report_failures("RTX", scene, run.rtx, luisa::span{scene.rays}, ref_scene);
    }
    if (options.verbose) {
        std::printf("       software hits %zu, reference hits %zu, rtx hits %zu\n",
                    run.software_hits, run.reference_hits, run.rtx_hits);
    }
}

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

// Builds every scene of `scenes` into one shared storage (which is the shape the
// library is designed for: all trees of a scene share their node/primitive
// buffers) and checks each of them.
void run_family(Device &device, Stream &stream, const Options &options, const char *family,
                luisa::vector<TestScene> &scenes, const RtxShader &rtx_shader,
                Totals &totals, bool determinism_scenes) noexcept {
    if (scenes.empty()) { return; }
    size_t total_triangles = 0u;
    size_t total_instances = 0u;
    size_t total_blases = 0u;
    for (auto &&scene : scenes) {
        total_triangles += scene.triangles.size();
        total_instances += scene.instances.size();
        total_blases += scene.meshes.size();
    }
    SoftwareLbvh lbvh{device, SoftwareLbvh::estimate(total_triangles, total_instances,
                                                     total_blases)};
    size_t running_nodes = 0u;
    size_t running_primitives = 0u;
    size_t running_blases = 0u;
    for (auto &&scene : scenes) {
        auto ref = make_ref_scene(scene);
        Clock clock;
        clock.tic();
        auto run = run_scene(device, stream, lbvh, scene, ref, rtx_shader,
                             /*with_rtx=*/true, options.verbose, running_nodes,
                             running_primitives, running_blases);
        if (determinism_scenes) {
            auto repeat = run_determinism(device, stream, scene, options);
            run.tree_problems += repeat.tree_problems;
            run.heap_problems += repeat.heap_problems;
            run.repeat_problems += repeat.repeat_problems;
            run.repeat_hit_problems += repeat.repeat_hit_problems;
            run.repeat_node_problems += repeat.repeat_node_problems;
        }
        // Storage compaction (P5): every scene is rebuilt into its own storage
        // (the family's shared one must not be compacted, because a compacted
        // storage has no spare capacity for the next scene's trees), compacted
        // and checked - structure, heap, hit identity, exact copy, no-op.
        {
            auto compaction = run_compaction_checks(device, stream, scene,
                                                    SoftwareLbvh::CompactionPolicy::as_built);
            run.compaction_problems += compaction.contract_problems +
                                       compaction.tree_problems + compaction.heap_problems +
                                       compaction.hit_problems + compaction.node_problems;
        }
        auto elapsed_ms = clock.toc();
        report_check(family, scene, run, options, elapsed_ms, &ref);
        totals.checks++;
        totals.failed += run.fatal() == 0u ? 0u : 1u;
        totals.tree_problems += run.tree_problems;
        totals.range_problems += run.range_problems;
        totals.heap_problems += run.heap_problems;
        totals.contract_problems += run.contract_problems;
        totals.slice_problems += run.slice_problems;
        totals.repeat_problems += run.repeat_problems;
        totals.compaction_problems += run.compaction_problems;
        totals.reference_hit_problems += run.reference_hit_problems;
        totals.software_mismatches += run.software.fatal();
        totals.rtx_mismatches += run.rtx.fatal();
        totals.compared += run.software.compared;
        totals.ties += run.software.ties + run.rtx.ties;
        totals.grazing += run.grazing();
        totals.boundary += run.boundary();
        totals.conditioned += run.conditioned();
        totals.guard_decided += run.guard_decided();
        totals.rtx_checks += run.rtx_ran ? 1u : 0u;
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// (8) compaction-specific family: the no-op boundary (a storage sized exactly
// like the scene), repeated compactions of independent storages, and the
// `as_built` copy/identity contract - next to the per-scene compaction check
// that `run_family` already runs.
// ---------------------------------------------------------------------------

void run_compaction_family(Device &device, Stream &stream, const Options &options,
                           luisa::vector<TestScene> &scenes, Totals &totals) noexcept {
    for (auto &&scene : scenes) {
        auto ref = make_ref_scene(scene);
        Clock clock;
        clock.tic();
        SceneRun run;
        auto noop = run_compaction_noop_check(device, stream, scene);
        run.compaction_problems += noop.contract_problems + noop.hit_problems +
                                   noop.tree_problems;
        auto compaction = run_compaction_checks(device, stream, scene,
                                                SoftwareLbvh::CompactionPolicy::as_built);
        run.compaction_problems += compaction.contract_problems + compaction.tree_problems +
                                   compaction.heap_problems + compaction.hit_problems +
                                   compaction.node_problems;
        if (options.verbose) {
            std::printf("       compaction %-16s trees=%zu nodes %zu -> %zu reclaimed=%zu "
                        "(no-op %s)\n",
                        scene.label.c_str(), compaction.trees, compaction.nodes_before,
                        compaction.nodes_after, compaction.reclaimed_bytes,
                        noop.contract_problems == 0u ? "ok" : "bad");
        }
        auto elapsed_ms = clock.toc();
        report_check("compact", scene, run, options, elapsed_ms, &ref);
        totals.checks++;
        totals.failed += run.fatal() == 0u ? 0u : 1u;
        totals.compaction_problems += run.compaction_problems;
        std::fflush(stdout);
    }
    if (!scenes.empty()) {
        // the build-scratch release (P6, opt-in): after it the storage is
        // traverse-only and the structure must still validate and trace
        SceneRun scratch_run;
        TestScene scratch_labelled;
        scratch_labelled.label = "release-scratch";
        Clock scratch_clock;
        scratch_clock.tic();
        auto scratch = run_scratch_release_check(device, stream, scenes.front());
        scratch_run.compaction_problems += scratch.contract_problems +
                                           scratch.tree_problems + scratch.heap_problems +
                                           scratch.hit_problems;
        auto scratch_ms = scratch_clock.toc();
        report_check("compact", scratch_labelled, scratch_run, options, scratch_ms, nullptr);
        totals.checks++;
        totals.failed += scratch_run.fatal() == 0u ? 0u : 1u;
        totals.compaction_problems += scratch_run.compaction_problems;
        std::fflush(stdout);
    }
    if (!scenes.empty()) {
        // repeated compactions of independent storages (P5.6): the retirement
        // callbacks must all fire and nothing may dangle.
        auto repeats = options.quick ? 2u : 3u;
        SceneRun run;
        TestScene labelled;
        labelled.label = luisa::format("repeat x{}", repeats);
        Clock clock;
        clock.tic();
        auto repeat = run_compaction_repeat_check(device, stream, scenes.front(), repeats);
        run.compaction_problems += repeat.contract_problems + repeat.tree_problems +
                                   repeat.heap_problems + repeat.hit_problems;
        auto elapsed_ms = clock.toc();
        report_check("compact", labelled, run, options, elapsed_ms, nullptr);
        totals.checks++;
        totals.failed += run.fatal() == 0u ? 0u : 1u;
        totals.compaction_problems += run.compaction_problems;
        std::fflush(stdout);
    }
}

[[nodiscard]] luisa::vector<uint> boundary_sizes(bool quick) noexcept {
    if (quick) {
        return {1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u,
                255u, 256u, 257u, 4097u};
    }
    return {1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u,
            127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 513u, 1000u, 4095u, 4096u,
            4097u};
}

[[nodiscard]] int run_all(Device &device, Stream &stream, const Options &options) noexcept {
    auto rtx_shader = device.compile(make_rtx_trace_kernel());
    auto ray_budget = options.quick ? 96u : 256u;
    Totals totals;
    Clock total_clock;
    total_clock.tic();

    {
        luisa::vector<TestScene> scenes;
        for (auto n : boundary_sizes(options.quick)) {
            scenes.emplace_back(make_boundary_scene(n, 0x1000ull + n, ray_budget));
        }
        run_family(device, stream, options, "boundary", scenes, rtx_shader, totals, false);
    }
    {
        auto scenes = make_degenerate_scenes(options.seed + 0x2000ull, ray_budget / 2u);
        run_family(device, stream, options, "degenerate", scenes, rtx_shader, totals, false);
    }
    {
        luisa::vector<TestScene> scenes;
        for (auto blas_count : {1u, 2u, 3u, 5u, 8u, 17u}) {
            scenes.emplace_back(make_multi_blas_scene(
                blas_count, 0x3000ull + blas_count, ray_budget,
                luisa::format("blas={} inst={}", blas_count, std::max(blas_count, 8u)).c_str()));
        }
        run_family(device, stream, options, "multi-blas", scenes, rtx_shader, totals, false);
    }
    {
        luisa::vector<TestScene> scenes;
        auto seeds = options.quick ? 8u : 32u;
        for (auto i = 0u; i < seeds; i++) {
            scenes.emplace_back(make_random_scene(options.seed + 0x4000ull + i, options.quick,
                                                  ray_budget));
        }
        run_family(device, stream, options, "random", scenes, rtx_shader, totals, false);
    }
    {
        // (8) storage compaction: the no-op boundary (a storage sized exactly like
        // the scene), the 1/2/17-BLAS and count==1 cases, and repeated
        // compactions.  `run_family` already compacts every scene of every family
        // above; this family adds the cases that need a purpose-built storage.
        luisa::vector<TestScene> scenes;
        scenes.emplace_back(make_boundary_scene(1u, 0x6000ull, ray_budget));
        scenes.emplace_back(make_boundary_scene(17u, 0x6001ull, ray_budget));
        scenes.emplace_back(make_multi_blas_scene(1u, 0x6002ull, ray_budget, "blas=1"));
        scenes.emplace_back(make_multi_blas_scene(2u, 0x6003ull, ray_budget, "blas=2"));
        scenes.emplace_back(make_multi_blas_scene(17u, 0x6004ull, ray_budget, "blas=17"));
        scenes.emplace_back(make_random_scene(options.seed + 0x6005ull, options.quick,
                                              ray_budget));
        run_compaction_family(device, stream, options, scenes, totals);
    }
    {
        // (6)/(7): a scene whose primitives all coincide (so the tree order is
        // observable in the reported `prim`), a random one and a multi-instance one
        luisa::vector<TestScene> scenes;
        scenes.emplace_back(make_coincident_scene(0x5000ull, ray_budget, "coincident-64"));
        scenes.emplace_back(make_random_scene(options.seed + 0x5001ull, options.quick,
                                              ray_budget));
        scenes.emplace_back(make_multi_blas_scene(5u, 0x5002ull, ray_budget, "multi-5"));
        run_family(device, stream, options, "determinism", scenes, rtx_shader, totals, true);
    }

    auto elapsed = total_clock.toc();
    std::printf("summary: %zu check(s), %zu passed, %zu failed in %.1f s (backend %s, seed %llu%s)\n",
                totals.checks, totals.checks - totals.failed, totals.failed, elapsed / 1000.0,
                options.backend.c_str(), static_cast<unsigned long long>(options.seed),
                options.quick ? ", quick" : "");
    std::printf("summary: problems: tree %zu, ranges %zu, heap %zu, hit contract %zu, slices %zu, "
                "repeats %zu, compaction %zu, empty reference %zu\n",
                totals.tree_problems, totals.range_problems, totals.heap_problems,
                totals.contract_problems,
                totals.slice_problems, totals.repeat_problems, totals.compaction_problems,
                totals.reference_hit_problems);
    std::printf("summary: software-vs-reference mismatches %zu, RTX-vs-reference mismatches %zu "
                "(%zu RTX cross-checks); compared %zu hit(s), ties %zu; not comparable: "
                "grazing %zu, boundary %zu, ill-conditioned %zu, determinant-guard %zu\n",
                totals.software_mismatches, totals.rtx_mismatches, totals.rtx_checks,
                totals.compared, totals.ties, totals.grazing, totals.boundary,
                totals.conditioned, totals.guard_decided);
    std::fflush(stdout);
    return totals.failed == 0u ? 0 : 1;
}

}// namespace

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(executable);
        return 1;
    }
    Context context{executable};
    Device device = context.create_device(options.backend);
    Stream stream = device.create_stream();
    std::printf("lbvh_test: backend=%s seed=%llu quick=%d\n", options.backend.c_str(),
                static_cast<unsigned long long>(options.seed), options.quick ? 1 : 0);
    std::fflush(stdout);
    return run_all(device, stream, options);
}
