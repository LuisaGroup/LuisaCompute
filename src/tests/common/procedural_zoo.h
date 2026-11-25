// Shared procedural-primitive "zoo" for ray-query sampling tests and
// benchmarks.
//
// Included by
//   src/tests/integration/runtime/test_procedural_sampling.cpp
//   src/tests/benchmark/benchmark_procedural_ray_query.cpp
//
// It provides
//   * ProcPrim, a POD record describing one implicit primitive.  It is also a
//     DSL struct, so the very same bytes are read by the device intersection
//     handler (only `dims`/`kind` matter there; `center`/`ex`/`ey`/`ez` are the
//     world transform),
//   * builders for the twelve zoo kinds, each with an orthonormal local frame
//     and a conservative world-space AABB (the procedural BLAS bounds),
//   * host-side signed-distance bounds (double precision) and analytic volumes,
//     used as the independent reference,
//   * device-side analytic first-hit code shared by both binaries.
//
// The zoo kinds are deliberately different *classes* of implicit geometry:
//
//   kind            local implicit                       analytic device hit
//   Sphere          |p| = r                              quadratic
//   Ellipsoid       |p/dims| = 1                          unit sphere in scaled space
//   Box             |p_i| <= h_i                          slab clip
//   Capsule         dist(p, segment) <= r                 cylinder body + two caps
//   Cylinder        |p_xy| <= r, |p_z| <= h               quadratic + two caps
//   CappedCone      r(z) = a + b z, |z| <= h              cone quadric + two caps
//   Bicone          two cones sharing a base plane        two cone quadrics
//   Tetrahedron     4 half spaces                         convex plane clip
//   HexPrism        3 slabs in xy + 2 caps                convex plane clip
//   Octahedron      8 half spaces                         convex plane clip
//   SphereChain     16 spheres on a circle                min of 16 spheres
//   HelixTube       12 capsules along a helix             min of 12 capsules
//
// Everything the device does is analytic (no sphere tracing), so the test can
// compare against a host reference that marches the exact implicit function.

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/aabb.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace luisa::test::zoo {

using namespace luisa::compute;

using luisa::float3;
using luisa::make_float3;
using luisa::uint;

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr double pi_d = 3.14159265358979323846;

// Number of chained elements of the two "loop" kinds.  Constant so the device
// handlers can be fully unrolled.
inline constexpr uint sphere_chain_count = 16u;
inline constexpr uint helix_segment_count = 12u;

enum Kind : uint {
    Sphere = 0u,
    Ellipsoid = 1u,
    Box = 2u,
    Capsule = 3u,
    Cylinder = 4u,
    CappedCone = 5u,
    Bicone = 6u,
    Tetrahedron = 7u,
    HexPrism = 8u,
    Octahedron = 9u,
    SphereChain = 10u,
    HelixTube = 11u,
    kind_count = 12u
};

struct ProcPrim {
    float3 center; // world-space origin of the local frame
    float3 ex;     // local x axis (unit)
    float3 ey;     // local y axis (unit)
    float3 ez;     // local z axis (unit, the "main" axis)
    float3 dims;   // kind specific, see make_* below
    uint kind;
};

}// namespace luisa::test::zoo

LUISA_STRUCT(luisa::test::zoo::ProcPrim, center, ex, ey, ez, dims, kind) {};

namespace luisa::test::zoo {

// ---------------------------------------------------------------------------
// Host-side (double precision) geometry
// ---------------------------------------------------------------------------
// The host reference deliberately lives in its own namespace: its free
// functions (dot, normalize, min/max, operator-, ...) would otherwise hide the
// global DSL operators of the same name inside the zoo namespace.
namespace host {


struct DVec3 {
    double x{};
    double y{};
    double z{};
};

[[nodiscard]] inline DVec3 operator+(DVec3 a, DVec3 b) noexcept { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
[[nodiscard]] inline DVec3 operator-(DVec3 a, DVec3 b) noexcept { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
[[nodiscard]] inline DVec3 operator*(DVec3 a, double s) noexcept { return {a.x * s, a.y * s, a.z * s}; }
[[nodiscard]] inline DVec3 operator*(double s, DVec3 a) noexcept { return a * s; }
[[nodiscard]] inline DVec3 operator-(DVec3 a) noexcept { return {-a.x, -a.y, -a.z}; }
[[nodiscard]] inline double dot(DVec3 a, DVec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] inline DVec3 cross(DVec3 a, DVec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
[[nodiscard]] inline double length(DVec3 a) noexcept { return std::sqrt(dot(a, a)); }
[[nodiscard]] inline DVec3 normalize(DVec3 a) noexcept {
    auto l = length(a);
    return l > 0.0 ? a * (1.0 / l) : DVec3{0.0, 0.0, 1.0};
}
[[nodiscard]] inline DVec3 abs(DVec3 a) noexcept { return {std::abs(a.x), std::abs(a.y), std::abs(a.z)}; }
[[nodiscard]] inline DVec3 max(DVec3 a, DVec3 b) noexcept {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
[[nodiscard]] inline double max_component(DVec3 a) noexcept { return std::max(a.x, std::max(a.y, a.z)); }
[[nodiscard]] inline double clamp(double x, double lo, double hi) noexcept { return std::min(std::max(x, lo), hi); }

struct Box3 {
    DVec3 lo{};
    DVec3 hi{};
};

[[nodiscard]] inline ProcPrim make_prim(uint kind, float3 center, float3 axis, float3 dims) noexcept {
    auto w = normalize(DVec3{axis.x, axis.y, axis.z});
    // pick the axis least aligned with w, then orthonormalize
    auto helper = std::abs(w.x) < 0.6 ? DVec3{1.0, 0.0, 0.0} : DVec3{0.0, 1.0, 0.0};
    auto u = normalize(cross(helper, w));
    auto v = cross(w, u);
    ProcPrim p{};
    p.center = center;
    p.ex = make_float3(static_cast<float>(u.x), static_cast<float>(u.y), static_cast<float>(u.z));
    p.ey = make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
    p.ez = make_float3(static_cast<float>(w.x), static_cast<float>(w.y), static_cast<float>(w.z));
    p.dims = dims;
    p.kind = kind;
    return p;
}

// dims = (radius, 0, 0)
[[nodiscard]] inline ProcPrim make_sphere(float3 c, float r) noexcept {
    return make_prim(Sphere, c, make_float3(0.f, 0.f, 1.f), make_float3(r, 0.f, 0.f));
}
// dims = (a, b, c)
[[nodiscard]] inline ProcPrim make_ellipsoid(float3 c, float3 radii, float3 axis) noexcept {
    return make_prim(Ellipsoid, c, axis, radii);
}
// dims = (hx, hy, hz)
[[nodiscard]] inline ProcPrim make_box(float3 c, float3 half_size, float3 axis) noexcept {
    return make_prim(Box, c, axis, half_size);
}
// dims = (radius, half_height, 0), axis = ez
[[nodiscard]] inline ProcPrim make_capsule(float3 c, float r, float half_height, float3 axis) noexcept {
    return make_prim(Capsule, c, axis, make_float3(r, half_height, 0.f));
}
// dims = (radius, half_height, 0), axis = ez
[[nodiscard]] inline ProcPrim make_cylinder(float3 c, float r, float half_height, float3 axis) noexcept {
    return make_prim(Cylinder, c, axis, make_float3(r, half_height, 0.f));
}
// dims = (radius at -h, radius at +h, half_height), axis = ez
[[nodiscard]] inline ProcPrim make_capped_cone(float3 c, float r_bottom, float r_top, float half_height, float3 axis) noexcept {
    return make_prim(CappedCone, c, axis, make_float3(r_bottom, r_top, half_height));
}
// dims = (base radius, half_height, 0); base plane at the origin, apexes at +-h
[[nodiscard]] inline ProcPrim make_bicone(float3 c, float r_base, float half_height, float3 axis) noexcept {
    return make_prim(Bicone, c, axis, make_float3(r_base, half_height, 0.f));
}
// dims = (s, 0, 0); vertices (+-s, +-s, +-s) with an even number of minus signs
[[nodiscard]] inline ProcPrim make_tetrahedron(float3 c, float s, float3 axis) noexcept {
    return make_prim(Tetrahedron, c, axis, make_float3(s, 0.f, 0.f));
}
// dims = (circumradius, half_height, 0)
[[nodiscard]] inline ProcPrim make_hex_prism(float3 c, float circumradius, float half_height, float3 axis) noexcept {
    return make_prim(HexPrism, c, axis, make_float3(circumradius, half_height, 0.f));
}
// dims = (s, 0, 0); |x| + |y| + |z| <= s
[[nodiscard]] inline ProcPrim make_octahedron(float3 c, float s, float3 axis) noexcept {
    return make_prim(Octahedron, c, axis, make_float3(s, 0.f, 0.f));
}
// dims = (major radius, sphere radius, 0); spheres sit in the local xy plane
[[nodiscard]] inline ProcPrim make_sphere_chain(float3 c, float major, float r, float3 axis) noexcept {
    return make_prim(SphereChain, c, axis, make_float3(major, r, 0.f));
}
// dims = (helix radius, tube radius, half_height)
[[nodiscard]] inline ProcPrim make_helix_tube(float3 c, float helix_radius, float tube_radius, float half_height, float3 axis) noexcept {
    return make_prim(HelixTube, c, axis, make_float3(helix_radius, tube_radius, half_height));
}

// Part of the zoo that the Monte-Carlo volume test can validate: convex bodies
// only (the inside test used there is `hit in +x && hit in -x`).
[[nodiscard]] inline bool is_convex(uint kind) noexcept {
    switch (kind) {
        case Sphere:
        case Ellipsoid:
        case Box:
        case Capsule:
        case Cylinder:
        case CappedCone:
        case Bicone:
        case Tetrahedron:
        case HexPrism:
        case Octahedron: return true;
        default: return false;
    }
}

// iq's exact signed distance of a z-aligned capped cone (radius r1 at z = -h,
// radius r2 at z = +h), expressed in the (radius, z) half plane.
[[nodiscard]] inline double sd_capped_cone_z(double rr, double z, double r1, double r2, double h) noexcept {
    double k1x = r2, k1y = h;
    double k2x = r2 - r1, k2y = 2.0 * h;
    double cax = rr - std::min(rr, z < 0.0 ? r1 : r2);
    double cay = std::abs(z) - h;
    double t = clamp(((k1x - rr) * k2x + (k1y - z) * k2y) / (k2x * k2x + k2y * k2y), 0.0, 1.0);
    double cbx = rr - k1x + k2x * t;
    double cby = z - k1y + k2y * t;
    auto s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * std::sqrt(std::min(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

// Signed distance (a conservative bound outside, exact inside) of the local
// implicit function, in double precision.
[[nodiscard]] inline double sdf(const ProcPrim &p, DVec3 q) noexcept {
    auto rel = q - DVec3{p.center.x, p.center.y, p.center.z};
    auto e0 = DVec3{p.ex.x, p.ex.y, p.ex.z};
    auto e1 = DVec3{p.ey.x, p.ey.y, p.ey.z};
    auto e2 = DVec3{p.ez.x, p.ez.y, p.ez.z};
    DVec3 s{dot(rel, e0), dot(rel, e1), dot(rel, e2)}; // local coordinates
    double x = s.x, y = s.y, z = s.z;
    auto d = [&](int i) noexcept { return static_cast<double>(i == 0 ? p.dims.x : (i == 1 ? p.dims.y : p.dims.z)); };
    switch (p.kind) {
        case Sphere: return std::sqrt(x * x + y * y + z * z) - d(0);
        case Ellipsoid: {
            auto a = d(0), b = d(1), c = d(2);
            auto k0 = std::sqrt((x / a) * (x / a) + (y / b) * (y / b) + (z / c) * (z / c));
            auto k1 = std::sqrt((x / (a * a)) * (x / (a * a)) + (y / (b * b)) * (y / (b * b)) + (z / (c * c)) * (z / (c * c)));
            return k1 > 0.0 ? k0 * (k0 - 1.0) / k1 : -std::min({a, b, c});
        }
        case Box: {
            auto qx = std::abs(x) - d(0), qy = std::abs(y) - d(1), qz = std::abs(z) - d(2);
            auto outside = std::sqrt(std::max(qx, 0.0) * std::max(qx, 0.0) +
                                      std::max(qy, 0.0) * std::max(qy, 0.0) +
                                      std::max(qz, 0.0) * std::max(qz, 0.0));
            auto inside = std::min(std::max(qx, std::max(qy, qz)), 0.0);
            return outside + inside;
        }
        case Capsule: {
            auto h = d(1), r = d(0);
            auto zc = clamp(z, -h, h);
            auto dz = z - zc;
            return std::sqrt(x * x + y * y + dz * dz) - r;
        }
        case Cylinder: {
            auto r = d(0), h = d(1);
            auto dr = std::sqrt(x * x + y * y) - r;
            auto dz = std::abs(z) - h;
            auto outside = std::sqrt(std::max(dr, 0.0) * std::max(dr, 0.0) + std::max(dz, 0.0) * std::max(dz, 0.0));
            return outside + std::min(std::max(dr, dz), 0.0);
        }
        case CappedCone: {
            auto rr = std::sqrt(x * x + y * y);
            return sd_capped_cone_z(rr, z, d(0), d(1), d(2));
        }
        case Bicone: {
            auto r = d(0), h = d(1), half = 0.5 * h;
            auto rr = std::sqrt(x * x + y * y);
            return std::min(sd_capped_cone_z(rr, z - half, r, 0.0, half),
                            sd_capped_cone_z(rr, -z - half, r, 0.0, half));
        }
        case Tetrahedron: {
            auto s = d(0) / std::sqrt(3.0);
            // planes: dot(v_i, p) <= s for v_i = (+-1, +-1, +-1) with product +1
            double f = -1e30;
            for (int i = 0; i < 4; i++) {
                double sx = (i == 0 || i == 3) ? 1.0 : -1.0;
                double sy = (i == 0 || i == 2) ? 1.0 : -1.0;
                double sz = (i == 0 || i == 1) ? 1.0 : -1.0;
                f = std::max(f, (sx * x + sy * y + sz * z) / std::sqrt(3.0) - s);
            }
            return f;
        }
        case HexPrism: {
            auto R = d(0), h = d(1);
            auto inr = R * std::sqrt(3.0) / 2.0;
            auto c = std::cos(pi_d / 3.0), sn = std::sin(pi_d / 3.0);
            auto f = std::max(std::abs(x), std::abs(x * c + y * sn));
            f = std::max(f, std::abs(x * c - y * sn));
            auto fxy = f - inr;
            auto fz = std::abs(z) - h;
            auto outside = std::sqrt(std::max(fxy, 0.0) * std::max(fxy, 0.0) + std::max(fz, 0.0) * std::max(fz, 0.0));
            return outside + std::min(std::max(fxy, fz), 0.0);
        }
        case Octahedron: {
            auto s = d(0);
            return (std::abs(x) + std::abs(y) + std::abs(z) - s) / std::sqrt(3.0);
        }
        case SphereChain: {
            auto R = d(0), r = d(1);
            double best = 1e30;
            for (uint i = 0; i < sphere_chain_count; i++) {
                auto a = 2.0 * pi_d * static_cast<double>(i) / static_cast<double>(sphere_chain_count);
                auto cx = R * std::cos(a), cy = R * std::sin(a);
                auto dx = x - cx, dy = y - cy;
                best = std::min(best, std::sqrt(dx * dx + dy * dy + z * z) - r);
            }
            return best;
        }
        case HelixTube: {
            auto R = d(0), r = d(1), h = d(2);
            auto point = [&](uint i) noexcept {
                auto t = static_cast<double>(i) / static_cast<double>(helix_segment_count);
                auto a = 2.0 * pi_d * t * 1.5;
                return DVec3{R * std::cos(a), R * std::sin(a), h * (2.0 * t - 1.0)};
            };
            double best = 1e30;
            for (uint i = 0; i + 1u < helix_segment_count; i++) {
                auto a = point(i), b = point(i + 1u);
                auto ba = b - a, pa = DVec3{x, y, z} - a;
                auto baba = dot(ba, ba);
                auto tt = clamp(dot(pa, ba) / baba, 0.0, 1.0);
                auto q = pa - ba * tt;
                best = std::min(best, length(q) - r);
            }
            return best;
        }
        default: return 1e30;
    }
}

// Conservative world-space bounds of one primitive.
[[nodiscard]] inline Box3 bounds(const ProcPrim &p) noexcept {
    auto c = DVec3{p.center.x, p.center.y, p.center.z};
    auto d = [&](int i) noexcept { return static_cast<double>(i == 0 ? p.dims.x : (i == 1 ? p.dims.y : p.dims.z)); };
    double r = 0.0;
    switch (p.kind) {
        case Sphere: r = d(0); break;
        case Ellipsoid: r = std::max(d(0), std::max(d(1), d(2))); break;
        case Box: r = std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2)); break;
        case Capsule: r = d(0) + d(1); break;
        case Cylinder: r = std::sqrt(d(0) * d(0) + d(1) * d(1)); break;
        case CappedCone: r = std::sqrt(std::max(d(0), d(1)) * std::max(d(0), d(1)) + d(2) * d(2)); break;
        case Bicone: r = std::sqrt(d(0) * d(0) + d(1) * d(1)); break;
        case Tetrahedron: r = d(0) * std::sqrt(3.0); break;
        case HexPrism: r = std::sqrt(d(0) * d(0) + d(1) * d(1)); break;
        case Octahedron: r = d(0); break;
        case SphereChain: r = d(0) + d(1); break;
        case HelixTube: r = std::sqrt(d(0) * d(0) + d(2) * d(2)) + d(1); break;
        default: break;
    }
    return {DVec3{c.x - r, c.y - r, c.z - r}, DVec3{c.x + r, c.y + r, c.z + r}};
}

// Analytic volume of the primitive (0 when it is not analytically convenient).
[[nodiscard]] inline double analytic_volume(const ProcPrim &p) noexcept {
    auto d = [&](int i) noexcept { return static_cast<double>(i == 0 ? p.dims.x : (i == 1 ? p.dims.y : p.dims.z)); };
    switch (p.kind) {
        case Sphere: return 4.0 / 3.0 * pi_d * d(0) * d(0) * d(0);
        case Ellipsoid: return 4.0 / 3.0 * pi_d * d(0) * d(1) * d(2);
        case Box: return 8.0 * d(0) * d(1) * d(2);
        case Capsule: return pi_d * d(0) * d(0) * 2.0 * d(1) + 4.0 / 3.0 * pi_d * d(0) * d(0) * d(0);
        case Cylinder: return pi_d * d(0) * d(0) * 2.0 * d(1);
        case CappedCone: {
            auto r1 = d(0), r2 = d(1), h = d(2);
            return pi_d * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2) * 2.0;
        }
        case Bicone: {
            auto r = d(0), h = d(1);
            return 2.0 * (pi_d * r * r * h / 3.0);
        }
        case Tetrahedron: {
            // vertices (+-s,+-s,+-s) (even number of minus signs)
            auto s = d(0);
            return 8.0 * s * s * s / 3.0;
        }
        case HexPrism: {
            auto R = d(0), h = d(1);
            return 3.0 * std::sqrt(3.0) / 2.0 * R * R * 2.0 * h;
        }
        case Octahedron: {
            auto s = d(0);
            return 4.0 / 3.0 * s * s * s;
        }
        default: return 0.0;
    }
}

// Ray/box interval test (double precision).
[[nodiscard]] inline bool ray_box(const Box3 &b, DVec3 o, DVec3 d, double t_min, double t_max,
                                  double &t0_out, double &t1_out) noexcept {
    auto t0 = t_min, t1 = t_max;
    for (int axis = 0; axis < 3; axis++) {
        auto oa = axis == 0 ? o.x : (axis == 1 ? o.y : o.z);
        auto da = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
        auto lo = axis == 0 ? b.lo.x : (axis == 1 ? b.lo.y : b.lo.z);
        auto hi = axis == 0 ? b.hi.x : (axis == 1 ? b.hi.y : b.hi.z);
        if (std::abs(da) < 1e-300) {
            if (oa < lo || oa > hi) { return false; }
        } else {
            auto inv = 1.0 / da;
            auto a = (lo - oa) * inv;
            auto c = (hi - oa) * inv;
            if (a > c) { std::swap(a, c); }
            t0 = std::max(t0, a);
            t1 = std::min(t1, c);
        }
    }
    t0_out = t0;
    t1_out = t1;
    return t0 <= t1;
}

// The whole zoo: one primitive of every kind.
[[nodiscard]] inline luisa::vector<ProcPrim> make_zoo(float spacing = 2.3f) noexcept {
    luisa::vector<ProcPrim> prims;
    prims.reserve(kind_count);
    auto at = [&](uint column, uint row) noexcept {
        return make_float3((static_cast<float>(column) - 1.5f) * spacing,
                           (static_cast<float>(row) - 1.f) * spacing,
                           0.f);
    };
    auto tilt = make_float3(0.35f, 0.45f, 0.82f);
    prims.emplace_back(make_sphere(at(0u, 0u), 0.55f));
    prims.emplace_back(make_ellipsoid(at(1u, 0u), make_float3(0.62f, 0.34f, 0.45f), tilt));
    prims.emplace_back(make_box(at(2u, 0u), make_float3(0.48f, 0.38f, 0.30f), tilt));
    prims.emplace_back(make_capsule(at(3u, 0u), 0.26f, 0.42f, make_float3(0.3f, 0.9f, 0.2f)));
    prims.emplace_back(make_cylinder(at(0u, 1u), 0.42f, 0.40f, tilt));
    prims.emplace_back(make_capped_cone(at(1u, 1u), 0.52f, 0.16f, 0.44f, make_float3(0.2f, 1.f, 0.15f)));
    prims.emplace_back(make_bicone(at(2u, 1u), 0.48f, 0.52f, tilt));
    prims.emplace_back(make_tetrahedron(at(3u, 1u), 0.52f, tilt));
    prims.emplace_back(make_hex_prism(at(0u, 2u), 0.52f, 0.34f, make_float3(0.1f, 0.2f, 1.f)));
    prims.emplace_back(make_octahedron(at(1u, 2u), 0.62f, tilt));
    prims.emplace_back(make_sphere_chain(at(2u, 2u), 0.58f, 0.185f, make_float3(0.05f, 0.35f, 0.94f)));
    prims.emplace_back(make_helix_tube(at(3u, 2u), 0.44f, 0.115f, 0.52f, make_float3(0.f, 0.f, 1.f)));
    return prims;
}

[[nodiscard]] inline AABB to_aabb(const Box3 &b) noexcept {
    AABB aabb{};
    aabb.packed_min = {static_cast<float>(b.lo.x), static_cast<float>(b.lo.y), static_cast<float>(b.lo.z)};
    aabb.packed_max = {static_cast<float>(b.hi.x), static_cast<float>(b.hi.y), static_cast<float>(b.hi.z)};
    return aabb;
}

}// namespace host

// ---------------------------------------------------------------------------
// Device side: analytic first hit
// ---------------------------------------------------------------------------

// Accumulates the nearest hit distance in [t_min, best).  `best` is a device
// local, so the assignment stays a store and survives control flow.
inline void zoo_consider(Var<float> &best, Expr<float> t, Expr<float> t_min) noexcept {
    $if (t >= t_min & t < best) { best = t; };
}

inline void zoo_sphere(Expr<float3> o, Expr<float3> d, Expr<float3> c, Expr<float> r,
                       Var<float> &best, Expr<float> t_min) noexcept {
    auto oc = o - c;
    auto a = dot(d, d);
    auto hb = dot(oc, d);
    auto cc = dot(oc, oc) - r * r;
    auto disc = hb * hb - a * cc;
    $if (disc >= 0.f) {
        auto sq = sqrt(disc);
        auto inv = 1.f / a;
        auto t0 = (-hb - sq) * inv;
        auto t1 = (-hb + sq) * inv;
        zoo_consider(best, select(t1, t0, t0 >= t_min), t_min);
    };
}

inline void zoo_capsule_segment(Expr<float3> o, Expr<float3> d, Expr<float3> a, Expr<float3> b,
                                Expr<float> r, Var<float> &best, Expr<float> t_min) noexcept {
    // lateral surface: both roots, restricted to the segment span, so that a ray
    // starting inside the capsule still reports the exit
    auto ba = b - a;
    auto oa = o - a;
    auto baba = dot(ba, ba);
    auto bard = dot(ba, d);
    auto baoa = dot(ba, oa);
    auto rdoa = dot(d, oa);
    auto oaoa = dot(oa, oa);
    auto qa = baba - bard * bard;
    auto qb = baba * rdoa - baoa * bard;
    auto qc = baba * oaoa - baoa * baoa - r * r * baba;
    auto disc = qb * qb - qa * qc;
    $if (qa > 1e-12f & disc >= 0.f) {
        auto sq = sqrt(disc);
        auto t0 = (-qb - sq) / qa;
        auto t1 = (-qb + sq) / qa;
        auto y0 = baoa + t0 * bard;
        auto y1 = baoa + t1 * bard;
        $if (y0 >= 0.f & y0 <= baba) { zoo_consider(best, t0, t_min); };
        $if (y1 >= 0.f & y1 <= baba) { zoo_consider(best, t1, t_min); };
    };
    // the two spherical caps (they also cover the ray-parallel-to-the-axis case)
    zoo_sphere(o, d, a, r, best, t_min);
    zoo_sphere(o, d, b, r, best, t_min);
}

inline void zoo_plane(Var<float> &tn, Var<float> &tf, Var<bool> &ok,
                      Expr<float3> o, Expr<float3> d,
                      Expr<float3> n, Expr<float> w) noexcept {
    auto denom = dot(n, d);
    auto dist = dot(n, o) - w;
    $if (abs(denom) < 1e-8f) {
        $if (dist > 0.f) { ok = false; };
    } $else {
        auto t = -dist / denom;
        $if (denom < 0.f) { tn = max(tn, t); } $else { tf = min(tf, t); };
    };
}

inline void zoo_clip_result(Expr<float> tn, Expr<float> tf, Expr<bool> ok,
                            Var<float> &best, Expr<float> t_min) noexcept {
    $if (ok & tn <= tf & tf >= t_min) {
        zoo_consider(best, max(tn, t_min), t_min);
    };
}

// local z-aligned cone (radius r1 at z = -h, r2 at z = +h) with caps
inline void zoo_cone_z(Expr<float3> o, Expr<float3> d, Expr<float> r1, Expr<float> r2, Expr<float> h,
                       Var<float> &best, Expr<float> t_min) noexcept {
    auto a0 = (r1 + r2) * 0.5f;
    auto b0 = (r2 - r1) / (2.f * h);
    auto r_at_o = a0 + b0 * o.z;
    auto qa = d.x * d.x + d.y * d.y - b0 * b0 * d.z * d.z;
    auto qb = 2.f * (o.x * d.x + o.y * d.y) - 2.f * b0 * d.z * r_at_o;
    auto qc = o.x * o.x + o.y * o.y - r_at_o * r_at_o;
    auto disc = qb * qb - 4.f * qa * qc;
    $if (abs(qa) < 1e-9f) {
        // ray parallel to a generator: linear equation
        $if (abs(qb) > 1e-9f) {
            auto t = -qc / qb;
            auto z = o.z + t * d.z;
            $if (abs(z) <= h) { zoo_consider(best, t, t_min); };
        };
    } $else {
        $if (disc >= 0.f) {
            auto sq = sqrt(disc);
            auto inv = 0.5f / qa;
            auto t0 = (-qb - sq) * inv;
            auto t1 = (-qb + sq) * inv;
            auto z0 = o.z + t0 * d.z;
            auto z1 = o.z + t1 * d.z;
            $if (abs(z0) <= h) { zoo_consider(best, t0, t_min); };
            $if (abs(z1) <= h) { zoo_consider(best, t1, t_min); };
        };
    };
    // the two caps
    $if (abs(d.z) > 1e-9f) {
        auto t0 = (-h - o.z) / d.z;
        auto p0 = o + d * t0;
        $if (p0.x * p0.x + p0.y * p0.y <= r1 * r1) { zoo_consider(best, t0, t_min); };
        auto t1 = (h - o.z) / d.z;
        auto p1 = o + d * t1;
        $if (p1.x * p1.x + p1.y * p1.y <= r2 * r2) { zoo_consider(best, t1, t_min); };
    };
}

// Analytic first hit of `p` along the world-space ray (o, d), restricted to
// [t_min, t_max).  On return `result` holds the closest hit distance, or
// t_max when the ray misses the primitive.
inline void zoo_hit(Expr<float3> o, Expr<float3> d, const Var<ProcPrim> &p,
                    Expr<float> t_min, Expr<float> t_max, Var<float> &result) noexcept {
    // the stored frame is orthonormal, so t is preserved by the change of basis
    auto rel = o - p.center;
    auto lo = make_float3(dot(rel, p.ex), dot(rel, p.ey), dot(rel, p.ez));
    auto ld = make_float3(dot(d, p.ex), dot(d, p.ey), dot(d, p.ez));
    auto best = def(t_max);
    auto kind = p.kind;
    $if (kind == static_cast<uint>(Sphere)) {
        zoo_sphere(lo, ld, make_float3(0.f), p.dims.x, best, t_min);
    } $elif (kind == static_cast<uint>(Ellipsoid)) {
        auto inv = make_float3(1.f / p.dims.x, 1.f / p.dims.y, 1.f / p.dims.z);
        zoo_sphere(lo * inv, ld * inv, make_float3(0.f), 1.f, best, t_min);
    } $elif (kind == static_cast<uint>(Box)) {
        // slab clip
        auto tn = def(-1e30f);
        auto tf = def(1e30f);
        auto ok = def(true);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(1.f, 0.f, 0.f), p.dims.x);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-1.f, 0.f, 0.f), p.dims.x);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, 1.f, 0.f), p.dims.y);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, -1.f, 0.f), p.dims.y);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, 0.f, 1.f), p.dims.z);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, 0.f, -1.f), p.dims.z);
        zoo_clip_result(tn, tf, ok, best, t_min);
    } $elif (kind == static_cast<uint>(Capsule)) {
        zoo_capsule_segment(lo, ld, make_float3(0.f, 0.f, -p.dims.y), make_float3(0.f, 0.f, p.dims.y), p.dims.x, best, t_min);
    } $elif (kind == static_cast<uint>(Cylinder)) {
        // lateral quadratic
        auto qa = ld.x * ld.x + ld.y * ld.y;
        auto qb = 2.f * (lo.x * ld.x + lo.y * ld.y);
        auto qc = lo.x * lo.x + lo.y * lo.y - p.dims.x * p.dims.x;
        $if (qa > 1e-12f) {
            auto disc = qb * qb - 4.f * qa * qc;
            $if (disc >= 0.f) {
                auto sq = sqrt(disc);
                auto inv = 0.5f / qa;
                auto t0 = (-qb - sq) * inv;
                auto t1 = (-qb + sq) * inv;
                $if (abs(lo.z + t0 * ld.z) <= p.dims.y) { zoo_consider(best, t0, t_min); };
                $if (abs(lo.z + t1 * ld.z) <= p.dims.y) { zoo_consider(best, t1, t_min); };
            };
        };
        $if (abs(ld.z) > 1e-9f) {
            auto r = p.dims.x;
            auto t0 = (-p.dims.y - lo.z) / ld.z;
            auto p0 = lo + ld * t0;
            $if (p0.x * p0.x + p0.y * p0.y <= r * r) { zoo_consider(best, t0, t_min); };
            auto t1 = (p.dims.y - lo.z) / ld.z;
            auto p1 = lo + ld * t1;
            $if (p1.x * p1.x + p1.y * p1.y <= r * r) { zoo_consider(best, t1, t_min); };
        };
    } $elif (kind == static_cast<uint>(CappedCone)) {
        zoo_cone_z(lo, ld, p.dims.x, p.dims.y, p.dims.z, best, t_min);
    } $elif (kind == static_cast<uint>(Bicone)) {
        auto h = p.dims.y;
        auto half = 0.5f * h;
        auto o0 = lo - make_float3(0.f, 0.f, half);
        zoo_cone_z(o0, ld, p.dims.x, 0.f, half, best, t_min);
        auto o1 = make_float3(lo.x, lo.y, -lo.z) - make_float3(0.f, 0.f, half);
        auto d1 = make_float3(ld.x, ld.y, -ld.z);
        zoo_cone_z(o1, d1, p.dims.x, 0.f, half, best, t_min);
    } $elif (kind == static_cast<uint>(Tetrahedron)) {
        auto tn = def(-1e30f);
        auto tf = def(1e30f);
        auto ok = def(true);
        auto s = p.dims.x;
        auto inv_sqrt3 = 0.57735026919f;
        // vertices (+-1,+-1,+-1) with an even number of minus signs
        zoo_plane(tn, tf, ok, lo, ld, make_float3(inv_sqrt3, inv_sqrt3, inv_sqrt3), s * inv_sqrt3);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(inv_sqrt3, -inv_sqrt3, -inv_sqrt3), s * inv_sqrt3);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-inv_sqrt3, inv_sqrt3, -inv_sqrt3), s * inv_sqrt3);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-inv_sqrt3, -inv_sqrt3, inv_sqrt3), s * inv_sqrt3);
        zoo_clip_result(tn, tf, ok, best, t_min);
    } $elif (kind == static_cast<uint>(HexPrism)) {
        auto tn = def(-1e30f);
        auto tf = def(1e30f);
        auto ok = def(true);
        auto inr = p.dims.x * 0.86602540378f;
        auto c30 = 0.5f;
        auto s60 = 0.86602540378f;
        zoo_plane(tn, tf, ok, lo, ld, make_float3(1.f, 0.f, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-1.f, 0.f, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(c30, s60, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-c30, -s60, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(c30, -s60, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(-c30, s60, 0.f), inr);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, 0.f, 1.f), p.dims.y);
        zoo_plane(tn, tf, ok, lo, ld, make_float3(0.f, 0.f, -1.f), p.dims.y);
        zoo_clip_result(tn, tf, ok, best, t_min);
    } $elif (kind == static_cast<uint>(Octahedron)) {
        auto tn = def(-1e30f);
        auto tf = def(1e30f);
        auto ok = def(true);
        auto s = p.dims.x * 0.57735026919f;
        auto inv_sqrt3 = 0.57735026919f;
        for (int i = 0; i < 8; i++) {
            auto nx = (i & 1) ? -inv_sqrt3 : inv_sqrt3;
            auto ny = (i & 2) ? -inv_sqrt3 : inv_sqrt3;
            auto nz = (i & 4) ? -inv_sqrt3 : inv_sqrt3;
            zoo_plane(tn, tf, ok, lo, ld, make_float3(nx, ny, nz), s);
        }
        zoo_clip_result(tn, tf, ok, best, t_min);
    } $elif (kind == static_cast<uint>(SphereChain)) {
        auto major = p.dims.x;
        auto radius = p.dims.y;
        $for (i, 0u, sphere_chain_count) {
            auto a = i.cast<float>() * (2.f * pi / static_cast<float>(sphere_chain_count));
            auto c = make_float3(cos(a) * major, sin(a) * major, 0.f);
            zoo_sphere(lo, ld, c, radius, best, t_min);
        };
    } $elif (kind == static_cast<uint>(HelixTube)) {
        auto major = p.dims.x;
        auto radius = p.dims.y;
        auto height = p.dims.z;
        $for (i, 0u, helix_segment_count - 1u) {
            auto t0 = i.cast<float>() / static_cast<float>(helix_segment_count);
            auto t1 = (i.cast<float>() + 1.f) / static_cast<float>(helix_segment_count);
            auto a0 = t0 * (3.f * pi);
            auto a1 = t1 * (3.f * pi);
            auto p0 = make_float3(cos(a0) * major, sin(a0) * major, height * (2.f * t0 - 1.f));
            auto p1 = make_float3(cos(a1) * major, sin(a1) * major, height * (2.f * t1 - 1.f));
            zoo_capsule_segment(lo, ld, p0, p1, radius, best, t_min);
        };
    };
    result = best;
}

// ---------------------------------------------------------------------------
// Device side: sampling algorithms
// ---------------------------------------------------------------------------
namespace sampling {

// Stateless PCG-style hash: the base of every random sequence below.
inline Var<uint> hash(Expr<uint> v) noexcept {
    auto state = v * 747796405u + 2891336453u;
    auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline Var<float> to_float(Expr<uint> h) noexcept {
    return cast<float>(h >> 8u) * (1.f / 16777216.f);
}

// Two uniform values in [0,1) from one (seed, index) pair.
inline Var<float2> pcg2(Expr<uint> seed, Expr<uint> index) noexcept {
    auto h0 = hash(seed ^ (index * 2654435761u));
    auto h1 = hash(h0 ^ 0x9e3779b9u);
    return make_float2(to_float(h0), to_float(h1));
}

// Van der Corput radical inverse in the given base.
inline Var<float> radical_inverse(Expr<uint> index, Expr<uint> base) noexcept {
    auto x = def(index);
    auto f = def(1.f);
    auto r = def(0.f);
    auto inv_base = 1.f / cast<float>(base);
    $while (x > 0u) {
        f = f * inv_base;
        r = r + f * cast<float>(x % base);
        x = x / base;
    };
    return r;
}

// algorithm 0: PCG hash stream
// algorithm 1: R2 low-discrepancy sequence (Cranley-Patterson rotated)
// algorithm 2: Halton (2, 3, 5) with a per-seed rotation
// algorithm 3: 8x8 stratified, one jittered sample per stratum
inline Var<float3> sample3(Expr<uint> algorithm, Expr<uint> seed, Expr<uint> index) noexcept {
    auto result = def(make_float3(0.5f));
    $if (algorithm == 0u) {
        auto a = pcg2(seed, index);
        auto b = pcg2(seed ^ 0x85ebca6bu, index);
        result = make_float3(a, b.x);
    } $elif (algorithm == 1u) {
        // R3/Weyl sequence in exact 32-bit fixed point (a float32 `alpha * i`
        // loses the low bits of the fraction for large i and stops being
        // low-discrepancy after a few thousand samples)
        auto weyl = make_uint3(index * 3242174889u, index * 2447445415u, index * 3518437209u);
        auto shift = make_float3(pcg2(seed, 0u), to_float(hash(seed ^ 0x51ed270bu)));
        result = fract(make_float3(to_float(weyl.x), to_float(weyl.y), to_float(weyl.z)) + shift);
    } $elif (algorithm == 2u) {
        auto shift = make_float3(pcg2(seed, 1u), to_float(hash(seed ^ 0x27d4eb2du)));
        auto i = index + 1u;
        result = fract(make_float3(radical_inverse(i, 2u),
                                   radical_inverse(i, 3u),
                                   radical_inverse(i, 5u)) +
                       shift);
    } $else {
        auto stratum = index % 64u;
        auto jitter = pcg2(seed, index);
        result = make_float3((cast<float>(stratum % 8u) + jitter.x) * 0.125f,
                             (cast<float>(stratum / 8u) + jitter.y) * 0.125f,
                             to_float(hash(index ^ seed)));
    };
    return result;
}

inline Var<float2> sample2(Expr<uint> algorithm, Expr<uint> seed, Expr<uint> index) noexcept {
    return sample3(algorithm, seed, index).xy();
}

// Branchless orthonormal basis around a unit vector (Duff et al. 2017).
inline void orthonormal_basis(Expr<float3> n, Var<float3> &t, Var<float3> &b) noexcept {
    auto sign_ = select(1.f, -1.f, n.z < 0.f);
    auto a = -1.f / (sign_ + n.z);
    auto bb = n.x * n.y * a;
    t = make_float3(1.f + sign_ * n.x * n.x * a, sign_ * bb, -sign_ * n.x);
    b = make_float3(bb, sign_ + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sample around `n`: pdf = cos(theta)/pi.
inline Var<float3> cosine_hemisphere(Expr<float2> u, Expr<float3> n) noexcept {
    auto r = sqrt(u.x);
    auto phi = 2.f * pi * u.y;
    auto local = make_float3(r * cos(phi), r * sin(phi), sqrt(max(0.f, 1.f - u.x)));
    auto t = def(make_float3(1.f, 0.f, 0.f));
    auto b = def(make_float3(0.f, 1.f, 0.f));
    orthonormal_basis(n, t, b);
    return normalize(t * local.x + b * local.y + n * local.z);
}

// GGX visible-normal distribution sampling (Heitz 2018).  `v_local` is the view
// direction in the local shading frame (z = normal); the result is a half vector
// in the same frame.
inline Var<float3> ggx_vndf(Expr<float2> u, Expr<float3> v_local, Expr<float> alpha) noexcept {
    auto vh = normalize(make_float3(alpha * v_local.x, alpha * v_local.y, v_local.z));
    auto lensq = vh.x * vh.x + vh.y * vh.y;
    auto t1 = select(make_float3(1.f, 0.f, 0.f),
                     make_float3(-vh.y, vh.x, 0.f) * rsqrt(max(lensq, 1e-12f)),
                     lensq > 0.f);
    auto t2 = cross(vh, t1);
    auto r = sqrt(u.x);
    auto phi = 2.f * pi * u.y;
    auto t1s = r * cos(phi);
    auto t2s = r * sin(phi);
    auto s = 0.5f * (1.f + vh.z);
    auto t2s_ = (1.f - s) * sqrt(max(0.f, 1.f - t1s * t1s)) + s * t2s;
    auto nh = t1 * t1s + t2 * t2s_ + vh * sqrt(max(0.f, 1.f - t1s * t1s - t2s_ * t2s_));
    return normalize(make_float3(alpha * nh.x, alpha * nh.y, max(1e-6f, nh.z)));
}

}// namespace sampling

}// namespace luisa::test::zoo