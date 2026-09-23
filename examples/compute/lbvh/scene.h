// Host-side geometry of the demo scene: a small set of triangle meshes that the
// example intersects (a UV sphere, a box and a torus).  This is pure host code;
// the meshes are uploaded once and used by both the software LBVH and the Luisa
// RTX reference.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/runtime/rtx/triangle.h>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// A triangle soup in object space, with the bounding box of its vertices.
struct HostMesh {
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    float3 lo{1.0e30f};
    float3 hi{-1.0e30f};

    void update_bounds() noexcept {
        for (auto v : vertices) {
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    void add_triangle(uint a, uint b, uint c) noexcept {
        if (a == b || b == c || a == c) { return; }// degenerate (sphere poles)
        auto pa = vertices[a];
        auto pb = vertices[b];
        auto pc = vertices[c];
        if (length(cross(pb - pa, pc - pa)) < 1.0e-9f) { return; }
        triangles.emplace_back(Triangle{a, b, c});
    }
};

[[nodiscard]] inline HostMesh make_uv_sphere(float radius, uint n_u, uint n_v) noexcept {
    HostMesh mesh;
    for (auto v = 0u; v <= n_v; v++) {
        auto theta = pi * static_cast<float>(v) / static_cast<float>(n_v);
        for (auto u = 0u; u <= n_u; u++) {
            auto phi = 2.0f * pi * static_cast<float>(u) / static_cast<float>(n_u);
            mesh.vertices.emplace_back(radius * make_float3(sin(theta) * cos(phi),
                                                            cos(theta),
                                                            sin(theta) * sin(phi)));
        }
    }
    auto stride = n_u + 1u;
    for (auto v = 0u; v < n_v; v++) {
        for (auto u = 0u; u < n_u; u++) {
            auto i0 = v * stride + u;
            auto i1 = i0 + 1u;
            auto i2 = i0 + stride;
            auto i3 = i2 + 1u;
            mesh.add_triangle(i0, i2, i1);
            mesh.add_triangle(i1, i2, i3);
        }
    }
    mesh.update_bounds();
    return mesh;
}

[[nodiscard]] inline HostMesh make_box(float3 half_extent) noexcept {
    HostMesh mesh;
    for (auto f = 0u; f < 6u; f++) {
        auto axis = f / 2u;
        auto sign = (f % 2u == 0u) ? -1.0f : 1.0f;
        auto u = (axis + 1u) % 3u;
        auto v = (axis + 2u) % 3u;
        for (auto c = 0u; c < 4u; c++) {
            auto su = (c == 0u || c == 3u) ? -1.0f : 1.0f;
            auto sv = (c < 2u) ? -1.0f : 1.0f;
            auto p = make_float3(0.0f);
            p[axis] = sign * half_extent[axis];
            p[u] = su * half_extent[u];
            p[v] = sv * half_extent[v];
            mesh.vertices.emplace_back(p);
        }
        auto base = f * 4u;
        mesh.add_triangle(base + 0u, base + 1u, base + 2u);
        mesh.add_triangle(base + 0u, base + 2u, base + 3u);
    }
    mesh.update_bounds();
    return mesh;
}

[[nodiscard]] inline HostMesh make_torus(float major_radius, float minor_radius,
                                         uint n_u, uint n_v) noexcept {
    HostMesh mesh;
    for (auto u = 0u; u < n_u; u++) {
        auto phi = 2.0f * pi * static_cast<float>(u) / static_cast<float>(n_u);
        for (auto v = 0u; v < n_v; v++) {
            auto theta = 2.0f * pi * static_cast<float>(v) / static_cast<float>(n_v);
            auto r = major_radius + minor_radius * cos(theta);
            mesh.vertices.emplace_back(make_float3(r * cos(phi),
                                                   minor_radius * sin(theta),
                                                   r * sin(phi)));
        }
    }
    for (auto u = 0u; u < n_u; u++) {
        for (auto v = 0u; v < n_v; v++) {
            auto u1 = (u + 1u) % n_u;
            auto v1 = (v + 1u) % n_v;
            auto i0 = u * n_v + v;
            auto i1 = u1 * n_v + v;
            auto i2 = u * n_v + v1;
            auto i3 = u1 * n_v + v1;
            mesh.add_triangle(i0, i1, i2);
            mesh.add_triangle(i1, i3, i2);
        }
    }
    mesh.update_bounds();
    return mesh;
}

}// namespace luisa::example::lbvh
