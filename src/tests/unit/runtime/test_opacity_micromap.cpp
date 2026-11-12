// Device test for hardware opacity micromaps ("cut-out masks"), i.e.
// VK_KHR_opacity_micromap / VK_EXT_opacity_micromap exposed through
// `OpacityMicromap`, `OpacityMicromap::bake()` and `Device::create_mesh(...,
// MeshCutoutMask)`.
//
// The geometry is a de-indexed quad covering [-1, 1]^2 in the z = 0 plane with a
// *global* UV map `uv = (x + 1) / 2, (y + 1) / 2`, so every point of the mesh has a
// known texel. The mask is opaque everywhere except a disc of radius 0.25 in UV
// space around the centre.
//
// The assertions are structural rather than pixel exact, because an opacity micromap
// quantises the cut-out to micro triangles:
//
// 1. a ray through the middle of the hole must miss *only* for the micromapped mesh
//    - the same mesh without a cut-out mask hits, so the micromap is what changed the
//    result;
// 2. a ray outside the hole must still hit;
// 3. the host I/O path (`upload` / `readback`) round-trips the packed bits, which is
//    the D10 contract and the only way to inspect a bake without a debug kernel.
//
// The traversal has to use `Accel::traverse()` (`RAY_TRACING_QUERY_ALL`, no ray
// flags). `intersect()` / `intersect_any()` emit `OpaqueKHR`, which by spec overrides
// every geometry flag *and* the micromap, so they deliberately bypass this feature.
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_opacity_micromap_detail {

constexpr auto mask_size = 64u;
// The hole sits strictly inside the *first* triangle, far from both the quad
// diagonal and the UV box border, so no boundary micro triangle can decide the
// result.
constexpr auto hole_center_u = 0.25f;
constexpr auto hole_center_v = 0.25f;
constexpr auto hole_radius = 0.15f;

// `alpha = 0` inside the disc, `1` outside, replicated over RGBA so the bake can pick
// any channel.
[[nodiscard]] std::vector<float> make_mask() {
    std::vector<float> pixels(mask_size * mask_size * 4u, 1.0f);
    for (auto y = 0u; y < mask_size; y++) {
        for (auto x = 0u; x < mask_size; x++) {
            auto u = (static_cast<float>(x) + 0.5f) / static_cast<float>(mask_size);
            auto v = (static_cast<float>(y) + 0.5f) / static_cast<float>(mask_size);
            auto du = u - hole_center_u;
            auto dv = v - hole_center_v;
            auto cut_out = std::sqrt(du * du + dv * dv) < hole_radius;
            auto index = (y * mask_size + x) * 4u;
            for (auto c = 0u; c < 4u; c++) {
                pixels[index + c] = cut_out ? 0.0f : 1.0f;
            }
        }
    }
    return pixels;
}

// `fract(4 u) < 0.5`: four vertical stripes over the UV square, replicated over RGBA.
// One stripe is 1/4 of `u`, i.e. four subdivision columns at level 4, which is coarse
// enough for the "all three corners" rule to keep a clear interior, yet fine enough
// that it cuts much less than the per-hit-point (centroid) rule - that difference is
// the regression test for the rule's polarity.
[[nodiscard]] std::vector<float> make_stripe_mask() {
    std::vector<float> pixels(mask_size * mask_size * 4u, 1.0f);
    for (auto y = 0u; y < mask_size; y++) {
        for (auto x = 0u; x < mask_size; x++) {
            auto u = (static_cast<float>(x) + 0.5f) / static_cast<float>(mask_size);
            auto f = std::fmod(u * 4.0f, 1.0f);
            auto value = f < 0.5f ? 0.0f : 1.0f;
            auto index = (y * mask_size + x) * 4u;
            for (auto c = 0u; c < 4u; c++) { pixels[index + c] = value; }
        }
    }
    return pixels;
}

// Two triangles of the quad, de-indexed: a per-triangle parameterisation is the only
// one a mesh can express (Mesh has no UV input).
[[nodiscard]] std::array<float3, 6> make_vertices() {    return {make_float3(-1.0f, -1.0f, 0.0f),
            make_float3(1.0f, -1.0f, 0.0f),
            make_float3(-1.0f, 1.0f, 0.0f),
            make_float3(1.0f, -1.0f, 0.0f),
            make_float3(1.0f, 1.0f, 0.0f),
            make_float3(-1.0f, 1.0f, 0.0f)};
}

[[nodiscard]] std::array<float2, 6> make_uvs() {
    // uv = ((x + 1) / 2, (y + 1) / 2) for every vertex of both triangles.
    return {make_float2(0.0f, 0.0f),
            make_float2(1.0f, 0.0f),
            make_float2(0.0f, 1.0f),
            make_float2(1.0f, 0.0f),
            make_float2(1.0f, 1.0f),
            make_float2(0.0f, 1.0f)};
}

[[nodiscard]] std::array<Triangle, 2> make_triangles() {
    return {Triangle{0u, 1u, 2u}, Triangle{3u, 4u, 5u}};
}

}// namespace test_opacity_micromap_detail

void test_opacity_micromap(Device &device) {

    using namespace test_opacity_micromap_detail;

    if (!OpacityMicromap::is_supported(device)) {
        LUISA_WARNING("This device has no opacity micromap support; skipping.");
        return;
    }

    auto stream = device.create_stream();
    auto vertices = make_vertices();
    auto uvs = make_uvs();
    auto triangles = make_triangles();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto uv_buffer = device.create_buffer<float2>(uvs.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mask_pixels = make_mask();
    auto mask = device.create_image<float>(PixelStorage::FLOAT4, mask_size, mask_size);
    // `luisa::span` keeps this working in a safe-mode build, where the raw-pointer
    // copy overloads do not exist.
    stream << vertex_buffer.copy_from(luisa::span<const float3>{vertices})
           << uv_buffer.copy_from(luisa::span<const float2>{uvs})
           << triangle_buffer.copy_from(luisa::span<const Triangle>{triangles})
           << mask.copy_from(luisa::span<const float>{mask_pixels})
           << synchronize();

    // Two rays: one through the centre of the hole, one outside it.
    // uv (0.28, 0.28) -> xy (-0.44, -0.44): inside the hole and inside the interior of
    // one micro triangle of triangle 0. The exact hole centre would land on a
    // micro-vertex of the level-4 grid, where the hardware's tie-break rule decides
    // which micro triangle is sampled.
    // uv (0.8, 0.8) -> xy (0.6, 0.6): outside the hole, inside triangle 1.
    std::array origins{make_float3(-0.44f, -0.44f, 1.0f), make_float3(0.6f, 0.6f, 1.0f)};
    std::array directions{make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 0.0f, -1.0f)};
    auto origin_buffer = device.create_buffer<float3>(origins.size());
    auto direction_buffer = device.create_buffer<float3>(directions.size());
    auto result_buffer = device.create_buffer<uint>(origins.size());
    stream << origin_buffer.copy_from(luisa::span<const float3>{origins})
           << direction_buffer.copy_from(luisa::span<const float3>{directions})
           << synchronize();

    // `traverse()` is the micromap-aware loop form: `intersect()` would emit
    // OpaqueKHR and ignore the cut-out entirely.
    auto trace_shader = device.compile<1>(
        [](AccelVar accel, BufferFloat3 origins, BufferFloat3 directions,
           BufferUInt results) noexcept {
            auto i = dispatch_id().x;
            auto ray = make_ray(origins.read(i), directions.read(i));
            auto hit = accel.traverse(ray, {})
                           .on_surface_candidate([](SurfaceCandidate &candidate) noexcept {
                               candidate.commit();
                           })
                           .trace();
            results.write(i, ite(hit->miss(), 0u, 1u));
        });
    auto read_results = [&] {
        std::array<uint, 2> values{};
        stream << result_buffer.copy_to(luisa::span<uint>{values}) << synchronize();
        return values;
    };
    auto trace = [&](const Accel &accel) {
        stream << trace_shader(accel, origin_buffer, direction_buffer, result_buffer)
                      .dispatch(static_cast<uint>(origins.size()))
               << synchronize();
        return read_results();
    };

    // 1. Baseline: without a cut-out mask the centre ray hits the quad.
    auto plain_mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto plain_accel = device.create_accel();
    plain_accel.emplace_back(plain_mesh, make_float4x4(1.0f));
    stream << plain_mesh.build(Mesh::BuildRequest::FORCE_BUILD)
           << plain_accel.build() << synchronize();
    auto baseline = trace(plain_accel);
    expect(baseline[0] == 1u) << "the un-masked quad must be hit at its centre";
    expect(baseline[1] == 1u) << "the un-masked quad must be hit off its centre";

    // 2. Bake the cut-out mask into a hardware micromap and attach it.
    auto micromap = device.create_opacity_micromap(
        static_cast<uint32_t>(triangles.size()),
        {.format = OpacityMicromap::Format::TwoState,
         .subdivision_level = 4u,
         .resolve_boundary_in_any_hit = false,
         .alpha_cutoff = 0.5f,
         .conservative = true,
         .index_format = OpacityMicromap::IndexFormat::None});
    expect(micromap.ready() == false)
        << "a freshly created micromap has no bits yet";
    stream << micromap.bake(mask.view(0u), uv_buffer.view(), triangle_buffer.view()).commit()
           << synchronize();
    expect(micromap.ready())
        << "recording a bake makes the micromap ready";

    // Sanity check of the bake itself: the cut-out disc covers
    // pi * 0.15^2 ~= 7% of the UV square, so roughly that share of the
    // 2 * 4^4 micro triangles must have been written as transparent (0).
    {
        std::vector<uint> baked(micromap.data().size(), 0u);
        stream << micromap.readback(luisa::span<uint>{baked}) << synchronize();
        auto total = 2ull * OpacityMicromap::micro_triangle_count(micromap.subdivision_level());
        auto cut = 0ull;
        for (auto i = 0ull; i < total; i++) {
            cut += (baked[i / 32u] >> (i % 32u)) & 1u ? 0ull : 1ull;
        }
        auto fraction = static_cast<double>(cut) / static_cast<double>(total);
        LUISA_INFO("baked micromap: {}/{} micro triangles cut out ({:.3f}).",
                   cut, total, fraction);
        expect(fraction > 0.01 && fraction < 0.20)
            << "the bake must cut out roughly the area of the mask's hole";
    }
    auto masked_mesh = device.create_mesh(vertex_buffer, triangle_buffer,
                                          MeshCutoutMask{micromap});
    auto masked_accel = device.create_accel();
    masked_accel.emplace_back(masked_mesh, make_float4x4(1.0f));
    stream << masked_mesh.build(Mesh::BuildRequest::FORCE_BUILD)
           << masked_accel.build() << synchronize();
    auto masked = trace(masked_accel);
    expect(masked[0] == 0u)
        << "a ray through the centre of the hole must miss the micromapped mesh";
    expect(masked[1] == 1u)
        << "a ray outside the hole must still hit the micromapped mesh";

    // 3. The host I/O path: the packed bits must round-trip unchanged.
    std::vector<uint> bits(micromap.data().size(), 0u);
    std::vector<uint> constant(bits.size(), 0x5au);
    stream << micromap.upload(luisa::span<const uint>{constant})
           << micromap.readback(luisa::span<uint>{bits})
           << synchronize();
    expect(bits == constant)
        << "OpacityMicromap::upload() must store exactly the bytes readback() returns";

    // 4. The sampling rule of the baker. `conservative` has to be the *safe* rule: a
    // micro triangle is cut out only when all three of its corners are transparent,
    // so it can never cut more than the centroid rule. On a stripe mask the two
    // rules differ by a wide margin, which is what turns this into a polarity
    // regression test (the inverted rule cuts ~4x too much).
    {
        auto stripe_pixels = make_stripe_mask();
        auto stripe_image = device.create_image<float>(PixelStorage::FLOAT4, mask_size, mask_size);
        stream << stripe_image.copy_from(luisa::span<const float>{stripe_pixels})
               << synchronize();
        auto bake_and_measure = [&](bool conservative) {
            auto map = device.create_opacity_micromap(
                static_cast<uint32_t>(triangles.size()),
                {.format = OpacityMicromap::Format::TwoState,
                 .subdivision_level = 4u,
                 .resolve_boundary_in_any_hit = false,
                 .alpha_cutoff = 0.5f,
                 .conservative = conservative,
                 .index_format = OpacityMicromap::IndexFormat::None});
            stream << map.bake(stripe_image.view(0u), uv_buffer.view(), triangle_buffer.view()).commit()
                   << synchronize();
            std::vector<uint> data(map.data().size(), 0u);
            stream << map.readback(luisa::span<uint>{data}) << synchronize();
            auto total = 2ull * OpacityMicromap::micro_triangle_count(map.subdivision_level());
            auto cut = 0ull;
            for (auto i = 0ull; i < total; i++) {
                cut += (data[i / 32u] >> (i % 32u)) & 1u ? 0ull : 1ull;
            }
            return static_cast<double>(cut) / static_cast<double>(total);
        };
        auto conservative_cut = bake_and_measure(true);
        auto centroid_cut = bake_and_measure(false);
        LUISA_INFO("stripe mask: the conservative rule cuts {:.3f}, the centroid rule "
                   "cuts {:.3f} of the micro triangles.",
                   conservative_cut, centroid_cut);
        expect(centroid_cut > 0.40 && centroid_cut < 0.60)
            << "the centroid rule must cut out roughly the mask's transparent half";
        expect(conservative_cut > 0.10 && conservative_cut < 0.35)
            << "the conservative rule must cut out the clear interior of each stripe "
               "and nothing more";
        expect(conservative_cut < centroid_cut)
            << "conservative sampling must be the safe rule (a subset of the centroid "
               "rule), not the aggressive one";
    }

    // 5. Four-state boundary resolution. With `resolve_boundary_in_any_hit` the
    // boundary micro triangles are reported as "unknown" (2) so an any-hit shader can
    // resolve them; without it the same bake has to stay two-valued.
    {
        auto four_state_level = std::min(
            2u, OpacityMicromap::max_subdivision_level(
                    device, OpacityMicromap::Format::FourState));
        auto count_unknown = [&](bool resolve_boundary) {
            auto map = device.create_opacity_micromap(
                static_cast<uint32_t>(triangles.size()),
                {.format = OpacityMicromap::Format::FourState,
                 .subdivision_level = four_state_level,
                 .resolve_boundary_in_any_hit = resolve_boundary,
                 .alpha_cutoff = 0.5f,
                 .conservative = true,
                 .index_format = OpacityMicromap::IndexFormat::None});
            stream << map.bake(mask.view(0u), uv_buffer.view(), triangle_buffer.view()).commit()
                   << synchronize();
            std::vector<uint> data(map.data().size(), 0u);
            stream << map.readback(luisa::span<uint>{data}) << synchronize();
            auto total = 2ull * OpacityMicromap::micro_triangle_count(map.subdivision_level());
            auto unknown = 0ull;
            for (auto i = 0ull; i < total; i++) {
                auto unit = i * 2ull;
                auto value = (data[unit / 32u] >> (unit % 32u)) & 3u;
                unknown += value == 2u ? 1ull : 0ull;
            }
            LUISA_INFO("four-state micromap at level {} (boundary resolution {}): "
                       "{}/{} micro triangles unknown.",
                       four_state_level, resolve_boundary, unknown, total);
            return unknown;
        };
        expect(count_unknown(true) > 0ull)
            << "boundary resolution must report the mask's edge micro triangles as "
               "unknown";
        expect(count_unknown(false) == 0ull)
            << "without boundary resolution a four-state micromap must stay "
               "two-valued";
    }
}


int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_opacity_micromap(dc->device);
}
