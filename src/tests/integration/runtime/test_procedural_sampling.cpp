// Test for procedural primitives and sampling algorithms through ray queries.
// This test covers:
//  - twelve implicit procedural primitive kinds, each intersected analytically
//    inside the ray-query procedural handler (sphere, ellipsoid, box, capsule,
//    cylinder, capped cone, bicone, tetrahedron, hex prism, octahedron, sphere
//    chain, helix tube),
//  - an independent host reference (dense march of the exact implicit function
//    followed by bisection) checked against every device hit,
//  - Monte-Carlo volume estimation of the ten convex kinds with three sampling
//    algorithms (PCG hash stream, R2 low-discrepancy sequence, Halton 2/3/5),
//  - the analytic form factor (r/d)^2 of a sphere estimated with cosine-weighted
//    hemisphere sampling,
//  - sampler statistics: uniformity, stratification and low discrepancy,
//  - a four-bounce path trace of the zoo (cosine-weighted diffuse), checked for
//    convergence between 16 and 128 samples per pixel and written to
//    test_procedural_sampling.png.

#include "ut/ut.hpp"
#include "test_device.h"
#include "procedural_zoo.h"
#include "reference_image.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

using luisa::test::zoo::host::Box3;
using luisa::test::zoo::host::DVec3;
using luisa::test::zoo::ProcPrim;

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

struct HostRng {
    uint64_t state;
    explicit HostRng(uint64_t seed) noexcept : state{seed * 6364136223846793005ull + 1442695040888963407ull} {}
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        auto x = static_cast<uint32_t>((state >> 33u) ^ (state >> 11u));
        x ^= x >> 16u;
        x *= 0x7feb352du;
        x ^= x >> 15u;
        x *= 0x846ca68bu;
        x ^= x >> 16u;
        return x;
    }
    [[nodiscard]] float uniform() noexcept { return static_cast<float>(next() >> 8u) * (1.f / 16777216.f); }
    [[nodiscard]] float uniform(float lo, float hi) noexcept { return lo + (hi - lo) * uniform(); }
};

[[nodiscard]] DVec3 to_dvec(float3 v) noexcept { return {v.x, v.y, v.z}; }
[[nodiscard]] float3 to_float3(DVec3 v) noexcept {
    return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}

// Reference first hit of the whole zoo along one ray, obtained by marching the
// exact implicit function and bisecting the first sign change.  This is fully
// independent of the analytic intersection code the device runs.
struct RefHit {
    bool hit{false};
    double t{0.0};
    uint prim{0u};
    bool ambiguous{false}; // grazing: no sign change but the surface is within 1 mm
};

[[nodiscard]] RefHit reference_hit(const luisa::vector<ProcPrim> &prims,
                                   const luisa::vector<Box3> &boxes,
                                   size_t prim_count,
                                   DVec3 o, DVec3 d, double t_min, double t_max) noexcept {
    RefHit best;
    best.t = t_max;
    for (size_t i = 0; i < prim_count; i++) {
        double t0{}, t1{};
        if (!luisa::test::zoo::host::ray_box(boxes[i], o, d, t_min, t_max, t0, t1)) { continue; }
        auto span = t1 - t0;
        if (span <= 0.0) { continue; }
        auto steps = static_cast<int>(std::clamp(span / 0.002, 64.0, 4096.0));
        auto f = [&](double t) noexcept { return luisa::test::zoo::host::sdf(prims[i], o + d * t); };
        auto prev_t = t0;
        auto f0 = f(t0);
        auto min_abs = std::abs(f0);
        if (f0 <= 0.0) { // origin already inside this primitive
            if (t0 < best.t) { best = {true, t0, static_cast<uint>(i), false}; }
            continue;
        }
        auto found = false;
        auto root = 0.0;
        for (int k = 1; k <= steps; k++) {
            auto t = t0 + span * static_cast<double>(k) / static_cast<double>(steps);
            auto fv = f(t);
            min_abs = std::min(min_abs, std::abs(fv));
            if (fv <= 0.0) {
                auto lo = prev_t, hi = t;
                for (int it = 0; it < 60; it++) {
                    auto mid = 0.5 * (lo + hi);
                    if (f(mid) > 0.0) { lo = mid; } else { hi = mid; }
                }
                root = 0.5 * (lo + hi);
                found = true;
                break;
            }
            prev_t = t;
        }
        if (found) {
            if (root < best.t) { best = {true, root, static_cast<uint>(i), false}; }
        } else if (min_abs < 1e-3) {
            best.ambiguous = true;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Device structs
// ---------------------------------------------------------------------------

struct RayInput {
    float3 origin;
    float3 direction;
};

struct TraceResult {
    float t;
    uint prim;
    uint kind;
    uint hit;
};

}// namespace

LUISA_STRUCT(RayInput, origin, direction) {};
LUISA_STRUCT(TraceResult, t, prim, kind, hit) {};

namespace {

constexpr uint kTraceRaysPerKind = 128u;
constexpr uint kVolumeSamples = 65536u;
constexpr uint kFormFactorSamples = 65536u;
constexpr uint kSamplerSamples = 65536u;
constexpr uint kSampleAlgorithms = 4u; // pcg, R2, Halton, stratified

}// namespace

void test_procedural_sampling(Device &device) {

    // --- scene -----------------------------------------------------------------
    auto prims = luisa::test::zoo::host::make_zoo();
    // an extra sphere used by the form-factor test, far behind the zoo
    auto form_factor_prim = static_cast<uint>(prims.size());
    prims.emplace_back(luisa::test::zoo::host::make_sphere(make_float3(0.f, 0.f, -8.f), 0.5f));
    luisa::vector<AABB> aabbs;
    luisa::vector<Box3> boxes;
    aabbs.reserve(prims.size());
    boxes.reserve(prims.size());
    for (auto &p : prims) {
        auto b = luisa::test::zoo::host::bounds(p);
        boxes.emplace_back(b);
        aabbs.emplace_back(luisa::test::zoo::host::to_aabb(b));
    }

    auto prim_buffer = device.create_buffer<ProcPrim>(prims.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto zoo_accel = device.create_accel();
    auto zoo_primitive = device.create_procedural_primitive(
        aabb_buffer.view().subview(0u, luisa::test::zoo::kind_count));
    zoo_accel.emplace_back(zoo_primitive);
    auto form_factor_accel = device.create_accel();
    auto form_factor_primitive = device.create_procedural_primitive(
        aabb_buffer.view().subview(form_factor_prim, 1u));
    form_factor_accel.emplace_back(form_factor_primitive);

    Stream stream = device.create_stream();
    stream << prim_buffer.copy_from(luisa::span{prims})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << zoo_primitive.build()
           << form_factor_primitive.build()
           << zoo_accel.build()
           << form_factor_accel.build()
           << synchronize();

    // -----------------------------------------------------------------------
    // 1. analytic hits vs. the host reference
    // -----------------------------------------------------------------------
    {
        auto kernel = device.compile<1>(
            [&](BufferVar<RayInput> rays, BufferVar<TraceResult> results) noexcept {
                auto i = dispatch_id().x;
                auto r = rays.read(i);
                auto t_best = def(1e30f);
                auto prim_best = def(~0u);
                auto kind_best = def(~0u);
                auto ray = make_ray(r->origin, r->direction, 1e-4f, 1e30f);
                zoo_accel->traverse(ray, {})
                    .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                        auto h = candidate.hit();
                        auto prim = prim_buffer->read(h.prim);
                        auto cr = candidate.ray();
                        auto t = def(cr->t_max());
                        luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                                  cr->t_min(), t_best, t);
                        $if (t < t_best) {
                            t_best = t;
                            prim_best = h.prim;
                            kind_best = prim->kind;
                            candidate.commit(t);
                        };
                    })
                    .trace();
                results.write(i, def<TraceResult>(t_best, prim_best, kind_best,
                                                 select(0u, 1u, t_best < 1e29f)));
            });

        luisa::vector<RayInput> rays;
        rays.reserve(luisa::test::zoo::kind_count * kTraceRaysPerKind);
        HostRng rng{7u};
        for (uint k = 0u; k < luisa::test::zoo::kind_count; k++) {
            auto &prim = prims[k];
            auto &box = boxes[k];
            auto center = to_dvec(prim.center);
            for (uint j = 0u; j < kTraceRaysPerKind; j++) {
                // random direction on the unit sphere, camera at 2.4 units
                auto z = rng.uniform(-1.f, 1.f);
                auto phi = rng.uniform(0.f, 6.2831853f);
                auto rr = std::sqrt(std::max(0.f, 1.f - z * z));
                auto dir = DVec3{rr * std::cos(phi), rr * std::sin(phi), z};
                auto origin = center + dir * 2.4;
                // target: a random point of the AABB, biased towards its surface
                auto target = DVec3{
                    box.lo.x + (box.hi.x - box.lo.x) * rng.uniform(),
                    box.lo.y + (box.hi.y - box.lo.y) * rng.uniform(),
                    box.lo.z + (box.hi.z - box.lo.z) * rng.uniform()};
                if (j % 4u == 0u) { // grazing rays: aim at a random AABB corner
                    target = DVec3{
                        (j % 8u == 0u) ? box.lo.x : box.hi.x,
                        (j % 8u == 4u) ? box.lo.y : box.hi.y,
                        (j % 8u == 4u) ? box.lo.z : box.hi.z};
                }
                auto d = luisa::test::zoo::host::normalize(target - origin);
                // the reference and the device agree on "first hit" only when the
                // origin is strictly outside every primitive
                auto outside = true;
                for (auto i = 0u; i < luisa::test::zoo::kind_count; i++) {
                    auto &p = prims[i];
                    if (luisa::test::zoo::host::sdf(p, origin) < 0.02) {
                        outside = false;
                        break;
                    }
                }
                if (!outside) { continue; }
                rays.emplace_back(RayInput{to_float3(origin), to_float3(d)});
            }
        }

        auto ray_buffer = device.create_buffer<RayInput>(rays.size());
        auto result_buffer = device.create_buffer<TraceResult>(rays.size());
        luisa::vector<TraceResult> results(rays.size());
        stream << ray_buffer.copy_from(luisa::span{rays})
             << kernel(ray_buffer, result_buffer).dispatch(static_cast<uint>(rays.size()))
             << result_buffer.copy_to(luisa::span{results})
             << synchronize();

          uint mismatches = 0u;
        uint ambiguous = 0u;
        uint hits = 0u;
        uint logged = 0u;
        for (size_t i = 0; i < rays.size(); i++) {
            auto o = to_dvec(rays[i].origin);
            auto d = to_dvec(rays[i].direction);
            auto ref = reference_hit(prims, boxes, luisa::test::zoo::kind_count, o, d, 1e-4, 12.0);
            auto &got = results[i];
            auto got_hit = got.hit != 0u;
            if (ref.ambiguous) { ambiguous++; }
            if (ref.hit) { hits++; }
            auto ok = false;
            if (ref.hit && got_hit) {
                auto tol = 5e-3 * (1.0 + ref.t);
                ok = std::abs(static_cast<double>(got.t) - ref.t) <= tol && got.prim == ref.prim;
            } else if (!ref.hit && !got_hit) {
                ok = true;
            } else if (ref.ambiguous) {
                ok = true;
            }
            if (!ok) {
                mismatches++;
                if (logged++ < 8u) {
                    LUISA_WARNING("zoo mismatch [{}]: ref({{hit={} t={} prim={}}}) device({{hit={} t={} prim={} kind={}}}), o=({},{},{}), d=({},{},{})",
                                  i, ref.hit, ref.t, ref.prim, got_hit, got.t, got.prim, got.kind,
                                  o.x, o.y, o.z, d.x, d.y, d.z);
                }
            }
        }
        LUISA_INFO("Procedural reference check: {} rays, {} hits, {} ambiguous, {} mismatches.",
                   rays.size(), hits, ambiguous, mismatches);
        expect(mismatches == 0u) << "device analytic hits must match the host reference";
        expect(ambiguous * 50u < rays.size()) << "the grazing fraction must stay small";
        expect(hits * 3u > rays.size()) << "a third of the reference rays must hit";
    }

    // -----------------------------------------------------------------------
    // 2. Monte-Carlo volume of the convex kinds, three sampling algorithms
    // -----------------------------------------------------------------------
    {
        auto kernel = device.compile<1>(
            [&](BufferVar<uint> inside, UInt algorithm, Float3 box_lo, Float3 box_hi) noexcept {
                auto i = dispatch_id().x;
                auto u = luisa::test::zoo::sampling::sample3(algorithm, 0x51ed270bu + i, i);
                auto p = box_lo + (box_hi - box_lo) * u;
                auto hit_plus = def(false);
                auto hit_minus = def(false);
                {
                    // the inside test is valid for convex bodies: a point is inside
                    // iff the line through it meets the surface on both sides.
                    auto ray = make_ray(p, make_float3(1.f, 0.f, 0.f), 1e-5f, box_hi.x - p.x + 1e-3f);
                    hit_plus = !zoo_accel->traverse(ray, {})
                                    .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                        auto h = candidate.hit();
                                        auto prim = prim_buffer->read(h.prim);
                                        auto cr = candidate.ray();
                                        auto t = def(cr->t_max());
                                        luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                                                  cr->t_min(), cr->t_max(), t);
                                        $if (t < cr->t_max()) { candidate.commit(t); };
                                    })
                                    .trace()
                                    ->miss();
                }
                {
                    auto ray = make_ray(p, make_float3(-1.f, 0.f, 0.f), 1e-5f, p.x - box_lo.x + 1e-3f);
                    hit_minus = !zoo_accel->traverse(ray, {})
                                     .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                         auto h = candidate.hit();
                                         auto prim = prim_buffer->read(h.prim);
                                         auto cr = candidate.ray();
                                         auto t = def(cr->t_max());
                                         luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                                                   cr->t_min(), cr->t_max(), t);
                                         $if (t < cr->t_max()) { candidate.commit(t); };
                                     })
                                     .trace()
                                     ->miss();
                }
                inside.write(i, select(0u, 1u, hit_plus & hit_minus));
            });

        auto inside_buffer = device.create_buffer<uint>(kVolumeSamples);
        luisa::vector<uint> inside(kVolumeSamples);
        bool volume_ok = true;
        for (uint k = 0u; k < luisa::test::zoo::kind_count; k++) {
            if (!luisa::test::zoo::host::is_convex(k)) { continue; }
            auto &prim = prims[k];
            auto &box = boxes[k];
            auto exact = luisa::test::zoo::host::analytic_volume(prim);
            auto box_volume = (box.hi.x - box.lo.x) * (box.hi.y - box.lo.y) * (box.hi.z - box.lo.z);
            std::array<double, kSampleAlgorithms - 1u> estimates{};
            for (uint algorithm = 0u; algorithm < kSampleAlgorithms - 1u; algorithm++) {
                stream << kernel(inside_buffer, algorithm, to_float3(box.lo), to_float3(box.hi))
                              .dispatch(kVolumeSamples)
                       << inside_buffer.copy_to(luisa::span{inside})
                       << synchronize();
                auto count = std::count_if(inside.begin(), inside.end(), [](uint v) noexcept { return v != 0u; });
                auto estimate = box_volume * static_cast<double>(count) / static_cast<double>(kVolumeSamples);
                estimates[algorithm] = estimate;
                auto relative = std::abs(estimate - exact) / exact;
                LUISA_INFO("Volume kind={} algorithm={}: exact={:.5f} estimate={:.5f} (error {:.2f}%)",
                           k, algorithm, exact, estimate, relative * 100.0);
                if (relative > 0.035) {
                    volume_ok = false;
                    LUISA_WARNING("volume estimate is off by more than 3.5% (kind {}, algorithm {})", k, algorithm);
                }
            }
            for (size_t a = 1; a < estimates.size(); a++) {
                auto relative = std::abs(estimates[a] - estimates[0]) / exact;
                if (relative > 0.035) {
                    volume_ok = false;
                    LUISA_WARNING("samplers disagree by more than 3.5% (kind {}, sampler {})", k, a);
                }
            }
        }
        expect(volume_ok) << "Monte-Carlo volumes must match the analytic ones for every sampler";
    }

    // -----------------------------------------------------------------------
    // 3. analytic sphere form factor with cosine-weighted hemisphere sampling
    // -----------------------------------------------------------------------
    {
        auto kernel = device.compile<1>(
            [&](BufferVar<uint> visible, UInt algorithm, Float3 shading_point, Float3 normal) noexcept {
                auto i = dispatch_id().x;
                auto u = luisa::test::zoo::sampling::sample2(algorithm, 0x9e3779b9u + i, i);
                auto d = luisa::test::zoo::sampling::cosine_hemisphere(u, normal);
                auto ray = make_ray(shading_point + normal * 1e-4f, d, 1e-5f, 1e30f);
                auto hit = !form_factor_accel->traverse(ray, {})
                                .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                    auto h = candidate.hit();
                                    auto prim = prim_buffer->read(form_factor_prim + h.prim);
                                    auto cr = candidate.ray();
                                    auto t = def(cr->t_max());
                                    luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                                              cr->t_min(), cr->t_max(), t);
                                    $if (t < cr->t_max()) { candidate.commit(t); };
                                })
                                .trace()
                                ->miss();
                visible.write(i, select(0u, 1u, hit));
            });

        const float3 shading_point = make_float3(0.f, 0.f, -6.5f);
        const float3 normal = make_float3(0.f, 0.f, -1.f);
        // exact form factor of a fully visible sphere: (r/d)^2
        auto d = 1.5;
        auto exact = (0.5 / d) * (0.5 / d);
        auto visible_buffer = device.create_buffer<uint>(kFormFactorSamples);
        luisa::vector<uint> visible(kFormFactorSamples);
        bool form_factor_ok = true;
        for (uint algorithm = 0u; algorithm < kSampleAlgorithms - 1u; algorithm++) {
            stream << kernel(visible_buffer, algorithm, shading_point, normal)
                          .dispatch(kFormFactorSamples)
                   << visible_buffer.copy_to(luisa::span{visible})
                   << synchronize();
            auto count = std::count_if(visible.begin(), visible.end(), [](uint v) noexcept { return v != 0u; });
            auto estimate = static_cast<double>(count) / static_cast<double>(kFormFactorSamples);
            auto error = std::abs(estimate - exact);
            LUISA_INFO("Form factor algorithm={}: exact={:.6f} estimate={:.6f} (error {:.5f})",
                       algorithm, exact, estimate, error);
            if (error > 0.006) {
                form_factor_ok = false;
                LUISA_WARNING("form factor estimate is off by more than 5 sigma (algorithm {})", algorithm);
            }
        }
        expect(form_factor_ok) << "cosine-weighted hemisphere sampling must reproduce (r/d)^2";
    }

    // -----------------------------------------------------------------------
    // 4. sampler statistics
    // -----------------------------------------------------------------------
    {
        auto kernel = device.compile<1>(
            [&](BufferVar<float4> samples, UInt algorithm, UInt base) noexcept {
                auto i = dispatch_id().x;
                // one fixed seed for the whole block: a Cranley-Patterson rotation must be
                // constant across a sequence, otherwise a low-discrepancy set degrades
                // into independent random points
                auto s = luisa::test::zoo::sampling::sample3(algorithm, 0x1234567u, i);
                samples.write(base + i, make_float4(s, 0.f));
            });
        auto buffer = device.create_buffer<float4>(kSamplerSamples * kSampleAlgorithms);
        luisa::vector<float4> samples(buffer.size());
        for (uint algorithm = 0u; algorithm < kSampleAlgorithms; algorithm++) {
            stream << kernel(buffer, algorithm, algorithm * kSamplerSamples)
                          .dispatch(kSamplerSamples);
        }
        stream << buffer.copy_to(luisa::span{samples}) << synchronize();

        bool statistics_ok = true;
        for (uint algorithm = 0u; algorithm < kSampleAlgorithms; algorithm++) {
            auto offset = algorithm * kSamplerSamples;
            std::array<double, 16u> histogram{};
            double sum = 0.0;
            double sum_y = 0.0;
            for (uint i = 0u; i < kSamplerSamples; i++) {
                auto v = samples[offset + i];
                sum += v.x;
                sum_y += v.y;
                auto bin = std::min(15u, static_cast<uint>(v.x * 16.f));
                histogram[bin] += 1.0;
            }
            auto mean = sum / kSamplerSamples;
            auto mean_y = sum_y / kSamplerSamples;
            LUISA_INFO("Sampler {} first four: ({:.6f}, {:.6f}, {:.6f}) ({:.6f}, {:.6f}, {:.6f}) ({:.6f}, {:.6f}, {:.6f}) ({:.6f}, {:.6f}, {:.6f})",
                       algorithm,
                       samples[offset + 0u].x, samples[offset + 0u].y, samples[offset + 0u].z,
                       samples[offset + 1u].x, samples[offset + 1u].y, samples[offset + 1u].z,
                       samples[offset + 2u].x, samples[offset + 2u].y, samples[offset + 2u].z,
                       samples[offset + 3u].x, samples[offset + 3u].y, samples[offset + 3u].z);
            auto expected = 1.0 / 16.0;
            double max_bin_error = 0.0;
            for (auto h : histogram) {
                max_bin_error = std::max(max_bin_error, std::abs(h / kSamplerSamples - expected));
            }
            // discrepancy: the spread of the occupancy of a 32x32 grid.  For an
            // independent random sequence the occupancy is Poisson, so its
            // standard deviation equals sqrt(mean); a low-discrepancy sequence
            // keeps it far below that.
            constexpr uint grid = 32u;
            std::array<uint, grid * grid> cells{};
            for (uint i = 0u; i < kSamplerSamples; i++) {
                auto v = samples[offset + i];
                auto cx = std::min(grid - 1u, static_cast<uint>(v.x * static_cast<float>(grid)));
                auto cy = std::min(grid - 1u, static_cast<uint>(v.y * static_cast<float>(grid)));
                cells[cy * grid + cx]++;
            }
            auto cell_mean = static_cast<double>(kSamplerSamples) / static_cast<double>(grid * grid);
            double cell_variance = 0.0;
            uint max_cell = 0u;
            for (auto c : cells) {
                cell_variance += (static_cast<double>(c) - cell_mean) * (static_cast<double>(c) - cell_mean);
                max_cell = std::max(max_cell, c);
            }
            auto cell_std = std::sqrt(cell_variance / static_cast<double>(cells.size()));
            LUISA_INFO("Sampler {}: mean=({:.4f}, {:.4f}) max bin error={:.4f} cell mean={:.1f} std={:.2f} max={}",
                       algorithm, mean, mean_y, max_bin_error, cell_mean, cell_std, max_cell);
            if (std::abs(mean - 0.5) > 0.01 || std::abs(mean_y - 0.5) > 0.01) {
                statistics_ok = false;
                LUISA_WARNING("sampler {} is not uniform (mean)", algorithm);
            }
            if (max_bin_error > 0.008) {
                statistics_ok = false;
                LUISA_WARNING("sampler {} is not uniform (16-bin histogram)", algorithm);
            }
            if ((algorithm == 1u || algorithm == 2u) && cell_std > 4.0) {
                statistics_ok = false;
                LUISA_WARNING("sampler {} does not look low-discrepancy (cell std {:.2f}, random would be {:.2f})",
                              algorithm, cell_std, std::sqrt(cell_mean));
            }
            if (algorithm == 0u && cell_std < 4.0) {
                statistics_ok = false;
                LUISA_WARNING("the hash sampler must look random, but its cell std is only {:.2f}", cell_std);
            }
            if (algorithm == 3u) {
                // the stratified sampler must place exactly 4 samples in every
                // stratum of every 8x8 block
                bool stratified_ok = true;
                for (uint block = 0u; block < kSamplerSamples / 256u && stratified_ok; block++) {
                    std::array<uint, 64u> counts{};
                    for (uint i = 0u; i < 256u; i++) {
                        auto v = samples[offset + block * 256u + i];
                        auto sx = std::min(7u, static_cast<uint>(v.x * 8.f));
                        auto sy = std::min(7u, static_cast<uint>(v.y * 8.f));
                        counts[sy * 8u + sx]++;
                    }
                    for (auto c : counts) {
                        if (c != 4u) { stratified_ok = false; }
                    }
                }
                if (!stratified_ok) {
                    statistics_ok = false;
                    LUISA_WARNING("the stratified sampler does not fill the 8x8 strata evenly");
                }
            }
        }
        expect(statistics_ok) << "all sampling algorithms must be uniform (and stratified/LDS as advertised)";
    }

    // -----------------------------------------------------------------------
    // 5. path-traced render of the zoo, convergence check and PNG output
    // -----------------------------------------------------------------------
    {
        constexpr uint width = 320u;
        constexpr uint height = 180u;
        auto kernel = device.compile<2>(
            [&](ImageFloat image, Float3 camera, UInt spp, UInt frame, Float sky, Float albedo) noexcept {
                auto coord = dispatch_id().xy();
                auto size = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
                auto aspect = size.x / size.y;
                auto sum = def(make_float3(0.f));
                $for (s, 0u, spp) {
                    auto u = luisa::test::zoo::sampling::sample2(0u, frame * 9781u + s, s);
                    auto p = (make_float2(coord) + u) / size * 2.f - 1.f;
                    auto origin = def(camera);
                    auto direction = def(normalize(make_float3(p * tan(0.5f * 0.7f) * make_float2(aspect, 1.f), -1.f)));
                    auto radiance = def(make_float3(0.f));
                    auto throughput = def(make_float3(1.f));
                    auto done = def(false);
                    $for (bounce, 0u, 4u) {
                        $if (!done) {
                            auto t_hit = def(1e30f);
                            auto prim_hit = def(~0u);
                            auto ray = make_ray(origin, direction, 1e-4f, 1e30f);
                            zoo_accel->traverse(ray, {})
                                .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                    auto h = candidate.hit();
                                    auto prim = prim_buffer->read(h.prim);
                                    auto cr = candidate.ray();
                                    auto t = def(cr->t_max());
                                    luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                                              cr->t_min(), t_hit, t);
                                    $if (t < t_hit) {
                                        t_hit = t;
                                        prim_hit = h.prim;
                                        candidate.commit(t);
                                    };
                                })
                                .trace();
                            $if (t_hit > 1e29f) {
                                radiance = radiance + throughput * sky;
                                done = true;
                            } $else {
                                auto prim = prim_buffer->read(prim_hit);
                                auto p_hit = origin + direction * t_hit;
                                auto n = normalize(p_hit - prim->center);
                                // a per-kind tint so the image is readable
                                auto hue = cast<float>(prim->kind) * (1.f / 12.f);
                                auto tint = 0.5f + 0.5f * cos(6.2831853f * (hue + make_float3(0.f, 0.33f, 0.67f)));
                                auto local_albedo = tint * albedo;
                                throughput = throughput * local_albedo;
                                auto next_u = luisa::test::zoo::sampling::sample2(0u, frame * 9781u + s * 4u + bounce, s + bounce * 65536u);
                                direction = luisa::test::zoo::sampling::cosine_hemisphere(next_u, n);
                                origin = p_hit + n * 2e-3f;
                            };
                        };
                    };
                    sum = sum + radiance;
                };
                image.write(coord, make_float4(sum / cast<float>(spp), 1.f));
            });

        auto ldr = device.create_image<float>(PixelStorage::BYTE4, width, height);
        auto blit = device.compile<2>([](ImageFloat image_in, ImageFloat image_out) noexcept {
            auto p = dispatch_id().xy();
            auto c = image_in.read(p).xyz();
            auto g = sqrt(max(c, make_float3(0.f)));
            image_out.write(p, make_float4(clamp(g, 0.f, 1.f), 1.f));
        });

        const float3 camera = make_float3(0.f, 0.f, 9.f);
        auto image16 = device.create_image<float>(PixelStorage::FLOAT4, width, height);
        auto image128 = device.create_image<float>(PixelStorage::FLOAT4, width, height);
        auto render = [&](Image<float> &image, uint spp) noexcept {
            luisa::vector<float4> host(width * height);
            stream << kernel(image, camera, spp, spp, 0.65f, 0.75f).dispatch(width, height)
                   << image.copy_to(luisa::span{host})
                   << synchronize();
            return host;
        };
        auto host16 = render(image16, 16u);
        auto host128 = render(image128, 128u);

        double mean_diff = 0.0;
        double max_diff = 0.0;
        double mean_value = 0.0;
        bool finite = true;
        for (size_t i = 0; i < host16.size(); i++) {
            auto a = host16[i];
            auto b = host128[i];
            auto diff = std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
            mean_diff += diff / 3.0;
            max_diff = std::max(max_diff, static_cast<double>(diff) / 3.0);
            mean_value += (b.x + b.y + b.z) / 3.0;
            finite = finite && std::isfinite(a.x + a.y + a.z + b.x + b.y + b.z);
        }
        mean_diff /= static_cast<double>(host16.size());
        mean_value /= static_cast<double>(host128.size());
        LUISA_INFO("Procedural zoo render: mean={:.4f}, mean|16spp-128spp|={:.4f}, max|diff|={:.4f}",
                   mean_value, mean_diff, max_diff);
        expect(finite) << "the path tracer must not produce NaN or Inf";
        expect(mean_value > 0.05 && mean_value < 0.65) << "the zoo render must be lit by the sky";
        expect(mean_diff < 0.06) << "16 and 128 samples per pixel must agree";

        luisa::vector<std::array<uint8_t, 4u>> pixels(width * height);
        stream << blit(image128, ldr).dispatch(width, height)
               << ldr.copy_to(luisa::span{pixels})
               << synchronize();
        stbi_write_png("test_procedural_sampling.png", width, height, 4, pixels.data(), 0);
        LUISA_INFO("Wrote test_procedural_sampling.png.");
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_procedural_sampling(device);
}