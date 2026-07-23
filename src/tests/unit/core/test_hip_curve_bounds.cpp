#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <luisa/core/basic_types.h>

#include "hip_curve_bounds.h"
#include "ut/ut.hpp"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] float4 evaluate_curve(
    CurveBasis basis, const std::array<float4, 4u> &p, float u) noexcept {
    auto u2 = u * u;
    auto u3 = u2 * u;
    switch (basis) {
        case CurveBasis::PIECEWISE_LINEAR:
            return p[0] * (1.0f - u) + p[1] * u;
        case CurveBasis::CUBIC_BSPLINE:
            return (p[0] * (1.0f - 3.0f * u + 3.0f * u2 - u3) +
                    p[1] * (4.0f - 6.0f * u2 + 3.0f * u3) +
                    p[2] * (1.0f + 3.0f * u + 3.0f * u2 - 3.0f * u3) +
                    p[3] * u3) /
                   6.0f;
        case CurveBasis::CATMULL_ROM:
            return (p[1] * 2.0f +
                    (p[2] - p[0]) * u +
                    (p[0] * 2.0f - p[1] * 5.0f + p[2] * 4.0f - p[3]) * u2 +
                    (-p[0] + p[1] * 3.0f - p[2] * 3.0f + p[3]) * u3) *
                   0.5f;
        case CurveBasis::BEZIER: {
            auto v = 1.0f - u;
            return p[0] * (v * v * v) +
                   p[1] * (3.0f * v * v * u) +
                   p[2] * (3.0f * v * u2) +
                   p[3] * u3;
        }
    }
    return {};
}

[[nodiscard]] std::array<float4, 4u> controls_for_basis(
    CurveBasis basis, const std::array<float4, 4u> &b) noexcept {
    switch (basis) {
        case CurveBasis::PIECEWISE_LINEAR:
            return {b[0], b[3], float4{}, float4{}};
        case CurveBasis::CUBIC_BSPLINE:
            return {
                b[0] * 6.0f - b[1] * 7.0f + b[2] * 2.0f,
                b[1] * 2.0f - b[2],
                b[2] * 2.0f - b[1],
                b[3] * 6.0f + b[1] * 2.0f - b[2] * 7.0f};
        case CurveBasis::CATMULL_ROM:
            return {
                b[3] - b[1] * 6.0f + b[0] * 6.0f,
                b[0],
                b[3],
                b[0] + b[3] * 6.0f - b[2] * 6.0f};
        case CurveBasis::BEZIER:
            return b;
    }
    return {};
}

[[nodiscard]] bool contains_swept_point(
    const HIPCurveAABB &bounds, float4 p) noexcept {
    auto radius = std::max(std::abs(p.w), 1.0e-7f);
    return bounds.min[0] <= p.x - radius && p.x + radius <= bounds.max[0] &&
           bounds.min[1] <= p.y - radius && p.y + radius <= bounds.max[1] &&
           bounds.min[2] <= p.z - radius && p.z + radius <= bounds.max[2];
}

[[nodiscard]] HIPCurveAABB legacy_sampled_bounds(
    CurveBasis basis, const std::array<float4, 4u> &p) noexcept {
    HIPCurveAABB bounds{};
    for (auto axis = 0u; axis < 3u; axis++) {
        bounds.min[axis] = std::numeric_limits<float>::infinity();
        bounds.max[axis] = -std::numeric_limits<float>::infinity();
    }
    auto steps = basis == CurveBasis::PIECEWISE_LINEAR ? 1u : 16u;
    for (auto i = 0u; i <= steps; i++) {
        auto q = evaluate_curve(basis, p, static_cast<float>(i) / static_cast<float>(steps));
        auto radius = std::max(std::abs(q.w), 1.0e-7f);
        for (auto axis = 0u; axis < 3u; axis++) {
            bounds.min[axis] = std::min(bounds.min[axis], q[axis] - radius);
            bounds.max[axis] = std::max(bounds.max[axis], q[axis] + radius);
        }
    }
    return bounds;
}

}// namespace

static auto test_hip_curve_bounds = [] {
    "HIP curve bounds enclose curves and capsule chains"_test = [] {
        std::array bezier{
            make_float4(0.0f, 1.0f, -2.0f, 0.5f),
            make_float4(100.0f, -30.0f, 20.0f, 0.5f),
            make_float4(-100.0f, 40.0f, -25.0f, 0.5f),
            make_float4(2.0f, -1.0f, 3.0f, 0.5f)};
        constexpr std::array bases{
            CurveBasis::PIECEWISE_LINEAR,
            CurveBasis::CUBIC_BSPLINE,
            CurveBasis::CATMULL_ROM,
            CurveBasis::BEZIER};
        for (auto basis : bases) {
            auto controls = controls_for_basis(basis, bezier);
            // Keep the geometric center curve adversarial while using valid,
            // positive, smoothly varying source radii for every basis. The
            // inverse center conversion above is intentionally not applied to
            // the radius channel.
            constexpr std::array source_radii{0.4f, 0.5f, 0.6f, 0.7f};
            auto control_count = segment_control_point_count(basis);
            for (auto i = 0u; i < control_count; i++) {
                controls[i].w = source_radii[i];
            }
            auto bounds = compute_hip_curve_aabb(basis, controls.data());
            auto dense_is_enclosed = true;
            constexpr auto dense_steps = 65536u;
            for (auto i = 0u; i <= dense_steps; i++) {
                auto u = static_cast<float>(i) / static_cast<float>(dense_steps);
                dense_is_enclosed &= contains_swept_point(
                    bounds, evaluate_curve(basis, controls, u));
            }
            expect(dense_is_enclosed);

            auto capsule_chain_is_enclosed = true;
            auto capsule_steps = basis == CurveBasis::PIECEWISE_LINEAR ? 1u : 16u;
            auto p0 = evaluate_curve(basis, controls, 0.0f);
            for (auto i = 0u; i < capsule_steps; i++) {
                auto u1 = static_cast<float>(i + 1u) /
                          static_cast<float>(capsule_steps);
                auto p1 = evaluate_curve(basis, controls, u1);
                auto radius = std::max(
                    0.5f * (std::abs(p0.w) + std::abs(p1.w)), 1.0e-7f);
                for (auto axis = 0u; axis < 3u; axis++) {
                    capsule_chain_is_enclosed &=
                        bounds.min[axis] <= std::min(p0[axis], p1[axis]) - radius &&
                        std::max(p0[axis], p1[axis]) + radius <= bounds.max[axis];
                }
                p0 = p1;
            }
            expect(capsule_chain_is_enclosed);
        }
    };

    "HIP curve bounds catch extrema between legacy samples"_test = [] {
        std::array bezier{
            make_float4(0.0f, 0.0f, 0.0f, 0.25f),
            make_float4(100.0f, 0.0f, 0.0f, 0.25f),
            make_float4(-100.0f, 0.0f, 0.0f, 0.25f),
            make_float4(0.0f, 0.0f, 0.0f, 0.25f)};
        constexpr std::array cubic_bases{
            CurveBasis::CUBIC_BSPLINE,
            CurveBasis::CATMULL_ROM,
            CurveBasis::BEZIER};
        for (auto basis : cubic_bases) {
            auto controls = controls_for_basis(basis, bezier);
            auto conservative = compute_hip_curve_aabb(basis, controls.data());
            auto legacy = legacy_sampled_bounds(basis, controls);
            auto legacy_misses = false;
            auto conservative_contains = true;
            constexpr auto dense_steps = 65536u;
            for (auto i = 0u; i <= dense_steps; i++) {
                auto u = static_cast<float>(i) / static_cast<float>(dense_steps);
                auto p = evaluate_curve(basis, controls, u);
                legacy_misses |= !contains_swept_point(legacy, p);
                conservative_contains &= contains_swept_point(conservative, p);
            }
            expect(legacy_misses);
            expect(conservative_contains);
        }
    };

    "HIP curve bounds enclose the runtime capsule radius"_test = [] {
        std::array controls{
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            make_float4(100.0f, 0.0f, 0.0f, 2.0f),
            float4{},
            float4{}};
        auto conservative = compute_hip_curve_aabb(
            CurveBasis::PIECEWISE_LINEAR, controls.data());
        auto legacy = legacy_sampled_bounds(
            CurveBasis::PIECEWISE_LINEAR, controls);
        // The device represents this segment by a radius-1 capsule (the
        // average of the endpoint radii). The old endpoint-sphere bounds start
        // near x=0 and therefore miss the capsule's x=-1 extent.
        constexpr auto capsule_min_x = -1.0f;
        expect(legacy.min[0] > capsule_min_x);
        expect(conservative.min[0] <= capsule_min_x);
    };

    "HIP curve bounds tolerate signed radius data"_test = [] {
        std::array controls{
            make_float4(0.0f, 0.0f, 0.0f, 0.1f),
            make_float4(1.0f, 2.0f, 3.0f, -3.0f),
            make_float4(2.0f, -1.0f, 1.0f, 0.5f),
            make_float4(3.0f, 0.0f, -2.0f, -0.2f)};
        auto bounds = compute_hip_curve_aabb(CurveBasis::BEZIER, controls.data());
        auto enclosed = true;
        constexpr auto dense_steps = 65536u;
        for (auto i = 0u; i <= dense_steps; i++) {
            auto u = static_cast<float>(i) / static_cast<float>(dense_steps);
            enclosed &= contains_swept_point(
                bounds, evaluate_curve(CurveBasis::BEZIER, controls, u));
        }
        expect(enclosed);
    };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
