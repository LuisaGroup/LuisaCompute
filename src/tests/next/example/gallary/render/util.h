#pragma once
/**
 * @file: tests/next/example/gallary/render/path_tracer_util.h
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the necessary struct used in path tracer
*/

#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

struct Material {
    float3 albedo;
    float3 emission;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

}// namespace luisa::test

LUISA_STRUCT(luisa::test::Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

LUISA_STRUCT(luisa::test::Material, albedo, emission) {};

namespace luisa::test {

Callable<Onb(float3)> make_onb_callable();
Callable<float3(float2)> cosine_sample_hemisphere_callable();
Callable<float(float, float)> balanced_heuristic_callable();
Callable<uint(uint, uint)> tea_callable();

}// namespace luisa::test