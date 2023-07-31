#pragma once
/**
 * @file: tests/next/example/gallary/render/path_tracer_util.h
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the necessary struct used in path tracer
*/

#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {
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
