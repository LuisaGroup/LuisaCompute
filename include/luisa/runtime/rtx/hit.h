#pragma once

#include <luisa/core/basic_types.h>

namespace luisa::compute {

enum class HitType : uint32_t {
    Miss = 0,
    Triangle = 1,
    Procedural = 2
};

// Hit classes used by DSL, DSL module see src/dsl/rtx/ray_query.h
// Return type of RayQuery::committed_hit(), it represents a hit that can be a triangle, a procedural-primitive or nothing
struct CommittedHit {
    uint inst;
    uint prim;
    float2 bary;
    uint hit_type; // HitType
    float committed_ray_t;
};
static_assert(sizeof(CommittedHit) == 24u, "CommittedHit size mismatch");
static_assert(alignof(CommittedHit) == 8u, "CommittedHit align mismatch");
// Return type of Accel::trace_closest() and RayQuery::triangle_candidate(), it represents a hit that can be a triangle or nothing
struct TriangleHit {
    uint inst;
    uint prim;
    float2 bary;
    float committed_ray_t;
};
static_assert(sizeof(TriangleHit) == 24u, "TriangleHit size mismatch");
static_assert(alignof(TriangleHit) == 8u, "TriangleHit align mismatch");
// Return type of RayQuery::procedural_candidate(), it represents a hit of procedural-primitive
struct ProceduralHit {
    uint inst;
    uint prim;
};

}// namespace luisa::compute

