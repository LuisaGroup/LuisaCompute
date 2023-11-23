#pragma once

#include <luisa/core/basic_types.h>

namespace luisa::compute {

enum class HitType : uint32_t {
    Miss = 0,
    Surface = 1,// triangle or curve
    Procedural = 2,   // bounding box (for procedural primitives)

    // legacy names
    Triangle = Surface,
};

// Hit classes used by DSL, DSL module see src/dsl/rtx/ray_query.h
// Return type of RayQuery::committed_hit(), it represents a hit that can be a triangle, a procedural-primitive or nothing
struct CommittedHit {
    uint inst;
    uint prim;
    float2 bary;
    uint hit_type;// HitType
    float committed_ray_t;
};
static_assert(sizeof(CommittedHit) == 24u, "CommittedHit size mismatch");
static_assert(alignof(CommittedHit) == 8u, "CommittedHit align mismatch");

// Return type of Accel::trace_closest() and RayQuery::triangle_candidate(), it represents a hit that can be a triangle or curve
// if bary.v < 0.f then it's a curve hit, otherwise it's a triangle hit
struct SurfaceHit {
    uint inst;
    uint prim;
    float2 bary;
    float committed_ray_t;
};
static_assert(sizeof(SurfaceHit) == 24u, "SurfaceHit size mismatch");
static_assert(alignof(SurfaceHit) == 8u, "SurfaceHit align mismatch");

// Return type of RayQuery::procedural_candidate(), it represents a bounding-box hit of procedural-primitive
struct AABBHit {
    uint inst;
    uint prim;
};

// legacy names, provided for compatibility
using TriangleHit = SurfaceHit;
using ProceduralHit = AABBHit;

}// namespace luisa::compute
