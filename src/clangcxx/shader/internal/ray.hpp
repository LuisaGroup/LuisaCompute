#pragma once
#include "vec.hpp"

namespace luisa::shader {
struct Ray {
    float3 origin;
    float3 dir;
    float t_min;
    float t_max;
    // Ray() = default;
    Ray(float3 origin,
        float3 dir,
        float t_min = 0.0f,
        float t_max = 1e30f) : origin(origin), dir(dir), t_min(t_min), t_max(t_min) {}
};

/* TODO
enum struct HitType : uint32 {
    Miss = 0,
    HitTriangle = 1,
    HitProcedural = 2
};
*/

using HitType = uint32;
trait HitTypes {
    static constexpr HitType Miss = 0;
    static constexpr HitType HitTriangle = 1;
    static constexpr HitType HitProcedural = 2;
};

struct CommittedHit {
    uint32 inst;
    uint32 prim;
    float2 bary;
    HitType hit_type;
};

struct TriangleHit {
    uint32 inst;
    uint32 prim;
    float2 bary;
    float ray_t;
};

struct ProceduralHit {
    uint32 inst;
    uint32 prim;
};
}// namespace luisa::shader