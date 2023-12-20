#pragma once
#include "vec.hpp"
#include "array.hpp"

namespace luisa::shader {
struct Ray {
    Array<float, 3> _origin;
    float t_min;
    Array<float, 3> _dir;
    float t_max;
    // Ray() = default;
    Ray(float3 origin,
        float3 dir,
        float t_min = 0.0f,
        float t_max = 1e30f) : t_min(t_min), t_max(t_max) {
        _origin[0] = origin.x;
        _origin[1] = origin.y;
        _origin[2] = origin.z;
        _dir[0] = dir.x;
        _dir[1] = dir.y;
        _dir[2] = dir.z;
    }
    float3 origin() const {
        return float3(_origin[0], _origin[1], _origin[2]);
    }
    float3 dir() const {
        return float3(_dir[0], _dir[1], _dir[2]);
    }
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
    bool miss() const {
        return hit_type == HitTypes::Miss;
    }
    bool hit_triangle() const {
        return hit_type == HitTypes::HitTriangle;
    }
    bool hit_procedural() const {
        return hit_type == HitTypes::HitProcedural;
    }
};

struct TriangleHit {
    uint32 inst;
    uint32 prim;
    float2 bary;
    float ray_t;
    bool miss() const {
        return inst == 4294967295u;
    }
    bool hitted() const {
        return inst != 4294967295u;
    }
    template<typename T>
        requires(is_float_family<T>::value)
    T interpolate(T a, T b, T c) {
        return T(1.0f - bary.x - bary.y) * a + T(bary.x) * b + T(bary.y) * c;
    }
};

struct ProceduralHit {
    uint32 inst;
    uint32 prim;
};
}// namespace luisa::shader