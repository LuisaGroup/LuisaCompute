#pragma once
#include "vec.hpp"
#include "array.hpp"
#include <luisa/numeric.hpp>
namespace luisa::shader {
struct alignas(16) Ray {
	Ray() = default;
	Ray(const float3& origin, const float3& dir, float t_min = 0.0f, float t_max = 1e30f)
		: t_min(t_min), t_max(t_max) {
		_origin.set<0>(origin.x, origin.y, origin.z);
		_dir.set<0>(dir.x, dir.y, dir.z);
	}
	[[nodiscard]] float3 origin() const {
		return float3(_origin[0], _origin[1], _origin[2]);
	}
	[[nodiscard]] float3 dir() const {
		return float3(_dir[0], _dir[1], _dir[2]);
	}

	Array<float, 3> _origin;
	float t_min = 0.0f;
	Array<float, 3> _dir;
	float t_max = 1e30f;
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
	float ray_t;
	[[nodiscard]] bool miss() const {
		return hit_type == HitTypes::Miss;
	}
	[[nodiscard]] bool hit_triangle() const {
		return hit_type == HitTypes::HitTriangle;
	}
	[[nodiscard]] bool hit_procedural() const {
		return hit_type == HitTypes::HitProcedural;
	}
	template<concepts::float_family T>
	T interpolate(const T& a, const T& b, const T& c) {
		return T(1.0f - bary.x - bary.y) * a + T(bary.x) * b + T(bary.y) * c;
	}
};

struct TriangleHit {
	uint32 inst;
	uint32 prim;
	float2 bary;
	float ray_t;
	[[nodiscard]] bool miss() const {
		return inst == max_uint32;
	}
	[[nodiscard]] bool hitted() const {
		return inst != max_uint32;
	}
	template<concepts::float_family T>
	T interpolate(const T& a, const T& b, const T& c) {
		return T(1.0f - bary.x - bary.y) * a + T(bary.x) * b + T(bary.y) * c;
	}
};
template<concepts::float_family T>
T interpolate(float2 bary, const T& a, const T& b, const T& c) {
	return T(1.0f - bary.x - bary.y) * a + T(bary.x) * b + T(bary.y) * c;
}

struct ProceduralHit {
	uint32 inst;
	uint32 prim;
};
}// namespace luisa::shader