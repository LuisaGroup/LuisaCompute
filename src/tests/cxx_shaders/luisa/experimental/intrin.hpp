#pragma once
#include <luisa/type_traits.hpp>
#include <luisa/types.hpp>
#include <luisa/functions/math.hpp>

namespace luisa::shader 
{
	template<concepts::vec V, concepts::matrix M>
	V mul(const V& v, const M& m) 
	{
		return v * m;
	}

	template<concepts::vec V, concepts::matrix M>
	V mul(const M& m, const V& v) 
	{
		return m * v;
	}

	template<concepts::matrix M>
	M mul(const M& m1, const M& m2) 
	{
		return m1 * m2;
	}

	template<concepts::vec V>
	V lerp(const float& a, const V& b, const float& factor) 
	{
		return lerp(V(a), b, factor);
	}

	template<concepts::vec V>
	V lerp(const V& a, const float& b, const float& factor) 
	{
		return lerp(a, V(b), factor);
	}

	template<concepts::vec V>
	V min(const V& a, const float& b) 
	{
		return min(a, V(b));
	}

	template<concepts::vec V>
	V min(const float& a, const V& b) 
	{
		return min(V(a), b);
	}

	template<concepts::vec V>
	V max(const V& a, const float& b) 
	{
		return max(a, V(b));
	}

	template<concepts::vec V>
	V max(const float& a, const V& b) 
	{
		return max(V(a), b);
	}

	float Square(const float& x)
	{
		return x * x;
	}

	float2 Square(const float2& x)
	{
		return x * x;
	}

	float3 Square(const float3& x)
	{
		return x * x;
	}

	float4 Square(const float4& x)
	{
		return x * x;
	}
}// namespace luisa::shader