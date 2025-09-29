#pragma once
#include "./../attributes.hpp"
#include "./../type_traits.hpp"
#include "./../types/vec.hpp"

namespace luisa::shader {
trait_struct Filter {
	static constexpr uint32 POINT = 0;
	static constexpr uint32 LINEAR_POINT = 1;
	static constexpr uint32 LINEAR_LINEAR = 2;
	static constexpr uint32 ANISOTROPIC = 3;
};

trait_struct Address {
	static constexpr uint32 EDGE = 0;
	static constexpr uint32 REPEAT = 1;
	static constexpr uint32 MIRROR = 2;
	static constexpr uint32 ZERO = 3;
};
template<concepts::arithmetic_scalar T, uint32 cache_flag = CacheFlags::None>
struct [[builtin("image")]] Image {
	using ElementType = T;

	[[callop("TEXTURE_READ")]] vec<T, 4> load(uint2 coord);
	[[callop("TEXTURE_WRITE")]] void store(uint2 coord, vec<T, 4> val);
	[[callop("TEXTURE_SIZE")]] uint2 size();
	[[ignore]] Image() = delete;
	[[ignore]] Image(Image const&) = delete;
	[[ignore]] Image& operator=(Image const&) = delete;
};

template<concepts::arithmetic_scalar T, uint32 cache_flag = CacheFlags::None>
struct [[builtin("volume")]] Volume {
	using ElementType = T;

	[[callop("TEXTURE_READ")]] vec<T, 4> load(uint3 coord);
	[[callop("TEXTURE_WRITE")]] void store(uint3 coord, vec<T, 4> val);
	[[callop("TEXTURE_SIZE")]] uint3 size();
	[[ignore]] Volume() = delete;
	[[ignore]] Volume(Volume const&) = delete;
	[[ignore]] Volume& operator=(Volume const&) = delete;
};

template<>
struct [[builtin("image")]] Image<float, CacheFlags::ReadOnly> {
	[[callop("TEXTURE_READ")]] float4 load(uint2 coord);
	[[callop("TEXTURE_SIZE")]] uint2 size();
	[[callop("TEXTURE2D_SAMPLE")]] float4 sample(float2 uv, uint filter, uint address);
	[[callop("TEXTURE2D_SAMPLE_LEVEL")]] float4 sample_level(float2 uv, uint level, uint filter, uint address);
	[[callop("TEXTURE2D_SAMPLE_GRAD")]] float4 sample_grad(float2 uv, float2 ddx, float2 ddy, uint filter, uint address);
	[[callop("TEXTURE2D_SAMPLE_GRAD_LEVEL")]] float4 sample_grad_level(float2 uv, float2 ddx, float2 ddy, float min_mip_level, uint filter, uint address);
	[[ignore]] Image() = delete;
	[[ignore]] Image(Image const&) = delete;
	[[ignore]] Image& operator=(Image const&) = delete;
};

template<>
struct [[builtin("volume")]] Volume<float, CacheFlags::ReadOnly> {
	[[callop("TEXTURE_READ")]] float4 load(uint3 coord);
	[[callop("TEXTURE_SIZE")]] uint3 size();
	[[callop("TEXTURE3D_SAMPLE")]] float4 sample(float3 uv, uint filter, uint address);
	[[callop("TEXTURE3D_SAMPLE_LEVEL")]] float4 sample_level(float3 uv, uint level, uint filter, uint address);
	[[callop("TEXTURE3D_SAMPLE_GRAD")]] float4 sample_grad(float3 uv, float3 ddx, float3 ddy, uint filter, uint address);
	[[callop("TEXTURE3D_SAMPLE_GRAD_LEVEL")]] float4 sample_grad_level(float3 uv, float3 ddx, float3 ddy, float min_mip_level, uint filter, uint address);
	[[ignore]] Volume() = delete;
	[[ignore]] Volume(Volume const&) = delete;
	[[ignore]] Volume& operator=(Volume const&) = delete;
};
using SampleImage = Image<float, CacheFlags::ReadOnly>;
using SampleVolume = Volume<float, CacheFlags::ReadOnly>;

}// namespace luisa::shader