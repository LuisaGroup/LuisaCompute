#pragma once
#include "./../types/vec.hpp"

namespace luisa::shader {
struct [[builtin("bindless_array")]] BindlessVolume {
	[[callop("TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER")]] float4 volume_sample(uint32 volume_index, float3 uv, uint filter, uint address);
	[[callop("TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER")]] float4 volume_sample_level(uint32 volume_index, float3 uv, float mip_level, uint filter, uint address);
	[[callop("TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER")]] float4 volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy, uint filter, uint address);
	[[callop("TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER")]] float4 volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy, float min_mip_level, uint filter, uint address);

	[[callop("TYPED_BINDLESS_TEXTURE3D_READ")]] float4 volume_load(uint32 volume_index, uint3 coord);
	[[callop("TYPED_BINDLESS_TEXTURE3D_READ_LEVEL")]] float4 volume_load_level(uint32 volume_index, uint3 coord, uint32 mip_level);
	[[callop("TYPED_BINDLESS_TEXTURE3D_SIZE")]] uint3 volume_size(uint32 volume_index);
	[[callop("TYPED_BINDLESS_TEXTURE3D_SIZE_LEVEL")]] uint3 volume_size_level(uint32 volume_index, uint32 mip_level);

	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER")]] float4 uniform_idx_volume_sample(uint32 volume_index, float3 uv, uint filter, uint address);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER")]] float4 uniform_idx_volume_sample_level(uint32 volume_index, float3 uv, float mip_level, uint filter, uint address);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER")]] float4 uniform_idx_volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy, uint filter, uint address);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER")]] float4 uniform_idx_volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy, float min_mip_level, uint filter, uint address);

	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ")]] float4 uniform_idx_volume_load(uint32 volume_index, uint3 coord);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL")]] float4 uniform_idx_volume_load_level(uint32 volume_index, uint3 coord, uint32 mip_level);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE")]] uint3 uniform_idx_volume_size(uint32 volume_index);
	[[callop("TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL")]] uint3 uniform_idx_volume_size_level(uint32 volume_index, uint32 mip_level);
};
}// namespace luisa::shader