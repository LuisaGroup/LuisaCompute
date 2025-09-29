#pragma once
#include "./../types/vec.hpp"

namespace luisa::shader {
struct [[builtin("bindless_array")]] BindlessImage {
    [[callop("TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER")]] float4 image_sample(uint32 image_index, float2 uv, uint filter, uint address);
    [[callop("TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER")]] float4 image_sample_level(uint32 image_index, float2 uv, float mip_level, uint filter, uint address);
    [[callop("TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER")]] float4 image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy, uint filter, uint address);
    [[callop("TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER")]] float4 image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy, float min_mip_level, uint filter, uint address);

    [[callop("TYPED_BINDLESS_TEXTURE2D_READ")]] float4 image_load(uint32 image_index, uint2 coord);
    [[callop("TYPED_BINDLESS_TEXTURE2D_READ_LEVEL")]] float4 image_load_level(uint32 image_index, uint2 coord, uint32 mip_level);
    [[callop("TYPED_BINDLESS_TEXTURE2D_SIZE")]] uint2 image_size(uint32 image_index);
    [[callop("TYPED_BINDLESS_TEXTURE2D_SIZE_LEVEL")]] uint2 image_size_level(uint32 image_index, uint32 mip_level);

    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER")]] float4 uniform_idx_image_sample(uint32 image_index, float2 uv, uint filter, uint address);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER")]] float4 uniform_idx_image_sample_level(uint32 image_index, float2 uv, float mip_level, uint filter, uint address);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER")]] float4 uniform_idx_image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy, uint filter, uint address);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER")]] float4 uniform_idx_image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy, float min_mip_level, uint filter, uint address);

    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ")]] float4 uniform_idx_image_load(uint32 image_index, uint2 coord);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL")]] float4 uniform_idx_image_load_level(uint32 image_index, uint2 coord, uint32 mip_level);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE")]] uint2 uniform_idx_image_size(uint32 image_index);
    [[callop("TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL")]] uint2 uniform_idx_image_size_level(uint32 image_index, uint32 mip_level);
};
}// namespace luisa::shader