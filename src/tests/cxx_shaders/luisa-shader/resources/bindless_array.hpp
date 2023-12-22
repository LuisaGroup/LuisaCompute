#pragma once
#include "./../types/vec.hpp"

namespace luisa::shader {
struct [[builtin("bindless_array")]] BindlessArray {
    [[callop("BINDLESS_TEXTURE2D_SAMPLE")]] float4 image_sample(uint32 image_index, float2 uv);
    [[callop("BINDLESS_TEXTURE2D_SAMPLE_LEVEL")]] float4 image_sample_level(uint32 image_index, float2 uv, float mip_level);
    [[callop("BINDLESS_TEXTURE2D_SAMPLE_GRAD")]] float4 image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy);
    [[callop("BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL")]] float4 image_sample_grad(uint32 image_index, float2 uv, float2 ddx, float2 ddy, float min_mip_level);
    [[callop("BINDLESS_TEXTURE2D_READ")]] float4 image_load(uint32 image_index, uint2 coord);
    [[callop("BINDLESS_TEXTURE2D_READ_LEVEL")]] float4 image_load_level(uint32 image_index, uint2 coord, uint32 mip_level);
    [[callop("BINDLESS_TEXTURE2D_SIZE")]] uint2 image_size(uint32 image_index);
    [[callop("BINDLESS_TEXTURE2D_SIZE_LEVEL")]] uint2 image_size_level(uint32 image_index, uint32 mip_level);
    [[callop("BINDLESS_TEXTURE3D_SAMPLE")]] float4 volume_sample(uint32 volume_index, float3 uv);
    [[callop("BINDLESS_TEXTURE3D_SAMPLE_LEVEL")]] float4 volume_sample_level(uint32 volume_index, float3 uv, float mip_level);
    [[callop("BINDLESS_TEXTURE3D_SAMPLE_GRAD")]] float4 volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy);
    [[callop("BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL")]] float4 volume_sample_grad(uint32 volume_index, float3 uv, float3 ddx, float3 ddy, float min_mip_level);
    [[callop("BINDLESS_TEXTURE3D_READ")]] float4 volume_load(uint32 volume_index, uint3 coord);
    [[callop("BINDLESS_TEXTURE3D_READ_LEVEL")]] float4 volume_load_level(uint32 volume_index, uint3 coord, uint32 mip_level);
    [[callop("BINDLESS_TEXTURE3D_SIZE")]] uint3 volume_size(uint32 volume_index);
    [[callop("BINDLESS_TEXTURE3D_SIZE_LEVEL")]] uint3 volume_size_level(uint32 volume_index, uint32 mip_level);
    template <typename T>
    [[callop("BINDLESS_BUFFER_READ")]] T buffer_read(uint32 buffer_index, uint32 elem_index);
    template <typename T>
    [[callop("BINDLESS_BYTE_BUFFER_READ")]]T byte_buffer_read(uint32 buffer_index, uint32 byte_offset);
    [[callop("BINDLESS_BUFFER_SIZE")]] uint32 buffer_size(uint32 buffer_index);
};
}// namespace luisa::shader