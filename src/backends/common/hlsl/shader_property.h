#pragma once
#include <luisa/core/basic_traits.h>
namespace lc::hlsl {
using namespace luisa;
enum class ShaderVariableType : uint8_t {
    ConstantBuffer,    // e.g. cbuffer MyConstantBuffer : register(b0)
    SRVTextureHeap,    // e.g. Texture2D<float4> textures[] : register(t0, space1)
    UAVTextureHeap,    // e.g. RWTexture2D<float4> textures[] : register(u0, space1)
    SRVBufferHeap,     // e.g. StructuredBuffer<MyStruct> buffers[] : register(t0, space1)
    UAVBufferHeap,     // e.g. RWStructuredBuffer<MyStruct> buffers[] : register(u0, space1)
    CBVBufferHeap,     // e.g. ConstantBuffer<MyStruct> cbuffers[] : register(b0, space1)
    SamplerHeap,       // e.g. SamplerState samplers[16] : register(s0)
    StructuredBuffer,  // e.g. StructuredBuffer<MyStruct> : register(t0)
    RWStructuredBuffer,// e.g. RWStructuredBuffer<MyStruct> : register(u0)
    ConstantValue      // e.g. uint value : register(b0)
};
struct Property {
    ShaderVariableType type;
    uint space_index;
    uint register_index;
    uint array_size;
};
}// namespace lc::hlsl
