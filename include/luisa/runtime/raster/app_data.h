#pragma once

#include <array>
#include <luisa/core/basic_types.h>

namespace luisa::compute {

// The vertex input a DSL `RasterKernel`'s vertex stage receives (`Var<AppData>`).
//
// The member order *is* the pipeline's vertex-input contract: it must describe the
// same attributes, in the same order, as the draw's `MeshFormat` (position, normal,
// tangent, color, uv0..uv3). `vertex_id` and `instance_id` are the draw's builtins,
// filled by the backend. The mesh's `object_id` is not here: it reaches the shader
// through the push-constant block.
//
// This type belongs to the DSL path only. A native raster shader
// (`NativeShaderExt::compile_raster`) declares its own inputs, bound at
// `location = static_cast<uint32_t>(VertexAttributeType)`.
struct AppData {
    float3 position;
    float3 normal;
    float4 tangent;
    float4 color;
    std::array<float2, 4> uv;
    uint32_t vertex_id;
    uint32_t instance_id;
};

}// namespace luisa::compute
