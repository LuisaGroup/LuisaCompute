#pragma once
#include <luisa/vstl/common.h>
#include <Windows.h>
#include <d3dx12.h>
#include "../../common/hlsl/shader_property.h"
namespace lc::dx {

enum class HitGroupFunctionType : uint8_t {
    RayGeneration,
    ClosestHit,
    AnyHit,
    Miss,
    Intersect,
    Num
};
struct DXRHitGroup {
    D3D12_HIT_GROUP_TYPE shaderType;
    D3D12_GPU_VIRTUAL_ADDRESS missVAddress;
    uint64 missSize;
    D3D12_GPU_VIRTUAL_ADDRESS rayGenVAddress;
    uint64 rayGenSize;
    D3D12_GPU_VIRTUAL_ADDRESS hitGroupVAddress;
    uint64 hitGroupSize;
};
struct ShaderVariable {
    vstd::string name;
    hlsl::ShaderVariableType type;
    uint tableSize;
    uint registerPos;
    uint space;
    ShaderVariable() {}
    ShaderVariable(
        const vstd::string &name,
        hlsl::ShaderVariableType type,
        uint tableSize,
        uint registerPos,
        uint space)
        : name(name),
          type(type),
          tableSize(tableSize),
          registerPos(registerPos),
          space(space) {}
};
}// namespace lc::dx
