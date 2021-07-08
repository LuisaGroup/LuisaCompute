#pragma once
#include <Common/GFXUtil.h>
#include <Struct/ShaderVariableType.h>
namespace SCompile {
enum class ShaderType : uint8_t {
	ComputeShader,
	VertexShader,
	PixelShader,
	HullShader,
	DomainShader,
	GeometryShader,
	RayTracingShader,
	ShaderTypeNum
};
enum class ShaderFileType : uint8_t {
	None,
	Shader,
	ComputeShader,
	DXRShader
};
struct Pass {
	vstd::string vsShaderr;
	vstd::string psShader;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};

enum class HitGroupFunctionType : uint8_t {
	RayGeneration,
	ClosestHit,
	AnyHit,
	Miss,
	Intersect,
	Num
};


struct PassDescriptor {
	vstd::string name;
	vstd::string vertex;
	vstd::string fragment;
	vstd::string hull;
	vstd::string domain;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};
}// namespace SCompile