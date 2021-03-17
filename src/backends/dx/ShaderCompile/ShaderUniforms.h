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
	vengine::string vsShaderr;
	vengine::string psShader;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};

struct KernelDescriptor {
	vengine::string name;
	ObjectPtr<vengine::vector<vengine::string>> macros;
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
	vengine::string name;
	vengine::string vertex;
	vengine::string fragment;
	vengine::string hull;
	vengine::string domain;
	ObjectPtr<vengine::vector<vengine::string>> macros;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};
}// namespace SCompile