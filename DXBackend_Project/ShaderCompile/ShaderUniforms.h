#pragma once
#include <Common/GFXUtil.h>
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
enum class ShaderVariableType : uint8_t {
	ConstantBuffer,
	SRVDescriptorHeap,
	UAVDescriptorHeap,
	StructuredBuffer,
	RWStructuredBuffer
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
struct DXRHitGroup {
	vengine::string name;
	uint shaderType;
	vengine::string functionIndex[(uint8_t)HitGroupFunctionType::Num];
};
struct ShaderVariable {
	vengine::string name;
	ShaderVariableType type;
	uint tableSize;
	uint registerPos;
	uint space;
	ShaderVariable() {}
	ShaderVariable(
		const vengine::string& name,
		ShaderVariableType type,
		uint tableSize,
		uint registerPos,
		uint space) : name(name),
					  type(type),
					  tableSize(tableSize),
					  registerPos(registerPos),
					  space(space) {}
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