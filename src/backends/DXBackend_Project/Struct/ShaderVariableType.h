#pragma once
#include <Common/Common.h>
enum class ShaderVariableType : uint8_t {
	ConstantBuffer,
	SRVDescriptorHeap,
	UAVDescriptorHeap,
	StructuredBuffer,
	RWStructuredBuffer
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
	vengine::string functions[(uint8_t)HitGroupFunctionType::Num];

	GpuAddress missVAddress;
	uint64 missSize;
	GpuAddress rayGenVAddress;
	uint64 rayGenSize;
	GpuAddress hitGroupVAddress;
	uint64 hitGroupSize;
};

struct CompileDXRHitGroup {
	vengine::string name;
	uint shaderType;
	vengine::string functions[(uint8_t)HitGroupFunctionType::Num];
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
	ShaderVariable(
		vengine::string&& name,
		ShaderVariableType type,
		uint tableSize,
		uint registerPos,
		uint space) : name(std::move(name)),
					  type(type),
					  tableSize(tableSize),
					  registerPos(registerPos),
					  space(space) {}
};