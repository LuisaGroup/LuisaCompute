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
	vstd::string name;
	uint shaderType;
	vstd::string functions[(uint8_t)HitGroupFunctionType::Num];

	GpuAddress missVAddress;
	uint64 missSize;
	GpuAddress rayGenVAddress;
	uint64 rayGenSize;
	GpuAddress hitGroupVAddress;
	uint64 hitGroupSize;
};

struct CompileDXRHitGroup {
	vstd::string name;
	uint shaderType;
	vstd::string functions[(uint8_t)HitGroupFunctionType::Num];
};
struct ShaderVariable {
	uint varID;
	ShaderVariableType type;
	uint tableSize;
	uint registerPos;
	uint space;
	ShaderVariable() {}
	ShaderVariable(
		uint varID,
		ShaderVariableType type,
		uint tableSize,
		uint registerPos,
		uint space) : varID(varID),
					  type(type),
					  tableSize(tableSize),
					  registerPos(registerPos),
					  space(space) {}
};