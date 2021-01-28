#pragma once
#include "../Common/Common.h"
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
	Num
};
struct DXRHitGroup {
	vengine::string name;
	vengine::string functions[(uint8_t)HitGroupFunctionType::Num];
	GpuAddress missVAddress;
	uint64 missSize;
	GpuAddress rayGenVAddress;
	uint64 rayGenSize;
	GpuAddress hitGroupVAddress;
	uint64 hitGroupSize;
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
		uint space)
		: name(name),
		  type(type),
		  tableSize(tableSize),
		  registerPos(registerPos),
		  space(space) {}
};