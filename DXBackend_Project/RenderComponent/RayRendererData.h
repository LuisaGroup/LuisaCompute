#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/VObject.h"
#include "IMesh.h"
class StructuredBuffer;
class RayRendererData {
public:
	struct MeshObject {
		uint vboDescIndex;
		uint iboDescIndex;
		uint vertexOffset;
		uint indexOffset;
		uint shaderID;
		uint materialID;
	};

	ObjectPtr<IMesh> mesh;
	uint subMeshIndex;
	D3D12_RAYTRACING_INSTANCE_DESC instanceDesc;
	MeshObject meshObj;
	uint listIndex = -1;
	RayRendererData(ObjectPtr<IMesh>&& mesh)
		: mesh(std::move(mesh)) {}
	KILL_COPY_CONSTRUCT(RayRendererData)
};