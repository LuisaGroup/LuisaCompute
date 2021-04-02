#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include "IMesh.h"
#include <RenderComponent/Utility/SeparableRenderer.h>

class Transform;
class StructuredBuffer;
class RayRendererData final : public SeparableRenderer {
public:
	struct MeshObject {
		uint vboDescIndex;
		uint iboDescIndex;
		uint vertexOffset;
		uint indexOffset;
		uint shaderID;
		uint materialID;
	};
	Transform* trans;
	ObjectPtr<IMesh> mesh;
	uint subMeshIndex;
	D3D12_RAYTRACING_INSTANCE_DESC instanceDesc;
	MeshObject meshObj;
	~RayRendererData() {}
	RayRendererData(ObjectPtr<IMesh>&& mesh)
		: mesh(std::move(mesh)) {}
	KILL_COPY_CONSTRUCT(RayRendererData)
};