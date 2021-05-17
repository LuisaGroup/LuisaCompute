#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <RenderComponent/IMesh.h>
#include <RenderComponent/Utility/SeparableRenderer.h>
#include <core/mathematics.h>
class Transform;
class StructuredBuffer;
namespace luisa::compute {
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
	float4x4 transformMatrix;
	ObjectPtr<IMesh> mesh;
	uint subMeshIndex;
	D3D12_RAYTRACING_INSTANCE_DESC instanceDesc;
	MeshObject meshObj;
	~RayRendererData() {}
	RayRendererData(ObjectPtr<IMesh>&& mesh)
		: mesh(std::move(mesh)) {}
	KILL_COPY_CONSTRUCT(RayRendererData)
};
}// namespace luisa::compute