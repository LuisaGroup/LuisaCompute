#pragma once
#include <Common/GFXUtil.h>
#include <util/VObject.h>
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
	IMesh* mesh;
	D3D12_RAYTRACING_INSTANCE_DESC instanceDesc;
	MeshObject meshObj;
	~RayRendererData() {}
	RayRendererData(IMesh* mesh)
		: mesh(mesh) {}
	VSTL_DELETE_COPY_CONSTRUCT(RayRendererData)
};
}// namespace luisa::compute
