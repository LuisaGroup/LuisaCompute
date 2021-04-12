#pragma once
#include <RenderComponent/GPUResourceBase.h>
#include <Common/IObjectReference.h>
#include <RenderComponent/IGPUResourceState.h>
struct SubMesh {
	float3 boundingCenter;
	float3 boundingExtent;
	uint materialIndex;
	uint vertexOffset;
	uint indexOffset;
	uint indexCount;
};
class IMesh : public IObjectReference {
public:
	virtual uint GetIndexCount() const = 0;
	virtual uint GetIndexFormat() const = 0;
	virtual uint GetLayoutIndex() const = 0;
	virtual uint GetVertexCount() const = 0;
	virtual uint GetVBOSRVDescIndex(GFXDevice* device) const = 0;
	virtual uint GetIBOSRVDescIndex(GFXDevice* device) const = 0;
	virtual GFXVertexBufferView const* VertexBufferViews() const = 0;
	virtual uint VertexBufferViewCount() const = 0;
	virtual GFXIndexBufferView IndexBufferView() const = 0;
	virtual float3 GetBoundingCenter() const = 0;
	virtual float3 GetBoundingExtent() const = 0;

	virtual uint GetSubMeshCount() const = 0;
	virtual SubMesh const& GetSubMesh(uint i) const = 0;
};