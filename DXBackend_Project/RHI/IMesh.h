#pragma once
#include "../Common/Common.h"
namespace lc_rhi {
class IGFXDevice;
class IGpuResource;
struct SubMesh {
	float3 boundingCenter;
	float3 boundingExtent;
	uint materialIndex;
	uint vertexOffset;
	uint indexOffset;
	uint indexCount;
};
class IMesh{
public:
	virtual uint GetIndexCount() const = 0;
	virtual uint GetVertexCount() const = 0;
	virtual uint GetVertexStride() const = 0;
	virtual float3 GetBoundingCenter() const = 0;
	virtual float3 GetBoundingExtent() const = 0;
	virtual void SetBoundingCenter(float3 const& center) = 0;
	virtual void SetBoundingExtent(float3 const& extent) = 0;
	virtual uint GetSubMeshCount() const = 0;
	virtual SubMesh const& GetSubMesh(uint i) const = 0;
};
}// namespace lc_rhi