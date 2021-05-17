#pragma once
#include <Common/GFXUtil.h>
#include <RenderComponent/IMesh.h>
#include <Common/VObject.h>
class StructuredBuffer;
namespace luisa::compute {
class LCMesh final : public VObject, public IMesh {
public:
	GFXFormat GetIndexFormat() const override { return GFXFormat_R32_UInt; }//LC use fixed index buffer format
	uint GetLayoutIndex() const override { return layoutIndex; }
	uint GetVertexCount() const override;
	uint GetVBOSRVDescIndex(GFXDevice* device) const override;
	uint GetIBOSRVDescIndex(GFXDevice* device) const override;
	GFXVertexBufferView const* VertexBufferViews() const override { return &vertView; }
	uint VertexBufferViewCount() const override { return 1; }
	GFXIndexBufferView const* IndexBufferView() const override { return &indView; }
	float3 GetBoundingCenter() const override { return subMesh.boundingCenter; }
	float3 GetBoundingExtent() const override { return subMesh.boundingExtent; }
	uint GetSubMeshCount() const override { return 1; }
	SubMesh const& GetSubMesh(uint i) const override { return subMesh; }
	uint GetIndexCount() const override;

	LCMesh(
		SubMesh const& mesh,
		StructuredBuffer const* vertBuffer,
		StructuredBuffer const* indexBuffer);
	VENGINE_IOBJREF_OVERRIDE
private:
	uint layoutIndex;
	StructuredBuffer const* vertBuffer;
	StructuredBuffer const* indBuffer;
	GFXIndexBufferView indView;
	GFXVertexBufferView vertView;
	SubMesh subMesh;
};
}// namespace luisa::compute