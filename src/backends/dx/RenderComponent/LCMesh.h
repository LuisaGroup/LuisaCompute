#pragma once
#include <Common/GFXUtil.h>
#include <RenderComponent/IMesh.h>
#include <core/vstl/VObject.h>
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
	uint GetIndexCount() const override;

	LCMesh(
		StructuredBuffer const* vertBuffer,
		StructuredBuffer const* indexBuffer,
		size_t vertBufferOffset,
		size_t indBufferOffset,
		uint vertCount,
		uint indCount);
	VENGINE_IOBJREF_OVERRIDE
private:
	uint layoutIndex;
	StructuredBuffer const* vertBuffer;
	StructuredBuffer const* indBuffer;
	GFXIndexBufferView indView;
	GFXVertexBufferView vertView;
};
}// namespace luisa::compute