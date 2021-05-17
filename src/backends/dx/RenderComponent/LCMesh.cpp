#include <RenderComponent/LCMesh.h>
#include <RenderComponent/StructuredBuffer.h>
#include <Singleton/MeshLayout.h>
namespace luisa::compute {
uint LCMesh::GetVertexCount() const {
	return static_cast<uint>(vertBuffer->GetElementCount(0));
}
uint LCMesh::GetVBOSRVDescIndex(GFXDevice* device) const {
	return vertBuffer->GetSRVDescIndex(device);
}
uint LCMesh::GetIBOSRVDescIndex(GFXDevice* device) const {
	return indBuffer->GetSRVDescIndex(device);
}

LCMesh::LCMesh(StructuredBuffer const* vertBuffer, StructuredBuffer const* indexBuffer)
	: vertBuffer(vertBuffer),
	  indBuffer(indexBuffer) {
	assert(vertBuffer->GetStride(0) == sizeof(float3));
	vertView.BufferLocation = vertBuffer->GetAddress(0, 0).address;
	vertView.SizeInBytes = vertBuffer->GetByteSize();
	vertView.StrideInBytes = sizeof(float3);
	indView.BufferLocation = indBuffer->GetAddress(0, 0).address;
	indView.SizeInBytes = indBuffer->GetByteSize();
	indView.Format = static_cast<DXGI_FORMAT>(GFXFormat_R32_UInt);
}
}// namespace luisa::compute