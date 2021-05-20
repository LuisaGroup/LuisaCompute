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
uint LCMesh::GetIndexCount() const {
	return indBuffer->GetElementCount(0);
}
LCMesh::LCMesh(
	StructuredBuffer const* vertBuffer,
	StructuredBuffer const* indexBuffer,
	size_t vertBufferOffset,
	size_t indBufferOffset,
	uint vertCount,
	uint indCount)
	: vertBuffer(vertBuffer),
	  indBuffer(indexBuffer) {
	vertView.BufferLocation = vertBuffer->GetAddress(0, vertBufferOffset).address;
	vertView.SizeInBytes = vertCount * sizeof(float3);
	vertView.StrideInBytes = sizeof(float3);
	indView.BufferLocation = indBuffer->GetAddress(0, indBufferOffset).address;
	indView.SizeInBytes = indCount * sizeof(uint);
	indView.Format = static_cast<DXGI_FORMAT>(GFXFormat_R32_UInt);
}
}// namespace luisa::compute