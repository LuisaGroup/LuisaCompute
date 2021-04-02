#include "IShader.h"
#include "DescriptorHeap.h"
#include "UploadBuffer.h"
#include "StructuredBuffer.h"
#include "Mesh.h"
#include "RenderTexture.h"
#include "TextureBase.h"
bool IShader::SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* targetObj, uint64 indexOffset) const {
	return SetRes(commandList, id, targetObj, indexOffset, ResourceType::DESCRIPTOR_HEAP);
}
bool IShader::SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* targetObj, uint64 indexOffset) const {
	return SetRes(commandList, id, targetObj, indexOffset, ResourceType::UPLOAD_BUFFER);
}
bool IShader::SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* targetObj, uint64 indexOffset) const {
	return SetRes(commandList, id, targetObj, indexOffset, ResourceType::STRUCTURE_BUFFER);
}
bool IShader::SetResource(ThreadCommand* commandList, uint id, Mesh const* targetObj, uint64 indexOffset) const {
	return SetRes(commandList, id, targetObj, indexOffset, ResourceType::MESH);
}
bool IShader::SetResource(ThreadCommand* commandList, uint id, TextureBase const* targetObj) const {
	return SetRes(commandList, id, targetObj, 0, ResourceType::TEXTURE);
}
bool IShader::SetResource(ThreadCommand* commandList, uint id, RenderTexture const* targetObj, uint64 uavMipLevel) const {
	return SetRes(commandList, id, targetObj, uavMipLevel, ResourceType::TEXTURE);
}