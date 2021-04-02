#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <CJsonObject/SerializedObject.h>
#include <Struct/ShaderVariableType.h>
class DescriptorHeap;
class StructuredBuffer;
class UploadBuffer;
class Mesh;
class TextureBase;
class RenderTexture;
class ShaderLoader;
class ThreadCommand;
class VENGINE_DLL_RENDERER IShader : public VObject {
public:
	enum class ResourceType : uint {
		NONE = 0,			 
		DESCRIPTOR_HEAP = 1, 
		TEXTURE = 2,		 
		STRUCTURE_BUFFER = 4,
		UPLOAD_BUFFER = 8,	 
		MESH = 16			 
	};

protected:
	virtual bool SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, ResourceType tyid) const = 0;

public:
	~IShader() {}
	virtual SerializedObject const* GetJsonObject() const = 0;
	virtual size_t VariableLength() const = 0;
	virtual int32_t GetPropertyRootSigPos(uint id) const = 0;
	virtual void BindShader(ThreadCommand* commandList) const = 0;
	virtual void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const = 0;
	virtual bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const = 0;
	virtual vengine::string const& GetName() const = 0;
	bool SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* descHeap, uint64 elementOffset) const;
	bool SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* buffer, uint64 elementOffset) const;
	bool SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* buffer, uint64 elementOffset) const;
	bool SetResource(ThreadCommand* commandList, uint id, Mesh const* mesh, uint64 byteOffset) const;
	bool SetResource(ThreadCommand* commandList, uint id, TextureBase const* texture) const;
	bool SetResource(ThreadCommand* commandList, uint id, RenderTexture const* renderTexture, uint64 uavMipLevel) const;
};