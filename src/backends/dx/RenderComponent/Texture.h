#pragma once
#include <Common/VObject.h>
#include <Common/vector.h>
#include <RenderComponent/TextureBase.h>
#include <Common/Memory.h>
class DescriptorHeap;
class UploadBuffer;
class TextureHeap;
class ITextureAllocator;
class IBufferAllocator;
class BinaryReader;
class ThreadCommand;

class VENGINE_DLL_RENDERER TextureData
{
public:
	uint width;
	uint height;
	uint depth;
	TextureDimension textureType;
	uint mipCount;
	enum LoadFormat
	{
		LoadFormat_R8G8B8A8_UNorm = 0,
		LoadFormat_R16G16B16A16_UNorm = 1,
		LoadFormat_R16G16B16A16_SFloat = 2,
		LoadFormat_R32G32B32A32_SFloat = 3,
		LoadFormat_R16G16_SFloat = 4,
		LoadFormat_R16G16_UNorm = 5,
		LoadFormat_BC7 = 6,
		LoadFormat_BC6H = 7,
		LoadFormat_R32_UInt = 8,
		LoadFormat_R32G32_UInt = 9,
		LoadFormat_R32G32B32A32_UInt = 10,
		LoadFormat_R16_UNorm = 11,
		LoadFormat_BC5U = 12,
		LoadFormat_BC5S = 13,
		LoadFormat_R16_UInt = 14,
		LoadFormat_R16G16_UInt = 15,
		LoadFormat_R16G16B16A16_UInt = 16,
		LoadFormat_R8_UInt = 17,
		LoadFormat_R8G8_UInt = 18,
		LoadFormat_R8G8B8A8_UInt = 19,
		LoadFormat_R32_SFloat = 20,
		LoadFormat_BC4U = 21,
		LoadFormat_BC4S = 22,
		LoadFormat_Num = 23
	};
	LoadFormat format;
	static void ReadTextureDataFromFile(BinaryReader& ifs,TextureData& result);
};

class VENGINE_DLL_RENDERER Texture final : public TextureBase
{
private:
	void GetResourceViewDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const;
	ITextureAllocator* allocator = nullptr;
	bool loaded = false;
	//Loading Parameters
	TextureData::LoadFormat fileLoadFormat;
	bool isStartLoad = false;
	uint64 fileReadOffset = 0;
	uint64 fileReadSize = 0;
	ObjectPtr<GFXResource*> resTracker;
	vengine::string targetFilePath;
	GPUResourceState initState = GPUResourceState_Common;
	Texture() {}
	D3D12_RESOURCE_DESC CreateWithoutResource(
		TextureData& data,
		GFXDevice* device,
		const vengine::string& filePath,
		bool startLoadNow,
		bool alreadyHaveTextureData,
		ArrayList<char>* datasPtr,
		TextureDimension type = TextureDimension::Tex2D,
		uint32_t maximumLoadMipmap = -1,
		uint32_t startMipMap = 0
	);

public:
	KILL_COPY_CONSTRUCT(Texture)
	//Async Load
	Texture(
		GFXDevice* device,
		const vengine::string& filePath,
		bool startLoading = true,
		TextureDimension type = TextureDimension::Tex2D,
		uint32_t maximumLoadMipmap = -1,
		uint32_t startMipMap = 0,
		TextureHeap* placedHeap = nullptr,
		uint64_t placedOffset = 0,
		ArrayList<char>* datasPtr = nullptr
	);
	Texture(
		GFXDevice* device,
		const vengine::string& filePath,
		ITextureAllocator* allocator,
		IBufferAllocator* bufferAllocator = nullptr,
		bool startLoading = true,
		TextureDimension type = TextureDimension::Tex2D,
		uint32_t maximumLoadMipmap = -1,
		uint32_t startMipMap = 0,
		ArrayList<char>* datasPtr = nullptr
	);

	Texture(
		GFXDevice* device,
		const vengine::string& filePath,
		const TextureData& texData,
		bool startLoading = true,
		uint32_t maximumLoadMipmap = -1,
		uint32_t startMipMap = 0,
		TextureHeap* placedHeap = nullptr,
		uint64_t placedOffset = 0,
		ArrayList<char>* datasPtr = nullptr
	);
	Texture(
		GFXDevice* device,
		const vengine::string& filePath,
		const TextureData& texData,
		ITextureAllocator* allocator,
		IBufferAllocator* bufferAllocator = nullptr,
		bool startLoading = true,
		uint32_t maximumLoadMipmap = -1,
		uint32_t startMipMap = 0,
		ArrayList<char>* datasPtr = nullptr
	);

	Texture(
		GFXDevice* device,
		uint width,
		uint height,
		uint depth,
		TextureDimension textureType,
		uint mipCount,
		GFXFormat format,
		TextureHeap* placedHeap = nullptr,
		uint64_t placedOffset = 0
	);

	//Sync Copy
	Texture(
		GFXDevice* device,
		ThreadCommand* commandList,
		UploadBuffer* buffer,
		uint width,
		uint height,
		uint depth,
		TextureDimension textureType,
		uint mipCount,
		TextureData::LoadFormat format,
		TextureHeap* placedHeap = nullptr,
		uint64_t placedOffset = 0
	);
	~Texture();
	static uint64_t GetSizeFromProperty(
		GFXDevice* device,
		uint width,
		uint height,
		uint depth,
		TextureDimension textureType,
		uint mipCount,
		GFXFormat format);
	void LoadTexture(GFXDevice* device, IBufferAllocator* allocator, ArrayList<char>* datasPtr);
	bool IsLoaded() const { return loaded; }
	bool IsStartLoad() const { return isStartLoad; }
	virtual void BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const;
	virtual GPUResourceState GetInitState() const
	{
		return initState;
	}
};
