//#endif
#include <RenderComponent/Texture.h>
#include <RenderComponent/DescriptorHeap.h>
#include <fstream>
#include <RenderComponent/ComputeShader.h>

#include <RenderComponent/RenderCommand.h>
#include <Singleton/ShaderID.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/TextureHeap.h>
#include <Common/Pool.h>
#include <RenderComponent/Utility/ITextureAllocator.h>
#include <Utility/BinaryReader.h>
#include <RenderComponent/UploadBuffer.h>
#include <PipelineComponent/ThreadCommand.h>
using Microsoft::WRL::ComPtr;
using namespace DirectX;
namespace TextureGlobal {
struct TextureFormat_LoadData {
	GFXFormat format;
	uint pixelSize;
	bool bcCompress;
};
//BC Format Have Different Unit
//BC Format / 4 = Pixel Size
TextureFormat_LoadData Texture_GetFormat(TextureData::LoadFormat loadFormat) {
	TextureFormat_LoadData loadData;
	loadData.bcCompress = false;
	switch (loadFormat) {
		case TextureData::LoadFormat_R8G8B8A8_UNorm:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R8G8B8A8_UNorm;
			break;
		case TextureData::LoadFormat_R16G16B16A16_UNorm:
			loadData.pixelSize = 8;
			loadData.format = GFXFormat_R16G16B16A16_UNorm;
			break;
		case TextureData::LoadFormat_R16G16B16A16_SFloat:
			loadData.pixelSize = 8;
			loadData.format = GFXFormat_R16G16B16A16_Float;
			break;
		case TextureData::LoadFormat_R32G32B32A32_SFloat:
			loadData.pixelSize = 16;
			loadData.format = GFXFormat_R32G32B32A32_Float;
			break;
		case TextureData::LoadFormat_R16G16_UNorm:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R16G16_UNorm;
			break;
		case TextureData::LoadFormat_R16G16_SFloat:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R16G16_Float;
			break;
		case TextureData::LoadFormat_BC7:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_BC7_UNorm;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_BC6H:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_BC6H_UF16;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_BC5U:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_BC5_UNorm;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_BC5S:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_BC5_SNorm;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_BC4U:
			loadData.pixelSize = 2;
			loadData.format = GFXFormat_BC4_UNorm;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_BC4S:
			loadData.pixelSize = 2;
			loadData.format = GFXFormat_BC4_SNorm;
			loadData.bcCompress = true;
			break;
		case TextureData::LoadFormat_R32_UInt:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R32_UInt;
			break;
		case TextureData::LoadFormat_R32G32_UInt:
			loadData.pixelSize = 8;
			loadData.format = GFXFormat_R32G32_UInt;
			break;
		case TextureData::LoadFormat_R32G32B32A32_UInt:
			loadData.pixelSize = 16;
			loadData.format = GFXFormat_R32G32B32A32_UInt;
			break;
		case TextureData::LoadFormat_R16_UNorm:
			loadData.pixelSize = 2;
			loadData.format = GFXFormat_R16_UNorm;
			break;
		case TextureData::LoadFormat_R16_UInt:
			loadData.pixelSize = 2;
			loadData.format = GFXFormat_R16_UInt;
			break;
		case TextureData::LoadFormat_R16G16_UInt:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R16G16_UInt;
			break;
		case TextureData::LoadFormat_R16G16B16A16_UInt:
			loadData.pixelSize = 8;
			loadData.format = GFXFormat_R16G16B16A16_UInt;
			break;
		case TextureData::LoadFormat_R8_UInt:
			loadData.pixelSize = 1;
			loadData.format = GFXFormat_R8_UInt;
			break;
		case TextureData::LoadFormat_R8G8_UInt:
			loadData.pixelSize = 2;
			loadData.format = GFXFormat_R8G8_UInt;
			break;
		case TextureData::LoadFormat_R8G8B8A8_UInt:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R8G8B8A8_UInt;
			break;
		case TextureData::LoadFormat_R32_SFloat:
			loadData.pixelSize = 4;
			loadData.format = GFXFormat_R32_Float;
			break;
	}
	return loadData;
}
void GetData(ArrayList<char>& datas, BinaryReader& ifs, uint64 offset, uint64 size) {
	datas.resize(size);
	ifs.SetPos(offset);
	ifs.Read(datas.data(), size);
}
void GetData(ArrayList<char>& datas, const vstd::string& path, uint64 offset, uint64 size) {
	BinaryReader ifs(path);
	GetData(datas, ifs, offset, size);
}
void ReadData(const vstd::string& str, ArrayList<char>& datas,
			  TextureData& headerResult, uint startMipLevel,
			  uint maximumMipLevel, uint64& targetOffset,
			  uint64& targetSize, bool startLoading, bool alreadyHaveHeader) {
	BinaryReader ifs(str);
	if (!ifs) {
		VEngine_Log("Texture Error! File Not Exists!\n");
		VENGINE_EXIT;
	}
	if (alreadyHaveHeader)
		ifs.SetPos(sizeof(TextureData));
	else
		TextureData::ReadTextureDataFromFile(ifs, headerResult);
	//Set Mip
	headerResult.mipCount = Max<uint>(headerResult.mipCount, 1);
	startMipLevel = Min(startMipLevel, headerResult.mipCount - 1);
	maximumMipLevel = Max<uint>(maximumMipLevel, 1);
	maximumMipLevel = Min(maximumMipLevel, headerResult.mipCount - startMipLevel);
	headerResult.mipCount = maximumMipLevel;
	uint formt = (uint)headerResult.format;
	if (formt >= (uint)(TextureData::LoadFormat_Num) || (uint)headerResult.textureType >= (uint)TextureDimension::Num) {
		VEngine_Log("Texture Error! Invalide Format!\n");
		VENGINE_EXIT;
	}
	uint stride = 0;
	TextureGlobal::TextureFormat_LoadData loadData = TextureGlobal::Texture_GetFormat(headerResult.format);
	stride = loadData.pixelSize;
	if (headerResult.depth != 1 && startMipLevel != 0) {
		VEngine_Log("Texture Error! Non-2D map can not use mip streaming!\n");
		VENGINE_EXIT;
	}
	uint64_t size = 0;
	uint64_t offsetSize = 0;
	uint depth = headerResult.depth;
	for (uint j = 0; j < depth; ++j) {
		uint width = headerResult.width;
		uint height = headerResult.height;
		for (uint i = 0; i < startMipLevel; ++i) {
			uint64_t currentChunkSize = 0;
			if (loadData.bcCompress)
				currentChunkSize = ((stride * (uint64_t)width + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * height / 4;
			else
				currentChunkSize = ((stride * (uint64_t)width + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * height;
			offsetSize += currentChunkSize;
			width /= 2;
			height /= 2;
			width = Max<uint>(1, width);
			height = Max<uint>(1, height);
		}
		headerResult.width = width;
		headerResult.height = height;
		for (uint i = 0; i < headerResult.mipCount; ++i) {
			uint64_t currentChunkSize = 0;
			if (loadData.bcCompress)
				currentChunkSize = ((stride * width + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * height / 4;
			else
				currentChunkSize = ((stride * (uint64_t)width + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * height;
			size += currentChunkSize;
			width /= 2;
			height /= 2;
			width = Max<uint>(1, width);
			height = Max<uint>(1, height);
		}
	}
	targetOffset = offsetSize + sizeof(TextureData);
	targetSize = size;
	if (startLoading) {
		GetData(datas, ifs, targetOffset, targetSize);
	}
}
class TextureLoadCommand : public RenderCommand {
private:
	StackObject<UploadBuffer, true> containedUbuffer;
	UploadBuffer* ubuffer;
	ObjWeakPtr<GFXResource*> resPtr;
	TextureData::LoadFormat loadFormat;
	uint width;
	uint height;
	uint mip;
	uint arraySize;
	TextureDimension type;
	bool* flag;
	GPUResourceState* initState;
	Texture* tex;

public:
	TextureLoadCommand(
		Texture* tex,
		GFXDevice* device,
		uint element,
		void* dataPtr,
		const ObjectPtr<GFXResource*>& resPtr,
		TextureData::LoadFormat loadFormat,
		uint width,
		uint height,
		uint mip,
		uint arraySize,
		TextureDimension type, IBufferAllocator* bufferAllocator, bool* flag,
		GPUResourceState* initState)
		: resPtr(resPtr), loadFormat(loadFormat),
		  tex(tex),
		  width(width), height(height),
		  mip(mip), arraySize(arraySize), type(type),
		  flag(flag), initState(initState) {
		containedUbuffer.New(device, element, false, 1, bufferAllocator);
		ubuffer = containedUbuffer;
		ubuffer->CopyDatas(0, element, dataPtr);
	}
	TextureLoadCommand(
		UploadBuffer* buffer,
		const ObjectPtr<GFXResource*> resPtr,
		TextureData::LoadFormat loadFormat,
		uint width,
		uint height,
		uint mip,
		uint arraySize,
		TextureDimension type, bool* flag,
		GPUResourceState* initState) : resPtr(resPtr), loadFormat(loadFormat),
									   width(width), height(height), mip(mip), arraySize(arraySize), type(type), ubuffer(buffer),
									   flag(flag), initState(initState) {
	}
	void Execute(
		GFXDevice* device,
		ThreadCommand* directCommandList,
		ThreadCommand* copyCommandList) override {
		if (!resPtr) return;
		auto res = *resPtr;
		uint offset = 0;
		TextureGlobal::TextureFormat_LoadData loadData = TextureGlobal::Texture_GetFormat(loadFormat);
		if (type == TextureDimension::Tex3D) {
			uint curWidth = width;
			uint curHeight = height;
			for (uint i = 0; i < mip; ++i) {
				Graphics::CopyBufferToTexture(
					copyCommandList,
					ubuffer,
					offset,
					res,
					i,
					curWidth, curHeight,
					arraySize,
					loadData.format, loadData.pixelSize);
				uint chunkOffset = (((loadData.pixelSize * curWidth) + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * curHeight;
				offset += chunkOffset;
				curWidth /= 2;
				curHeight /= 2;
			}
		} else {
			for (uint j = 0; j < arraySize; ++j) {
				uint curWidth = width;
				uint curHeight = height;
				for (uint i = 0; i < mip; ++i) {
					if (loadData.bcCompress) {
						Graphics::CopyBufferToBCTexture(
							copyCommandList,
							ubuffer,
							offset,
							res,
							(j * mip) + i,
							curWidth, curHeight,
							1,
							loadData.format, loadData.pixelSize);
					} else {
						Graphics::CopyBufferToTexture(
							copyCommandList,
							ubuffer,
							offset,
							res,
							(j * mip) + i,
							curWidth, curHeight,
							1,
							loadData.format, loadData.pixelSize);
					}
					uint chunkOffset = ((loadData.pixelSize * curWidth + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) * curHeight / 4;
					offset += chunkOffset;
					curWidth /= 2;
					curHeight /= 2;
				}
			}
		}
		uint64_t ofst = offset;
		uint64_t size = ubuffer->GetElementCount();
		*flag = true;
		*initState = GPUResourceState_GenericRead;
		directCommandList->UpdateResState(GPUResourceState_Common, GPUResourceState_GenericRead, tex);
	}
};
}// namespace TextureGlobal
void TextureData::ReadTextureDataFromFile(BinaryReader& ifs, TextureData& result) {
	ifs.SetPos(0);
	ifs.Read((char*)&result, sizeof(TextureData));
}
uint64_t Texture::GetSizeFromProperty(
	GFXDevice* device,
	uint width,
	uint height,
	uint depth,
	TextureDimension textureType,
	uint mipCount,
	GFXFormat format) {
	mipCount = Max<uint>(1, mipCount);
	if (textureType == TextureDimension::Cubemap)
		depth = 6;
	D3D12_RESOURCE_DESC texDesc;
	ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
	texDesc.Dimension = textureType == TextureDimension::Tex3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = width;
	texDesc.Height = height;
	texDesc.DepthOrArraySize = depth;
	texDesc.MipLevels = mipCount;
	texDesc.Format = (DXGI_FORMAT)format;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	return device->device()->GetResourceAllocationInfo(
							   0, 1, &texDesc)
		.SizeInBytes;
}
Texture::Texture(
	GFXDevice* device,
	ThreadCommand* commandList,
	UploadBuffer* buffer,
	uint width,
	uint height,
	uint depth,
	TextureDimension textureType,
	uint mipCount,
	TextureData::LoadFormat format,
	TextureHeap* placedHeap,
	uint64_t placedOffset) : TextureBase(device, nullptr) {
	dimension = textureType;
	fileLoadFormat = format;
	if (textureType == TextureDimension::Cubemap)
		depth = 6;
	auto loadData = TextureGlobal::Texture_GetFormat(format);
	mFormat = loadData.format;
	this->depthSlice = depth;
	this->mWidth = width;
	this->mHeight = height;
	mipCount = Max<uint>(1, mipCount);
	this->mipCount = mipCount;
	D3D12_RESOURCE_DESC texDesc;
	ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
	texDesc.Dimension = textureType == TextureDimension::Tex3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = width;
	texDesc.Height = height;
	texDesc.DepthOrArraySize = depth;
	texDesc.MipLevels = mipCount;
	texDesc.Format = (DXGI_FORMAT)mFormat;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	resourceSize = device->device()->GetResourceAllocationInfo(
									   0, 1, &texDesc)
					   .SizeInBytes;
	if (placedHeap) {
		ThrowIfFailed(device->device()->CreatePlacedResource(
			placedHeap->GetHeap(),
			placedOffset,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->device()->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	resTracker = ObjectPtr<GFXResource*>::MakePtrNoMemoryFree(Resource.GetAddressOf());
	TextureGlobal::TextureLoadCommand cmd(
		buffer,
		resTracker,
		format,
		width,
		height,
		mipCount,
		depth,
		dimension, &loaded, &initState);
	cmd.Execute(device, commandList, commandList);
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
D3D12_RESOURCE_DESC Texture::CreateWithoutResource(
	TextureData& data,
	GFXDevice* device,
	const vstd::string& filePath,
	bool startLoadNow,
	bool alreadyHaveTextureData,
	ArrayList<char>* datasPtr,
	TextureDimension type,
	uint32_t maximumLoadMipmap,
	uint32_t startMipMap) {
	dimension = type;
	targetFilePath = filePath;
	TextureGlobal::ReadData(filePath, *datasPtr, data, startMipMap, maximumLoadMipmap, fileReadOffset, fileReadSize, startLoadNow, alreadyHaveTextureData);
	if ((size_t)data.format >= (size_t)TextureData::LoadFormat_Num || data.textureType != type) {
		VEngine_Log("Texture Type Not Match Exception\n");
		VENGINE_EXIT;
	}
	if (type == TextureDimension::Cubemap && data.depth != 6) {
		VEngine_Log("Cubemap's tex size must be 6\n");
		VENGINE_EXIT;
	}
	if (data.mipCount > 14) {
		VEngine_Log("Too Many Mipmap!");
		VENGINE_EXIT;
	}
	if (data.width > 8192 || data.height > 8192) {
		VEngine_Log("Texture Too Large!");
		VENGINE_EXIT;
	}
	auto loadData = TextureGlobal::Texture_GetFormat(data.format);
	mFormat = loadData.format;
	this->depthSlice = data.depth;
	this->mWidth = data.width;
	this->mHeight = data.height;
	this->mipCount = data.mipCount;
	D3D12_RESOURCE_DESC texDesc;
	ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
	texDesc.Dimension = type == TextureDimension::Tex3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = data.width;
	texDesc.Height = data.height;
	texDesc.DepthOrArraySize = data.depth;
	texDesc.MipLevels = data.mipCount;
	texDesc.Format = (DXGI_FORMAT)mFormat;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	resourceSize = device->device()->GetResourceAllocationInfo(
									   0, 1, &texDesc)
					   .SizeInBytes;
	return texDesc;
}
void Texture::LoadTexture(GFXDevice* device, IBufferAllocator* allocator, ArrayList<char>* datasPtr) {
	if (isStartLoad) return;
	isStartLoad = true;
	ArrayList<char> datas;
	if (datasPtr->empty())
		TextureGlobal::GetData(*datasPtr, targetFilePath, fileReadOffset, fileReadSize);
	TextureGlobal::TextureLoadCommand* cmd;
	resTracker = ObjectPtr<GFXResource*>::MakePtrNoMemoryFree(Resource.GetAddressOf());
	cmd = new TextureGlobal::TextureLoadCommand(
		this,
		device,
		datasPtr->size(),
		datasPtr->data(),
		resTracker,
		fileLoadFormat,
		mWidth,
		mHeight,
		mipCount,
		depthSlice,
		dimension, allocator, &loaded, &initState);
	RenderCommand::UpdateResState(cmd);
}
Texture::Texture(
	GFXDevice* device,
	uint width,
	uint height,
	uint depth,
	TextureDimension textureType,
	uint mipCount,
	GFXFormat format,
	TextureHeap* placedHeap,
	uint64_t placedOffset) : TextureBase(device, nullptr) {
	dimension = textureType;
	if (textureType == TextureDimension::Cubemap)
		depth = 6;
	mFormat = format;
	this->depthSlice = depth;
	this->mWidth = width;
	this->mHeight = height;
	mipCount = Max<uint>(1, mipCount);
	this->mipCount = mipCount;
	D3D12_RESOURCE_DESC texDesc;
	ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
	texDesc.Dimension = textureType == TextureDimension::Tex3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = width;
	texDesc.Height = height;
	texDesc.DepthOrArraySize = depth;
	texDesc.MipLevels = mipCount;
	texDesc.Format = (DXGI_FORMAT)mFormat;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	resourceSize = device->device()->GetResourceAllocationInfo(
									   0, 1, &texDesc)
					   .SizeInBytes;
	if (placedHeap) {
		ThrowIfFailed(device->device()->CreatePlacedResource(
			placedHeap->GetHeap(),
			placedOffset,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->device()->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
Texture::Texture(
	GFXDevice* device,
	const vstd::string& filePath,
	ITextureAllocator* allocator,
	IBufferAllocator* bufferAllocator,
	bool startLoading,
	TextureDimension type,
	uint32_t maximumLoadMipmap,
	uint32_t startMipMap,
	ArrayList<char>* datasPtr) : allocator(allocator), TextureBase(device, allocator) {
	ArrayList<char> datas;
	if (!datasPtr) datasPtr = &datas;
	else
		datasPtr->clear();
	ID3D12Heap* placedHeap;
	uint64_t placedOffset;
	TextureData data;
	auto texDesc = CreateWithoutResource(data, device, filePath, startLoading, false, datasPtr, type, maximumLoadMipmap, startMipMap);
	fileLoadFormat = data.format;
	allocator->AllocateTextureHeap(device, mFormat, mWidth, mHeight, data.depth, type, data.mipCount, &placedHeap, &placedOffset, false, GetInstanceID());
	ThrowIfFailed(device->device()->CreatePlacedResource(
		placedHeap,
		placedOffset,
		&texDesc,
		D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
		nullptr,
		IID_PPV_ARGS(&Resource)));
	if (startLoading) {
		LoadTexture(device, bufferAllocator, datasPtr);
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
Texture::Texture(
	GFXDevice* device,
	const vstd::string& filePath,
	bool startLoading,
	TextureDimension type,
	uint32_t maximumLoadMipmap,
	uint32_t startMipMap,
	TextureHeap* placedHeap,
	uint64_t placedOffset,
	ArrayList<char>* datasPtr) : TextureBase(device, nullptr) {
	TextureData data;
	ArrayList<char> datas;
	if (!datasPtr) datasPtr = &datas;
	else
		datasPtr->clear();
	auto texDesc = CreateWithoutResource(data, device, filePath, startLoading, false, datasPtr, type, maximumLoadMipmap, startMipMap);
	fileLoadFormat = data.format;
	if (placedHeap) {
		ThrowIfFailed(device->device()->CreatePlacedResource(
			placedHeap->GetHeap(),
			placedOffset,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->device()->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	if (startLoading) {
		LoadTexture(device, nullptr, datasPtr);
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
Texture::Texture(
	GFXDevice* device,
	const vstd::string& filePath,
	const TextureData& texData,
	bool startLoading,
	uint32_t maximumLoadMipmap,
	uint32_t startMipMap,
	TextureHeap* placedHeap,
	uint64_t placedOffset,
	ArrayList<char>* datasPtr) : TextureBase(device, nullptr) {
	TextureData data;
	memcpy(&data, &texData, sizeof(TextureData));
	ArrayList<char> datas;
	if (!datasPtr) datasPtr = &datas;
	else
		datasPtr->clear();
	auto texDesc = CreateWithoutResource(data, device, filePath, startLoading, true, datasPtr, data.textureType, maximumLoadMipmap, startMipMap);
	fileLoadFormat = data.format;
	if (placedHeap) {
		ThrowIfFailed(device->device()->CreatePlacedResource(
			placedHeap->GetHeap(),
			placedOffset,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->device()->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	if (startLoading) {
		LoadTexture(device, nullptr, datasPtr);
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
Texture::Texture(
	GFXDevice* device,
	const vstd::string& filePath,
	const TextureData& texData,
	ITextureAllocator* allocator,
	IBufferAllocator* bufferAllocator,
	bool startLoading,
	uint32_t maximumLoadMipmap,
	uint32_t startMipMap,
	ArrayList<char>* datasPtr) : allocator(allocator), TextureBase(device, allocator) {
	ID3D12Heap* placedHeap;
	uint64_t placedOffset;
	TextureData data;
	memcpy(&data, &texData, sizeof(TextureData));
	ArrayList<char> datas;
	if (!datasPtr) datasPtr = &datas;
	else
		datasPtr->clear();
	auto texDesc = CreateWithoutResource(data, device, filePath, startLoading, true, datasPtr, data.textureType, maximumLoadMipmap, startMipMap);
	fileLoadFormat = data.format;
	allocator->AllocateTextureHeap(device, mFormat, mWidth, mHeight, data.depth, data.textureType, data.mipCount, &placedHeap, &placedOffset, false, GetInstanceID());
	ThrowIfFailed(device->device()->CreatePlacedResource(
		placedHeap,
		placedOffset,
		&texDesc,
		D3D12_RESOURCE_STATE_COMMON,//GPUResourceState_UnorderedAccess,
		nullptr,
		IID_PPV_ARGS(&Resource)));
	if (startLoading) {
		LoadTexture(device, bufferAllocator, datasPtr);
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
Texture::~Texture() {
	if (allocator) {
		allocator->Release(GetInstanceID());
	}
}
void Texture::GetResourceViewDescriptor(D3D12_SHADER_RESOURCE_VIEW_DESC& srvDesc) const {
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	auto format = mFormat;
	switch (dimension) {
		case TextureDimension::Tex2D:
			srvDesc.Format = (DXGI_FORMAT)format;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Texture2D.MipLevels = mipCount;
			srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Tex3D:
			srvDesc.Format = (DXGI_FORMAT)format;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
			srvDesc.Texture3D.MipLevels = mipCount;
			srvDesc.Texture3D.MostDetailedMip = 0;
			srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Cubemap:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
			srvDesc.TextureCube.MostDetailedMip = 0;
			srvDesc.TextureCube.MipLevels = mipCount;
			srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
			srvDesc.Format = (DXGI_FORMAT)format;
			break;
	}
}
void Texture::BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	GetResourceViewDescriptor(srvDesc);
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
