#include <Singleton/Graphics.h>

#include <Singleton/ShaderID.h>
#include <RenderComponent/ComputeShader.h>
#include <Singleton/MeshLayout.h>
#include <RenderComponent/DescriptorHeap.h>
#include <RenderComponent/StructuredBuffer.h>
#include <RenderComponent/RenderTexture.h>
#include <RenderComponent/Texture.h>
#include <Common/Memory.h>
#include <Utility/QuickSort.h>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/UploadBuffer.h>
#include <RenderComponent/ReadbackBuffer.h>
#include <Utility/QuickSort.h>
#define MAXIMUM_HEAP_COUNT 65536
thread_local Graphics* Graphics::current = nullptr;

Graphics::Graphics(GFXDevice* device)
	: usedDescs(MAXIMUM_HEAP_COUNT),
	  unusedDescs(MAXIMUM_HEAP_COUNT) {
	//using namespace GraphicsGlobalN;
	/*_ReadBuffer_K = ShaderID::PropertyToID("_ReadBuffer_K");
	_WriteBuffer_K = ShaderID::PropertyToID("_WriteBuffer_K");
	_ReadBuffer = ShaderID::PropertyToID("_ReadBuffer");
	_WriteBuffer = ShaderID::PropertyToID("_WriteBuffer");
	copyBufferCS = ShaderLoader::GetComputeShader("CopyBufferRegion");*/

	MeshLayout::Initialize();
	globalDescriptorHeap = std::unique_ptr<DescriptorHeap>(new DescriptorHeap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, MAXIMUM_HEAP_COUNT, true));
	static constexpr uint INIT_RTV_SIZE = 2048;
#pragma loop(hint_parallel(0))
	for (uint i = 0; i < MAXIMUM_HEAP_COUNT; ++i) {
		unusedDescs[i] = i;
	}
	std::array<float3, 3> vertex;
	std::array<float2, 3> uv;
	vertex[0] = {-3, -1, 1};
	vertex[1] = {1, -1, 1};
	vertex[2] = {1, 3, 1};
	uv[0] = {-1, 1};
	uv[1] = {1, 1};
	uv[2] = {1, -1};
	std::array<INT32, 3> indices = {0, 1, 2};
	
}
uint Graphics::GetDescHeapIndexFromPool() {
	std::lock_guard lck(current->mtx);
	if (current->unusedDescs.empty()) {
		VEngine_Log("No Global Descriptor Index Lefted!\n");
		throw 0;
	}
	uint value = current->unusedDescs.erase_last();
	current->usedDescs[value] = true;
	return value;
}
void Graphics::CopyTexture(
	ThreadCommand* commandList,
	RenderTexture const* source, uint sourceSlice, uint sourceMipLevel,
	RenderTexture const* dest, uint destSlice, uint destMipLevel) {
	commandList->ExecuteResBarrier();
	if (source->GetDimension() == TextureDimension::Tex2D) sourceSlice = 0;
	if (dest->GetDimension() == TextureDimension::Tex2D) destSlice = 0;
	D3D12_TEXTURE_COPY_LOCATION sourceLocation;
	sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	sourceLocation.SubresourceIndex = sourceSlice * source->GetMipCount() + sourceMipLevel;
	D3D12_TEXTURE_COPY_LOCATION destLocation;
	destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	destLocation.SubresourceIndex = destSlice * dest->GetMipCount() + destMipLevel;
	sourceLocation.pResource = source->GetResource();
	destLocation.pResource = dest->GetResource();
	commandList->GetCmdList()->CopyTextureRegion(
		&destLocation,
		0, 0, 0,
		&sourceLocation,
		nullptr);
}
void Graphics::CopyBufferToBCTexture(
	ThreadCommand* commandList,
	UploadBuffer* sourceBuffer, size_t sourceBufferOffset,
	GFXResource* textureResource, uint targetMip,
	uint width, uint height, uint depth, GFXFormat targetFormat, uint pixelSize) {
	commandList->ExecuteResBarrier();
	D3D12_TEXTURE_COPY_LOCATION sourceLocation;
	sourceLocation.pResource = sourceBuffer->GetResource();
	sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	sourceLocation.PlacedFootprint.Offset = sourceBufferOffset;
	sourceLocation.PlacedFootprint.Footprint =
		{
			(DXGI_FORMAT)targetFormat,							  //GFXFormat Format;
			width,												  //uint Width;
			height,												  //uint Height;
			depth,												  //uint Depth;
			GFXUtil::CalcConstantBufferByteSize(width * pixelSize)//uint RowPitch;
		};
	D3D12_TEXTURE_COPY_LOCATION destLocation;
	destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	destLocation.SubresourceIndex = targetMip;
	destLocation.pResource = textureResource;
	commandList->GetCmdList()->CopyTextureRegion(
		&destLocation,
		0, 0, 0,
		&sourceLocation,
		nullptr);
}
static D3D12_TEXTURE_COPY_LOCATION Graphics_GetTextureCopy(GFXResource* textureResource, uint targetMip) {
	D3D12_TEXTURE_COPY_LOCATION destLocation;
	destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	destLocation.SubresourceIndex = targetMip;
	destLocation.pResource = textureResource;
	return destLocation;
}
static D3D12_TEXTURE_COPY_LOCATION Graphics_GetBufferCopy(
	GFXFormat targetFormat,
	GPUResourceBase* sourceBuffer,
	size_t sourceBufferOffset,
	uint width, uint height, uint depth,
	uint pixelSize) {
	D3D12_TEXTURE_COPY_LOCATION sourceLocation;
	sourceLocation.pResource = sourceBuffer->GetResource();
	sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	sourceLocation.PlacedFootprint.Offset = sourceBufferOffset;
	sourceLocation.PlacedFootprint.Footprint =
		{
			(DXGI_FORMAT)targetFormat,							  //DXGI_FORMAT Format;
			width,												  //uint Width;
			height,												  //uint Height;
			depth,												  //uint Depth;
			GFXUtil::CalcConstantBufferByteSize(width * pixelSize)//uint RowPitch;
		};
	return sourceLocation;
}
void Graphics::CopyTextureToBuffer(
	ThreadCommand* commandList,
	ReadbackBuffer* destBuffer, size_t destBufferOffset,
	GFXResource* textureResource, uint targetMip,
	uint width, uint height, uint depth, GFXFormat targetFormat, uint pixelSize) {
	commandList->ExecuteResBarrier();
	D3D12_TEXTURE_COPY_LOCATION destLocation = Graphics_GetBufferCopy(
		targetFormat,
		destBuffer,
		destBufferOffset,
		width, height, depth,
		pixelSize);
	D3D12_TEXTURE_COPY_LOCATION sourceLocation = Graphics_GetTextureCopy(
		textureResource, targetMip);
	commandList->GetCmdList()->CopyTextureRegion(
		&destLocation,
		0, 0, 0,
		&sourceLocation,
		nullptr);
}
void Graphics::CopyBufferToTexture(
	ThreadCommand* commandList,
	UploadBuffer* sourceBuffer, size_t sourceBufferOffset,
	GFXResource* textureResource, uint targetMip,
	uint width, uint height, uint depth, GFXFormat targetFormat, uint pixelSize) {
	commandList->ExecuteResBarrier();
	D3D12_TEXTURE_COPY_LOCATION sourceLocation = Graphics_GetBufferCopy(
		targetFormat,
		sourceBuffer,
		sourceBufferOffset,
		width, height, depth,
		pixelSize);
	D3D12_TEXTURE_COPY_LOCATION destLocation = Graphics_GetTextureCopy(
		textureResource, targetMip);
	commandList->GetCmdList()->CopyTextureRegion(
		&destLocation,
		0, 0, 0,
		&sourceLocation,
		nullptr);
}
void Graphics::ReturnDescHeapIndexToPool(uint target) {

	std::lock_guard lck(current->mtx);
	auto ite = current->usedDescs[target];
	if (ite) {
		current->unusedDescs.push_back(target);
		ite = false;
	}
}
void Graphics::ForceCollectAllHeapIndex() {

	std::lock_guard lck(current->mtx);
	for (uint i = 0; i < MAXIMUM_HEAP_COUNT; ++i) {
		current->unusedDescs[i] = i;
	}
	current->usedDescs.Reset(false);
}
void Graphics::CopyBufferRegion(
	ThreadCommand* commandList,
	IBuffer const* dest,
	uint64 destOffset,
	IBuffer const* source,
	uint64 sourceOffset,
	uint64 byteSize) {
	commandList->ExecuteResBarrier();
	commandList->GetCmdList()->CopyBufferRegion(
		dest->GetResource(),
		destOffset,
		source->GetResource(),
		sourceOffset,
		byteSize);
}
Graphics::~Graphics() {
	globalDescriptorHeap = nullptr;
	unusedDescs.dispose();
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	RenderTexture const* const* renderTargets,
	uint rtCount,
	RenderTexture const* depthTex) {
	D3D12_CPU_DESCRIPTOR_HANDLE* ptr = nullptr;
	if (rtCount > 0) {
		ptr = (D3D12_CPU_DESCRIPTOR_HANDLE*)alloca(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * rtCount);
		for (uint i = 0; i < rtCount; ++i) {
			ptr[i] = renderTargets[i]->GetColorDescriptor(0, 0);
		}
	}
	if (depthTex) {
		auto col = depthTex->GetColorDescriptor(0, 0);
		if (commandList->UpdateRenderTarget(rtCount, ptr, &col)) {
			depthTex->SetViewport(commandList);
		}
	} else if (commandList->UpdateRenderTarget(rtCount, ptr, nullptr)) {
		renderTargets[0]->SetViewport(commandList);
	}
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTexture const*>& rtLists,
	RenderTexture const* depthTex) {
	RenderTexture const* const* renderTargets = rtLists.begin();
	size_t rtCount = rtLists.size();
	SetRenderTarget(
		commandList, renderTargets, rtCount,
		depthTex);
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	const RenderTarget* renderTargets,
	uint rtCount,
	const RenderTarget& depth) {
	D3D12_CPU_DESCRIPTOR_HANDLE* ptr = nullptr;
	if (rtCount > 0) {
		renderTargets[0].rt->SetViewport(commandList, renderTargets[0].mipCount);
		ptr = (D3D12_CPU_DESCRIPTOR_HANDLE*)alloca(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * rtCount);
		for (uint i = 0; i < rtCount; ++i) {
			const RenderTarget& rt = renderTargets[i];
			ptr[i] = rt.rt->GetColorDescriptor(rt.depthSlice, rt.mipCount);
		}
	} else if (depth.rt) {
		depth.rt->SetViewport(commandList, depth.mipCount);
	}
	if (depth.rt) {
		auto rrt = depth.rt->GetColorDescriptor(depth.depthSlice, depth.mipCount);
		commandList->UpdateRenderTarget(rtCount, ptr, &rrt);
	} else {
		commandList->UpdateRenderTarget(rtCount, ptr, nullptr);
	}
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTarget>& init,
	const RenderTarget& depth) {
	const RenderTarget* renderTargets = init.begin();
	const size_t rtCount = init.size();
	SetRenderTarget(commandList, renderTargets, rtCount, depth);
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	const RenderTarget* renderTargets,
	uint rtCount) {
	if (rtCount > 0) {
		renderTargets[0].rt->SetViewport(commandList, renderTargets[0].mipCount);
		D3D12_CPU_DESCRIPTOR_HANDLE* ptr = (D3D12_CPU_DESCRIPTOR_HANDLE*)alloca(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * rtCount);
		for (uint i = 0; i < rtCount; ++i) {
			const RenderTarget& rt = renderTargets[i];
			ptr[i] = rt.rt->GetColorDescriptor(rt.depthSlice, rt.mipCount);
		}
		commandList->UpdateRenderTarget(rtCount, ptr, nullptr);
	}
}
void Graphics::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTarget>& init) {
	const RenderTarget* renderTargets = init.begin();
	const size_t rtCount = init.size();
	SetRenderTarget(commandList, renderTargets, rtCount);
}
#undef MAXIMUM_HEAP_COUNT
