#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/MetaLib.h>
#include <Common/HashMap.h>
#include <Common/BitArray.h>
#include <Common/vector.h>
#include <Common/RandomVector.h>
#include <RenderComponent/Mesh.h>
#include <Struct/RenderTarget.h>
#include <Utility/ElementAllocator.h>
#include <RenderComponent/DescriptorHeap.h>
class IBuffer;
class Shader;
class RenderTexture;
class UploadBuffer;

enum BackBufferState {
	BackBufferState_Present = 0,
	BackBufferState_RenderTarget = 1
};
class Texture;
class StructuredBuffer;
class TextureBase;
class ThreadCommand;
class DescriptorHeapRoot;
namespace luisa::compute {
class DXDevice;
}

class Graphics {
	friend class luisa::compute::DXDevice;
	friend class UploadBuffer;
	friend class PSOContainer;
	friend class StructuredBuffer;
	friend class Texture;
	friend class RenderTexture;
	friend class TextureBase;
	friend class DescriptorHeap;
	friend class DescriptorHeapRoot;
	friend class Mesh;

private:
	static thread_local Graphics* current;
	spin_mutex mtx;
	StackObject<Mesh, true> fullScreenMesh;
	std::unique_ptr<DescriptorHeap> globalDescriptorHeap;
	StackObject<BitArray, true> usedDescs;
	ArrayList<uint, false> unusedDescs;
	ObjectPtr<Mesh> cubeMesh;
	StackObject<ElementAllocator, true> srvAllocator;
	StackObject<ElementAllocator, true> rtvAllocator;
	StackObject<ElementAllocator, true> dsvAllocator;

	static void SetRenderTarget(
		ThreadCommand* commandList,
		RenderTexture const* const* renderTargets,
		uint rtCount,
		RenderTexture const* depthTex = nullptr);
	static void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTexture const*>& renderTargets,
		RenderTexture const* depthTex = nullptr);
	static void SetRenderTarget(
		ThreadCommand* commandList,
		const RenderTarget* renderTargets,
		uint rtCount,
		const RenderTarget& depth);
	static void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTarget>& init,
		const RenderTarget& depth);
	static void SetRenderTarget(
		ThreadCommand* commandList,
		const RenderTarget* renderTargets,
		uint rtCount);
	static void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTarget>& init);
	static inline DescriptorHeap* GetGlobalDescHeapNonConst() {
		return current->globalDescriptorHeap.operator->();
	}
	static uint GetDescHeapIndexFromPool();
	static void ReturnDescHeapIndexToPool(uint targetIndex);
	static void ForceCollectAllHeapIndex();

public:
	static void SetGraphics(Graphics* graphics) {
		current = graphics;
	}
	static inline DescriptorHeap const* GetGlobalDescHeap() {
		return current->globalDescriptorHeap.operator->();
	}
	static Mesh const* GetFullScreenMesh() {
		return current->fullScreenMesh;
	}
	Graphics(GFXDevice* device);
	static void Blit(
		ThreadCommand* commandList,
		GFXDevice* device,
		D3D12_CPU_DESCRIPTOR_HANDLE* renderTarget,
		GFXFormat* renderTargetFormats,
		uint renderTargetCount,
		D3D12_CPU_DESCRIPTOR_HANDLE* depthTarget,
		GFXFormat depthFormat,
		uint width, uint height,
		const Shader* shader, uint pass);
	static void Blit(
		ThreadCommand* commandList,
		GFXDevice* device,
		const std::initializer_list<RenderTarget>& renderTarget,
		const RenderTarget& depthTarget,
		const Shader* shader, uint pass);
	static void Blit(
		ThreadCommand* commandList,
		GFXDevice* device,
		RenderTarget const* renderTargets,
		uint renderTargetCount,
		const RenderTarget& depthTarget,
		const Shader* shader, uint pass);
	/*
	template <uint count>
	inline  static void UAVBarriers(
		ThreadCommand* commandList,
		const std::initializer_list<GFXResource*>& resources)
	{
		D3D12_RESOURCE_BARRIER barrier[count];
		GFXResource* const* ptr = resources.begin();
		auto func = [&](uint i)->void
		{
			barrier[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			barrier[i].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			barrier[i].UAV = { ptr[i] };
		};
		InnerLoop<decltype(func), count>(func);
		commandList->ResourceBarrier(count, barrier);
	}*/
	static void CopyTexture(
		ThreadCommand* commandList,
		RenderTexture const* source, uint sourceSlice, uint sourceMipLevel,
		RenderTexture const* dest, uint destSlice, uint destMipLevel);

	static void CopyBufferToTexture(
		ThreadCommand* commandList,
		UploadBuffer* sourceBuffer, size_t sourceBufferOffset,
		GFXResource* textureResource, uint targetMip,
		uint width, uint height, uint depth, GFXFormat targetFormat, uint pixelSize);
	static void CopyBufferToBCTexture(
		ThreadCommand* commandList,
		UploadBuffer* sourceBuffer, size_t sourceBufferOffset,
		GFXResource* textureResource, uint targetMip,
		uint width, uint height, uint depth, GFXFormat targetFormat, uint pixelSize);
	static void DrawMesh(
		GFXDevice* device,
		ThreadCommand* commandList,
		const IMesh* mesh,
		const Shader* shader, uint pass, uint instanceCount = 1);

	static void DrawMeshes(
		GFXDevice* device,
		ThreadCommand* commandList,
		IMesh const** mesh, uint meshCount,
		const Shader* shader, uint pass, bool sort);
	static void CopyBufferRegion(
		ThreadCommand* commandList,
		IBuffer const* dest,
		uint64 destOffset,
		IBuffer const* source,
		uint64 sourceOffset,
		uint64 byteSize);
	/*
	static void CopyBufferRegion_Compute(
		ThreadCommand* commandList,
		IBuffer const* dest,
		uint64 destOffset,
		IBuffer const* source,
		uint64 sourceOffset,
		uint64 byteSize);*/
	static Mesh const* GetCubeMesh() { return current->cubeMesh; }
	static uint GetLeftedPoolIndicies() {
		return current->unusedDescs.size();
	}
	~Graphics();
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};