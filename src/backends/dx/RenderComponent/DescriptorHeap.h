#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/vector.h>
#include <RenderComponent/DescriptorHeapRoot.h>
#include <Utility/ElementAllocator.h>
class GPUResourceBase;
class ThreadCommand;
class VENGINE_DLL_RENDERER DescriptorHeap : public VObject
{
public:
	DescriptorHeap(
		GFXDevice* pDevice,
		D3D12_DESCRIPTOR_HEAP_TYPE Type,
		uint64 NumDescriptors,
		bool bShaderVisible = false);
	DescriptorHeap() : rootPtr(nullptr), size(0), offset(0)
	{

	}
	void Create(GFXDevice* pDevice,
		D3D12_DESCRIPTOR_HEAP_TYPE Type,
		uint64 NumDescriptors,
		bool bShaderVisible = false);
	ID3D12DescriptorHeap* Get() const { return rootPtr->Get(); }
	D3D12_GPU_DESCRIPTOR_HANDLE hGPU(uint64 index) const
	{
		return rootPtr->hGPU(index + offset);
	}
	void SetDescriptorHeap(ThreadCommand* commandList) const;
	D3D12_DESCRIPTOR_HEAP_DESC GetDesc() const { return rootPtr->GetDesc(); };
	uint64 Size() const { return size; }
	void CreateUAV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, uint64 index);
	void CreateSRV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, uint64 index);
	void CreateRTV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, uint64 index);
	void CreateDSV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, uint64 index);
	void ClearView(uint64 index) const;
	BindType GetBindType(uint64 index) const;
	~DescriptorHeap();
	D3D12_CPU_DESCRIPTOR_HANDLE hCPU(uint64 index) const
	{
		return rootPtr->hCPU(index + offset);
	}
private:
	ElementAllocator* allocator;
	ElementAllocator::AllocateHandle handle;
	DescriptorHeapRoot* rootPtr;
	uint64 size;
	uint64 offset;
	void InternalCreate(
		GFXDevice* pDevice,
		D3D12_DESCRIPTOR_HEAP_TYPE Type,
		uint64 NumDescriptors,
		bool bShaderVisible);
	void InternalDispose();
};