//#endif
#include <RenderComponent/Utility/DefaultTextureAllocator.h>
#include <RenderComponent/Utility/D3D12MemoryAllocator/D3D12MemAlloc.h>
#include <RenderComponent/RenderTexture.h>
#include <RenderComponent/Texture.h>
DefaultTextureAllocator::~DefaultTextureAllocator() {
	if (allocator)
		allocator->Release();
}
void DefaultTextureAllocator::AllocateTextureHeap(
	GFXDevice* device,
	GFXFormat format,
	uint32_t width,
	uint32_t height,
	uint32_t depthSlice,
	TextureDimension dimension,
	uint32_t mipCount,
	ID3D12Heap** heap, uint64_t* offset,
	bool isRenderTexture,
	TextureBase* currentPtr) {
	D3D12_HEAP_FLAGS heapFlag = isRenderTexture ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
	D3D12MA::ALLOCATION_DESC desc;
	desc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
	desc.Flags = D3D12MA::ALLOCATION_FLAGS::ALLOCATION_FLAG_NONE;
	desc.ExtraHeapFlags = heapFlag;
	desc.CustomPool = nullptr;
	D3D12_RESOURCE_ALLOCATION_INFO info;
	info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
	if (isRenderTexture) {
		info.SizeInBytes = RenderTexture::GetSizeFromProperty(
			device,
			width,
			height,
			RenderTextureFormat::GetColorFormat(format),
			dimension,
			depthSlice,
			mipCount,
			RenderTextureState::Generic_Read);
	} else {
		info.SizeInBytes = Texture::GetSizeFromProperty(
			device,
			width,
			height,
			depthSlice,
			dimension,
			mipCount,
			format);
	}
	D3D12MA::Allocation* alloc;
	lockGuard lck(mtx);
	allocator->AllocateMemory(&desc, &info, &alloc);
	allocatedTexs.Insert(currentPtr->GetInstanceID(), alloc);
	*heap = alloc->GetHeap();
	*offset = alloc->GetOffset();
	allocatedTexs.Insert(currentPtr->GetInstanceID(), alloc);
}
DefaultTextureAllocator::DefaultTextureAllocator(
	GFXDevice* device,
	IDXGIAdapter* adapter) : allocatedTexs(36) {

	D3D12MA::ALLOCATOR_DESC desc;
	desc.Flags = D3D12MA::ALLOCATOR_FLAGS::ALLOCATOR_FLAG_SINGLETHREADED;
	desc.pAdapter = adapter;
	desc.pAllocationCallbacks = nullptr;
	desc.pDevice = device->device();
	desc.PreferredBlockSize = 1;
	desc.PreferredBlockSize <<= 30;//1G
	D3D12MA::CreateAllocator(&desc, &allocator);
}
void DefaultTextureAllocator::ReturnTexture(TextureBase* tex) {
	lockGuard lck(mtx);
	auto ite = allocatedTexs.Find(tex->GetInstanceID());
	if (!ite) {
#if defined(DEBUG)
		throw "Non Exist Resource!";
#endif
		return;
	}
	ite.Value()->Release();
	allocatedTexs.Remove(ite);
}
