#include <RenderComponent/CBufferAllocator.h>
CBufferChunk CBufferAllocator::Allocate(uint64_t size) noexcept {
	if (size > D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT)
		throw "Too Large!";
	size = (size + (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) / D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
	size = Max<uint>(1, size);
	CBufferChunk cb;
	ElementAllocator::AllocateHandle& handle = cb.node;
	if (singleThread) {
		handle = buddyAlloc->Allocate(size);
	} else {
		std::lock_guard<decltype(mtx)> lck(mtx);
		handle = buddyAlloc->Allocate(size);
	}
	cb.size = size;
	return cb;
}
void CBufferChunk::CopyConstBuffer(ConstBuffer const* ptr) {
	void* ff = node.GetBlockResource<UploadBuffer>()->GetMappedDataPtr(node.GetPosition());
	ptr->CopyDataTo(ff);
}

void CBufferChunk::CopyData(void const* ptr, size_t sz) const noexcept {
	node.GetBlockResource<UploadBuffer>()->CopyData(node.GetPosition(), ptr, sz);
}

void CBufferAllocator::Release(CBufferChunk const& chunk) noexcept {
#if defined(VSTL_DEBUG)
	if (!chunk.GetBuffer() || !chunk.node)
		throw "Null Chunk!";
#endif
	if (singleThread) {
		buddyAlloc->Release(chunk.node);
	} else {
		std::lock_guard<decltype(mtx)> lck(mtx);
		buddyAlloc->Release(chunk.node);
	}
}
CBufferAllocator::~CBufferAllocator() noexcept {}
CBufferAllocator::CBufferAllocator(GFXDevice* device, bool singleThread) noexcept
	: buddyAlloc(
		new ElementAllocator(
			D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT / D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, [=](uint64_t size) -> void* { return new UploadBuffer(device, size, true, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT); },
			[=](void* ptr) -> void {
				delete (static_cast<UploadBuffer*>(ptr));
			})),
	  singleThread(singleThread) {
}
