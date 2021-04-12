#pragma once
#include <Common/GFXUtil.h>
#include <RenderComponent/UploadBuffer.h>
#include <Utility/ElementAllocator.h>
class BuddyAllocator;
class BuddyNode;
class CBufferAllocator;
class CBufferChunk {
	friend class CBufferAllocator;

private:
	ElementAllocator::AllocateHandle node;
	uint64_t size;
	CBufferChunk(
		ElementAllocator::AllocateHandle node)
		: node(node) {}
	void CopyData(void const* ptr, size_t sz) const noexcept {
		node.GetBlockResource<UploadBuffer>()->CopyData(node.GetPosition(), ptr, sz);
	}

public:
	CBufferChunk() noexcept {}
	UploadBuffer const* GetBuffer() const noexcept {
		return node.GetBlockResource<UploadBuffer>();
	}
	uint64 GetOffset() const noexcept {
		return node.GetPosition();
	}
	uint64 GetSize() const noexcept {
		return size;
	}
	template<typename T>
	void CopyData(T const* ptr) const noexcept {
		CopyData(ptr, sizeof(T));
	}
};
class VENGINE_DLL_RENDERER CBufferAllocator {
private:
	std::unique_ptr<ElementAllocator> buddyAlloc;
	std::mutex mtx;
	bool singleThread;

public:
	CBufferChunk Allocate(uint64_t size) noexcept;
	void Release(CBufferChunk const& chunk) noexcept;
	CBufferAllocator(GFXDevice* device, bool singleThread = false) noexcept;
	~CBufferAllocator() noexcept;
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};