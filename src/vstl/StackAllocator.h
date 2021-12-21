#pragma once
#include <vstl/Common.h>
namespace vstd {
class StackAllocatorVisitor {
public:
	virtual uint64 Allocate(uint64 size) = 0;
	virtual void DeAllocate(uint64 handle) = 0;
};
class VENGINE_DLL_COMMON StackAllocator {
	StackAllocatorVisitor* visitor;
	uint64 capacity;
	struct Buffer {
		uint64 handle;
		uint64 fullSize;
		uint64 leftSize;
	};
	vstd::vector<Buffer> allocatedBuffers;

public:
	StackAllocator(
		uint64 initCapacity,
		StackAllocatorVisitor* visitor);
	~StackAllocator();
	struct Chunk {
		uint64 handle;
		uint64 offset;
	};
	Chunk Allocate(
		uint64 targetSize);
	void Clear();
};
}// namespace vstd