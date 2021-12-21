#pragma vengine_package vengine_dll
#include <vstl/StackAllocator.h>
namespace vstd {
StackAllocator::StackAllocator(
    uint64 initCapacity,
    StackAllocatorVisitor *visitor)
    : capacity(initCapacity),
      visitor(visitor) {
}
StackAllocator::Chunk StackAllocator::Allocate(uint64 targetSize) {
    Buffer *bf = nullptr;
    uint64 minSize = std::numeric_limits<uint64>::max();
    for (auto &&i : allocatedBuffers) {
        if (i.leftSize >= targetSize && i.leftSize < minSize) {
            minSize = i.leftSize;
            bf = &i;
        }
    }
    if (bf) {
        auto ofst = bf->fullSize - bf->leftSize;
        bf->leftSize -= targetSize;
        return {
            bf->handle,
            ofst};
    }
    while (capacity < targetSize) {
        capacity = std::max<uint64>(capacity + 1, capacity * 1.5);
    }
    auto newHandle = visitor->Allocate(capacity);
    allocatedBuffers.push_back(Buffer{
        newHandle,
        capacity,
        capacity - targetSize});
    return {
        newHandle,
        0};
    //TODO: return
}
void StackAllocator::Clear() {
    for (auto &&i : allocatedBuffers) {
        i.leftSize = i.fullSize;
    }
}
StackAllocator::~StackAllocator() {
    for (auto &&i : allocatedBuffers) {
        visitor->DeAllocate(i.handle);
    }
}

}// namespace vstd
