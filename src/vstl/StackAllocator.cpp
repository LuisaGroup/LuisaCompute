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
StackAllocator::Chunk StackAllocator::Allocate(
    uint64 targetSize,
    uint64 align) {
    targetSize = std::max(targetSize, align);
    Buffer *bf = nullptr;
    uint64 offset = 0;
    uint64 minLeftSize = std::numeric_limits<uint64>::max();
    auto CalcAlign = [](uint64 value, uint64 align) -> uint64 {
        return (value + (align - 1)) & ~(align - 1);
    };
    struct Result {
        uint64 offset;
        uint64 leftSize;
    };
    auto GetLeftSize = [&](uint64 leftSize, uint64 size) -> vstd::optional<Result> {
        uint64 offset = size - leftSize;
        uint64 alignedOffset = CalcAlign(offset, align);
        if (alignedOffset > size) return {};
        return Result{alignedOffset, size - alignedOffset};
    };
    for (auto &&i : allocatedBuffers) {
        auto result = GetLeftSize(i.leftSize, i.fullSize);
        if (!result) continue;
        if (result->leftSize < minLeftSize) {
            minLeftSize = result->leftSize;
            offset = result->offset;
            bf = &i;
        }
    }
    if (bf) {
        bf->leftSize = minLeftSize;
        return {
            bf->handle,
            offset};
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
