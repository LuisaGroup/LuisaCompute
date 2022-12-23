#include <vstl/StackAllocator.h>
namespace vstd {
StackAllocator::StackAllocator(
    uint64 initCapacity,
    StackAllocatorVisitor *visitor)
    : visitor(visitor),
      capacity(initCapacity),
      initCapacity(initCapacity) {
}
StackAllocator::Chunk StackAllocator::Allocate(uint64 targetSize) {
    for (auto &&i : allocatedBuffers) {
        int64 leftSize = (i.fullSize - i.position);
        if (leftSize >= targetSize) {
            auto ofst = i.position;
            i.position += targetSize;
            return {i.handle, ofst};
        }
    }
    if (capacity < targetSize) {
        capacity = std::max<uint64>(targetSize, capacity * 1.5);
    }
    auto newHandle = visitor->Allocate(capacity);
    allocatedBuffers.push_back(Buffer{
        .handle = newHandle,
        .fullSize = capacity,
        .position = targetSize});
    return {newHandle, 0};
}
StackAllocator::Chunk StackAllocator::Allocate(
    uint64 targetSize,
    uint64 align) {
    targetSize = std::max(targetSize, align);
    auto CalcAlign = [](uint64 value, uint64 align) -> uint64 {
        return (value + (align - 1)) & ~(align - 1);
    };
    for (auto &&i : allocatedBuffers) {
        auto position = CalcAlign(i.position, align);
        int64 leftSize = (i.fullSize - position);
        if (leftSize >= targetSize) {
            auto ofst = position;
            i.position = position + targetSize;
            return {i.handle, ofst};
        }
    }
    if (capacity < targetSize) {
        capacity = std::max<uint64>(targetSize, capacity * 1.5);
    }
    auto newHandle = visitor->Allocate(capacity);
    allocatedBuffers.push_back(Buffer{
        .handle = newHandle,
        .fullSize = capacity,
        .position = targetSize});
    return {
        newHandle,
        0};
}
void StackAllocator::Dispose() {
    capacity = initCapacity;
    if (allocatedBuffers.empty()) return;
    if (allocatedBuffers.size() > 1) {
        for (auto i : vstd::range(1, allocatedBuffers.size())) {
            visitor->DeAllocate(allocatedBuffers[i].handle);
        }
        allocatedBuffers.resize(1);
    }
    auto &first = allocatedBuffers[0];
    if (first.fullSize > capacity) {
        visitor->DeAllocate(first.handle);
        first.handle = visitor->Allocate(capacity);
        first.fullSize = capacity;
    }
    first.position = 0;
}
void StackAllocator::Clear() {
    switch (allocatedBuffers.size()) {
        case 0: break;
        case 1: {
            auto &&i = allocatedBuffers[0];
            i.position = 0;
        } break;
        default: {
            size_t sumSize = 0;
            for (auto &&i : allocatedBuffers) {
                sumSize += i.fullSize;
                visitor->DeAllocate(i.handle);
            }
            allocatedBuffers.clear();
            allocatedBuffers.push_back(Buffer{
                .handle = visitor->Allocate(sumSize),
                .fullSize = sumSize,
                .position = 0});
        } break;
    }
}
StackAllocator::~StackAllocator() {
    for (auto &&i : allocatedBuffers) {
        visitor->DeAllocate(i.handle);
    }
}
uint64 DefaultMallocVisitor::Allocate(uint64 size) {
    return reinterpret_cast<uint64>(malloc(size));
}
void DefaultMallocVisitor::DeAllocate(uint64 handle) {
    free(reinterpret_cast<void *>(handle));
}
uint64 VEngineMallocVisitor::Allocate(uint64 size) {
    return reinterpret_cast<uint64>(vengine_malloc(size));
}
void VEngineMallocVisitor::DeAllocate(uint64 handle) {
    vengine_free(reinterpret_cast<void *>(handle));
}
}// namespace vstd
