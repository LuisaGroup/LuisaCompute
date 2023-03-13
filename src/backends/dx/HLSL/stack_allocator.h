#pragma once
#include <vstl/common.h>
namespace vstd {
class StackAllocatorVisitor {
public:
    virtual uint64 Allocate(uint64 size) = 0;
    virtual void DeAllocate(uint64 handle) = 0;
};
class StackAllocator {
    StackAllocatorVisitor *visitor;
    uint64 capacity;
    uint64 initCapacity;
    struct Buffer {
        uint64 handle;
        uint64 fullSize;
        uint64 position;
    };
    vstd::vector<Buffer> allocatedBuffers;

public:
    StackAllocator(
        uint64 initCapacity,
        StackAllocatorVisitor *visitor);
    ~StackAllocator();
    struct Chunk {
        uint64 handle;
        uint64 offset;
    };
    Chunk Allocate(
        uint64 targetSize);
    Chunk Allocate(
        uint64 targetSize,
        uint64 align);
    void Clear();
    void Dispose();
    template<typename T, bool clearMemory = true>
        requires(std::is_trivially_constructible_v<T>)
    T *AllocateMemory() {
        constexpr size_t align = alignof(T);
        Chunk chunk;
        if constexpr (align > 1) {
            chunk = Allocate(sizeof(T), align);
        } else {
            chunk = Allocate(sizeof(T));
        }
        T *ptr = reinterpret_cast<T *>(chunk.handle + chunk.offset);
        if constexpr (clearMemory) {
            memset(ptr, 0, sizeof(T));
        }
        return ptr;
    }
};
class DefaultMallocVisitor : public StackAllocatorVisitor {
public:
    uint64 Allocate(uint64 size) override;
    void DeAllocate(uint64 handle) override;
};
class VEngineMallocVisitor : public StackAllocatorVisitor {
public:
    uint64 Allocate(uint64 size) override;
    void DeAllocate(uint64 handle) override;
};
}// namespace vstd