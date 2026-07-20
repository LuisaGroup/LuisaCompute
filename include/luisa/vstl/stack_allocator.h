#pragma once
#include <luisa/vstl/common.h>
namespace vstd {
class StackAllocatorVisitor {
public:
    virtual uint64 allocate(uint64 size) = 0;
    virtual void deallocate(uint64 handle) = 0;
};
class LUISA_VSTL_API StackAllocator {
public:
    struct Buffer {
        uint64 handle;
        uint64 fullSize;
        uint64 position;
    };
private:
    StackAllocatorVisitor *visitor;
    uint64 capacity;
    uint64 initCapacity;
    double capaExpanRate;

    vstd::vector<Buffer> allocatedBuffers{};

public:
    StackAllocator(
        uint64 initCapacity,
        StackAllocatorVisitor *visitor,
        double capaExpanRate = 1.5);
    ~StackAllocator();
    struct Chunk {
        uint64 handle;
        uint64 offset;
    };
    luisa::span<Buffer const> allocated_buffer() const {
        return allocatedBuffers;
    }
    Chunk
    allocate(
        uint64 targetSize);
    Chunk allocate(
        uint64 targetSize,
        uint64 align);
    void clear();
    void soft_clear();
    void dispose();
    template<typename T, bool clearMemory = true>
        requires(std::is_trivially_constructible_v<T>)
    T *allocate_memory() {
        constexpr size_t align = alignof(T);
        Chunk chunk;
        if constexpr (align > 1) {
            chunk = allocate(sizeof(T), align);
        } else {
            chunk = allocate(sizeof(T));
        }
        T *ptr = reinterpret_cast<T *>(chunk.handle + chunk.offset);
        if constexpr (clearMemory) {
            std::memset(ptr, 0, sizeof(T));
        }
        return ptr;
    }
};
class LUISA_VSTL_API DefaultMallocVisitor : public StackAllocatorVisitor {
public:
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
};
class LUISA_VSTL_API VEngineMallocVisitor : public StackAllocatorVisitor {
public:
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
};
}// namespace vstd
