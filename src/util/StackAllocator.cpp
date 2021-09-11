#pragma vengine_package vengine_dll
#include <util/StackAllocator.h>
#include <util/Log.h>
namespace vstd {

namespace valloc {

struct StackData {
    static constexpr size_t VENGINE_STACK_LENGTH = 1024 * 256;
    uint8_t *data = nullptr;
    uint8_t *offset = nullptr;
    ~StackData() {
        if (data) delete data;
    }
    template<size_t align>
    static constexpr size_t CalcAlign(size_t value) {
        return (value + (align - 1)) & ~(align - 1);
    }
    void *GetCurrent() {
        if (!data) {
            data = new uint8_t[VENGINE_STACK_LENGTH];
            offset = data;
        }
        return offset;
    }
    template<size_t align>
    void *Alloc(size_t sz) {
        if (!data) {
            data = new uint8_t[VENGINE_STACK_LENGTH];
            offset = data;
        }
        offset = reinterpret_cast<uint8_t *>(CalcAlign<align>(reinterpret_cast<size_t>(offset)));
        void *ptr = offset;
        offset += sz;
#ifdef DEBUG
        if (reinterpret_cast<size_t>(offset) - reinterpret_cast<size_t>(data) > VENGINE_STACK_LENGTH) {
            VEngine_Log("Stack-Overflow!\n");
            VENGINE_EXIT;
        }
#endif
        return ptr;
    }
    void ReleaseTo(uint8_t *result) {
        auto Min = [](auto &&a, auto &&b) {
            return (a < b) ? a : b;
        };
        offset = reinterpret_cast<uint8_t *>(
            Min(reinterpret_cast<size_t>(offset), reinterpret_cast<size_t>(result)));
    }
};

static thread_local StackData data;
}// namespace valloc

void *StackBuffer::stack_malloc(size_t sz) {
    return valloc::data.Alloc<16>(sz);
}
void StackBuffer::stack_free(void *ptr) {
    valloc::data.ReleaseTo(reinterpret_cast<uint8_t *>(ptr));
}
void *StackBuffer::GetCurrentPtr() {
    return valloc::data.GetCurrent();
}

StackBuffer::~StackBuffer() {
    if (ptr)
        valloc::data.ReleaseTo(reinterpret_cast<uint8_t *>(ptr));
}

StackBuffer StackBuffer::Allocate(size_t size) {
    return StackBuffer(valloc::data.Alloc<8>(size), size);
}
StackBuffer StackBuffer::Allocate_Align16(size_t size) {
    return StackBuffer(valloc::data.Alloc<16>(size), size);
}
StackBuffer StackBuffer::Allocate_Align32(size_t size) {
    return StackBuffer(valloc::data.Alloc<32>(size), size);
}

StackBuffer::StackBuffer(StackBuffer &&stk) {
    ptr = stk.ptr;
    mLength = stk.mLength;
    stk.ptr = nullptr;
}

}// namespace vstd