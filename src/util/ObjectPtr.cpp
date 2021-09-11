#pragma vengine_package vengine_dll
#include <util/ObjectPtr.h>
#include <mutex>
#include <util/spin_mutex.h>
namespace vstd {

namespace detail {
#define PRINT_SIZE 256
Pool<LinkHeap, VEngine_AllocType::VEngine, true> allocPool(PRINT_SIZE, false);
luisa::spin_mutex globalMtx;
}// namespace detail
void PtrWeakLink::Dispose() noexcept {
    auto a = heapPtr;
    heapPtr = nullptr;
    if (a) {
        a->WeakDestructor();
    }
}
void LinkHeap::Destroy() {
    void *a;
    funcPtr_t<void(void *)> disposer;
    {
        std::unique_lock lck(mtx);
        a = ptr;
        disposer = this->disposer;
        this->refCount = 0;
        ptr = nullptr;
        auto looseRefCount = --this->looseRefCount;
        if (looseRefCount == 0) {
            lck.unlock();
            LinkHeap::ReturnHeap(this);
        }
    }
    if (a) {
        disposer(a);
    }
}
void LinkHeap::Destructor() {
    void *a;
    funcPtr_t<void(void *)> disposer;
    {
        std::unique_lock lck(mtx);
        auto refCount = --this->refCount;
        auto looseRefCount = --this->looseRefCount;
        if (refCount == 0) {
            a = ptr;
            ptr = nullptr;
            disposer = this->disposer;
        } else {
            a = nullptr;
            disposer = nullptr;
        }
        if (looseRefCount == 0) {
            lck.unlock();
            LinkHeap::ReturnHeap(this);
        }
    }
    if (a) {
        disposer(a);
    }
}

void LinkHeap::WeakDestructor() {
    std::unique_lock lck(mtx);
    if (--this->looseRefCount == 0) {
        lck.unlock();
        LinkHeap::ReturnHeap(this);
    }
}

void PtrLink::Destroy() noexcept {
    auto bb = heapPtr;
    heapPtr = nullptr;
    if (bb) {
        bb->Destroy();
    }
}
PtrLink::PtrLink(const PtrLinkBase &p) noexcept {
    offset = p.offset;
    auto func = [&]() {
        if (!p.heapPtr) return false;
        std::lock_guard lck(p.heapPtr->mtx);
        if (p.heapPtr->ptr) {
            ++p.heapPtr->refCount;
            ++p.heapPtr->looseRefCount;
            return true;
        }
        return false;
    };
    if (func()) {
        heapPtr = p.heapPtr;
    } else {
        heapPtr = nullptr;
    }
}
PtrLink::PtrLink(PtrLinkBase &&p) noexcept {
    offset = p.offset;
    heapPtr = p.heapPtr;
    p.heapPtr = nullptr;
}
void PtrLink::operator=(const PtrLinkBase &p) noexcept {
    if (&p == this) return;
    this->~PtrLink();
    new (this) PtrLink(p);
}
void PtrLink::operator=(PtrLinkBase &&p) noexcept {
    if (&p == this) return;
    this->~PtrLink();
    new (this) PtrLink(std::move(p));
}
void PtrWeakLink::Destroy() noexcept {
    auto bb = heapPtr;
    heapPtr = nullptr;
    if (bb) {
        bb->Destroy();
    }
}
void PtrLink::Dispose() noexcept {
    auto a = heapPtr;
    heapPtr = nullptr;
    if (a) {
        a->Destructor();
    }
}
LinkHeap *LinkHeap::GetHeap(void *obj, funcPtr_t<void(void *)> disp) noexcept {
    LinkHeap *ptr = detail::allocPool.New_Lock(detail::globalMtx);
    ptr->ptr = obj;
    ptr->disposer = disp;
    return ptr;
}
void LinkHeap::ReturnHeap(LinkHeap *value) noexcept {
    detail::allocPool.Delete_Lock(detail::globalMtx, value);
}
PtrWeakLink::PtrWeakLink(const PtrLinkBase &p) noexcept {
    offset = p.offset;
    auto func = [&]() {
        if (!p.heapPtr) return false;
        std::lock_guard lck(p.heapPtr->mtx);
        if (p.heapPtr->ptr) {
            ++p.heapPtr->looseRefCount;
            return true;
        }
        return false;
    };
    if (func()) {
        heapPtr = p.heapPtr;
    } else {
        heapPtr = nullptr;
    }
}
PtrWeakLink::PtrWeakLink(PtrLinkBase &&p) noexcept {
    heapPtr = p.heapPtr;
    offset = p.offset;
    p.heapPtr = nullptr;
}
void PtrWeakLink::operator=(const PtrLinkBase &p) noexcept {
    if (&p == this) return;
    this->~PtrWeakLink();
    new (this) PtrWeakLink(p);
}
void PtrWeakLink::operator=(PtrLinkBase &&p) noexcept {
    if (&p == this) return;
    this->~PtrWeakLink();
    new (this) PtrWeakLink(std::move(p));
}

#undef PRINT_SIZE

}// namespace vstd