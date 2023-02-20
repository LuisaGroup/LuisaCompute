#pragma once
#include <stddef.h>
namespace toolhub::directx {
template<typename T>
class DxPtr {
    T *ptr;
    bool contained;
    void Dispose() {
        if (ptr && contained) {
            ptr->Release();
        }
    }

public:
    bool Contained() const { return contained; }
    void Clear() {
        Dispose();
        ptr = nullptr;
        contained = false;
    }
    DxPtr(T *ptr, bool contained)
        : ptr{ptr}, contained{contained} {}
    DxPtr() : DxPtr{nullptr, false} {}
    DxPtr(DxPtr const &) = delete;
    DxPtr(DxPtr &&rhs) {
        ptr = rhs.ptr;
        contained = rhs.contained;
        rhs.ptr = nullptr;
        rhs.contained = false;
    }
    DxPtr &operator=(DxPtr const &) = delete;
    DxPtr &operator=(DxPtr &&rhs) {
        Dispose();
        ptr = rhs.ptr;
        contained = rhs.contained;
        rhs.ptr = nullptr;
        rhs.contained = false;
        return *this;
    }
    T **GetAddressOf() {
        Dispose();
        ptr = nullptr;
        contained = true;
        return &ptr;
    }
    T *Get() const {
        return ptr;
    }
    T *operator->() const {
        return ptr;
    }
    T &operator*() const {
        return *ptr;
    }
    ~DxPtr() {
        Dispose();
    }
    operator T *() const {
        return ptr;
    }
};
}// namespace toolhub::directx