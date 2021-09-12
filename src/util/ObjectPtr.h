#pragma once
#include <util/Memory.h>
#include <util/Pool.h>
namespace vstd {

class PtrLink;
class PtrWeakLink;
struct LUISA_DLL LinkHeap {
    friend class PtrLink;
    friend class PtrWeakLink;

private:
    funcPtr_t<void(void *)> disposer;
    static LinkHeap *GetHeap(void *obj, funcPtr_t<void(void *)> disp) noexcept;
    static void ReturnHeap(LinkHeap *value) noexcept;

public:
    uint32_t refCount = 1;
    uint32_t looseRefCount = 1;
    luisa::spin_mutex mtx;
    void *ptr;
    void Destroy();
    void Destructor();
    void WeakDestructor();
};

class VEngine;
class PtrWeakLink;
class PtrLinkBase {
public:
    LinkHeap *heapPtr;
    size_t offset = 0;
};
class LUISA_DLL PtrLink : public PtrLinkBase {
    friend class VEngine;
    friend class PtrWeakLink;

public:
    inline PtrLink() noexcept {
        heapPtr = nullptr;
    }
    void Dispose() noexcept;
    template<typename T>
    PtrLink(T *obj, funcPtr_t<void(void *)> disposer) noexcept {
        heapPtr = LinkHeap::GetHeap(obj, disposer);
    }

    PtrLink(const PtrLinkBase &p) noexcept;
    PtrLink(PtrLinkBase &&p) noexcept;

    PtrLink(PtrLink const &p);
    PtrLink(PtrLink &&p);
    PtrLink(PtrWeakLink const &p);
    PtrLink(PtrWeakLink &&p);

    void operator=(const PtrLinkBase &p) noexcept;
    void operator=(PtrLinkBase &&p) noexcept;

    inline void operator=(const PtrLink &p) noexcept;
    inline void operator=(PtrLink &&p) noexcept;
    inline void operator=(const PtrWeakLink &p) noexcept;
    inline void operator=(PtrWeakLink &&p) noexcept;
    void Destroy() noexcept;
    ~PtrLink() noexcept {
        Dispose();
    }
};
class LUISA_DLL PtrWeakLink : public PtrLinkBase {
public:
    PtrWeakLink() noexcept {
        heapPtr = nullptr;
    }

    void Dispose() noexcept;
    PtrWeakLink(const PtrLinkBase &p) noexcept;
    PtrWeakLink(PtrLinkBase &&p) noexcept;//TODO
    void operator=(const PtrLinkBase &p) noexcept;
    void operator=(PtrLinkBase &&p) noexcept;

    PtrWeakLink(PtrLink const &p) : PtrWeakLink(static_cast<const PtrLinkBase &>(p)) {}
    PtrWeakLink(PtrLink &&p) : PtrWeakLink(static_cast<PtrLinkBase &&>(p)) {}
    PtrWeakLink(PtrWeakLink const &p) : PtrWeakLink(static_cast<const PtrLinkBase &>(p)) {}
    PtrWeakLink(PtrWeakLink &&p) : PtrWeakLink(static_cast<PtrLinkBase &&>(p)) {}

    void operator=(const PtrLink &p) noexcept {
        operator=(static_cast<const PtrLinkBase &>(p));
    }
    void operator=(PtrLink &&p) noexcept {
        operator=(static_cast<PtrLinkBase &&>(p));
    }
    void operator=(const PtrWeakLink &p) noexcept {
        operator=(static_cast<const PtrLinkBase &>(p));
    }
    void operator=(PtrWeakLink &&p) noexcept {
        operator=(static_cast<PtrLinkBase &&>(p));
    }

    void Destroy() noexcept;

    inline ~PtrWeakLink() noexcept {
        Dispose();
    }
};
namespace vstd {
template<typename T, typename F>
size_t GetOffset() {
    T *const ptr = reinterpret_cast<T *>(1);
    F *const fPtr = static_cast<F *>(ptr);
    return (size_t)fPtr - (size_t)1;
}
}// namespace vstd
template<typename T>
class ObjWeakPtr;
template<typename T>
class ObjectPtr;
class SharedWeakFlag;
class SharedFlag {
    friend class SharedWeakFlag;

private:
    PtrLink link;

public:
    SharedFlag(size_t value)
        : link(reinterpret_cast<char *>(value), [](void *) {}) {
    }
    SharedFlag() : link() {}
    SharedFlag(SharedFlag const &flag)
        : link(flag.link) {
    }
    SharedFlag(SharedFlag &&flag)
        : link(std::move(flag.link)) {
    }
    inline SharedFlag(SharedWeakFlag const &flag);
    inline SharedFlag(SharedWeakFlag &&flag);

    inline void Dispose() {
        link.Dispose();
    }
    inline operator size_t() const noexcept {
        if (link.heapPtr == nullptr) {
            return 0;
        }
        return reinterpret_cast<size_t>(link.heapPtr->ptr);
    }
    inline void operator=(const SharedWeakFlag &other) noexcept;
    inline void operator=(const SharedFlag &other) noexcept {
        link = other.link;
    }
    inline void operator=(SharedFlag &&other) noexcept {
        link = std::move(other.link);
    }
};

class SharedWeakFlag {
    friend class SharedFlag;

private:
    PtrWeakLink link;

public:
    SharedWeakFlag() : link() {}
    SharedWeakFlag(SharedFlag const &flag) : link(flag.link) {}
    SharedWeakFlag(SharedWeakFlag const &flag) : link(flag.link) {}
    SharedWeakFlag(SharedWeakFlag &&flag) : link(std::move(flag.link)) {}
    inline void operator=(const SharedWeakFlag &other) noexcept {
        link = other.link;
    }
    inline void operator=(SharedWeakFlag &&other) noexcept {
        link = std::move(other.link);
    }
    inline void operator=(const SharedFlag &other) noexcept {
        link = other.link;
    }
    inline operator size_t() const noexcept {
        if (link.heapPtr == nullptr) {
            return 0;
        }
        return reinterpret_cast<size_t>(link.heapPtr->ptr);
    }
};

void SharedFlag::operator=(const SharedWeakFlag &other) noexcept {
    link = other.link;
}

SharedFlag::SharedFlag(SharedWeakFlag const &flag)
    : link(flag.link) {
}
SharedFlag::SharedFlag(SharedWeakFlag &&flag)
    : link(std::move(flag.link)) {
}
template<typename T>
class ObjectPtr {
private:
    friend class ObjWeakPtr<T>;
    PtrLink link;
    inline ObjectPtr(T *ptr, funcPtr_t<void(void *)> disposer) noexcept : link(ptr, disposer) {
    }
    T *GetPtr() const noexcept {
        return reinterpret_cast<T *>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
    }

public:
    ObjectPtr(const PtrLink &link, size_t addOffset) noexcept : link(link) {
        this->link.offset += addOffset;
    }
    ObjectPtr(PtrLink &&link, size_t addOffset) noexcept : link(std::move(link)) {
        this->link.offset += addOffset;
    }
    inline ObjectPtr() noexcept : link() {}
    inline ObjectPtr(std::nullptr_t) noexcept : link() {
    }
    inline ObjectPtr(const ObjectPtr<T> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjectPtr(ObjectPtr<T> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    inline ObjectPtr(const ObjWeakPtr<T> &ptr) noexcept;
    inline ObjectPtr(ObjWeakPtr<T> &&ptr) noexcept;
    static ObjectPtr<T> MakePtr(T *ptr) noexcept {
        return ObjectPtr<T>(ptr, [](void *ptr) -> void {
            if constexpr (std::is_base_of_v<IDisposable, T>) {
                reinterpret_cast<T *>(ptr)->Dispose();
            } else {
                delete (reinterpret_cast<T *>(ptr));
            }
        });
    }
    static ObjectPtr<T> MakePtr(T *ptr, funcPtr_t<void(void *)> disposer) noexcept {
        return ObjectPtr<T>(ptr, disposer);
    }
    template<typename... Args>
    static ObjectPtr<T> NewObject(Args &&...args) {
        T *ptr = vstl_new<T>(std::forward<Args>(args)...);
        return ObjectPtr<T>(ptr, [](void *ptr) -> void {
            vstl_delete<T>(reinterpret_cast<T *>(ptr));
        });
    }
    static ObjectPtr<T> MakePtrNoMemoryFree(T *ptr) noexcept {
        return ObjectPtr<T>(ptr, [](void *ptr) -> void {
            if (std::is_trivially_destructible_v<T>)
                (reinterpret_cast<T *>(ptr))->~T();
        });
    }
    static ObjectPtr<T> MakePtr(ObjectPtr<T>) noexcept = delete;

    inline operator bool() const noexcept {
        return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
    }
    inline bool operator!() const {
        return !operator bool();
    }

    inline operator T *() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline void Destroy() noexcept {
        link.Destroy();
    }

    template<typename F>
    inline ObjectPtr<F> CastTo() const &noexcept {
        return ObjectPtr<F>(link, GetOffset<T, F>());
    }
   
    template<typename F>
    inline ObjectPtr<F> Reinterpret_CastTo(size_t offset) const &noexcept {
        return ObjectPtr<F>(link, offset);
    }
    template<typename F>
    inline ObjectPtr<F> CastTo() &&noexcept {
        return ObjectPtr<F>(std::move(link), GetOffset<T, F>());
    }
    template<typename F>
    inline ObjectPtr<F> Reinterpret_CastTo(size_t offset) &&noexcept {
        return ObjectPtr<F>(std::move(link), offset);
    }
    inline void operator=(const ObjWeakPtr<T> &other) noexcept;
    inline void operator=(const ObjectPtr<T> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjectPtr<T> &&other) noexcept {
        link = std::move(other.link);
    }
    inline void operator=(T *other) noexcept = delete;
    inline void operator=(void *other) noexcept = delete;
    inline void operator=(std::nullptr_t t) noexcept {
        link.Dispose();
    }

    inline T *operator->() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline T &operator*() noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }

    inline T const &operator*() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }

    inline bool operator==(const ObjectPtr<T> &ptr) const noexcept {
        return link.heapPtr == ptr.link.heapPtr;
    }
    inline bool operator!=(const ObjectPtr<T> &ptr) const noexcept {
        return link.heapPtr != ptr.link.heapPtr;
    }
};

template<typename T>
class ObjectPtr<T[]> {

private:
    friend class ObjWeakPtr<T[]>;
    PtrLink link;
    inline ObjectPtr(T *ptr, funcPtr_t<void(void *)> disposer) noexcept : link(ptr, disposer) {
    }
    T *GetPtr() const noexcept {
        return reinterpret_cast<T *>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
    }

public:
    ObjectPtr(const PtrLink &link, size_t addOffset) noexcept : link(link) {
        this->link.offset += addOffset;
    }
    ObjectPtr(PtrLink &&link, size_t addOffset) noexcept : link(std::move(link)) {
        this->link.offset += addOffset;
    }
    inline ObjectPtr() noexcept : link() {}
    inline ObjectPtr(std::nullptr_t) noexcept : link() {
    }
    inline ObjectPtr(const ObjectPtr<T[]> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjectPtr(ObjectPtr<T[]> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    inline ObjectPtr(const ObjWeakPtr<T[]> &ptr) noexcept;
    static ObjectPtr<T[]> MakePtr(T *ptr) noexcept {
        return ObjectPtr<T[]>(ptr, [](void *ptr) -> void {
            if constexpr (std::is_base_of_v<IDisposable, T>) {
                reinterpret_cast<T *>(ptr)->Dispose();
            } else {
                delete[](reinterpret_cast<T *>(ptr));
            }
        });
    }
    static ObjectPtr<T[]> MakePtr(T *ptr, funcPtr_t<void(void *)> disposer) noexcept {
        return ObjectPtr<T[]>(ptr, disposer);
    }
    static ObjectPtr<T[]> MakePtr(ObjectPtr<T[]>) noexcept = delete;

    inline operator bool() const noexcept {
        return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
    }
    bool operator!() const {
        return !operator bool();
    }

    inline operator T *() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline void Destroy() noexcept {
        link.Destroy();
    }

    template<typename F>
    inline ObjectPtr<F[]> Reinterpret_CastTo(size_t offset) const &noexcept {
        return ObjectPtr<F[]>(link, offset);
    }
    template<typename F>
    inline ObjectPtr<F[]> Reinterpret_CastTo(size_t offset) &&noexcept {
        return ObjectPtr<F[]>(std::move(link), offset);
    }
    inline void operator=(const ObjWeakPtr<T[]> &other) noexcept;
    inline void operator=(const ObjectPtr<T[]> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjectPtr<T[]> &&other) noexcept {
        link = std::move(other.link);
    }

    inline void operator=(T *other) noexcept = delete;
    inline void operator=(void *other) noexcept = delete;
    inline void operator=(std::nullptr_t t) noexcept {
        link.Dispose();
    }

    inline T *operator->() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline T &operator*() noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }

    inline T const &operator*() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }
    inline T &operator[](size_t key) noexcept {
        return GetPtr()[key];
    }

    inline T const &operator[](size_t key) const noexcept {
        return GetPtr()[key];
    }

    inline bool operator==(const ObjectPtr<T[]> &ptr) const noexcept {
        return link.heapPtr == ptr.link.heapPtr;
    }
    inline bool operator!=(const ObjectPtr<T[]> &ptr) const noexcept {
        return link.heapPtr != ptr.link.heapPtr;
    }
};

template<typename T>
class ObjWeakPtr {
private:
    friend class ObjectPtr<T>;
    PtrWeakLink link;
    T *GetPtr() const noexcept {
        return reinterpret_cast<T *>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
    }

public:
    inline ObjWeakPtr() noexcept : link() {}
    inline ObjWeakPtr(std::nullptr_t) noexcept : link() {
    }
    inline ObjWeakPtr(const ObjWeakPtr<T> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjWeakPtr(ObjWeakPtr<T> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    inline ObjWeakPtr(const ObjectPtr<T> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjWeakPtr(ObjectPtr<T> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    ObjWeakPtr(const PtrWeakLink &link, size_t addOffset) noexcept : link(link) {
        this->link.offset += addOffset;
    }
    ObjWeakPtr(PtrWeakLink &&link, size_t addOffset) noexcept : link(std::move(link)) {
        this->link.offset += addOffset;
    }

    inline operator bool() const noexcept {
        return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
    }
    bool operator!() const {
        return !operator bool();
    }

    inline operator T *() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline void Destroy() noexcept {
        link.Destroy();
    }

    template<typename F>
    inline ObjWeakPtr<F> CastTo() const &noexcept {
        return ObjWeakPtr<F>(link, GetOffset<T, F>());
    }
    template<typename F>
    inline ObjWeakPtr<F> Reinterpret_CastTo(size_t offset) const &noexcept {
        return ObjWeakPtr<F>(link, offset);
    }
    template<typename F>
    inline ObjWeakPtr<F> CastTo() &&noexcept {
        return ObjWeakPtr<F>(std::move(link), GetOffset<T, F>());
    }
    template<typename F>
    inline ObjWeakPtr<F> Reinterpret_CastTo(size_t offset) &&noexcept {
        return ObjWeakPtr<F>(std::move(link), offset);
    }
   
    inline void operator=(const ObjWeakPtr<T> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjWeakPtr<T> &&other) noexcept {
        link = std::move(other.link);
    }

    inline void operator=(const ObjectPtr<T> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjectPtr<T> &&other) noexcept {
        link = std::move(other.link);
    }

    inline void operator=(T *other) noexcept = delete;
    inline void operator=(void *other) noexcept = delete;
    inline void operator=(std::nullptr_t t) noexcept {
        link.Dispose();
    }

    inline T *operator->() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline T &operator*() noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }

    inline T const &operator*() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }

    inline bool operator==(const ObjWeakPtr<T> &ptr) const noexcept {
        return link.heapPtr == ptr.link.heapPtr;
    }
    inline bool operator!=(const ObjWeakPtr<T> &ptr) const noexcept {
        return link.heapPtr != ptr.link.heapPtr;
    }
};

template<typename T>
class ObjWeakPtr<T[]> {
private:
    friend class ObjectPtr<T[]>;
    PtrWeakLink link;
    T *GetPtr() const noexcept {
        return reinterpret_cast<T *>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
    }

public:
    inline ObjWeakPtr() noexcept : link() {}
    inline ObjWeakPtr(std::nullptr_t) noexcept : link() {
    }
    inline ObjWeakPtr(const ObjWeakPtr<T[]> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjWeakPtr(ObjWeakPtr<T[]> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    inline ObjWeakPtr(const ObjectPtr<T[]> &ptr) noexcept : link(ptr.link) {
    }
    inline ObjWeakPtr(ObjectPtr<T[]> &&ptr) noexcept : link(std::move(ptr.link)) {
    }
    ObjWeakPtr(const PtrWeakLink &link, size_t addOffset) noexcept : link(link) {
        this->link.offset += addOffset;
    }
    ObjWeakPtr(PtrWeakLink &&link, size_t addOffset) noexcept : link(std::move(link)) {
        this->link.offset += addOffset;
    }

    inline operator bool() const noexcept {
        return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
    }
    bool operator!() const {
        return !operator bool();
    }

    inline operator T *() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline void Destroy() noexcept {
        link.Destroy();
    }

    template<typename F>
    inline ObjWeakPtr<F[]> Reinterpret_CastTo(size_t offset) const noexcept {
        return ObjWeakPtr<F[]>(link, offset);
    }
    inline void operator=(const ObjWeakPtr<T[]> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjWeakPtr<T[]> &&other) noexcept {
        link = std::move(other.link);
    }

    inline void operator=(const ObjectPtr<T[]> &other) noexcept {
        link = other.link;
    }
    inline void operator=(ObjectPtr<T[]> &&other) noexcept {
        link = std::move(other.link);
    }
    inline void operator=(T *other) noexcept = delete;
    inline void operator=(void *other) noexcept = delete;
    inline void operator=(std::nullptr_t t) noexcept {
        link.Dispose();
    }

    inline T *operator->() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return GetPtr();
    }

    inline T &operator*() noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }
    inline T const &operator*() const noexcept {
#if defined(VSTL_DEBUG)
        //Null Check!
        assert(link.heapPtr != nullptr);
#endif
        return *GetPtr();
    }
    inline T &operator[](size_t key) noexcept {
        return GetPtr()[key];
    }

    inline T const &operator[](size_t key) const noexcept {
        return GetPtr()[key];
    }
    inline bool operator==(const ObjWeakPtr<T[]> &ptr) const noexcept {
        return link.heapPtr == ptr.link.heapPtr;
    }
    inline bool operator!=(const ObjWeakPtr<T[]> &ptr) const noexcept {
        return link.heapPtr != ptr.link.heapPtr;
    }
};
template<typename T>
inline ObjectPtr<T>::ObjectPtr(const ObjWeakPtr<T> &ptr) noexcept : link(ptr.link) {
}
template<typename T>
inline ObjectPtr<T>::ObjectPtr(ObjWeakPtr<T> &&ptr) noexcept : link(std::move(ptr.link)) {
}
template<typename T>
inline void ObjectPtr<T>::operator=(const ObjWeakPtr<T> &other) noexcept {
    link = other.link;
}

template<typename T>
inline ObjectPtr<T[]>::ObjectPtr(const ObjWeakPtr<T[]> &ptr) noexcept : link(ptr.link) {
}
template<typename T>
inline void ObjectPtr<T[]>::operator=(const ObjWeakPtr<T[]> &other) noexcept {
    link = other.link;
}

template<typename T>
inline static ObjectPtr<T> MakeObjectPtr(T *ptr) noexcept {
    return ObjectPtr<T>::MakePtr(ptr);
}

template<typename T>
inline static ObjectPtr<T> MakeObjectPtr(T *ptr, funcPtr_t<void(void *)> disposer) noexcept {
    return ObjectPtr<T>::MakePtr(ptr, disposer);
}

inline void PtrLink::operator=(const PtrLink &p) noexcept {
    operator=(static_cast<const PtrLinkBase &>(p));
}
inline void PtrLink::operator=(PtrLink &&p) noexcept {
    operator=(static_cast<PtrLinkBase &&>(p));
}
inline void PtrLink::operator=(const PtrWeakLink &p) noexcept {
    operator=(static_cast<const PtrLinkBase &>(p));
}
inline void PtrLink::operator=(PtrWeakLink &&p) noexcept {
    operator=(static_cast<PtrLinkBase &&>(p));
}
inline PtrLink::PtrLink(PtrLink const &p) : PtrLink(static_cast<const PtrLinkBase &>(p)) {}
inline PtrLink::PtrLink(PtrLink &&p) : PtrLink(static_cast<PtrLinkBase &&>(p)) {}
inline PtrLink::PtrLink(PtrWeakLink const &p) : PtrLink(static_cast<const PtrLinkBase &>(p)) {}
inline PtrLink::PtrLink(PtrWeakLink &&p) : PtrLink(static_cast<PtrLinkBase &&>(p)) {}
}// namespace vstd