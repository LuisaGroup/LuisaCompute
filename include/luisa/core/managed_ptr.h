#pragma once

#include <cstddef>
#include <atomic>

#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/memory.h>

#ifndef NDEBUG
#define LUISA_MANAGED_OBJECT_CANARY 0xDEADBEEF
#endif

namespace luisa {

template<typename T>
class ManagedPtr;

namespace detail {

struct ManagedPtrLowLevelOp;

class ManagedObject {

private:
    std::atomic<int32_t> _ref_count;
#ifdef LUISA_MANAGED_OBJECT_CANARY
    volatile uint32_t _canary{LUISA_MANAGED_OBJECT_CANARY};
#endif

public:
    ManagedObject() noexcept : _ref_count{1} {}
#ifdef LUISA_MANAGED_OBJECT_CANARY
    virtual ~ManagedObject() noexcept { _canary = 0u; }
    void validate_canary() const noexcept {
        assert(_canary == LUISA_MANAGED_OBJECT_CANARY &&
               "ManagedObject canary has been corrupted, "
               "object is likely already destroyed.");
    }
#else
    virtual ~ManagedObject() noexcept = default;
    void validate_canary() const noexcept { /* no canary to validate in release mode */ }
#endif
    ManagedObject(ManagedObject &&) = delete;
    ManagedObject(const ManagedObject &) = delete;
    ManagedObject &operator=(ManagedObject &&) = delete;
    ManagedObject &operator=(const ManagedObject &) = delete;

private:
    friend ManagedPtrLowLevelOp;
    ManagedObject *do_retain() noexcept {
        validate_canary();
        [[maybe_unused]] auto old_refcount = _ref_count.fetch_add(1, std::memory_order_relaxed);
        assert(old_refcount > 0 && "Retained object is likely already destroyed.");
        return this;
    }
    void do_release() noexcept {
        validate_canary();
        auto old_refcount = _ref_count.fetch_sub(1, std::memory_order_acq_rel);
        assert(old_refcount > 0 && "Releasing object is likely already destroyed.");
        if (old_refcount == 1) { luisa::delete_with_allocator(this); }
    }
};

struct ManagedPtrLowLevelOp {
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain_nonnull(T *o) noexcept {
        assert(o != nullptr && "Null pointer dereferenced.");
        return static_cast<T *>(o->do_retain());
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain(T *o) noexcept {
        return o == nullptr ? nullptr : retain_nonnull(o);
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    static void release(T *o) noexcept {
        if (o != nullptr) { o->do_release(); }
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    static void reset(ManagedPtr<T> &m, T *ptr) noexcept {
        m.reset(ptr);
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto transfer(ManagedPtr<T> &m) noexcept {
        return m.transfer();
    }
};

}// namespace detail

template<typename T>
class ManagedPtr : public ManagedPtr<const T> {

    template<typename U>
    friend class ManagedPtr;

public:
    ManagedPtr() noexcept = default;
    ManagedPtr(std::nullptr_t) noexcept {}
    ~ManagedPtr() noexcept = default;

    ManagedPtr(const ManagedPtr &) noexcept = default;
    ManagedPtr(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(const ManagedPtr &) noexcept = default;

    ManagedPtr &operator=(std::nullptr_t) noexcept {
        ManagedPtr<const T>::operator=(nullptr);
        return *this;
    }

    ManagedPtr(const ManagedPtr<const T> &) noexcept = delete;
    ManagedPtr(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(const ManagedPtr<const T> &) noexcept = delete;

    template<typename U>
        requires(!std::is_const_v<U> && std::derived_from<U, T>)
    ManagedPtr(ManagedPtr<U> &&other) noexcept
        : ManagedPtr<const T>{std::move(other)} {}

    template<typename U>
        requires(!std::is_const_v<U> && std::derived_from<U, T>)
    ManagedPtr(const ManagedPtr<U> &other) noexcept
        : ManagedPtr<const T>{other} {}

    template<typename U>
        requires(!std::is_const_v<U> && std::derived_from<U, T>)
    ManagedPtr &operator=(ManagedPtr<U> &&other) noexcept {
        ManagedPtr<const T>::operator=(std::move(other));
        return *this;
    }

    template<typename U>
        requires(!std::is_const_v<U> && std::derived_from<U, T>)
    ManagedPtr &operator=(const ManagedPtr<U> &other) noexcept {
        ManagedPtr<const T>::operator=(other);
        return *this;
    }

    [[nodiscard]] T *get() const noexcept { return const_cast<T *>(ManagedPtr<const T>::get()); }
    [[nodiscard]] T *operator->() const noexcept { return get(); }
    [[nodiscard]] T &operator*() const noexcept { return *get(); }

    template<typename U>
        requires requires(T *p) { static_cast<U *>(p); }
    [[nodiscard]] auto into() && noexcept {
        ManagedPtr<U> p;
        p.reset(static_cast<std::remove_const_t<U> *>(this->transfer()));
        return p;
    }
};

template<typename T>
class ManagedPtr<const T> {

    template<typename U>
    friend class ManagedPtr;

private:
    static_assert(std::derived_from<T, detail::ManagedObject>);
    T *_object{nullptr};

protected:
    friend detail::ManagedPtrLowLevelOp;
    [[nodiscard]] T *transfer() noexcept {
        return std::exchange(_object, nullptr);
    }
    void reset(T *new_object = nullptr) noexcept {
        luisa::detail::ManagedPtrLowLevelOp::release(
            std::exchange(_object, new_object));
    }

public:
    ManagedPtr() noexcept = default;
    ManagedPtr(std::nullptr_t) noexcept {}
    ~ManagedPtr() noexcept { reset(); }
    ManagedPtr(ManagedPtr &&other) noexcept {
        reset(other.transfer());
    }
    ManagedPtr(const ManagedPtr &other) noexcept {
        reset(luisa::detail::ManagedPtrLowLevelOp::
                  retain(const_cast<T *>(other.get())));
    }
    ManagedPtr &operator=(ManagedPtr &&other) noexcept {
        if (&other != this) {
            reset(other.transfer());
        }
        return *this;
    }
    ManagedPtr &operator=(const ManagedPtr &other) noexcept {
        if (&other != this) {
            if (auto p = const_cast<T *>(other.get()); p != this->get()) {
                reset(luisa::detail::ManagedPtrLowLevelOp::retain(p));
            }
        }
        return *this;
    }

    ManagedPtr &operator=(std::nullptr_t) noexcept {
        reset();
        return *this;
    }

    // from derived ManagedPtr<const T>
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    ManagedPtr(ManagedPtr<U> &&other) noexcept {
        reset(other.transfer());
    }
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    ManagedPtr(const ManagedPtr<U> &other) noexcept {
        reset(luisa::detail::ManagedPtrLowLevelOp::
                  retain(const_cast<T *>(other.get())));
    }
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    ManagedPtr &operator=(ManagedPtr<U> &&other) noexcept {
        if (other.get() != this->get()) {
            reset(other.transfer());
        }
        return *this;
    }
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    ManagedPtr &operator=(const ManagedPtr<U> &other) noexcept {
        if (auto p = const_cast<T *>(other.get()); p != this->get()) {
            reset(luisa::detail::ManagedPtrLowLevelOp::retain(p));
        }
        return *this;
    }

public:
    [[nodiscard]] const T *get() const noexcept { return _object; }
    [[nodiscard]] const T *operator->() const noexcept { return get(); }
    [[nodiscard]] const T &operator*() const noexcept { return *get(); }

    [[nodiscard]] bool is_engaged() const noexcept { return _object != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return is_engaged(); }

    template<typename U>
        requires requires(T *lhs, U *rhs) { lhs == rhs; }
    [[nodiscard]] bool operator==(const U *rhs) const noexcept {
        return get() == rhs;
    }

    template<typename U>
        requires requires(T *lhs, U *rhs) { lhs == rhs; }
    [[nodiscard]] bool operator==(const ManagedPtr<U> &rhs) const noexcept {
        return get() == rhs.get();
    }

    [[nodiscard]] bool operator==(std::nullptr_t) const { return !is_engaged(); }

    template<typename U>
        requires requires(T *p) { static_cast<const U *>(p); }
    [[nodiscard]] auto into() && noexcept {
        ManagedPtr<const U> p;
        p.reset(static_cast<std::remove_const_t<U> *>(this->transfer()));
        return p;
    }
};

template<typename T, typename Base = detail::ManagedObject>
    requires std::derived_from<Base, detail::ManagedObject>
class Managed : public Base {

public:
    static_assert(std::same_as<std::remove_cv_t<T>, T>);
    using Super = Managed;
    using Base::Base;

public:
    [[nodiscard]] auto lock() noexcept {
        auto self = detail::ManagedPtrLowLevelOp::
            retain_nonnull(static_cast<T *>(this));
        ManagedPtr<T> p;
        detail::ManagedPtrLowLevelOp::reset(p, self);
        return p;
    }
    [[nodiscard]] auto lock() const noexcept {
        return static_cast<ManagedPtr<const T>>(
            const_cast<Managed *>(this)->lock());
    }
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    [[nodiscard]] auto lock_into() noexcept {
        return lock().template into<U>();
    }
    template<typename U>
        requires std::derived_from<std::remove_const_t<U>, T>
    [[nodiscard]] auto lock_into() const noexcept {
        return lock().template into<const U>();
    }
};

template<typename T, typename... Args>
    requires std::derived_from<T, detail::ManagedObject>
[[nodiscard]] ManagedPtr<T> make_managed(Args &&...args) noexcept {
    auto o = luisa::new_with_allocator<std::remove_const_t<T>>(std::forward<Args>(args)...);
    assert(std::addressof(*o) == std::addressof(*static_cast<detail::ManagedObject *>(o)) &&
           "ManagedObject should be the first non-empty base class of its derived classes.");
    ManagedPtr<T> p;
    detail::ManagedPtrLowLevelOp::reset(p, o);
    return p;
}

template<typename T>
[[nodiscard]] luisa::span<T *> to_unowned_span(luisa::span<ManagedPtr<T>> managed_span) noexcept {
    auto data = managed_span.data();
    auto size = managed_span.size();
    return {reinterpret_cast<T **>(data), size};
}

template<typename T>
[[nodiscard]] luisa::span<const T *const> to_unowned_span(luisa::span<const ManagedPtr<T>> managed_span) noexcept {
    auto data = managed_span.data();
    auto size = managed_span.size();
    return {reinterpret_cast<const T *const *>(data), size};
}

template<typename T>
struct hash<ManagedPtr<T>> {
    template<typename U>
    [[nodiscard]] constexpr uint64_t
    operator()(const ManagedPtr<U> &ptr, uint64_t seed = hash64_default_seed) const noexcept {
        return hash<U *>{}(ptr.get(), seed);
    }
    template<typename U>
    [[nodiscard]] constexpr uint64_t
    operator()(U *ptr, uint64_t seed = hash64_default_seed) const noexcept {
        return hash<U *>{}(ptr, seed);
    }
};

}// namespace luisa
