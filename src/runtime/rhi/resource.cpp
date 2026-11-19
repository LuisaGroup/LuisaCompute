#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>
#include <luisa/core/pool.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] static auto allocate_resource_uid() noexcept {
    static std::atomic_uint64_t uid{0u};
    return ++uid;
}

/// Process-wide recycle bin for reference-count blocks.
/// \details Deliberately leaked instead of being a namespace-scope object: a resource
/// may outlive static destruction (a namespace-scope `Buffer`, or a `Device` released
/// from another static object's destructor) and would then return its block to a pool
/// that no longer exists. The pool's own storage is reclaimed by the process.
[[nodiscard]] static ConcurrentPool<ResourceRefCount> &ref_count_pool() noexcept {
    static auto pool = luisa::new_with_allocator<ConcurrentPool<ResourceRefCount>>();
    return *pool;
}

}// namespace detail

Resource::Resource(Resource &&rhs) noexcept
    : _device{std::move(rhs._device)},
      _info{rhs._info},
      _tag{rhs._tag},
      _uid{rhs._uid},
      _rc{rhs._rc.exchange(nullptr, std::memory_order_relaxed)} {
    // Ownership (including the shared block) moves to `this`; the source stays a
    // valid, destructible but invalid resource.
    rhs._info.invalidate();
}

Resource::Resource(const Resource &rhs) noexcept
    : _device{rhs._device},
      _info{rhs._info},
      _tag{rhs._tag},
      _uid{rhs._uid} {
    // _rc is picked up (and allocated if needed) by _share
    _share(rhs);
}

Resource::Resource(DeviceInterface *device,
                   Resource::Tag tag,
                   const ResourceCreationInfo &info) noexcept
    : _device{device->shared_from_this()},
      _info{info}, _tag{tag},
      _uid{detail::allocate_resource_uid()} {}

void Resource::set_name(luisa::string_view name) const noexcept {
    _device->set_name(_tag, _info.handle, name);
}

void Resource::_check_same_derived_types(const Resource &lhs,
                                         const Resource &rhs) noexcept {
    if (lhs && rhs) {
        LUISA_ASSERT(lhs._tag == rhs._tag,
                     "Cannot assign resources of different types.");
    }
}

void Resource::_error_invalid() noexcept {
    LUISA_ERROR_WITH_LOCATION("Invalid resource.");
}

void Resource::_share(const Resource &other) noexcept {
    // An invalid (default-constructed, moved-from or released) resource owns nothing,
    // so there is nothing to keep alive and no reason to pay for a shared block.
    if (!_info.valid()) { return; }
    auto rc = other._rc.load(std::memory_order_acquire);
    if (rc == nullptr) {
        // First copy of a single-owner resource: promote the source to shared ownership.
        // The block starts with the source's reference (count == 1) and is installed with
        // a compare-and-swap, because two threads may copy the same handle at the same
        // time; the loser hands its block straight back to the pool.
        auto fresh = detail::ref_count_pool().create();
        if (other._rc.compare_exchange_strong(rc, fresh,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            rc = fresh;
        } else {
            detail::ref_count_pool().destroy(fresh);
        }
    }
    rc->count.fetch_add(1, std::memory_order_relaxed);
    _rc.store(rc, std::memory_order_release);
}

void Resource::_release() noexcept {
    if (auto rc = _rc.exchange(nullptr, std::memory_order_acq_rel)) {
        if (rc->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // This instance was the last live owner: recycle the block *before* tearing
            // the object down, so that a `destroy()` implementation is free to copy a
            // resource of its own and allocate a block. `destroy()` is a plain virtual
            // call that lands on the concrete override because we are inside that
            // class's destructor body, and it runs while `*this` and its `_info` are
            // still intact -- every `destroy()` implementation guards its API call with
            // a validity check.
            detail::ref_count_pool().destroy(rc);
            destroy();
        }
        _info.invalidate();
        return;
    }
    // Single owner: destroy in place. The dynamic type is still the concrete class
    // here (every derived destructor calls _release() from its own body), so the
    // virtual dispatch reaches the right override.
    destroy();
    _info.invalidate();
}

Resource &Resource::operator=(Resource &&rhs) noexcept {
    if (this != &rhs) [[likely]] {
        _check_same_derived_types(*this, rhs);
        _release();// give up the current ownership first
        _device = std::move(rhs._device);
        _info = rhs._info;
        _tag = rhs._tag;
        _uid = rhs._uid;
        _rc.store(rhs._rc.exchange(nullptr, std::memory_order_relaxed),
                  std::memory_order_release);
        rhs._info.invalidate();
    }
    return *this;
}

Resource &Resource::operator=(const Resource &rhs) noexcept {
    if (this != &rhs) [[likely]] {
        _check_same_derived_types(*this, rhs);
        _release();// give up the current ownership first
        _device = rhs._device;
        _info = rhs._info;
        _tag = rhs._tag;
        _uid = rhs._uid;
        _share(rhs);
    }
    return *this;
}

Resource::~Resource() noexcept {
    // Derived destructors give up their ownership through `_release()` while the
    // dynamic type is still the concrete class -- that is the only point at which the
    // teardown can be dispatched. Both a surviving block and a still-valid handle mean
    // a subclass forgot that call, which would leak the underlying API object. `_info`
    // is checked as well because the single-owner path leaves no block behind, so
    // `_rc == nullptr` alone cannot tell "released" from "never released".
    //
    // Debug-only on purpose: in a release build the only consequence of a violation is
    // a leaked device object, which is not worth an abort in a shipping application --
    // but the check is free there, so nothing is paid for carrying it.
    LUISA_DEBUG_ASSERT(_rc.load(std::memory_order_relaxed) == nullptr && !_info.valid(),
                       "A Resource subclass must call Resource::_release() "
                       "from its destructor, but tag {} did not.",
                       static_cast<uint>(_tag));
    // manually reset to workaround "dispose"
    _device.reset();
}

ResourceCreationInfo Resource::release() noexcept {
    // `release()` hands the ownership of the underlying object to the caller, so this
    // instance has to be its only owner: for a shared resource the remaining copies
    // would destroy the object the caller now believes it owns.
    if (auto rc = _rc.load(std::memory_order_acquire)) {
        // Fail closed: the check runs before anything is mutated, so a violation leaves
        // the resource untouched instead of dropping a block other owners still use.
        LUISA_ASSERT(rc->count.load(std::memory_order_acquire) == 1,
                     "Cannot release a resource that is still shared by "
                     "{} other instance(s).",
                     rc->count.load(std::memory_order_relaxed) - 1);
        _rc.store(nullptr, std::memory_order_release);
        rc->count.store(0, std::memory_order_relaxed);
        detail::ref_count_pool().destroy(rc);
    }
    auto info = _info;
    _info.invalidate();
    return info;
}

}// namespace luisa::compute
