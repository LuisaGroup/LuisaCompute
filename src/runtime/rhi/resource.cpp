#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] static auto allocate_resource_uid() noexcept {
    static std::atomic_uint64_t uid{0u};
    return ++uid;
}

}// namespace detail

Resource::Resource(Resource &&rhs) noexcept
    : _device{std::move(rhs._device)},
      _info{rhs._info},
      _tag{rhs._tag},
      _uid{rhs._uid} { rhs._info.invalidate(); }

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
                     "Cannot move resources of different types.");
    }
}

void Resource::_error_invalid() noexcept {
    LUISA_ERROR_WITH_LOCATION("Invalid resource.");
}

Resource::~Resource() noexcept {
    // manually reset to workaround "dispose"
    _device.reset();
}

void Resource::dispose() noexcept {
    // trick here: we can not call derive class' destructor
    // Resource destructor might be called multiple times
    if (*this) {
        this->~Resource();
        _info.invalidate();
    }
}
}// namespace luisa::compute
