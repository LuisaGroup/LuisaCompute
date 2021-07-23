//
// Created by Mike Smith on 2021/6/24.
//

#include <ast/function_builder.h>
#include <runtime/shader.h>
#include <rtx/accel.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const Accel &accel) noexcept {
    auto v = _kernel.arguments()[_argument_index++].uid();
    _dispatch_command()->encode_accel(v, accel.handle());
    return *this;
}

}// namespace detail

Accel Device::create_accel() noexcept { return _create<Accel>(); }

Accel::Accel(Device::Handle device) noexcept
    : _device{std::move(device)},
      _handle{_device->create_accel()} {}

void Accel::_destroy() noexcept {
    if (*this) { _device->destroy_accel(_handle); }
}

Accel::~Accel() noexcept { _destroy(); }

Command *Accel::update() noexcept {
    if (!_built) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Geometry #{} is not built when updating.",
            _handle);
    }
    return AccelUpdateCommand::create(_handle);
}

Accel &Accel::operator=(Accel &&rhs) noexcept {
    if (&rhs != this) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _built = rhs._built;
    }
    return *this;
}

detail::Expr<Hit> Accel::trace_closest(detail::Expr<Ray> ray) const noexcept {
    return detail::Expr<Accel>{*this}.trace_closest(ray);
}

detail::Expr<bool> Accel::trace_any(detail::Expr<Ray> ray) const noexcept {
    return detail::Expr<Accel>{*this}.trace_any(ray);
}

Command *Accel::update(
    size_t first,
    size_t count,
    const float4x4 *transforms) noexcept {

    if (!_built) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Geometry #{} is not built when updating.",
            _handle);
    }
    return AccelUpdateCommand::create(
        _handle,
        std::span{transforms, count},
        first);
}

Command *Accel::build(
    AccelBuildHint mode,
    std::span<const uint64_t> mesh_handles,
    std::span<const float4x4> transforms) noexcept {
    _built = true;
    return AccelBuildCommand::create(_handle, mode, mesh_handles, transforms);
}

}// namespace luisa::compute
