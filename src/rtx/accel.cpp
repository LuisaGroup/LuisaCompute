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

Accel::Accel(Device::Interface *device, AccelBuildHint hint) noexcept
    : Resource{device, Tag::ACCEL, device->create_accel(hint)} {}

Command *Accel::update() noexcept {
    if (!_built) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Geometry #{} is not built when updating.",
            handle());
    }
    return AccelUpdateCommand::create(handle());
}

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}

Command* Accel::build() noexcept{
    _requires_rebuild = false;
    _dirty_begin = 0u;
    _dirty_count = 0u;
return AccelBuildCommand::create(handle());
}

Command *Accel::update(
    size_t first,
    size_t count,
    const float4x4 *transforms) noexcept {

    if (!_built) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Geometry #{} is not built when updating.",
            handle());
    }
    return AccelUpdateCommand::create(
        handle(),
        std::span{transforms, count},
        first);
}

Command *Accel::build(
    AccelBuildHint mode,
    std::span<const uint64_t> mesh_handles,
    std::span<const float4x4> transforms) noexcept {
    _built = true;
    return AccelBuildCommand::create(handle(), mode, mesh_handles, transforms);
}

}// namespace luisa::compute
