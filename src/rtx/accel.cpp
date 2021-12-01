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

Accel Device::create_accel(AccelBuildHint hint) noexcept { return _create<Accel>(hint); }

Accel::Accel(Device::Interface *device, AccelBuildHint hint) noexcept
    : Resource{device, Tag::ACCEL, device->create_accel(hint)} {}

Command *Accel::update() noexcept {
    if (_requires_rebuild) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Accel #{} requires rebuild rather than update. "
            "Automatically replacing with AccelBuildCommand.",
            handle());
        return build();
    }
    return AccelUpdateCommand::create(handle());
}

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}

Accel &Accel::emplace_back(const Mesh &mesh, float4x4 transform) noexcept {
    _set_requires_rebuild();
    device()->emplace_back_instance_in_accel(handle(), mesh.handle(), transform);
    mesh._register(this);
    _meshes.emplace(&mesh);
    _size++;
    return *this;
}

Accel &Accel::set_transform(size_t index, float4x4 transform) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, handle());
    }
    device()->set_instance_transform_in_accel(handle(), index, transform);
    return *this;
}

Command *Accel::build() noexcept {
    _requires_rebuild = false;
    return AccelBuildCommand::create(handle());
}

Accel::~Accel() noexcept {
    for (auto m : _meshes) { m->_remove(this); }
}

}// namespace luisa::compute
