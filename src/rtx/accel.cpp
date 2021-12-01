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
    : _device{device->shared_from_this()},
      _handle{device->create_accel(hint)} {}

Accel::~Accel() noexcept { _destroy(); }

Command *Accel::update() noexcept {
    if (_requires_rebuild) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Accel #{} requires rebuild rather than update. "
            "Automatically replacing with AccelBuildCommand.",
            _handle);
        return build();
    }
    return AccelUpdateCommand::create(_handle);
}

Var<Hit> Accel::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_closest(ray);
}

Var<bool> Accel::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{*this}.trace_any(ray);
}

Accel &Accel::emplace_back(const Mesh &mesh, float4x4 transform) noexcept {
    _set_requires_rebuild();
    device()->emplace_back_instance_in_accel(_handle, mesh._handle, transform);
    mesh._register(this);
    _meshes.emplace(&mesh);
    _size++;
    return *this;
}

Accel &Accel::set_transform(size_t index, float4x4 transform) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid index {} in accel #{}.",
            index, _handle);
    }
    device()->set_instance_transform_in_accel(_handle, index, transform);
    return *this;
}

Command *Accel::build() noexcept {
    _requires_rebuild = false;
    return AccelBuildCommand::create(_handle);
}

void Accel::_set_requires_rebuild() noexcept {
    _requires_rebuild = true;
}

void Accel::_replace(const Mesh *prev, const Mesh *curr) noexcept {
    _meshes.erase(prev);
    _meshes.emplace(curr);
}

void Accel::_destroy() noexcept {
    if (*this) {
        for (auto m : _meshes) {
            m->_remove(this);
        }
        _device->destroy_accel(_handle);
    }
}

Accel::Accel(Accel &&another) noexcept
    : _device{std::move(another._device)},
      _handle{another._handle},
      _meshes{std::move(another._meshes)},
      _size{another._size},
      _requires_rebuild{another._requires_rebuild} {
    for (auto m : _meshes) {
        m->_remove(&another);
        m->_register(this);
    }
}

Accel &Accel::operator=(Accel &&rhs) noexcept {
    if (this != &rhs) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _meshes = std::move(rhs._meshes);
        _size = rhs._size;
        _requires_rebuild = rhs._requires_rebuild;
        for (auto m : _meshes) {
            m->_remove(&rhs);
            m->_register(this);
        }
    }
    return *this;
}

}// namespace luisa::compute
