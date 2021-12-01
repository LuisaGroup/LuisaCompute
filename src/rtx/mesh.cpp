//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>
#include <rtx/accel.h>

namespace luisa::compute {

Command *Mesh::update() noexcept {
    if (_requires_rebuild) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Mesh #{} requires rebuild rather than update. "
            "Automatically replacing with MeshRebuildCommand.",
            _handle);
        return build();
    }
    return MeshUpdateCommand::create(_handle);
}

Command *Mesh::build() noexcept {
    for (auto &&o : _observers) {
        o->_set_requires_rebuild();
    }
    _requires_rebuild = false;
    return MeshBuildCommand::create(_handle);
}

void Mesh::_register(Accel *accel) const noexcept {
    _observers.emplace(accel);
}

void Mesh::_remove(Accel *accel) const noexcept {
    _observers.erase(accel);
}

void Mesh::_destroy() noexcept {
    if (*this) {
        if (!_observers.empty()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Mesh #{} being destructed has non-empty observers.",
                _handle);
        }
        _device->destroy_mesh(_handle);
    }
}

Mesh::~Mesh() noexcept { _destroy(); }

Mesh::Mesh(Mesh &&another) noexcept
    : _device{std::move(another._device)},
      _handle{another._handle},
      _observers{std::move(another._observers)},
      _requires_rebuild{another._requires_rebuild} {
    for (auto o : _observers) {
        o->_replace(&another, this);
    }
}

Mesh &Mesh::operator=(Mesh &&rhs) noexcept {
    if (this != &rhs) [[likely]] {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _observers = std::move(rhs._observers);
        _requires_rebuild = rhs._requires_rebuild;
        for (auto o : _observers) {
            o->_replace(&rhs, this);
        }
    }
    return *this;
}

}// namespace luisa::compute
