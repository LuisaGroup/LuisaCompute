//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>

namespace luisa::compute {

Mesh Device::create_mesh() noexcept { return _create<Mesh>(); }

Command *Mesh::update() const noexcept { return MeshUpdateCommand::create(_handle); }

void Mesh::_destroy() noexcept {
    if (*this) { _device->destroy_mesh(_handle); }
}

Mesh &Mesh::operator=(Mesh &&rhs) noexcept {
    if (&rhs != this) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
    }
    return *this;
}

}
