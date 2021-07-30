//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>

namespace luisa::compute {

Mesh Device::create_mesh() noexcept { return _create<Mesh>(); }

Command *Mesh::update() noexcept {
    if (!_built) {
        LUISA_ERROR_WITH_LOCATION(
            "Mesh #{} is not built when updating.",
            handle());
    }
    return MeshUpdateCommand::create(handle());
}

}// namespace luisa::compute
