//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>
#include <rtx/accel.h>

namespace luisa::compute {

Command *Mesh::update() noexcept {
    return MeshUpdateCommand::create(_resource->handle());
}

Command *Mesh::build() noexcept {
    return MeshBuildCommand::create(_resource->handle());
}

}// namespace luisa::compute
