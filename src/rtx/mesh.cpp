//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>
#include <rtx/accel.h>

namespace luisa::compute {

Command *Mesh::build(Mesh::BuildRequest request) noexcept {
    return MeshBuildCommand::create(handle(), request, _v_buffer, _t_buffer);
}

}// namespace luisa::compute
