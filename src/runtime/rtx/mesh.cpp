
//
// Created by Mike Smith on 2021/7/22.
//

#include <runtime/rtx/mesh.h>
#include <runtime/rtx/accel.h>

namespace luisa::compute {

luisa::unique_ptr<Command> Mesh::build(Mesh::BuildRequest request) noexcept {
    return MeshBuildCommand::create(
        handle(), request,
        _v_buffer, _v_buffer_offset, _v_buffer_size, _v_stride,
        _t_buffer, _t_buffer_offset, _t_buffer_size);
}

Mesh::~Mesh() noexcept {
    if (*this) { device()->destroy_mesh(handle()); }
}

}// namespace luisa::compute
