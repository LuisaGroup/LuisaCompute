
//
// Created by Mike Smith on 2021/7/22.
//

#include <runtime/custom_pass.h>
#include <rtx/mesh.h>
#include <rtx/accel.h>

namespace luisa::compute {

luisa::unique_ptr<Command> Mesh::build(Mesh::BuildRequest request) noexcept {
    return MeshBuildCommand::create(
        handle(), request,
        _v_buffer, _v_buffer_offset, _v_buffer_size, _v_stride,
        _t_buffer, _t_buffer_offset, _t_buffer_size);
}

void CustomPass::_emplace(string &&name, Usage usage, const Mesh &v) noexcept {
    CustomCommand::ResourceBinding bindings;
    bindings.name = std::move(name);
    bindings.usage = usage;
    bindings.resource_view = CustomCommand::MeshView{
        .handle = v.handle()};
    _bindings.emplace_back(std::move(bindings));
}

}// namespace luisa::compute