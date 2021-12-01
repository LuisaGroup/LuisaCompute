//
// Created by Mike Smith on 2021/7/22.
//

#include <rtx/mesh.h>

namespace luisa::compute {

Command *Mesh::update() noexcept {
    if (_requires_rebuild) {
        LUISA_WARNING_WITH_LOCATION(
            "Mesh #{} requires rebuild rather than update. "
            "Automatically replacing with MeshRebuildCommand.",
            handle());
        return build();
    }
    return MeshUpdateCommand::create(handle());
}

Command *Mesh::build() noexcept {
    _requires_rebuild = false;
    return MeshBuildCommand::create(handle());
}

}// namespace luisa::compute
