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
            handle());
        return build();
    }
    return MeshUpdateCommand::create(handle());
}

Command *Mesh::build() noexcept {
    _subject->notify_all();
    _requires_rebuild = false;
    return MeshBuildCommand::create(handle());
}

}// namespace luisa::compute
