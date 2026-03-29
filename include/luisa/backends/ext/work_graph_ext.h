#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/work_graph/work_graph_program.h>
#include <luisa/backends/ext/work_graph_ext_interface.h>

namespace luisa::compute {

template<typename>
[[nodiscard]] WorkGraphProgram Device::compile(
    const WorkGraph &work_graph,
    const ShaderOption &option) noexcept {

    if (extension<WorkGraphExt>() == nullptr) {
        LUISA_ERROR("work graphs not supported on this device");
    }

    return _create<WorkGraphProgram>(extension<WorkGraphExt>(), work_graph, option);
}

} // namespace luisa::compute