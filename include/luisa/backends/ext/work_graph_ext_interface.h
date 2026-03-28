#pragma once

#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

class WorkGraphExt : public DeviceExtension {
protected:
    ~WorkGraphExt() noexcept = default;
public:
    static constexpr luisa::string_view name = "WorkGraphExt";
    // shader
    [[nodiscard]] virtual ResourceCreationInfo create_work_graph_program(const WorkGraph& work_graph, const ShaderOption& option) noexcept = 0;
    virtual void destroy_work_graph_program(uint64_t handle) = 0;
};


} // namespace luisa::compute