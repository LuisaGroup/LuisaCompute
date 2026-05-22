#pragma once

#include <luisa/backends/ext/work_graph_cmd.h>
#include <luisa/backends/ext/work_graph_ext_interface.h>

namespace luisa::compute {

namespace detail {

class WorkGraphProgramInvoke {
    uint64_t _handle;

public:
    explicit WorkGraphProgramInvoke(uint64_t handle) noexcept
        : _handle(handle) {}

    WorkGraphProgramInvoke(WorkGraphProgramInvoke &&) noexcept = default;
    WorkGraphProgramInvoke(const WorkGraphProgramInvoke &) noexcept = delete;
    WorkGraphProgramInvoke &operator=(WorkGraphProgramInvoke &&) noexcept = default;
    WorkGraphProgramInvoke &operator=(const WorkGraphProgramInvoke &) noexcept = delete;

    luisa::unique_ptr<Command> dispatch(size_t record_count, size_t record_stride, void* records) && noexcept {
        return luisa::make_unique<WorkGraphDispatchCommand>(_handle, record_count, record_stride, records);
    }

    luisa::unique_ptr<Command> dispatch(uint64_t gpu_input) && noexcept {
        return luisa::make_unique<WorkGraphDispatchCommand>(_handle, gpu_input);
    }
};
}// namespace detail

class WorkGraphProgram : public Resource {
    friend class WorkGraphExt;
    friend class Device;

private:
    WorkGraphExt* _work_graph_ext{};

    WorkGraphProgram(
        DeviceInterface *device,
        WorkGraphExt* work_graph_ext,
        const WorkGraph& work_graph,
        const ShaderOption &option
    ) noexcept
        : Resource(device, Tag::WORK_GRAPH_PROGRAM, work_graph_ext->create_work_graph_program(work_graph, option)),
          _work_graph_ext(work_graph_ext) {}


public:
    WorkGraphProgram() noexcept = default;
    WorkGraphProgram(WorkGraphProgram &&) noexcept = default;
    WorkGraphProgram(WorkGraphProgram const &) noexcept = delete;
    WorkGraphProgram &operator=(WorkGraphProgram &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    WorkGraphProgram &operator=(WorkGraphProgram const &) noexcept = delete;
    ~WorkGraphProgram() noexcept override {
        if (*this) { _work_graph_ext->destroy_work_graph_program(handle()); }
    }
    using Resource::operator bool;
    using Resource::release;
    [[nodiscard]] auto operator()() const noexcept {
        detail::WorkGraphProgramInvoke invoke {handle()};
        return std::move(invoke);
    }
};

} // namespace luisa::compute