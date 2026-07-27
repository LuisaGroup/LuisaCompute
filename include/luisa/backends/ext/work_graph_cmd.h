#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>

namespace luisa::compute {

class WorkGraphDispatchCommand final : public ShaderDispatchCommandBase, public CustomCommand {
public:
    struct NodeCPUInput {
        size_t _record_count;
        size_t _record_stride;
        void* _records;
    };

    struct NodeGPUInput {
        uint64_t _gpu_input;
    };

private:
    uint64_t _handle;
    luisa::variant<NodeCPUInput, NodeGPUInput> _records;


public:
    // CPU input: `records` is not copied; user must keep it alive until it is read by backend
    WorkGraphDispatchCommand(uint64_t handle,
                             size_t record_count,
                             size_t record_stride,
                             void* records) noexcept
        : ShaderDispatchCommandBase { handle, {}, 0 }, 
          _records(NodeCPUInput(record_count, record_stride, records)) {}

    WorkGraphDispatchCommand(uint64_t handle,
                             uint64_t gpu_input) noexcept
        : ShaderDispatchCommandBase { handle, {}, 0 },
          _records(NodeGPUInput(gpu_input)) {}

    WorkGraphDispatchCommand(WorkGraphDispatchCommand const &) noexcept = delete;
    WorkGraphDispatchCommand(WorkGraphDispatchCommand &&) noexcept = default;
    ~WorkGraphDispatchCommand() noexcept override = default;
    uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::WORK_GRAPH_DISPATCH); }

    [[nodiscard]] auto records() const noexcept { return _records; }

    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

}// namespace luisa::compute
