#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>

namespace luisa::compute {

class LUISA_RUNTIME_API WorkGraphDispatchCommand final : public CustomCommand {

private:
    uint64_t _handle;
    size_t _record_count;
    size_t _record_stride;
    void* _records;

public:
    // `records` is not copied; user must keep it alive until it is read by backend
    WorkGraphDispatchCommand(uint64_t handle,
                             size_t record_count,
                             size_t record_stride,
                             void* records) noexcept
        : _handle(handle), _record_count(record_count), _record_stride(record_stride), _records(records) {}

    WorkGraphDispatchCommand(WorkGraphDispatchCommand const &) noexcept = delete;
    WorkGraphDispatchCommand(WorkGraphDispatchCommand &&) noexcept = default;
    ~WorkGraphDispatchCommand() noexcept override = default;
    uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::WORK_GRAPH_DISPATCH); }

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto record_count() const noexcept { return _record_count; }
    [[nodiscard]] auto record_stride() const noexcept { return _record_stride; }
    [[nodiscard]] auto records() const noexcept { return _records; }

    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

}// namespace luisa::compute
