#pragma once
#include <luisa/core/concepts.h>
#include <luisa/runtime/rhi/tile_modification.h>
#include <luisa/runtime/stream_event.h>

namespace luisa::compute {
struct SparseCommandListCommit;
class LC_RUNTIME_API SparseCommandList : concepts::Noncopyable {
    friend struct SparseCommandListCommit;

private:
    luisa::vector<SparseUpdateTile> _update_cmd;

public:
    SparseCommandList() noexcept;
    ~SparseCommandList() noexcept;
    SparseCommandList(SparseCommandList const &) = delete;
    SparseCommandList(SparseCommandList &&) noexcept = default;
    SparseCommandList &operator=(SparseCommandList const &) = delete;
    SparseCommandList &operator=(SparseCommandList &&) = default;
    SparseCommandList &operator<<(SparseUpdateTile &&tile) noexcept;
    [[nodiscard]] SparseCommandListCommit commit() noexcept;
    [[nodiscard]] auto size() const noexcept { return _update_cmd.size(); }
    [[nodiscard]] auto empty() const noexcept { return _update_cmd.empty(); }
};
struct LC_RUNTIME_API SparseCommandListCommit {
    SparseCommandList cmd_list;
    void operator()(DeviceInterface *device, uint64_t stream_handle) noexcept;
};
LUISA_MARK_STREAM_EVENT_TYPE(SparseCommandListCommit)
}// namespace luisa::compute