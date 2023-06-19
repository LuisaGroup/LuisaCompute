#include <luisa/runtime/sparse_command_list.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

SparseCommandList::SparseCommandList() noexcept = default;

SparseCommandList::~SparseCommandList() noexcept {
    LUISA_ASSERT(_update_cmd.empty(),
                 "Destructing non-empty command list. "
                 "Did you forget to commit?");
}

SparseCommandList &SparseCommandList::operator<<(SparseUpdateTile &&tile) noexcept {
    _update_cmd.emplace_back(std::move(tile));
    return *this;
}

SparseCommandListCommit SparseCommandList::commit() noexcept {
    return {std::move(*this)};
}

void SparseCommandListCommit::operator()(DeviceInterface *device, uint64_t stream_handle) noexcept {
    device->update_sparse_resources(stream_handle, std::move(cmd_list._update_cmd));
    cmd_list._update_cmd.clear();
}

}// namespace luisa::compute
