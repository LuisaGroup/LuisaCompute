#include "SparseQueue.h"
#include <Resource/GpuAllocator.h>
#include <luisa/core/logging.h>
#include <Resource/SparseBuffer.h>
#include <Resource/SparseTexture.h>
namespace lc::dx {
SparseQueue::SparseQueue(
    Device *device)
    : CmdQueueBase(device, CmdQueueTag::Sparse),
      queue(device, device->defaultAllocator.get(), D3D12_COMMAND_LIST_TYPE_COPY) {
}
SparseQueue::~SparseQueue() {
}
void SparseQueue::Execute(
    CommandList &&cmdList) {
    vstd::vector<uint64> destroyList;
    for (auto &&i : cmdList.commands()) {
        if (i->tag() != Command::Tag::ESparseResourceUpdateCommand) [[unlikely]] {
            LUISA_ERROR("Illegal command.");
        }
        auto cmd = static_cast<SparseResourceUpdateCommand const *>(i.get());
        auto cmdQueue = queue.Queue();
        luisa::visit(
            [&]<typename T>(T const &t) {
                if constexpr (std::is_same_v<T, SparseTextureMapOperation>) {
                    auto tex = reinterpret_cast<SparseTexture *>(cmd->handle());
                    tex->AllocateTile(cmdQueue, t.start_tile, t.tile_count, t.mip_level);
                } else if constexpr (std::is_same_v<T, SparseBufferMapOperation>) {
                    auto buffer = reinterpret_cast<SparseBuffer *>(cmd->handle());
                    buffer->AllocateTile(cmdQueue, t.start_tile, t.tile_count);
                } else if constexpr (std::is_same_v<T, SparseTextureUnMapOperation>) {
                    auto tex = reinterpret_cast<SparseTexture *>(cmd->handle());
                    tex->DeAllocateTile(cmdQueue, t.start_tile, t.mip_level, destroyList);
                } else {
                    auto buffer = reinterpret_cast<SparseBuffer *>(cmd->handle());
                    buffer->DeAllocateTile(cmdQueue, t.start_tile, destroyList);
                }
            },
            cmd->operation());
    }
    queue.Signal(std::move(destroyList), cmdList.steal_callbacks());
}

void SparseQueue::Sync() {
    queue.Complete();
}
}// namespace lc::dx