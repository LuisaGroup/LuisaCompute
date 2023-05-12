#include "DStorageCommandQueue.h"
#include <DXApi/LCEvent.h>
#include <core/logging.h>
#include <backends/ext/dstorage_ext_interface.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
namespace lc::dx {
void DStorageCommandQueue::ExecuteThread() {
    while (enabled) {
        auto ExecuteAllocator = [&](auto &b) {
            device->WaitFence(cmdFence.Get(), b);
            {
                std::lock_guard lck(mtx);
                executedFrame = b;
            }
            mainCv.notify_all();
        };
        auto ExecuteCallbacks = [&](auto &vec) {
            for (auto &&i : vec.first) {
                i();
            }
            {
                std::lock_guard lck(mtx);
                executedFrame = vec.second;
            }
            mainCv.notify_all();
        };
        auto ExecuteEvent = [&](auto &pair) {
            auto evt = pair.first;
            auto tarFrame = pair.second;
            device->WaitFence(evt->fence.Get(), tarFrame);
            {
                std::lock_guard lck(evt->eventMtx);
                evt->finishedEvent = std::max(tarFrame, evt->finishedEvent);
            }
            evt->cv.notify_all();
        };
        while (auto b = executedAllocators.pop()) {
            b->multi_visit(
                ExecuteAllocator,
                ExecuteCallbacks,
                ExecuteEvent);
        }
        std::unique_lock lck(mtx);
        while (enabled && executedAllocators.length() == 0) {
            waitCv.wait(lck);
        }
    }
}
void DStorageCommandQueue::AddEvent(LCEvent const *evt) {
    executedAllocators.push(evt, evt->fenceIndex);
    mtx.lock();
    mtx.unlock();
    waitCv.notify_one();
}
uint64 DStorageCommandQueue::Execute(luisa::compute::CommandList &&list) {
    for (auto &&i : list.commands()) {
        if (i->tag() != luisa::compute::Command::Tag::ECustomCommand ||
            static_cast<luisa::compute::CustomCommand const *>(i.get())->uuid() != luisa::compute::dstorage_command_uuid) [[unlikely]] {
            LUISA_ERROR("Only DStorage command allowed in this stream.");
        }
        auto cmd = static_cast<luisa::compute::DStorageReadCommand const *>(i.get());
        DSTORAGE_REQUEST request{};
        auto file = reinterpret_cast<DStorageFileImpl *>(cmd->file_handle);
        request.Source.File.Source = file->file.Get();
        request.Source.File.Offset = cmd->file_offset;
        request.Source.File.Size = cmd->size_bytes;
        luisa::visit(
            [&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<DStorageReadCommand::GDeflateCompression, T>) {
                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                request.UncompressedSize = t.uncompressed_size;
            }
            },
            cmd->compress_option);
        luisa::visit(
            [&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::BufferEnqueue>) {
                request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                request.Destination.Buffer.Resource = reinterpret_cast<Buffer *>(t.buffer_handle)->GetResource();
                request.Destination.Buffer.Offset = t.buffer_offset;
                request.Destination.Buffer.Size = cmd->size_bytes;
            } else if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::ImageEnqueue>) {
                request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
                auto tex = reinterpret_cast<TextureBase *>(t.image_handle);
                request.Destination.Texture.SubresourceIndex = t.mip_level;
                request.Destination.Texture.Resource = tex->GetResource();
                request.Destination.Texture.Region = D3D12_BOX{
                    0u, 0u, 0u,
                    tex->Width(), tex->Height(), tex->Depth()};
            } else {
                request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                request.Destination.Memory.Buffer = t.dst_ptr;
                request.Destination.Memory.Size = cmd->size_bytes;
            } },
            cmd->enqueue_cmd());
        queue->EnqueueRequest(&request);
    }
    auto curFrame = ++lastFrame;
    queue->EnqueueSignal(cmdFence.Get(), curFrame);
    queue->Submit();
    executedAllocators.push(curFrame);
    curFrame = ++lastFrame;
    executedAllocators.push(std::move(list.steal_callbacks()), curFrame);
    mtx.lock();
    mtx.unlock();
    waitCv.notify_one();
    return curFrame;
}
void DStorageCommandQueue::Complete(uint64 fence) {
    std::unique_lock lck(mtx);
    while (executedFrame < fence) {
        mainCv.wait(lck);
    }
}
void DStorageCommandQueue::Complete() {
    std::unique_lock lck(mtx);
    while (executedAllocators.length() > 0) {
        mainCv.wait(lck);
    }
}
DStorageCommandQueue::DStorageCommandQueue(IDStorageFactory *factory, Device *device)
    : device(device),
      CmdQueueBase(CmdQueueTag::DStorage),
      thd([this] { ExecuteThread(); }) {
    DSTORAGE_QUEUE_DESC queue_desc{
        .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
        .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
        .Priority = DSTORAGE_PRIORITY_NORMAL,
        .Device = device->device.Get()};
    ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&cmdFence)));
}
DStorageCommandQueue::~DStorageCommandQueue() {
    {
        std::lock_guard lck(mtx);
        enabled = false;
    }
    waitCv.notify_one();
    thd.join();
}
}// namespace lc::dx