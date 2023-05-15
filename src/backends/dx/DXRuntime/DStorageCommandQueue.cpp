#include "DStorageCommandQueue.h"
#include <DXApi/LCEvent.h>
#include <core/logging.h>
#include <backends/ext/dstorage_ext_interface.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
namespace lc::dx {
void DStorageCommandQueue::ExecuteThread() {
    while (enabled) {
        uint64_t fence;
        bool wakeupThread;
        auto ExecuteAllocator = [&](WaitQueueHandle const &b) {
            for (auto &&i : b.handles) {
                if (i) {
                    WaitForSingleObject(i, INFINITE);
                    CloseHandle(i);
                }
            }
            if (wakeupThread) {
                {
                    std::lock_guard lck(mtx);
                    executedFrame = fence;
                }
                mainCv.notify_all();
            }
        };
        auto ExecuteCallbacks = [&](auto &vec) {
            for (auto &&i : vec) {
                i();
            }
            if (wakeupThread) {
                {
                    std::lock_guard lck(mtx);
                    executedFrame = fence;
                }
                mainCv.notify_all();
            }
        };
        auto ExecuteEvent = [&](auto &evt) {
            device->WaitFence(evt->fence.Get(), fence);
            {
                std::lock_guard lck(evt->eventMtx);
                evt->finishedEvent = std::max(fence, evt->finishedEvent);
            }
            evt->cv.notify_all();
        };
        while (auto b = executedAllocators.pop()) {
            fence = b->fence;
            wakeupThread = b->wakeupThread;
            b->evt.multi_visit(
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
    executedAllocators.push(evt, evt->fenceIndex, true);
    mtx.lock();
    mtx.unlock();
    waitCv.notify_one();
}
uint64 DStorageCommandQueue::Execute(luisa::compute::CommandList &&list) {
    size_t curFrame;
    bool memQueueUsed = false;
    bool fileQueueUsed = false;
    WaitQueueHandle waitQueueHandle;
    for (auto &&i : waitQueueHandle.handles) {
        i = nullptr;
    }
    {
        std::lock_guard lck{exec_mtx};
        for (auto &&i : list.commands()) {
            if (i->tag() != luisa::compute::Command::Tag::ECustomCommand ||
                static_cast<luisa::compute::CustomCommand const *>(i.get())->uuid() != luisa::compute::dstorage_command_uuid) [[unlikely]] {
                LUISA_ERROR("Only DStorage command allowed in this stream.");
            }
            auto cmd = static_cast<luisa::compute::DStorageReadCommand const *>(i.get());
            IDStorageQueue *queue;
            DSTORAGE_REQUEST request{};
            luisa::visit(
                [&]<typename T>(T const &t) {
                if constexpr (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                    auto file = reinterpret_cast<DStorageFileImpl *>(t.file_handle);
                    request.Source.File.Source = file->file.Get();
                    request.Source.File.Offset = t.file_offset;
                    request.Source.File.Size = cmd->size_bytes;
                    if (!fileQueueUsed) {
                        fileQueueUsed = true;
                        if (!fileQueue) {
                            DSTORAGE_QUEUE_DESC queue_desc{
                                .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
                                .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
                                .Priority = DSTORAGE_PRIORITY_NORMAL,
                                .Device = device->device.Get()};
                            ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(fileQueue.GetAddressOf())));
                        }
                    }
                    queue = fileQueue.Get();
                } else {
                    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
                    request.Source.Memory.Source = t.src_ptr;
                    request.Source.Memory.Size = cmd->size_bytes;
                    if (!memQueueUsed) {
                        memQueueUsed = true;
                        if (!memQueue) {
                            DSTORAGE_QUEUE_DESC queue_desc{
                                .SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY,
                                .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
                                .Priority = DSTORAGE_PRIORITY_NORMAL,
                                .Device = device->device.Get()};
                            ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(memQueue.GetAddressOf())));
                        }
                    }
                    queue = memQueue.Get();
                }
                },
                cmd->src);
            if (cmd->compression == DStorageReadCommand::Compression::GDeflate) {
                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                request.UncompressedSize = cmd->size_bytes;
            }
            luisa::visit(
                [&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::BufferEnqueue>) {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                request.Destination.Buffer.Resource = reinterpret_cast<Buffer *>(t.buffer_handle)->GetResource();
                request.Destination.Buffer.Offset = t.buffer_offset;
                request.Destination.Buffer.Size = cmd->size_bytes;
            } else if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::ImageEnqueue>) {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
                auto tex = reinterpret_cast<TextureBase *>(t.image_handle);
                request.Destination.Texture.SubresourceIndex = t.mip_level;
                request.Destination.Texture.Resource = tex->GetResource();
                request.Destination.Texture.Region = D3D12_BOX{
                    0u, 0u, 0u,
                    tex->Width(), tex->Height(), tex->Depth()};
            } else {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                request.Destination.Memory.Buffer = t.dst_ptr;
                request.Destination.Memory.Size = cmd->size_bytes;
            } },
                cmd->enqueue_cmd());
            queue->EnqueueRequest(&request);
        }
        auto setQueueHandle = [&](auto queue, bool used, auto &handle) {
            if (!used) return;
            handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            queue->EnqueueSetEvent(handle);
            queue->Submit();
        };
        setQueueHandle(fileQueue, fileQueueUsed, waitQueueHandle.handles[0]);
        setQueueHandle(memQueue, memQueueUsed, waitQueueHandle.handles[1]);
    }
    bool callbackEmpty = list.callbacks().empty();
    curFrame = ++lastFrame;
    executedAllocators.push(waitQueueHandle, curFrame, callbackEmpty);
    if (!callbackEmpty) {
        executedAllocators.push(std::move(list.steal_callbacks()), curFrame, true);
    }
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
    Complete(lastFrame);
}
DStorageCommandQueue::DStorageCommandQueue(IDStorageFactory *factory, Device *device)
    : device(device),
      factory{factory},
      CmdQueueBase(CmdQueueTag::DStorage),
      thd([this] { ExecuteThread(); }) {
}
void DStorageCommandQueue::Signal(ID3D12Fence *fence, UINT64 value) {
    std::lock_guard lck{exec_mtx};
    if (fileQueue) {
        fileQueue->EnqueueSignal(fence, value);
        fileQueue->Submit();
    }
    if (memQueue) {
        memQueue->EnqueueSignal(fence, value);
        memQueue->Submit();
    }
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