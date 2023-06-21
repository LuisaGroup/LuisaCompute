#include "DStorageCommandQueue.h"
#include <DXApi/LCEvent.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <Resource/SparseTexture.h>
#include <Resource/Buffer.h>
#include <luisa/backends/ext/dstorage_cmd.h>
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
                static_cast<luisa::compute::CustomCommand const *>(i.get())->uuid() != luisa::to_underlying(CustomCommandUUID::DSTORAGE_READ)) [[unlikely]] {
                LUISA_ERROR("Only DStorage command allowed in this stream.");
            }
            auto cmd = static_cast<luisa::compute::DStorageReadCommand const *>(i.get());
            IDStorageQueue *queue;
            DSTORAGE_REQUEST request{};
            luisa::visit(
                [&]<typename T>(T const &t) {
                    if constexpr (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                        auto file = reinterpret_cast<DStorageFileImpl *>(t.handle);
                        request.Source.File.Source = file->file.Get();
                        request.Source.File.Offset = t.offset_bytes;
                        request.Source.File.Size = t.size_bytes;
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
                        request.Source.Memory.Source = reinterpret_cast<void const *>(t.handle + t.offset_bytes);
                        request.Source.Memory.Size = t.size_bytes;
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
                cmd->source());
            auto set_compress = [&](size_t size) {
                if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                    request.UncompressedSize = size;
                }
            };
            luisa::visit(
                [&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::BufferRequest>) {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                request.Destination.Buffer.Resource = reinterpret_cast<Buffer *>(t.handle)->GetResource();
                request.Destination.Buffer.Offset = t.offset_bytes;
                request.Destination.Buffer.Size = t.size_bytes;
                set_compress(t.size_bytes);
            } else if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::TextureRequest>) {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
                auto tex = reinterpret_cast<TextureBase *>(t.handle);
                request.Destination.Texture.SubresourceIndex = t.level;
                request.Destination.Texture.Resource = tex->GetResource();
                auto tex_size = Resource::GetTexturePixelSize(tex->Format()) * t.size[0] * t.size[1] * t.size[2];
                auto row_size = (tex_size / (t.size[1] * t.size[2]));
                if(row_size < D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) [[unlikely]]{
                    LUISA_ERROR("DirectX direct-storage can not support texture destination with row size(width * pixel_size) less than {}, current row size: {}, try use buffer instead.", D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, row_size);
                }
                request.Destination.Texture.Region = D3D12_BOX{
                    t.offset[0], t.offset[1], t.offset[2],
                    t.offset[0] + t.size[0], t.offset[1] + t.size[1], t.offset[2] + t.size[2]};
                if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                    request.UncompressedSize = tex_size;
                }
            }
            else {
                request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                request.Destination.Memory.Buffer = t.data;
                request.Destination.Memory.Size = t.size_bytes;
                set_compress(t.size_bytes);
            } },
                cmd->request());
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
        executedAllocators.push(list.steal_callbacks(), curFrame, true);
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
    : CmdQueueBase(device, CmdQueueTag::DStorage),
      factory{factory},
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
