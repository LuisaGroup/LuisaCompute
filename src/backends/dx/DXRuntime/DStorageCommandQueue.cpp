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
            if (b.handle) {
                WaitForSingleObject(b.handle, INFINITE);
                CloseHandle(b.handle);
            }
            if (wakeupThread) {
                {
                    std::lock_guard lck(mtx);
                    executedFrame = std::max(executedFrame, fence);
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
                    executedFrame = std::max(executedFrame, fence);
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
        while (true) {
            vstd::optional<CallbackEvent> b;
            {
                std::lock_guard lck{mtx};
                b = executedAllocators.pop();
            }
            if (!b) break;
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
void DStorageCommandQueue::AddEvent(LCEvent const *evt, uint64 fenceIdx) {
    mtx.lock();
    executedAllocators.push(evt, fenceIdx, true);
    mtx.unlock();
    waitCv.notify_one();
}
uint64 DStorageCommandQueue::Execute(luisa::compute::CommandList &&list) {
    size_t curFrame;
    WaitQueueHandle waitQueueHandle;
    waitQueueHandle.handle = nullptr;
    {
        std::lock_guard lck{exec_mtx};
        for (auto &&i : list.commands()) {
            if (i->tag() != luisa::compute::Command::Tag::ECustomCommand ||
                static_cast<luisa::compute::CustomCommand const *>(i.get())->uuid() != luisa::to_underlying(CustomCommandUUID::DSTORAGE_READ)) [[unlikely]] {
                LUISA_ERROR("Only DStorage command allowed in this stream.");
            }
            auto cmd = static_cast<luisa::compute::DStorageReadCommand const *>(i.get());
            DSTORAGE_REQUEST request{};
            luisa::visit(
                [&]<typename T>(T const &t) {
                    if constexpr (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                        auto file = reinterpret_cast<DStorageFileImpl *>(t.handle);
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        request.Source.File.Source = file->file.Get();
                        request.Source.File.Offset = t.offset_bytes;
                        request.Source.File.Size = t.size_bytes;
                    } else {
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
                        request.Source.Memory.Source = reinterpret_cast<void const *>(t.handle + t.offset_bytes);
                        request.Source.Memory.Size = t.size_bytes;
                    }
                },
                cmd->source());
            if (request.Options.SourceType != sourceType) {
                LUISA_ERROR("Source type not match.");
            }
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
        if (!list.commands().empty()) {
            waitQueueHandle.handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            queue->EnqueueSetEvent(waitQueueHandle.handle);
            queue->Submit();
        }
    }
    bool callbackEmpty = list.callbacks().empty();
    curFrame = ++lastFrame;
    {
        std::unique_lock lck(mtx);
        executedAllocators.push(waitQueueHandle, curFrame, callbackEmpty);
        if (!callbackEmpty) {
            executedAllocators.push(list.steal_callbacks(), curFrame, true);
        }
    }
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
DStorageCommandQueue::DStorageCommandQueue(IDStorageFactory *factory, Device *device, luisa::compute::DStorageStreamSource source)
    : CmdQueueBase(device, CmdQueueTag::DStorage),
      thd([this] { ExecuteThread(); }) {
    switch (source) {
        case DStorageStreamSource::FileSource: {
            DSTORAGE_QUEUE_DESC queue_desc{
                .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
                .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
                .Priority = DSTORAGE_PRIORITY_NORMAL,
                .Device = device->device.Get()};
            sourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
        } break;
        case DStorageStreamSource::MemorySource: {
            DSTORAGE_QUEUE_DESC queue_desc{
                .SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY,
                .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
                .Priority = DSTORAGE_PRIORITY_NORMAL,
                .Device = device->device.Get()};
            sourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
            ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
        } break;
        default:
            LUISA_ERROR("Unsupported source type.");
            break;
    }
}
void DStorageCommandQueue::Signal(ID3D12Fence *fence, UINT64 value) {
    std::lock_guard lck{exec_mtx};
    queue->EnqueueSignal(fence, value);
    queue->Submit();
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
