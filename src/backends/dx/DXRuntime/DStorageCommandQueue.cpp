#include "DStorageCommandQueue.h"
#include <DXApi/LCEvent.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <Resource/SparseTexture.h>
#include <Resource/Buffer.h>
#include <luisa/backends/ext/dstorage_cmd.h>
namespace lc::dx {
void DStorageCommandQueue::ExecuteThread() {
    while (enabled || executedAllocators.length() != 0) {
        uint64_t fence;
        bool wakeupThread;
        auto max_fence = [&]() {
            uint64 prev_value = executedFrame;
            while (prev_value < fence && !executedFrame.compare_exchange_weak(prev_value, fence)) {
                std::this_thread::yield();
            }
        };
        auto ExecuteAllocator = [&](WaitQueueHandle const &b) {
            if (b.handle) {
                WaitForSingleObject(b.handle, INFINITE);
                CloseHandle(b.handle);
            }
            if (wakeupThread) {
                max_fence();
            }
        };
        auto ExecuteCallbacks = [&](auto &vec) {
            for (auto &&i : vec) {
                i();
            }
            if (wakeupThread) {
                max_fence();
            }
        };
        auto ExecuteEvent = [&](auto &evt) {
            device->WaitFence(evt->fence.Get(), fence);
            {
                std::lock_guard lck(evt->eventMtx);
                evt->finishedEvent = std::max(fence, evt->finishedEvent);
            }
            evt->cv.notify_all();
            if (wakeupThread) {
                executedFrame++;
            }
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
    ++lastFrame;
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
            size_t src_size{};
            uint yz_tex_offsets[] = {0, 0};
            auto set_request_dst = [&](size_t sub_offset, size_t sub_size) {
                size_t real_size = sub_size;
                if (request.Options.SourceType != sourceType) {
                    LUISA_ERROR("Source type not match.");
                }
                luisa::visit(
                    [&]<typename T>(T const &t) {
                        if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::BufferRequest>) {
                            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                            request.Destination.Buffer.Resource = reinterpret_cast<Buffer *>(t.handle)->GetResource();
                            request.Destination.Buffer.Offset = t.offset_bytes + sub_offset;
                            request.Destination.Buffer.Size = sub_size;
                            if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                                request.UncompressedSize = t.size_bytes;
                                LUISA_ASSERT(t.size_bytes <= staging_buffer_size, "Compressed buffer's size({} bytes) can-not be large than {} bytes", t.size_bytes, staging_buffer_size);
                            }

                        } else if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::TextureRequest>) {
                            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
                            auto tex = reinterpret_cast<TextureBase *>(t.handle);
                            request.Destination.Texture.SubresourceIndex = t.level;
                            request.Destination.Texture.Resource = tex->GetResource();
                            auto row_count = t.size[1] * t.size[2];
                            auto pixel_size = Resource::GetTexturePixelSize(tex->Format());
                            auto tex_size = pixel_size * t.size[0] * row_count;
                            auto row_size = CalcAlign(tex_size / row_count, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
                            uint local_offset[3] = {
                                t.offset[0],
                                t.offset[1] + yz_tex_offsets[0],
                                t.offset[2] + yz_tex_offsets[1],
                            };
                            uint local_size[3] = {t.size[0], t.size[1], t.size[2]};
                            if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                                LUISA_ASSERT(tex_size <= staging_buffer_size, "Compressed buffer's size({} bytes) can-not be large than {} bytes", tex_size, staging_buffer_size);
                            } else {
                                // Use slice copy
                                if (tex_size > sub_size) {
                                    auto plane_size = pixel_size * t.size[0] * t.size[1];
                                    // per x_row copy
                                    if (plane_size > sub_size) {
                                        if (yz_tex_offsets[0] == t.size[1]) {
                                            yz_tex_offsets[0] = 0;
                                            yz_tex_offsets[1] += 1;
                                        }
                                        uint max_allowed_height = sub_size / row_size;
                                        max_allowed_height = std::min<uint>(max_allowed_height, t.size[1] - yz_tex_offsets[0]);
                                        local_size[2] = 1;
                                        local_size[1] = max_allowed_height;
                                        yz_tex_offsets[0] += max_allowed_height;
                                        real_size = row_size * max_allowed_height;
                                    }
                                    // per xy_plane copy
                                    else {
                                        // copy full plane
                                        if (yz_tex_offsets[0] == 0) {
                                            uint max_allowed_depth = std::min<uint>(sub_size / plane_size, t.size[2] - yz_tex_offsets[1]);
                                            local_size[2] = max_allowed_depth;
                                            yz_tex_offsets[1] += max_allowed_depth;
                                            real_size = plane_size * max_allowed_depth;
                                        } else {
                                            local_size[2] = 1;
                                            local_size[1] = t.size[1] - yz_tex_offsets[0];
                                            yz_tex_offsets[0] = 0;
                                            real_size = pixel_size * row_size * local_size[1];
                                        }
                                    }
                                }
                            }

                            if (row_size < D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) [[unlikely]] {
                                LUISA_ERROR("DirectX direct-storage can not support texture destination with row size(width * pixel_size) less than {}, current row size: {}, try use buffer instead.", D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, row_size);
                            }

                            request.Destination.Texture.Region = D3D12_BOX{
                                local_offset[0], local_offset[1], local_offset[2],
                                local_offset[0] + local_size[0], local_offset[1] + local_size[1], local_offset[2] + local_size[2]};

                        } else {
                            // request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;

                            // if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                            //     request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                            //     request.UncompressedSize = t.size_bytes;
                            // }
                            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                            request.Destination.Memory.Buffer = reinterpret_cast<std::byte *>(t.data) + sub_offset;
                            request.Destination.Memory.Size = sub_size;
                            if (cmd->compression() == DStorageReadCommand::Compression::GDeflate) {
                                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
                                request.UncompressedSize = t.size_bytes;
                                LUISA_ASSERT(t.size_bytes <= staging_buffer_size, "Compressed buffer's size({} bytes) can-not be large than {} bytes", t.size_bytes, staging_buffer_size);
                            }
                        }
                    },
                    cmd->request());
                return real_size;
            };

            luisa::visit(
                [&]<typename T>(T const &t) {
                    src_size = t.size_bytes;
                    size_t sub_offset = 0;
                    if constexpr (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                        auto file = reinterpret_cast<DStorageFileImpl *>(t.handle);
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        request.Source.File.Source = file->file.Get();
                        request.Source.File.Offset = t.offset_bytes;
                        // if (luisa::holds_alternative<DStorageReadCommand::MemoryRequest>(cmd->request())) {
                        //     request.Source.File.Size = t.size_bytes;
                        //     set_request_dst(0, t.size_bytes);
                        //     queue->EnqueueRequest(&request);
                        // } else {
                        auto lefted_size = static_cast<int64_t>(t.size_bytes);
                        auto slice_size = std::min(t.size_bytes, staging_buffer_size);
                        while (lefted_size > 0) {
                            auto real_size = set_request_dst(sub_offset, slice_size);
                            request.Source.File.Size = real_size;
                            queue->EnqueueRequest(&request);
                            request.Source.File.Offset += real_size;
                            sub_offset += real_size;
                            lefted_size -= real_size;
                        }
                        // }
                    } else {
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
                        auto ptr = reinterpret_cast<std::byte const *>(t.handle + t.offset_bytes);
                        request.Source.Memory.Source = ptr;
                        request.Source.Memory.Size = t.size_bytes;
                        if (luisa::holds_alternative<DStorageReadCommand::MemoryRequest>(cmd->request())) {
                            request.Source.Memory.Size = t.size_bytes;
                            set_request_dst(0, t.size_bytes);
                            queue->EnqueueRequest(&request);
                        } else {
                            auto lefted_size = static_cast<int64_t>(t.size_bytes);
                            auto slice_size = std::min(t.size_bytes, staging_buffer_size);
                            while (lefted_size > 0) {
                                auto real_size = set_request_dst(sub_offset, slice_size);
                                request.Source.Memory.Size = real_size;
                                queue->EnqueueRequest(&request);
                                ptr += real_size;
                                request.Source.Memory.Source = ptr;
                                sub_offset += real_size;
                                lefted_size -= real_size;
                            }
                        }
                    }
                },
                cmd->source());
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
    while (executedFrame < fence) {
        std::this_thread::yield();
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
                .Priority = DSTORAGE_PRIORITY_LOW,
                .Device = device->device.Get()};
            sourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
        } break;
        case DStorageStreamSource::MemorySource: {
            DSTORAGE_QUEUE_DESC queue_desc{
                .SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY,
                .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
                .Priority = DSTORAGE_PRIORITY_LOW,
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
