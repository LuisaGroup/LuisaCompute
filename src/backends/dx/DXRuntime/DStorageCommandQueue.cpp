#include "DStorageCommandQueue.h"
#include <DXApi/LCEvent.h>
#include <DXApi/ext.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include <Resource/SparseTexture.h>
#include <Resource/Buffer.h>
#include <Resource/TextureBase.h>
namespace lc::dx {
namespace {
[[nodiscard]] bool dstorage_is_read_command(luisa::compute::Command const *cmd) noexcept {
    return cmd->tag() == luisa::compute::Command::Tag::ECustomCommand &&
           static_cast<luisa::compute::CustomCommand const *>(cmd)->custom_cmd_uuid() ==
               luisa::to_underlying(luisa::compute::CustomCommandUUID::DSTORAGE_READ);
}
}// namespace

void dstorage_split_texture_region(
    luisa::compute::PixelStorage storage,
    uint32_t const *offset, uint32_t const *size,
    size_t staging_buffer_size,
    vstd::vector<DStorageTextureSubRegion> &result) noexcept {
    result.clear();
    if (size[0] == 0u || size[1] == 0u || size[2] == 0u) { return; }
    auto pitch = luisa::compute::dstorage_texture_row_pitch(storage, size[0]);
    // Number of rows (spanning whole planes when possible) that fit into one
    // staging buffer.  `staging_buffer_size == 0` means "do not split".
    auto max_rows = (staging_buffer_size == 0u || pitch == 0u)
                        ? std::numeric_limits<size_t>::max()
                        : std::max<size_t>(1u, staging_buffer_size / pitch);
    auto plane_rows = static_cast<size_t>(size[1]);
    uint32_t y = 0u;
    for (uint32_t z = 0u; z < size[2];) {
        if (plane_rows <= max_rows) {
            // Take as many whole planes as fit.
            auto planes = std::max<size_t>(1u, max_rows / std::max<size_t>(1u, plane_rows));
            auto take = static_cast<uint32_t>(
                std::min<size_t>(planes, static_cast<size_t>(size[2] - z)));
            DStorageTextureSubRegion r{};
            r.offset[0] = offset[0];
            r.offset[1] = offset[1];
            r.offset[2] = offset[2] + z;
            r.size[0] = size[0];
            r.size[1] = size[1];
            r.size[2] = take;
            r.source_bytes = pitch * plane_rows * take;
            result.emplace_back(r);
            z += take;
            y = 0u;
        } else {
            // Split the current plane by rows.
            auto take = static_cast<uint32_t>(
                std::min<size_t>(max_rows, plane_rows - y));
            DStorageTextureSubRegion r{};
            r.offset[0] = offset[0];
            r.offset[1] = offset[1] + y;
            r.offset[2] = offset[2] + z;
            r.size[0] = size[0];
            r.size[1] = take;
            r.size[2] = 1u;
            r.source_bytes = pitch * take;
            result.emplace_back(r);
            y += take;
            if (y == size[1]) {
                y = 0u;
                ++z;
            }
        }
    }
}

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
            for (auto handle : b.handles) {
                if (handle) {
                    WaitForSingleObject(handle, INFINITE);
                    CloseHandle(handle);
                }
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
            device->wait_fence(evt->fence(), fence);
            {
                std::lock_guard lck(evt->event_mtx);
                evt->finished_event = std::max<uint64_t>(fence, evt->finished_event);
            }
            if (wakeupThread) {
                executedFrame++;
            }
        };
        while (true) {
            vstd::optional<CallbackEvent> b;
            {
                std::lock_guard lck{mtx};
                b = executedAllocators.dequeue();
            }
            if (!b) break;
            fence = b->fence;
            wakeupThread = b->wakeupThread;
            b->evt.multi_visit(
                ExecuteAllocator,
                ExecuteCallbacks,
                ExecuteEvent);
        }
        while (enabled && executedAllocators.length() == 0) {
            std::this_thread::yield();
        }
    }
}
void DStorageCommandQueue::AddEvent(LCEvent const *evt, uint64_t fenceIdx) {
    ++lastFrame;
    mtx.lock();
    executedAllocators.enqueue(evt, fenceIdx, true);
    mtx.unlock();
}
uint64_t DStorageCommandQueue::Execute(
    vstd::span<const luisa::unique_ptr<luisa::compute::Command>> commands,
    luisa::vector<luisa::move_only_function<void()>> &&funcs) {
    WaitQueueHandle waitQueueHandle;
    {
        std::lock_guard lck{exec_mtx};
        bool used_file = false;
        bool used_memory = false;
        auto enqueue_request = [&](DSTORAGE_REQUEST &request) {
            if (request.Options.SourceType == DSTORAGE_REQUEST_SOURCE_FILE) {
                if (!_file_queue) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION(
                        "A file-sourced DirectStorage request was dispatched on a "
                        "memory-only stream. Create the stream with "
                        "DStorageStreamSource::FileSource or AnySource.");
                }
                used_file = true;
                _file_pending = true;
                _file_queue->EnqueueRequest(&request);
            } else {
                if (!_memory_queue) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION(
                        "A memory-sourced DirectStorage request was dispatched on a "
                        "file-only stream. Create the stream with "
                        "DStorageStreamSource::MemorySource or AnySource.");
                }
                used_memory = true;
                _memory_pending = true;
                _memory_queue->EnqueueRequest(&request);
            }
        };
        auto check_staging = [&](size_t bytes, char const *what) {
            if (staging_buffer_size != 0u && bytes > staging_buffer_size) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "DirectStorage request {} ({} byte(s)) exceeds the process-global "
                    "staging buffer size ({} byte(s)). Create the first DirectStorage "
                    "stream with a larger DStorageStreamOption::staging_buffer_size.",
                    what, bytes, staging_buffer_size);
            }
        };
        for (auto &&i : commands) {
            if (!dstorage_is_read_command(i.get())) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Only DStorage commands are allowed in this stream.");
            }
            auto cmd = static_cast<luisa::compute::DStorageReadCommand const *>(i.get());
            auto compressed = cmd->is_compressed();
            auto transfer = cmd->effective_transfer_size();
            auto required = cmd->required_source_size_bytes();
            DSTORAGE_REQUEST request{};
            if (compressed) {
                request.Options.CompressionFormat =
                    DSTORAGE_COMPRESSION_FORMAT::DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
            }
            // ---- bind the source -------------------------------------------
            size_t src_offset = 0u;
            size_t src_size = 0u;
            bool from_file = false;
            std::byte const *memory_base = nullptr;
            luisa::visit(
                [&]<typename T>(T const &s) {
                    src_offset = s.offset_bytes;
                    src_size = s.size_bytes;
                    if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::FileSource>) {
                        from_file = true;
                        if (s.handle == luisa::compute::invalid_resource_handle) [[unlikely]] {
                            LUISA_ERROR_WITH_LOCATION(
                                "DStorage file source is invalid (the file failed to open).");
                        }
                        auto file = reinterpret_cast<DStorageFileImpl *>(s.handle);
                        // Per-request bounds check against the real file size
                        // (mirrors the reference implementation).
                        LUISA_ASSERT(src_offset <= file->size_bytes &&
                                         src_size <= file->size_bytes - src_offset,
                                     "DStorage file source out of range: offset {} + size {} "
                                     "exceeds file size {}.",
                                     src_offset, src_size, file->size_bytes);
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        request.Source.File.Source = file->file.Get();
                        request.Source.File.Offset = src_offset;
                    } else {
                        auto base = reinterpret_cast<std::byte const *>(s.handle);
                        if (base == nullptr) [[unlikely]] {
                            LUISA_ERROR_WITH_LOCATION(
                                "DStorage memory source is null (the pinned-memory handle "
                                "is invalid).");
                        }
                        if (_ext != nullptr) {
                            auto pinned = _ext->pinned_memory_size(s.handle);
                            if (pinned != 0u) {
                                LUISA_ASSERT(src_offset <= pinned &&
                                                 src_size <= pinned - src_offset,
                                             "DStorage memory source out of range: offset {} + "
                                             "size {} exceeds the pinned range {}.",
                                             src_offset, src_size, pinned);
                            }
                        }
                        memory_base = base;
                        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
                        request.Source.Memory.Source = base;
                    }
                },
                cmd->source());
            auto set_source_chunk = [&](size_t consumed, size_t bytes) {
                if (from_file) {
                    request.Source.File.Offset = src_offset + consumed;
                    request.Source.File.Size = bytes;
                } else {
                    request.Source.Memory.Source = memory_base + consumed;
                    request.Source.Memory.Size = bytes;
                }
            };

            // ---- bind the destination and emit requests ---------------------
            luisa::visit(
                [&]<typename T>(T const &dst) {
                    if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::BufferRequest>) {
                        auto *buffer = reinterpret_cast<Buffer *>(dst.handle);
                        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                        request.Destination.Buffer.Resource = buffer->GetResource();
                        if (compressed) {
                            // Compressed requests are *not* byte-sliceable: the
                            // source is one contiguous compressed blob and the
                            // destination gets the whole decompressed payload.
                            // `UncompressedSize` is the number of bytes written,
                            // so `Destination.Buffer.Size` is that same value
                            // ("number of bytes to write to the destination").
                            auto uncompressed = dst.size_bytes;
                            check_staging(uncompressed, "uncompressed size");
                            check_staging(src_size, "source size");
                            request.Destination.Buffer.Offset = dst.offset_bytes;
                            request.Destination.Buffer.Size = uncompressed;
                            request.UncompressedSize = static_cast<UINT32>(uncompressed);
                            set_source_chunk(0u, src_size);
                            enqueue_request(request);
                        } else {
                            size_t consumed = 0u;
                            while (consumed < transfer) {
                                auto chunk = std::min(transfer - consumed,
                                                      staging_buffer_size == 0u ? transfer : staging_buffer_size);
                                request.Destination.Buffer.Offset = dst.offset_bytes + consumed;
                                request.Destination.Buffer.Size = chunk;
                                set_source_chunk(consumed, chunk);
                                enqueue_request(request);
                                consumed += chunk;
                            }
                        }
                    } else if constexpr (std::is_same_v<T, luisa::compute::DStorageReadCommand::TextureRequest>) {
                        auto *tex = reinterpret_cast<TextureBase *>(dst.handle);
                        auto region = luisa::make_uint3(dst.size[0], dst.size[1], dst.size[2]);
                        // DirectStorage requires the source to be the D3D12
                        // "copyable footprint" of the region, i.e. rows padded
                        // to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256 bytes).
                        // `dstorage_texture_source_size` derives that pitch
                        // from the row (`align(width * pixel_size, 256)`), which
                        // D3D12 also uses for the subresource layout, so no
                        // extra pitch check is needed here.  This is exactly
                        // what the old `CalcAlign(tex_size / row_count, 256)`
                        // computed.
                        auto padded = luisa::compute::dstorage_texture_source_size(dst.storage, region);
                        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
                        request.Destination.Texture.Resource = tex->GetResource();
                        request.Destination.Texture.SubresourceIndex = dst.level;
                        if (compressed) {
                            check_staging(src_size, "compressed source size");
                            check_staging(padded, "uncompressed size");
                            request.Destination.Texture.Region = D3D12_BOX{
                                dst.offset[0], dst.offset[1], dst.offset[2],
                                dst.offset[0] + dst.size[0],
                                dst.offset[1] + dst.size[1],
                                dst.offset[2] + dst.size[2]};
                            request.UncompressedSize = static_cast<UINT32>(padded);
                            set_source_chunk(0u, src_size);
                            enqueue_request(request);
                        } else {
                            if (src_size < required) [[unlikely]] {
                                LUISA_ERROR_WITH_LOCATION(
                                    "DStorage texture source is too small: {} byte(s) available, "
                                    "but the pitch-aligned layout of a {}x{}x{} region needs {} "
                                    "byte(s). Lay out texture sources with "
                                    "dstorage_texture_row_pitch()/dstorage_texture_source_size(), "
                                    "or use a buffer destination.",
                                    src_size, region.x, region.y, region.z, required);
                            }
                            vstd::vector<DStorageTextureSubRegion> sub_regions;
                            dstorage_split_texture_region(
                                dst.storage, dst.offset, dst.size,
                                staging_buffer_size, sub_regions);
                            size_t consumed = 0u;
                            for (auto &&sub : sub_regions) {
                                request.Destination.Texture.Region = D3D12_BOX{
                                    sub.offset[0], sub.offset[1], sub.offset[2],
                                    sub.offset[0] + sub.size[0],
                                    sub.offset[1] + sub.size[1],
                                    sub.offset[2] + sub.size[2]};
                                set_source_chunk(consumed, sub.source_bytes);
                                enqueue_request(request);
                                consumed += sub.source_bytes;
                            }
                        }
                    } else {// MemoryRequest
                        auto *data = static_cast<std::byte *>(dst.data);
                        if (data == nullptr) [[unlikely]] {
                            LUISA_ERROR_WITH_LOCATION("DStorage memory destination is null.");
                        }
                        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                        if (compressed) {
                            auto uncompressed = dst.size_bytes;
                            request.Destination.Memory.Buffer = data;
                            request.Destination.Memory.Size = uncompressed;
                            request.UncompressedSize = static_cast<UINT32>(uncompressed);
                            set_source_chunk(0u, src_size);
                            enqueue_request(request);
                        } else {
                            size_t consumed = 0u;
                            while (consumed < transfer) {
                                auto chunk = std::min(transfer - consumed,
                                                      staging_buffer_size == 0u ? transfer : staging_buffer_size);
                                request.Destination.Memory.Buffer = data + consumed;
                                request.Destination.Memory.Size = chunk;
                                set_source_chunk(consumed, chunk);
                                enqueue_request(request);
                                consumed += chunk;
                            }
                        }
                    }
                },
                cmd->request());
        }
        if (used_file) {
            auto handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            _file_queue->EnqueueSetEvent(handle);
            _file_queue->Submit();
            waitQueueHandle.handles.emplace_back(handle);
        }
        if (used_memory) {
            auto handle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            _memory_queue->EnqueueSetEvent(handle);
            _memory_queue->Submit();
            waitQueueHandle.handles.emplace_back(handle);
        }
    }
    bool callbackEmpty = funcs.empty();
    auto curFrame = ++lastFrame;
    {
        std::unique_lock lck(mtx);
        executedAllocators.enqueue(std::move(waitQueueHandle), curFrame, callbackEmpty);
        if (!callbackEmpty) {
            executedAllocators.enqueue(std::move(funcs), curFrame, true);
        }
    }
    return curFrame;
}
void DStorageCommandQueue::Complete(uint64_t fence) {
    while (executedFrame < fence) {
        std::this_thread::yield();
    }
    if (fence >= lastFrame.load()) {
        // Every submitted batch has completed, so no native queue holds
        // uncovered work any more (`Signal` uses these flags to decide which
        // queues a fence must be enqueued on).
        std::lock_guard lck{exec_mtx};
        _file_pending = false;
        _memory_pending = false;
    }
}
void DStorageCommandQueue::Complete() {
    Complete(lastFrame);
}
DStorageCommandQueue::DStorageCommandQueue(DStorageExtImpl *ext,
                                           IDStorageFactory *factory,
                                           Device *device,
                                           luisa::compute::DStorageStreamSource source,
                                           size_t staging_buffer_size)
    : CmdQueueBase(device, CmdQueueTag::DStorage),
      _ext{ext},
      _source_hint{source},
      thd([this] { ExecuteThread(); }) {
    this->staging_buffer_size = staging_buffer_size;
    auto create_queue = [&](DSTORAGE_REQUEST_SOURCE_TYPE type,
                            ComPtr<IDStorageQueue2> &queue) {
        DSTORAGE_QUEUE_DESC queue_desc{
            .SourceType = type,
            .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
            .Priority = DSTORAGE_PRIORITY_LOW,
            .Device = device->device.Get()};
        ThrowIfFailed(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())));
    };
    switch (source) {
        case DStorageStreamSource::FileSource:
            create_queue(DSTORAGE_REQUEST_SOURCE_FILE, _file_queue);
            break;
        case DStorageStreamSource::MemorySource:
            create_queue(DSTORAGE_REQUEST_SOURCE_MEMORY, _memory_queue);
            break;
        case DStorageStreamSource::AnySource:
            create_queue(DSTORAGE_REQUEST_SOURCE_FILE, _file_queue);
            create_queue(DSTORAGE_REQUEST_SOURCE_MEMORY, _memory_queue);
            break;
        default:
            LUISA_ERROR("Unsupported DirectStorage source type.");
            break;
    }
}
void DStorageCommandQueue::Signal(ID3D12Fence *fence, UINT64 value) {
    std::lock_guard lck{exec_mtx};
    // Signal only the queues that still hold uncovered work: an *empty* native
    // queue executes `EnqueueSignal` immediately, which would let a shared
    // fence reach `value` before the queue that actually has work is done.
    auto signal_queue = [&](ComPtr<IDStorageQueue2> &queue) {
        queue->EnqueueSignal(fence, value);
        queue->Submit();
    };
    auto file = _file_pending;
    auto memory = _memory_pending;
    if (!file && !memory) {
        // Nothing outstanding: fire from the primary queue so waiters unblock.
        if (_file_queue) {
            signal_queue(_file_queue);
        } else if (_memory_queue) {
            signal_queue(_memory_queue);
        }
        return;
    }
    if (file && memory) {
        // A single DX fence cannot join two native queues: the fence reaches
        // `value` when the first one finishes.  `synchronize()` waits for both
        // (the worker thread waits on every submitted queue's set-event), so
        // use it for AnySource cross-stream ordering.
        if (!_signal_join_warned) {
            _signal_join_warned = true;
            LUISA_WARNING(
                "Signalling a DirectStorage event while an AnySource stream has "
                "outstanding file *and* memory work: a single DirectStorage fence "
                "cannot join two native queues, so the event may fire when the "
                "first queue completes. Use synchronize() for strict ordering.");
        }
    }
    if (file && _file_queue) {
        signal_queue(_file_queue);
        _file_pending = false;
    }
    if (memory && _memory_queue) {
        signal_queue(_memory_queue);
        _memory_pending = false;
    }
}
DStorageCommandQueue::~DStorageCommandQueue() {
    {
        std::lock_guard lck{mtx};
        enabled = false;
    }
    thd.join();
}
}// namespace lc::dx
