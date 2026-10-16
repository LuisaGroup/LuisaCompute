#pragma once
#include <dstorage/dstorage.h>
#include <DXRuntime/Device.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <DXRuntime/DxPtr.h>
#include <DXApi/CmdQueueBase.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>

namespace lc::dx {

class LCEvent;
class DStorageExtImpl;

class DStorageFileImpl : public vstd::IOperatorNewBase {
public:
    ComPtr<IDStorageFile> file;
    size_t size_bytes;
    DStorageFileImpl(ComPtr<IDStorageFile> &&file, size_t size_bytes) : file{std::move(file)}, size_bytes{size_bytes} {}
};

/// One GPU-side texture copy produced by splitting a DirectStorage texture
/// request so that its pitch-aligned source region fits into the staging
/// buffer.  `source_bytes` is the number of *padded* source bytes this
/// sub-request consumes (rows are aligned to
/// `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`).
struct DStorageTextureSubRegion {
    uint32_t offset[3];
    uint32_t size[3];
    size_t source_bytes;
};

/// Split a texture region into sub-regions whose pitch-aligned source size
/// (`dstorage_texture_row_pitch(storage, width) * rows`) never exceeds
/// `staging_buffer_size`.  Whole planes are taken when they fit; otherwise a
/// plane is split by rows.  Sub-regions come out in the source's row-major
/// (z, y, x) order so a caller may consume the padded source sequentially.
/// Pure function, unit-testable.
void dstorage_split_texture_region(
    luisa::compute::PixelStorage storage,
    uint32_t const *offset, uint32_t const *size,
    size_t staging_buffer_size,
    vstd::vector<DStorageTextureSubRegion> &result) noexcept;

class DStorageCommandQueue : public CmdQueueBase {
    struct WaitQueueHandle {
        // One event per native DirectStorage queue that was submitted; element
        // 0 belongs to the file-sourced queue, element 1 (if present) to the
        // memory-sourced queue.
        vstd::vector<HANDLE> handles;
    };
    struct CallbackEvent {
        using Variant = vstd::variant<
            WaitQueueHandle,
            vstd::vector<vstd::function<void()>>,
            LCEvent const *>;
        Variant evt;
        uint64_t fence;
        bool wakeupThread;
        template<typename Arg>
            requires(luisa::is_constructible_v<Variant, Arg &&>)
        CallbackEvent(Arg &&arg,
                      uint64_t fence,
                      bool wakeupThread)
            : evt{std::forward<Arg>(arg)}, fence{fence}, wakeupThread{wakeupThread} {}
    };
    std::atomic_bool enabled = true;
    std::atomic_uint64_t executedFrame = 0;
    std::atomic_uint64_t lastFrame = 0;
    luisa::spin_mutex mtx;
    luisa::spin_mutex exec_mtx;
    // `option.source == AnySource` creates both queues; otherwise exactly one
    // of them is created and requests of the wrong source type are rejected.
    DStorageExtImpl *_ext{nullptr};
    ComPtr<IDStorageQueue2> _file_queue;
    ComPtr<IDStorageQueue2> _memory_queue;
    luisa::compute::DStorageStreamSource _source_hint;
    // Whether a native queue holds work that no fence signal has covered yet.
    // `Signal` only signals queues that are actually in flight so an idle queue
    // (e.g. the memory queue of an AnySource stream serving a file read) cannot
    // advance the shared fence ahead of the real work.
    bool _file_pending{false};
    bool _memory_pending{false};
    bool _signal_join_warned{false};
    vstd::SingleThreadArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    size_t staging_buffer_size = DSTORAGE_STAGING_BUFFER_SIZE_32MB;
    void Signal(ID3D12Fence *fence, UINT64 value);
    uint64_t LastFrame() const { return lastFrame; }
    DStorageCommandQueue(DStorageExtImpl *ext,
                         IDStorageFactory *factory,
                         Device *device,
                         luisa::compute::DStorageStreamSource source,
                         size_t staging_buffer_size);
    void AddEvent(LCEvent const *evt, uint64_t fenceIdx);
    uint64_t Execute(
        vstd::span<const luisa::unique_ptr<luisa::compute::Command>> commands,
        luisa::vector<luisa::move_only_function<void()>> &&funcs);
    void Complete(uint64_t fence);
    void Complete();
    KILL_MOVE_CONSTRUCT(DStorageCommandQueue)
    KILL_COPY_CONSTRUCT(DStorageCommandQueue)
    ~DStorageCommandQueue();
private:
    // make sure thread always construct after all members
    std::thread thd;
};
}// namespace lc::dx
