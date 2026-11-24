#include <DXApi/dx_accel_archive_ext.h>

#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCDevice.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/Device.h>
#include <Resource/BottomAccel.h>
#include <Resource/DefaultBuffer.h>
#include <dstorage/dstorage.h>

#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/vstl/common.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <type_traits>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

namespace lc::dx {

namespace {

using namespace luisa::compute;

constexpr uint32_t kDxCompatMagic = 0x41415844u;// "DXAA"
constexpr uint32_t kDxCompatVersion = 1u;
constexpr luisa::string_view kDxLayoutTag = "dxr_serialized_raytracing_as_v1";
constexpr size_t kDxMaxPoolSlotsPerKind = 16u;
constexpr size_t kDxMaxPoolBytesPerKind = 2ull << 30u;// 2 GiB cap per pool
constexpr size_t kDxDStorageChunkBytes = 16ull << 20u;// 16 MiB chunks (<= default 32 MiB staging)

static_assert(sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) == 56u,
              "Unexpected D3D12 serialized AS header size");

[[nodiscard]] size_t round_up_pool_size(size_t needed) noexcept {
    size_t const min_sz = std::max<size_t>(needed, 65536u);
    if (min_sz <= (4ull << 20u)) {
        return CalcAlign(min_sz, 65536ull);
    }
    if (min_sz <= (64ull << 20u)) {
        return CalcAlign(min_sz, 1ull << 20u);
    }
    return CalcAlign(min_sz, 4ull << 20u);
}

enum class DxPoolKind : uint8_t {
    GpuUploadOrUpload,
    PinnedCachedHost,
    VramStagingDefault,
    ExportSerUav,
    ExportReadback
};

struct DxPoolSlot {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    D3D12_GPU_VIRTUAL_ADDRESS gpu_va{0u};
    std::byte *mapped_ptr{nullptr};
    size_t capacity{0u};
    uint64_t lru_tick{0u};
    uint64_t token{0u};
    uint32_t gpu_leases{0u};
    bool caller_owned{false};
    bool is_gpu_upload{false};
    bool is_write_back{false};
};

class DxBufferPool {
public:
    DxBufferPool(Device *dev, DxPoolKind kind, bool gpu_upload_supported) noexcept
        : _device{dev}, _kind{kind}, _gpu_upload_supported{gpu_upload_supported} {}

    ~DxBufferPool() noexcept {
        std::lock_guard lck{_mtx};
        for (auto &slot : _slots) {
            if (slot.resource && slot.mapped_ptr != nullptr && _kind != DxPoolKind::ExportReadback) {
                D3D12_RANGE write_range{0, 0};
                slot.resource->Unmap(0, &write_range);
                slot.mapped_ptr = nullptr;
            }
            slot.resource.Reset();
        }
        _slots.clear();
        _total_bytes = 0u;
    }

    struct Acquisition {
        size_t slot_index{0u};
        ID3D12Resource *resource{nullptr};
        D3D12_GPU_VIRTUAL_ADDRESS gpu_va{0u};
        std::byte *mapped_ptr{nullptr};
        size_t capacity{0u};
        uint64_t token{0u};
        bool is_gpu_upload{false};
        bool is_write_back{false};
    };

    [[nodiscard]] bool acquire(size_t min_bytes, bool for_caller, Acquisition &out) noexcept {
        std::lock_guard lck{_mtx};
        size_t const req_bytes = CalcAlign(std::max<size_t>(min_bytes, 256u), 256u);

        // Best-fit free slot >= req_bytes.
        size_t best_idx = _slots.size();
        for (size_t i = 0u; i < _slots.size(); ++i) {
            auto const &s = _slots[i];
            if (!s.caller_owned && s.gpu_leases == 0u && s.capacity >= req_bytes) {
                if (best_idx == _slots.size() || s.capacity < _slots[best_idx].capacity) {
                    best_idx = i;
                }
            }
        }
        if (best_idx < _slots.size()) {
            auto &s = _slots[best_idx];
            s.lru_tick = ++_tick;
            if (for_caller) {
                s.caller_owned = true;
            } else {
                ++s.gpu_leases;
            }
            out = Acquisition{
                .slot_index = best_idx,
                .resource = s.resource.Get(),
                .gpu_va = s.gpu_va,
                .mapped_ptr = s.mapped_ptr,
                .capacity = s.capacity,
                .token = s.token,
                .is_gpu_upload = s.is_gpu_upload,
                .is_write_back = s.is_write_back};
            return true;
        }

        size_t const alloc_bytes = round_up_pool_size(req_bytes);
        // Evict oldest idle slots if slot count or budget exceeded.
        auto evict_oldest_idle = [&]() noexcept -> bool {
            size_t victim = _slots.size();
            for (size_t i = 0u; i < _slots.size(); ++i) {
                if (!_slots[i].caller_owned && _slots[i].gpu_leases == 0u) {
                    if (victim == _slots.size() || _slots[i].lru_tick < _slots[victim].lru_tick) {
                        victim = i;
                    }
                }
            }
            if (victim == _slots.size()) return false;
            auto &v = _slots[victim];
            if (v.resource && v.mapped_ptr != nullptr && _kind != DxPoolKind::ExportReadback) {
                D3D12_RANGE write_range{0, 0};
                v.resource->Unmap(0, &write_range);
                v.mapped_ptr = nullptr;
            }
            _total_bytes -= v.capacity;
            if (victim + 1u < _slots.size()) {
                _slots[victim] = std::move(_slots.back());
            }
            _slots.pop_back();
            return true;
        };

        while (_slots.size() >= kDxMaxPoolSlotsPerKind ||
               (_total_bytes + alloc_bytes > kDxMaxPoolBytesPerKind && !_slots.empty())) {
            if (!evict_oldest_idle()) break;
        }

        DxPoolSlot created{};
        if (!_create_slot(alloc_bytes, created)) {
            return false;
        }
        created.lru_tick = ++_tick;
        created.token = (++_next_token) | (static_cast<uint64_t>(static_cast<uint8_t>(_kind) + 1u) << 56u);
        if (for_caller) {
            created.caller_owned = true;
        } else {
            created.gpu_leases = 1u;
        }
        _total_bytes += created.capacity;
        size_t const idx = _slots.size();
        _slots.push_back(std::move(created));
        auto const &s = _slots[idx];
        out = Acquisition{
            .slot_index = idx,
            .resource = s.resource.Get(),
            .gpu_va = s.gpu_va,
            .mapped_ptr = s.mapped_ptr,
            .capacity = s.capacity,
            .token = s.token,
            .is_gpu_upload = s.is_gpu_upload,
            .is_write_back = s.is_write_back};
        return true;
    }

    [[nodiscard]] bool match_caller_buffer_and_lease_gpu(
        const std::byte *ptr,
        size_t size,
        Acquisition &out,
        size_t &out_offset) noexcept {
        if (ptr == nullptr) return false;
        std::lock_guard lck{_mtx};
        for (size_t i = 0u; i < _slots.size(); ++i) {
            auto &s = _slots[i];
            if ((s.caller_owned || s.gpu_leases > 0u) && s.mapped_ptr != nullptr) {
                auto const *begin = s.mapped_ptr;
                auto const *end = begin + s.capacity;
                if (ptr >= begin && ptr <= end) {
                    size_t const off = static_cast<size_t>(ptr - begin);
                    if (off <= s.capacity && size <= s.capacity - off) {
                        ++s.gpu_leases;
                        s.lru_tick = ++_tick;
                        out = Acquisition{
                            .slot_index = i,
                            .resource = s.resource.Get(),
                            .gpu_va = s.gpu_va,
                            .mapped_ptr = s.mapped_ptr,
                            .capacity = s.capacity,
                            .token = s.token,
                            .is_gpu_upload = s.is_gpu_upload,
                            .is_write_back = s.is_write_back};
                        out_offset = off;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void release_caller(uint64_t token, const std::byte *mapped_data) noexcept {
        std::lock_guard lck{_mtx};
        for (auto &s : _slots) {
            if ((token != 0u && s.token == token) ||
                (mapped_data != nullptr && s.mapped_ptr == mapped_data)) {
                s.caller_owned = false;
                s.lru_tick = ++_tick;
                return;
            }
        }
    }

    void release_gpu_by_token(uint64_t token) noexcept {
        if (token == 0u) return;
        std::lock_guard lck{_mtx};
        for (auto &s : _slots) {
            if (s.token == token) {
                if (s.gpu_leases > 0u) {
                    --s.gpu_leases;
                }
                s.lru_tick = ++_tick;
                return;
            }
        }
    }

private:
    [[nodiscard]] bool _create_slot(size_t bytes, DxPoolSlot &out) noexcept {
        if (_device == nullptr || _device->device == nullptr) return false;
        auto *d3d_dev = _device->device.Get();
        bool const enhanced = _device->feature_check.enhanced_barriers_supported();
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
        if (_kind == DxPoolKind::VramStagingDefault || _kind == DxPoolKind::ExportSerUav) {
            flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        auto const desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);

        auto try_create = [&](D3D12_HEAP_PROPERTIES const &hp, D3D12_RESOURCE_STATES init_state) noexcept {
            out.resource.Reset();
            return SUCCEEDED(d3d_dev->CreateCommittedResource(
                &hp,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                init_state,
                nullptr,
                IID_PPV_ARGS(&out.resource)));
        };

        if (_kind == DxPoolKind::GpuUploadOrUpload) {
            if (_gpu_upload_supported) {
                auto const hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_GPU_UPLOAD);
                if (try_create(hp, D3D12_RESOURCE_STATE_GENERIC_READ)) {
                    out.is_gpu_upload = true;
                }
            }
            if (!out.resource) {
                auto const hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
                if (!try_create(hp, D3D12_RESOURCE_STATE_GENERIC_READ)) return false;
                out.is_gpu_upload = false;
            }
            D3D12_RANGE empty_read{0, 0};
            void *ptr = nullptr;
            if (FAILED(out.resource->Map(0, &empty_read, &ptr)) || ptr == nullptr) {
                out.resource.Reset();
                return false;
            }
            out.mapped_ptr = static_cast<std::byte *>(ptr);
        } else if (_kind == DxPoolKind::PinnedCachedHost) {
            D3D12_HEAP_PROPERTIES wb_hp{
                D3D12_HEAP_TYPE_CUSTOM,
                D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                D3D12_MEMORY_POOL_L0,
                0u,
                0u};
            if (try_create(wb_hp, D3D12_RESOURCE_STATE_GENERIC_READ)) {
                out.is_write_back = true;
            } else {
                auto const up_hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
                if (!try_create(up_hp, D3D12_RESOURCE_STATE_GENERIC_READ)) return false;
                out.is_write_back = false;
            }
            D3D12_RANGE empty_read{0, 0};
            void *ptr = nullptr;
            HRESULT map_hr = out.resource->Map(0, out.is_write_back ? nullptr : &empty_read, &ptr);
            if (FAILED(map_hr) || ptr == nullptr) {
                out.resource.Reset();
                return false;
            }
            out.mapped_ptr = static_cast<std::byte *>(ptr);
        } else if (_kind == DxPoolKind::VramStagingDefault) {
            auto const hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            if (!try_create(hp, D3D12_RESOURCE_STATE_COMMON)) return false;
        } else if (_kind == DxPoolKind::ExportSerUav) {
            auto const hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto const init_state = enhanced ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            if (!try_create(hp, init_state)) return false;
        } else if (_kind == DxPoolKind::ExportReadback) {
            auto const hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
            if (!try_create(hp, D3D12_RESOURCE_STATE_COPY_DEST)) return false;
        }

        out.capacity = bytes;
        out.gpu_va = out.resource->GetGPUVirtualAddress();
        return true;
    }

    Device *_device{nullptr};
    DxPoolKind _kind;
    bool _gpu_upload_supported{false};
    std::mutex _mtx;
    luisa::vector<DxPoolSlot> _slots;
    size_t _total_bytes{0u};
    uint64_t _tick{0u};
    uint64_t _next_token{0u};
};

void emit_resource_barrier(
    Device const *dev,
    CommandBufferBuilder &builder,
    ID3D12Resource *resource,
    D3D12_BARRIER_SYNC sync_before,
    D3D12_BARRIER_SYNC sync_after,
    D3D12_BARRIER_ACCESS access_before,
    D3D12_BARRIER_ACCESS access_after,
    D3D12_RESOURCE_STATES legacy_before,
    D3D12_RESOURCE_STATES legacy_after) {
    auto *cmdlist = builder.get_cb()->cmd_list();
    if (dev->feature_check.enhanced_barriers_supported()) {
        if (access_before == D3D12_BARRIER_ACCESS_COMMON && sync_before == D3D12_BARRIER_SYNC_NONE) {
            sync_before = D3D12_BARRIER_SYNC_ALL;
        }
        if (access_after == D3D12_BARRIER_ACCESS_COMMON && sync_after == D3D12_BARRIER_SYNC_NONE) {
            sync_after = D3D12_BARRIER_SYNC_ALL;
        }
        ComPtr<ID3D12GraphicsCommandList7> cmd7;
        ThrowIfFailed(cmdlist->QueryInterface(IID_PPV_ARGS(&cmd7)));
        D3D12_BUFFER_BARRIER barrier{
            .SyncBefore = sync_before,
            .SyncAfter = sync_after,
            .AccessBefore = access_before,
            .AccessAfter = access_after,
            .pResource = resource,
            .Offset = 0,
            .Size = UINT64_MAX};
        D3D12_BARRIER_GROUP group{
            .Type = D3D12_BARRIER_TYPE_BUFFER,
            .NumBarriers = 1,
            .pBufferBarriers = &barrier};
        cmd7->Barrier(1, &group);
    } else if (legacy_before != legacy_after) {
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            resource,
            legacy_before,
            legacy_after);
        cmdlist->ResourceBarrier(1, &barrier);
    }
}

[[nodiscard]] luisa::string wide_to_ascii(WCHAR const *wstr) {
    if (!wstr) return {};
    luisa::string out;
    for (size_t i = 0; wstr[i] != L'\0'; ++i) {
        wchar_t wc = wstr[i];
        out.push_back((wc >= 0x20 && wc < 0x7f) ? static_cast<char>(wc) : '?');
    }
    return out;
}

[[nodiscard]] bool utf8_to_wide(luisa::string_view utf8, luisa::vector<wchar_t> &out) noexcept {
    if (utf8.empty()) return false;
    int needed = MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (needed <= 0) return false;
    out.resize(static_cast<size_t>(needed) + 1u);
    int written = MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), out.data(), needed);
    if (written != needed) return false;
    out[static_cast<size_t>(needed)] = L'\0';
    return true;
}

[[nodiscard]] AccelArchiveResult check_stream_and_blas(
    LCDevice *device,
    uint64_t stream_handle,
    uint64_t mesh_handle,
    LCCmdBuffer *&out_cmd_buf,
    BottomAccel *&out_blas) noexcept {
    if (device == nullptr || device->use_fallback_rtx() || !device->hardware_raytracing_available()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 hardware DXR acceleration structure archiving is unavailable on this device.");
    }
    if (stream_handle == invalid_resource_handle || mesh_handle == invalid_resource_handle) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Invalid stream or mesh handle passed to DxAccelArchiveExt.");
    }
    if (device->owns_fallback_blas(mesh_handle)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "DX12 software fallback RTX BLAS cannot be archived via DXR serialization.");
    }
    auto *base = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (base->tag() != CmdQueueTag::MainCmd) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "stream_handle is not a main DX12 compute/graphics stream.");
    }
    auto *cmd_buf = static_cast<LCCmdBuffer *>(base);
    if (cmd_buf->queue.Type() == D3D12_COMMAND_LIST_TYPE_COPY) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "DXR AS serialize/deserialize requires a compute or direct queue.");
    }
    auto *blas = reinterpret_cast<BottomAccel *>(mesh_handle);
    if (blas->Option().allow_update) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static non-updateable triangle BLAS only.");
    }
    out_cmd_buf = cmd_buf;
    out_blas = blas;
    return {};
}

[[nodiscard]] bool options_match(AccelOption const &a, AccelOption const &b) noexcept {
    return a.hint == b.hint &&
           a.allow_compaction == b.allow_compaction &&
           a.allow_update == b.allow_update &&
           a.motion.keyframe_count == b.motion.keyframe_count;
}

}// namespace

struct DxAccelArchiveExt::Impl {
    // Declare DynamicModules BEFORE COM objects so COM refs drop first in ~Impl().
    luisa::DynamicModule dstorage_core_module;
    luisa::DynamicModule dstorage_module;
    std::mutex ds_mtx;
    bool ds_attempted{false};
    bool ds_available{false};
    Microsoft::WRL::ComPtr<IDStorageFactory> ds_factory;
    Microsoft::WRL::ComPtr<IDStorageQueue2> ds_queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> ds_fence;
    uint64_t ds_fence_value{0u};

    std::mutex export_info_mtx;
    Microsoft::WRL::ComPtr<ID3D12Resource> export_info_uav;
    Microsoft::WRL::ComPtr<ID3D12Resource> export_info_rb;

    std::shared_ptr<DxBufferPool> gpu_upload_pool;
    std::shared_ptr<DxBufferPool> pinned_cached_pool;
    std::shared_ptr<DxBufferPool> vram_staging_pool;
    std::shared_ptr<DxBufferPool> export_ser_uav_pool;
    std::shared_ptr<DxBufferPool> export_rb_pool;

    Impl(Device *native_dev, bool gpu_upload_supported) noexcept
        : gpu_upload_pool{std::make_shared<DxBufferPool>(native_dev, DxPoolKind::GpuUploadOrUpload, gpu_upload_supported)},
          pinned_cached_pool{std::make_shared<DxBufferPool>(native_dev, DxPoolKind::PinnedCachedHost, gpu_upload_supported)},
          vram_staging_pool{std::make_shared<DxBufferPool>(native_dev, DxPoolKind::VramStagingDefault, gpu_upload_supported)},
          export_ser_uav_pool{std::make_shared<DxBufferPool>(native_dev, DxPoolKind::ExportSerUav, gpu_upload_supported)},
          export_rb_pool{std::make_shared<DxBufferPool>(native_dev, DxPoolKind::ExportReadback, gpu_upload_supported)} {}

    ~Impl() noexcept {
        export_info_rb.Reset();
        export_info_uav.Reset();
        gpu_upload_pool.reset();
        pinned_cached_pool.reset();
        vram_staging_pool.reset();
        export_ser_uav_pool.reset();
        export_rb_pool.reset();
        ds_queue.Reset();
        ds_fence.Reset();
        ds_factory.Reset();
    }

    [[nodiscard]] bool ensure_export_info_buffers(Device *dev) noexcept {
        if (export_info_uav && export_info_rb) return true;
        bool const enhanced = dev->feature_check.enhanced_barriers_supported();
        auto const init_uav = enhanced ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        auto const desc_uav = CD3DX12_RESOURCE_DESC::Buffer(256u, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto const desc_rb = CD3DX12_RESOURCE_DESC::Buffer(256u, D3D12_RESOURCE_FLAG_NONE);
        auto const hp_def = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto const hp_rb = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        if (FAILED(dev->device->CreateCommittedResource(
                &hp_def, D3D12_HEAP_FLAG_NONE, &desc_uav, init_uav, nullptr,
                IID_PPV_ARGS(&export_info_uav)))) {
            return false;
        }
        if (FAILED(dev->device->CreateCommittedResource(
                &hp_rb, D3D12_HEAP_FLAG_NONE, &desc_rb, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&export_info_rb)))) {
            export_info_uav.Reset();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool ensure_dstorage(LCDevice *lc_dev) noexcept {
        std::lock_guard lck{ds_mtx};
        if (ds_attempted) return ds_available;
        ds_attempted = true;
        auto const &rt_dir = lc_dev->context().runtime_directory();
        dstorage_core_module = DynamicModule::load(rt_dir, "dstoragecore");
        dstorage_module = DynamicModule::load(rt_dir, "dstorage");
        if (!dstorage_module) return false;
        using DStorageGetFactoryFn = std::remove_pointer_t<decltype(&DStorageGetFactory)>;
        auto get_factory = dstorage_module.function<DStorageGetFactoryFn>("DStorageGetFactory");
        if (get_factory == nullptr) return false;
        if (FAILED(get_factory(IID_PPV_ARGS(ds_factory.GetAddressOf()))) || !ds_factory) {
            return false;
        }
        DSTORAGE_QUEUE_DESC qdesc{
            .SourceType = DSTORAGE_REQUEST_SOURCE_FILE,
            .Capacity = DSTORAGE_MAX_QUEUE_CAPACITY,
            .Priority = DSTORAGE_PRIORITY_NORMAL,
            .Name = "lc_accel_archive_ds",
            .Device = lc_dev->native_device.device.Get()};
        if (FAILED(ds_factory->CreateQueue(&qdesc, IID_PPV_ARGS(ds_queue.GetAddressOf()))) || !ds_queue) {
            ds_factory.Reset();
            return false;
        }
        if (FAILED(lc_dev->native_device.device->CreateFence(
                0u, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(ds_fence.GetAddressOf()))) ||
            !ds_fence) {
            ds_queue.Reset();
            ds_factory.Reset();
            return false;
        }
        ds_available = true;
        return true;
    }
};

DxAccelArchiveExt::DxAccelArchiveExt(LCDevice *device) noexcept
    : _device{device} {
    _supported = _device != nullptr &&
                 !_device->use_fallback_rtx() &&
                 _device->hardware_raytracing_available() &&
                 _device->native_device.adapter != nullptr &&
                 _device->native_device.device != nullptr;
    if (_supported) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(_device->native_device.adapter->GetDesc1(&desc))) {
            LARGE_INTEGER umd_version{};
            bool const has_umd = SUCCEEDED(_device->native_device.adapter->CheckInterfaceSupport(
                __uuidof(IDXGIDevice), &umd_version));
            auto adapter_name = wide_to_ascii(desc.Description);
            auto const rt_tier = static_cast<uint32_t>(_device->native_device.feature_check.raytracing_tier());
            auto const eb = _device->native_device.feature_check.enhanced_barriers_supported() ? 1u : 0u;

            D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16{};
            if (SUCCEEDED(_device->native_device.device->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16)))) {
                _gpu_upload_supported = options16.GPUUploadHeapSupported != FALSE;
            }

            _compat = CompatInfo{
                .vendor_id = desc.VendorId,
                .device_id = desc.DeviceId,
                .subsys_id = desc.SubSysId,
                .revision = desc.Revision,
                .magic = kDxCompatMagic,
                .version = kDxCompatVersion,
                .umd_version = has_umd ? static_cast<uint64_t>(umd_version.QuadPart) : 0ull,
                .raytracing_tier = rt_tier,
                .enhanced_barriers = eb,
                .deserialized_size = 0ull};
            _device_fingerprint = luisa::format("{}:0x{:04x}:0x{:04x}:rev{}",
                                                adapter_name, desc.VendorId, desc.DeviceId, desc.Revision);
            uint32_t const hi0 = static_cast<uint32_t>((_compat.umd_version >> 48u) & 0xffffu);
            uint32_t const hi1 = static_cast<uint32_t>((_compat.umd_version >> 32u) & 0xffffu);
            uint32_t const lo0 = static_cast<uint32_t>((_compat.umd_version >> 16u) & 0xffffu);
            uint32_t const lo1 = static_cast<uint32_t>(_compat.umd_version & 0xffffu);
            _driver_fingerprint = luisa::format("umd={}.{}.{}.{};rt_tier=0x{:x};eb={};gpu_up={}",
                                                hi0, hi1, lo0, lo1, rt_tier, eb, _gpu_upload_supported ? 1u : 0u);
            _impl = std::make_shared<Impl>(&_device->native_device, _gpu_upload_supported);
        } else {
            _supported = false;
        }
    } else if (_device != nullptr) {
        _device_fingerprint = "dx-fallback-or-no-dxr";
        _driver_fingerprint = "unsupported";
    }
}

DxAccelArchiveExt::~DxAccelArchiveExt() noexcept = default;

AccelArchiveCapabilities DxAccelArchiveExt::capabilities() const noexcept {
    AccelArchiveCapabilities caps{};
    caps.backend_kind = AccelArchiveBackendKind::DX;
    caps.envelope_version = accel_archive::kEnvelopeVersion;
    caps.device_fingerprint = _device_fingerprint;
    caps.driver_fingerprint = _driver_fingerprint;
    caps.supported = _supported;
    caps.static_triangle_blas = _supported;
    caps.supports_compacted_export = _supported;
    caps.requires_geometry_rebind = false;
    caps.import_is_async_on_stream = _supported;
    caps.supports_pinned_staging = _supported;
    caps.supports_direct_file_import = _supported;
    caps.layout_info = _supported ?
                           luisa::string{kDxLayoutTag} :
                           (_device != nullptr && _device->use_fallback_rtx() ?
                                "dx_fallback_rtx_disabled" :
                                "dxr_unavailable");
    return caps;
}

AccelArchiveStagingBuffer DxAccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    if (!_supported || !_impl || min_bytes == 0u) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    DxBufferPool::Acquisition acq{};
    if (!_impl->pinned_cached_pool->acquire(min_bytes, true, acq)) {
        return AccelArchiveExt::acquire_staging_buffer(min_bytes);
    }
    AccelArchiveStagingBuffer out{};
    out.handle = acq.token;
    out.mapped_data = acq.mapped_ptr;
    out.capacity = acq.capacity;
    out.alignment = 4096u;
    out.is_device_local = false;
    out.is_pinned_host = true;
    return out;
}

void DxAccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (_impl && buffer.handle != invalid_resource_handle && (buffer.handle >> 56u) != 0u) {
        _impl->pinned_cached_pool->release_caller(buffer.handle, buffer.mapped_data);
        buffer = {};
        return;
    }
    AccelArchiveExt::release_staging_buffer(std::move(buffer));
}

AccelArchiveResult DxAccelArchiveExt::_check_compat_and_header(
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload_prefix,
    uint64_t expected_payload_size,
    uint64_t meta_unpacked_size,
    uint64_t &out_deserialized_size) const noexcept {
    if (compat_blob.size() != sizeof(CompatInfo)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "DX12 AS archive compat blob has unexpected byte size.");
    }
    CompatInfo info{};
    std::memcpy(&info, compat_blob.data(), sizeof(CompatInfo));
    if (info.vendor_id != _compat.vendor_id ||
        info.device_id != _compat.device_id ||
        info.subsys_id != _compat.subsys_id ||
        info.revision != _compat.revision ||
        info.magic != kDxCompatMagic ||
        info.version != kDxCompatVersion ||
        info.raytracing_tier != _compat.raytracing_tier ||
        info.enhanced_barriers != _compat.enhanced_barriers) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            luisa::format(
                "DX12 AS archive targets vendor=0x{:04x}:0x{:04x}, current is 0x{:04x}:0x{:04x}.",
                info.vendor_id, info.device_id, _compat.vendor_id, _compat.device_id));
    }
    if (info.umd_version != _compat.umd_version) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "DX12 display driver UMD version does not match active device.");
    }
    if (expected_payload_size < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) ||
        payload_prefix.size() < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "DXR serialized BLAS payload is smaller than 56-byte header.");
    }
    D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER hdr{};
    std::memcpy(&hdr, payload_prefix.data(), sizeof(hdr));
    if (hdr.SerializedSizeInBytesIncludingHeader != expected_payload_size ||
        hdr.DeserializedSizeInBytes == 0u ||
        hdr.DeserializedSizeInBytes != meta_unpacked_size ||
        hdr.DeserializedSizeInBytes != info.deserialized_size ||
        hdr.NumBottomLevelAccelerationStructurePointersAfterHeader != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "DXR serialized BLAS header size or pointer count mismatch.");
    }
    auto const match_status = _device->native_device.device->CheckDriverMatchingIdentifier(
        D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE,
        &hdr.DriverMatchingIdentifier);
    if (match_status != D3D12_DRIVER_MATCHING_IDENTIFIER_COMPATIBLE_WITH_DEVICE) {
        auto const err_status =
            (match_status == D3D12_DRIVER_MATCHING_IDENTIFIER_INCOMPATIBLE_VERSION) ?
                AccelArchiveStatus::DRIVER_INCOMPATIBLE :
                AccelArchiveStatus::DEVICE_INCOMPATIBLE;
        return accel_archive::make_archive_error(
            err_status,
            luisa::format(
                "D3D12 CheckDriverMatchingIdentifier failed (status={}).",
                static_cast<uint32_t>(match_status)));
    }
    out_deserialized_size = hdr.DeserializedSizeInBytes;
    return {};
}

AccelArchiveResult DxAccelArchiveExt::validate_mesh_archive(
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata *out_meta) const noexcept {
    if (!capabilities().supported) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    auto env_res = accel_archive::decode_and_validate_envelope(
        archive_bytes,
        AccelArchiveBackendKind::DX,
        expected_desc,
        view);
    if (!env_res) {
        return env_res;
    }

    uint64_t deserialized_size = 0u;
    auto check_res = _check_compat_and_header(
        view.compat_blob,
        view.payload,
        view.payload.size(),
        view.meta.unpacked_as_size,
        deserialized_size);
    if (!check_res) {
        return check_res;
    }

    if (out_meta != nullptr) {
        *out_meta = view.meta;
    }
    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::NONE;
    res.unpacked_as_size = deserialized_size;
    res.serialized_payload_size = view.payload.size();
    res.is_compacted = view.meta.is_compacted;
    res.message = "validated";
    return res;
}

AccelArchiveResult DxAccelArchiveExt::export_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &desc,
    luisa::vector<std::byte> &out_archive) noexcept {
    out_archive.clear();
    if (!capabilities().supported || !_impl) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }
    auto desc_res = accel_archive::validate_v1_static_triangle_desc(desc);
    if (!desc_res) {
        return desc_res;
    }

    LCCmdBuffer *cmd_buf = nullptr;
    BottomAccel *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, cmd_buf, blas);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(blas->Option(), desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "DX12 BLAS AccelOption does not match AccelArchiveMeshDesc::option.");
    }

    // Drain any in-flight build and CheckAccel compaction on the caller stream first.
    cmd_buf->Sync();

    if (!blas->IsBuilt() || !blas->HasAccelBuffer()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::MESH_NOT_BUILT,
            "DX12 BLAS has not been built yet.");
    }
    if (!blas->IsTriangleMesh()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static triangle BLAS only.");
    }
    if (blas->LastTriangleCount() != desc.triangle_count) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::KEY_MISMATCH,
            luisa::format(
                "DX12 BLAS primitive count ({}) does not match expected triangle_count ({}).",
                blas->LastTriangleCount(), desc.triangle_count));
    }

    auto &native_dev = _device->native_device;
    bool const enhanced = native_dev.feature_check.enhanced_barriers_supported();
    auto const blas_va = blas->GetAccelBuffer()->GetAddress();

    // Pass 1: Query serialized byte size via persistent info UAV + Readback buffers.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC ser_desc{};
    {
        std::lock_guard info_lck{_impl->export_info_mtx};
        if (!_impl->ensure_export_info_buffers(&native_dev)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to allocate DX12 AS serialization size query buffer.");
        }
        auto *info_uav = _impl->export_info_uav.Get();
        auto *info_rb = _impl->export_info_rb.Get();
        {
            auto allocator = cmd_buf->queue.create_allocator(0);
            auto *cmd_buffer = allocator->get_buffer();
            {
                auto builder = cmd_buffer->build();
                auto *cmdlist = builder.get_cb()->cmd_list();
                emit_resource_barrier(
                    &native_dev, builder, info_uav,
                    D3D12_BARRIER_SYNC_NONE,
                    D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC post_info{
                    .DestBuffer = info_uav->GetGPUVirtualAddress(),
                    .InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION};
                cmdlist->EmitRaytracingAccelerationStructurePostbuildInfo(&post_info, 1u, &blas_va);
                emit_resource_barrier(
                    &native_dev, builder, info_uav,
                    D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmdlist->CopyBufferRegion(info_rb, 0u, info_uav, 0u, sizeof(ser_desc));
                emit_resource_barrier(
                    &native_dev, builder, info_uav,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_NONE,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    enhanced ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
            cmd_buf->queue.complete();
        }
        void *mapped = nullptr;
        D3D12_RANGE read_range{0, sizeof(ser_desc)};
        if (FAILED(info_rb->Map(0, &read_range, &mapped)) || mapped == nullptr) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to map DX12 serialization size query readback buffer.");
        }
        std::memcpy(&ser_desc, mapped, sizeof(ser_desc));
        D3D12_RANGE empty_write{0, 0};
        info_rb->Unmap(0, &empty_write);
    }

    if (ser_desc.SerializedSizeInBytes < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) ||
        ser_desc.NumBottomLevelAccelerationStructurePointers != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            luisa::format(
                "Invalid DXR serialized size query (size={}, blas_ptrs={}).",
                ser_desc.SerializedSizeInBytes, ser_desc.NumBottomLevelAccelerationStructurePointers));
    }

    // Pass 2: Serialize BLAS into pooled GPU UAV buffer -> pooled Readback buffer.
    auto const aligned_ser_size = CalcAlign(ser_desc.SerializedSizeInBytes, 256u);
    DxBufferPool::Acquisition ser_uav_acq{};
    DxBufferPool::Acquisition ser_rb_acq{};
    if (!_impl->export_ser_uav_pool->acquire(aligned_ser_size, false, ser_uav_acq)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to acquire DX12 AS serialization UAV buffer.");
    }
    if (!_impl->export_rb_pool->acquire(ser_desc.SerializedSizeInBytes, false, ser_rb_acq)) {
        _impl->export_ser_uav_pool->release_gpu_by_token(ser_uav_acq.token);
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to acquire DX12 AS serialization readback buffer.");
    }

    auto release_export_slots = vstd::scope_exit([&]() noexcept {
        _impl->export_ser_uav_pool->release_gpu_by_token(ser_uav_acq.token);
        _impl->export_rb_pool->release_gpu_by_token(ser_rb_acq.token);
    });

    {
        auto allocator = cmd_buf->queue.create_allocator(0);
        auto *cmd_buffer = allocator->get_buffer();
        {
            auto builder = cmd_buffer->build();
            auto *cmdlist = builder.get_cb()->cmd_list();
            emit_resource_barrier(
                &native_dev, builder, ser_uav_acq.resource,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cmdlist->CopyRaytracingAccelerationStructure(
                ser_uav_acq.gpu_va,
                blas_va,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE);
            emit_resource_barrier(
                &native_dev, builder, ser_uav_acq.resource,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmdlist->CopyBufferRegion(
                ser_rb_acq.resource, 0u,
                ser_uav_acq.resource, 0u,
                ser_desc.SerializedSizeInBytes);
            emit_resource_barrier(
                &native_dev, builder, ser_uav_acq.resource,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                enhanced ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
        cmd_buf->queue.complete();
    }

    void *rb_mapped = nullptr;
    D3D12_RANGE rb_read_range{0, ser_desc.SerializedSizeInBytes};
    if (FAILED(ser_rb_acq.resource->Map(0, &rb_read_range, &rb_mapped)) || rb_mapped == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to map DX12 serialization readback buffer.");
    }
    auto unmap_rb = vstd::scope_exit([&]() noexcept {
        D3D12_RANGE empty_write{0, 0};
        ser_rb_acq.resource->Unmap(0, &empty_write);
    });

    D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER hdr{};
    std::memcpy(&hdr, rb_mapped, sizeof(hdr));
    if (hdr.SerializedSizeInBytesIncludingHeader != ser_desc.SerializedSizeInBytes ||
        hdr.DeserializedSizeInBytes == 0u ||
        hdr.NumBottomLevelAccelerationStructurePointersAfterHeader != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "DXR serialized BLAS header failed post-readback sanity checks.");
    }

    CompatInfo compat_to_write = _compat;
    compat_to_write.deserialized_size = hdr.DeserializedSizeInBytes;

    bool const is_compacted = blas->IsCompacted();
    luisa::span<const std::byte> payload_span{
        static_cast<const std::byte *>(rb_mapped),
        ser_desc.SerializedSizeInBytes};
    auto encode_res = accel_archive::encode_envelope(
        AccelArchiveBackendKind::DX,
        desc,
        is_compacted,
        hdr.DeserializedSizeInBytes,
        luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(&compat_to_write),
            sizeof(compat_to_write)},
        payload_span,
        out_archive);
    if (!encode_res) {
        return encode_res;
    }

    encode_res.readiness = AccelArchiveReadiness::READY_IMMEDIATELY;
    encode_res.unpacked_as_size = hdr.DeserializedSizeInBytes;
    encode_res.serialized_payload_size = payload_span.size();
    encode_res.is_compacted = is_compacted;
    encode_res.message = "exported";
    return encode_res;
}

AccelArchiveResult DxAccelArchiveExt::import_mesh_archive(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes) noexcept {
    AccelArchiveImportOptions default_opts{};
    return import_mesh_archive_with_options(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        archive_bytes,
        default_opts);
}

AccelArchiveResult DxAccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &options) noexcept {
    if (!capabilities().supported || !_impl) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }

    accel_archive::AccelArchiveEnvelopeView view{};
    uint8_t expected_payload_checksum[16]{};
    auto env_res = accel_archive::decode_envelope_header_and_compat(
        archive_bytes,
        AccelArchiveBackendKind::DX,
        expected_desc,
        view,
        expected_payload_checksum);
    if (!env_res) {
        return env_res;
    }

    uint64_t deserialized_size = 0u;
    auto check_res = _check_compat_and_header(
        view.compat_blob,
        view.payload,
        view.payload.size(),
        view.meta.unpacked_as_size,
        deserialized_size);
    if (!check_res) {
        return check_res;
    }

    LCCmdBuffer *cmd_buf = nullptr;
    BottomAccel *blas = nullptr;
    auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, cmd_buf, blas);
    if (!handle_res) {
        return handle_res;
    }
    if (!options_match(blas->Option(), expected_desc.option)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target DX12 Mesh AccelOption does not match archive AccelOption.");
    }

    auto &native_dev = _device->native_device;
    auto const payload_size = view.payload.size();

    // Check if caller filled one of our persistently pinned staging buffers (zero CPU copy!).
    DxBufferPool::Acquisition caller_pinned{};
    size_t caller_offset = 0u;
    bool const is_pinned_zero_copy = _impl->pinned_cached_pool->match_caller_buffer_and_lease_gpu(
        view.payload.data(), payload_size, caller_pinned, caller_offset);

    if (is_pinned_zero_copy) {
        if (options.verify_payload_checksum) {
            auto csum_res = accel_archive::verify_payload_checksum(view.payload, expected_payload_checksum);
            if (!csum_res) {
                _impl->pinned_cached_pool->release_gpu_by_token(caller_pinned.token);
                return csum_res;
            }
        }

        // CopyBufferRegion SDMA from pinned L0 -> VRAM DEFAULT staging -> DESERIALIZE.
        DxBufferPool::Acquisition vram_acq{};
        if (!_impl->vram_staging_pool->acquire(payload_size, false, vram_acq)) {
            _impl->pinned_cached_pool->release_gpu_by_token(caller_pinned.token);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to acquire DX12 VRAM staging buffer for pinned import.");
        }

        auto allocator = cmd_buf->queue.create_allocator(native_dev.max_allocator_count);
        auto new_accel_buf = vstd::create_unique(new DefaultBuffer(
            &native_dev,
            CalcAlign(deserialized_size, 65536u),
            native_dev.default_allocator.get(),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            false,
            "blas-accel-buffer"));

        {
            auto *cmd_buffer = allocator->get_buffer();
            auto builder = cmd_buffer->build();
            auto *cmdlist = builder.get_cb()->cmd_list();
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST);
            cmdlist->CopyBufferRegion(
                vram_acq.resource, 0u,
                caller_pinned.resource, caller_offset,
                payload_size);
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmdlist->CopyRaytracingAccelerationStructure(
                new_accel_buf->GetAddress(),
                vram_acq.gpu_va,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
            BottomAccel::EmitPostCopyBarrier(builder, new_accel_buf.get());
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        }

        blas->ReplaceFromArchive(
            allocator.get(),
            std::move(new_accel_buf),
            deserialized_size,
            view.meta.is_compacted,
            expected_desc.triangle_count);

        allocator->execute_after_complete([
                                              pinned_pool = _impl->pinned_cached_pool,
                                              vram_pool = _impl->vram_staging_pool,
                                              pinned_tok = caller_pinned.token,
                                              vram_tok = vram_acq.token]() noexcept {
            pinned_pool->release_gpu_by_token(pinned_tok);
            vram_pool->release_gpu_by_token(vram_tok);
        });
        cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
    } else {
        // Pageable host memory: blast into persistent GPU_UPLOAD (ReBAR VRAM) or UPLOAD pool
        // in parallel while concurrently hashing the payload in cached RAM.
        DxBufferPool::Acquisition up_acq{};
        if (!_impl->gpu_upload_pool->acquire(payload_size, false, up_acq)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to acquire DX12 GPU upload staging buffer.");
        }

        if (!accel_archive::copy_payload_optimized(
                up_acq.mapped_ptr,
                view.payload,
                options.verify_payload_checksum,
                expected_payload_checksum)) {
            _impl->gpu_upload_pool->release_gpu_by_token(up_acq.token);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                "DX12 AS archive payload XXH3-128 checksum mismatch.");
        }

        auto allocator = cmd_buf->queue.create_allocator(native_dev.max_allocator_count);
        auto new_accel_buf = vstd::create_unique(new DefaultBuffer(
            &native_dev,
            CalcAlign(deserialized_size, 65536u),
            native_dev.default_allocator.get(),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            false,
            "blas-accel-buffer"));

        DxBufferPool::Acquisition vram_acq{};
        if (up_acq.is_gpu_upload) {
            // ReBAR GPU_UPLOAD lives directly in VRAM — deserialize directly with zero intermediate copy!
            auto *cmd_buffer = allocator->get_buffer();
            auto builder = cmd_buffer->build();
            builder.get_cb()->cmd_list()->CopyRaytracingAccelerationStructure(
                new_accel_buf->GetAddress(),
                up_acq.gpu_va,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
            BottomAccel::EmitPostCopyBarrier(builder, new_accel_buf.get());
        } else {
            // Non-ReBAR UPLOAD fallback: DMA CopyBufferRegion -> DEFAULT VRAM -> DESERIALIZE.
            if (!_impl->vram_staging_pool->acquire(payload_size, false, vram_acq)) {
                _impl->gpu_upload_pool->release_gpu_by_token(up_acq.token);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::BACKEND_ERROR,
                    "Failed to acquire DX12 VRAM staging buffer.");
            }
            auto *cmd_buffer = allocator->get_buffer();
            auto builder = cmd_buffer->build();
            auto *cmdlist = builder.get_cb()->cmd_list();
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_NONE,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST);
            cmdlist->CopyBufferRegion(
                vram_acq.resource, 0u,
                up_acq.resource, 0u,
                payload_size);
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmdlist->CopyRaytracingAccelerationStructure(
                new_accel_buf->GetAddress(),
                vram_acq.gpu_va,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
            BottomAccel::EmitPostCopyBarrier(builder, new_accel_buf.get());
            emit_resource_barrier(
                &native_dev, builder, vram_acq.resource,
                D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        }

        blas->ReplaceFromArchive(
            allocator.get(),
            std::move(new_accel_buf),
            deserialized_size,
            view.meta.is_compacted,
            expected_desc.triangle_count);

        allocator->execute_after_complete([
                                              up_pool = _impl->gpu_upload_pool,
                                              vram_pool = _impl->vram_staging_pool,
                                              up_tok = up_acq.token,
                                              vram_tok = vram_acq.token]() noexcept {
            up_pool->release_gpu_by_token(up_tok);
            vram_pool->release_gpu_by_token(vram_tok);
        });
        cmd_buf->queue.execute(std::move(allocator), {}, {}, false);
    }

    AccelArchiveResult res{};
    res.status = AccelArchiveStatus::OK;
    res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
    res.unpacked_as_size = deserialized_size;
    res.serialized_payload_size = payload_size;
    res.is_compacted = view.meta.is_compacted;
    res.message = is_pinned_zero_copy ? "imported_pinned_on_stream" : "imported_on_stream";
    return res;
}

AccelArchiveResult DxAccelArchiveExt::import_mesh_archive_file(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::string_view file_path,
    const AccelArchiveImportOptions &options) noexcept {
    if (!capabilities().supported || !_impl) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND,
            "DX12 DXR AS archive extension is not supported on this device.");
    }

    // When verification is disabled and DirectStorage is available, stream the payload
    // directly NVMe -> DEFAULT VRAM without touching host payload buffers!
    if (!options.verify_payload_checksum && _impl->ensure_dstorage(_device)) {
        LCCmdBuffer *cmd_buf = nullptr;
        BottomAccel *blas = nullptr;
        auto handle_res = check_stream_and_blas(_device, stream_handle, mesh_handle, cmd_buf, blas);
        if (!handle_res) {
            return handle_res;
        }
        if (!options_match(blas->Option(), expected_desc.option)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::OPTIONS_MISMATCH,
                "Target DX12 Mesh AccelOption does not match archive AccelOption.");
        }

        luisa::vector<wchar_t> wpath;
        if (!utf8_to_wide(file_path, wpath)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                "Invalid UTF-8 file path passed to DxAccelArchiveExt::import_mesh_archive_file.");
        }

        HANDLE hfile = CreateFileW(
            wpath.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
        if (hfile == INVALID_HANDLE_VALUE) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                luisa::format("Failed to open archive file: {}", file_path));
        }

        LARGE_INTEGER li_size{};
        if (!GetFileSizeEx(hfile, &li_size) || li_size.QuadPart <= 0) {
            CloseHandle(hfile);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_HEADER,
                "Archive file is empty or size query failed.");
        }
        uint64_t const total_file_size = static_cast<uint64_t>(li_size.QuadPart);
        size_t prefix_cap = static_cast<size_t>(std::min<uint64_t>(total_file_size, 4096ull));
        luisa::vector<std::byte> prefix(prefix_cap);
        DWORD bytes_read = 0u;
        if (!ReadFile(hfile, prefix.data(), static_cast<DWORD>(prefix_cap), &bytes_read, nullptr) ||
            bytes_read < sizeof(accel_archive::AccelArchiveEnvelopeHeader)) {
            CloseHandle(hfile);
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_HEADER,
                "Failed to read archive envelope header.");
        }
        prefix.resize(bytes_read);

        accel_archive::AccelArchiveEnvelopeHeader hdr_preview{};
        std::memcpy(&hdr_preview, prefix.data(), sizeof(hdr_preview));
        size_t const min_prefix_needed =
            static_cast<size_t>(hdr_preview.header_size) +
            static_cast<size_t>(hdr_preview.compat_blob_size) +
            sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER);
        if (min_prefix_needed > prefix.size() && min_prefix_needed <= total_file_size && min_prefix_needed <= (1ull << 20u)) {
            LARGE_INTEGER zero{};
            SetFilePointerEx(hfile, zero, nullptr, FILE_BEGIN);
            prefix.resize(min_prefix_needed);
            DWORD extra_read = 0u;
            if (!ReadFile(hfile, prefix.data(), static_cast<DWORD>(min_prefix_needed), &extra_read, nullptr)) {
                CloseHandle(hfile);
                return accel_archive::make_archive_error(
                    AccelArchiveStatus::CORRUPT_HEADER,
                    "Failed to read extended archive envelope prefix.");
            }
            prefix.resize(extra_read);
        }
        CloseHandle(hfile);

        AccelArchiveMetadata meta{};
        uint64_t payload_offset = 0u;
        luisa::span<const std::byte> compat_blob;
        uint8_t dummy_checksum[16]{};
        auto pre_res = accel_archive::decode_envelope_prefix(
            prefix,
            total_file_size,
            AccelArchiveBackendKind::DX,
            expected_desc,
            meta,
            payload_offset,
            compat_blob,
            dummy_checksum);
        if (!pre_res) {
            return pre_res;
        }
        if (prefix.size() < payload_offset + sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER)) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::CORRUPT_PAYLOAD,
                "Archive file prefix does not cover DXR serialized header.");
        }
        luisa::span<const std::byte> payload_hdr_span{
            prefix.data() + payload_offset,
            prefix.size() - static_cast<size_t>(payload_offset)};

        uint64_t deserialized_size = 0u;
        auto check_res = _check_compat_and_header(
            compat_blob,
            payload_hdr_span,
            meta.payload_size,
            meta.unpacked_as_size,
            deserialized_size);
        if (!check_res) {
            return check_res;
        }

        Microsoft::WRL::ComPtr<IDStorageFile> ds_file;
        if (SUCCEEDED(_impl->ds_factory->OpenFile(wpath.data(), IID_PPV_ARGS(ds_file.GetAddressOf()))) &&
            ds_file) {
            DxBufferPool::Acquisition vram_acq{};
            if (_impl->vram_staging_pool->acquire(meta.payload_size, false, vram_acq)) {
                uint64_t sig_val = 0u;
                {
                    std::lock_guard lck{_impl->ds_mtx};
                    size_t consumed = 0u;
                    while (consumed < meta.payload_size) {
                        size_t const chunk = std::min<size_t>(meta.payload_size - consumed, kDxDStorageChunkBytes);
                        DSTORAGE_REQUEST req{};
                        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                        req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                        req.Source.File.Source = ds_file.Get();
                        req.Source.File.Offset = payload_offset + consumed;
                        req.Source.File.Size = static_cast<UINT32>(chunk);
                        req.Destination.Buffer.Resource = vram_acq.resource;
                        req.Destination.Buffer.Offset = consumed;
                        req.Destination.Buffer.Size = static_cast<UINT32>(chunk);
                        req.UncompressedSize = static_cast<UINT32>(chunk);
                        _impl->ds_queue->EnqueueRequest(&req);
                        consumed += chunk;
                    }
                    sig_val = ++_impl->ds_fence_value;
                    _impl->ds_queue->EnqueueSignal(_impl->ds_fence.Get(), sig_val);
                    _impl->ds_queue->Submit();
                }

                // Queue GPU fence wait BEFORE our DESERIALIZE command list on cmd_buf's queue.
                auto &native_dev = _device->native_device;
                cmd_buf->queue.queue()->Wait(_impl->ds_fence.Get(), sig_val);

                auto allocator = cmd_buf->queue.create_allocator(native_dev.max_allocator_count);
                auto new_accel_buf = vstd::create_unique(new DefaultBuffer(
                    &native_dev,
                    CalcAlign(deserialized_size, 65536u),
                    native_dev.default_allocator.get(),
                    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                    false,
                    "blas-accel-buffer"));

                {
                    auto *cmd_buffer = allocator->get_buffer();
                    auto builder = cmd_buffer->build();
                    auto *cmdlist = builder.get_cb()->cmd_list();
                    emit_resource_barrier(
                        &native_dev, builder, vram_acq.resource,
                        D3D12_BARRIER_SYNC_ALL,
                        D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_ACCESS_COMMON,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    cmdlist->CopyRaytracingAccelerationStructure(
                        new_accel_buf->GetAddress(),
                        vram_acq.gpu_va,
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
                    BottomAccel::EmitPostCopyBarrier(builder, new_accel_buf.get());
                    emit_resource_barrier(
                        &native_dev, builder, vram_acq.resource,
                        D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_SYNC_ALL,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                        D3D12_BARRIER_ACCESS_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON);
                }

                blas->ReplaceFromArchive(
                    allocator.get(),
                    std::move(new_accel_buf),
                    deserialized_size,
                    meta.is_compacted,
                    expected_desc.triangle_count);

                auto file_holder = std::make_shared<Microsoft::WRL::ComPtr<IDStorageFile>>(std::move(ds_file));
                allocator->execute_after_complete([
                                                      vram_pool = _impl->vram_staging_pool,
                                                      vram_tok = vram_acq.token,
                                                      file_holder]() noexcept {
                    file_holder->Reset();
                    vram_pool->release_gpu_by_token(vram_tok);
                });
                cmd_buf->queue.execute(std::move(allocator), {}, {}, false);

                AccelArchiveResult res{};
                res.status = AccelArchiveStatus::OK;
                res.readiness = AccelArchiveReadiness::READY_ON_STREAM;
                res.unpacked_as_size = deserialized_size;
                res.serialized_payload_size = meta.payload_size;
                res.is_compacted = meta.is_compacted;
                res.message = "imported_dstorage_on_stream";
                return res;
            }
        }
    }

    // Verified path (or DirectStorage unavailable fallback): base class reads unbuffered
    // directly into our WRITE_BACK L0 pinned pool and calls import_mesh_archive_with_options().
    return AccelArchiveExt::import_mesh_archive_file(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        file_path,
        options);
}

}// namespace lc::dx
