#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include "cuda_error.h"
#include "cuda_buffer.h"

namespace luisa::compute::cuda::cuda_buffer_detail {

/**
 * @brief Process-wide cache of freed device blocks, keyed by (context, exact size).
 *
 * cuMemAlloc is a driver round trip and cuMemFree
 * **synchronises the device**, so constructing and destroying a buffer costs two
 * implicit barriers (and a free in the middle of a frame flushes the pending command
 * list).  Reusing a block of exactly the requested size makes that a free-list pop.
 *
 * Only blocks that came from cuMemAlloc pass through here: host-mapped buffers,
 * externally owned memory and virtual-address reservations keep their own paths.
 * The cache is bounded by LUISA_CUDA_ALLOC_CACHE_MB (default 256, 0 disables it) and
 * is flushed before an allocation reports out-of-memory, so it can never be the reason
 * a request fails.
 *
 * Blocks are keyed by the CUDA context rather than by the device: a device pointer is
 * only valid inside the address space of the context that allocated it, and a context
 * that is destroyed (last `cuDevicePrimaryCtxRelease`) and re-created on the same
 * device must never inherit the previous context's blocks.
 */
class DeviceBlockCache {

private:
    std::mutex _mutex;
    std::map<std::pair<CUcontext, size_t>, std::vector<CUdeviceptr>> _blocks;
    size_t _cached_bytes{0u};
    size_t _cap_bytes{0u};
    bool _enabled{false};

private:
    DeviceBlockCache() noexcept {
        auto env = std::getenv("LUISA_CUDA_ALLOC_CACHE_MB");
        auto mb = env == nullptr ? 256 : std::atoi(env);
        _enabled = mb > 0;
        _cap_bytes = static_cast<size_t>(_enabled ? mb : 0) << 20;
    }

    /// The context bound to the calling thread, i.e. the address space that
    /// `cuMemAlloc`/`cuMemFree` operate on; `nullptr` when nothing is bound.
    [[nodiscard]] static CUcontext _current_context() noexcept {
        CUcontext context{nullptr};
        return cuCtxGetCurrent(&context) == CUDA_SUCCESS ? context : nullptr;
    }

    /// Detach every block cached for `context`; the caller frees them.
    [[nodiscard]] std::vector<CUdeviceptr> _detach(CUcontext context) noexcept {
        std::vector<CUdeviceptr> detached;
        std::scoped_lock lock{_mutex};
        for (auto it = _blocks.begin(); it != _blocks.end();) {
            if (it->first.first != context) {
                ++it;
                continue;
            }
            for (auto block : it->second) { detached.emplace_back(block); }
            _cached_bytes -= it->second.size() * it->first.second;
            it = _blocks.erase(it);
        }
        return detached;
    }

public:
    [[nodiscard]] static DeviceBlockCache &instance() noexcept {
        static DeviceBlockCache cache;
        return cache;
    }

    [[nodiscard]] CUdeviceptr acquire(size_t size) noexcept {
        if (!_enabled || size == 0u) { return 0u; }
        auto context = _current_context();
        if (context == nullptr) { return 0u; }
        std::scoped_lock lock{_mutex};
        auto it = _blocks.find({context, size});
        if (it == _blocks.end() || it->second.empty()) { return 0u; }
        auto block = it->second.back();
        it->second.pop_back();
        _cached_bytes -= size;
        return block;
    }

    void release(CUdeviceptr ptr, size_t size) noexcept {
        auto cached = false;
        if (_enabled && size != 0u) {
            if (auto context = _current_context(); context != nullptr) {
                std::scoped_lock lock{_mutex};
                if (_cached_bytes + size <= _cap_bytes) {
                    _blocks[{context, size}].emplace_back(ptr);
                    _cached_bytes += size;
                    cached = true;
                }
            }
        }
        if (!cached) { LUISA_CHECK_CUDA(cuMemFree(ptr)); }
    }

    /// Hand the cached blocks of the calling thread's context back to the driver.  Used
    /// when a fresh allocation fails (our own cache must never be the reason a request
    /// is refused) and when the device owning that context is torn down.
    void flush_current_context() noexcept {
        auto blocks = _detach(_current_context());
        for (auto block : blocks) { LUISA_CHECK_CUDA(cuMemFree(block)); }
    }
};

void flush_cached_device_blocks() noexcept {
    DeviceBlockCache::instance().flush_current_context();
}

}// namespace luisa::compute::cuda::cuda_buffer_detail

namespace luisa::compute::cuda {

CUDABufferBase::CUDABufferBase(size_t size_bytes, Location loc, int host_alloc_flags) noexcept
    : _device_address{}, _size_bytes{size_bytes}, _external_memory{false}, _is_reserved_memory{false} {
    switch (loc) {
        case Location::FORCE_HOST: {
            LUISA_CHECK_CUDA(cuMemHostAlloc(&_host_address, size_bytes, host_alloc_flags));
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&_device_address, _host_address, 0));
            break;
        }
        case Location::PREFER_DEVICE: {
            if (auto cached = cuda_buffer_detail::DeviceBlockCache::instance().acquire(size_bytes)) {
                _device_address = cached;
                break;
            }
            auto ret = cuMemAlloc(&_device_address, size_bytes);
            if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
                // Our own cache may hold exactly what was just refused: hand it back
                // and try once more before degrading to host memory.
                cuda_buffer_detail::DeviceBlockCache::instance().flush_current_context();
                ret = cuMemAlloc(&_device_address, size_bytes);
            }
            if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
                LUISA_WARNING(
                    "CUDA allocation out of device memory. Falling back to host memory.\n"
                    " THIS MAY CAUSE SIGNIFICANT PERFORMANCE DEGRADATION\n"
                    " PLEASE CONSIDER REDUCING THE WORKING SET OF YOUR APPLICATION");
                LUISA_CHECK_CUDA(cuMemHostAlloc(&_host_address, size_bytes,
                                                CU_MEMHOSTALLOC_DEVICEMAP |
                                                    CU_MEMHOSTALLOC_WRITECOMBINED));
                LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&_device_address, _host_address, 0));
            } else {
                LUISA_CHECK_CUDA(ret);
            }
            break;
        }
        case Location::RESERVED_MEMORY: {
            if (auto ret = cuMemAddressReserve(&_device_address, size_bytes, 0, 0, 0);
                ret == CUDA_ERROR_OUT_OF_MEMORY) {
                LUISA_ERROR("CUDA allocation out of device memory.");
            }
            _is_reserved_memory = true;
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Invalid CUDABufferBase::Location.");
    }
    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated CUDA buffer: {} bytes @ {}",
        size_bytes, reinterpret_cast<void *>(_device_address));
}

CUDABufferBase::~CUDABufferBase() noexcept {
    // we do not own external memory so do not free it
    if (_external_memory) { return; }
    if (_host_address != nullptr) {
        LUISA_CHECK_CUDA(cuMemFreeHost(_host_address));
    } else if (_is_reserved_memory) {
        LUISA_CHECK_CUDA(cuMemAddressFree(_device_address, _size_bytes));
    } else {
                    cuda_buffer_detail::DeviceBlockCache::instance().release(_device_address, _size_bytes);
    }
    auto size = _size_bytes;
    LUISA_VERBOSE_WITH_LOCATION(
        "Freed CUDA buffer: {} bytes @ {}",
        size, reinterpret_cast<void *>(_device_address));
}

CUDABufferBase::CUDABufferBase(CUdeviceptr external_ptr, size_t size_bytes) noexcept
    : _device_address{external_ptr},
      _size_bytes{size_bytes},
      _external_memory{true},
      _is_reserved_memory{false} {}

CUDABuffer::Binding CUDABuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= size_bytes(), "CUDABuffer::binding() out of range.");
    return Binding{device_address() + offset, size};
}

CUDAIndirectDispatchBuffer::CUDAIndirectDispatchBuffer(size_t capacity) noexcept
    : CUDABufferBase{sizeof(Header) + sizeof(Dispatch) * capacity},
      _capacity{capacity} {}

CUDAIndirectDispatchBuffer::Binding
CUDAIndirectDispatchBuffer::binding(size_t offset, size_t size) const noexcept {
    size = std::min(size, std::numeric_limits<size_t>::max() - offset);// prevent overflow
    return {device_address(),
            static_cast<uint>(offset),
            static_cast<uint>(std::min(offset + size, capacity()))};
}

}// namespace luisa::compute::cuda
