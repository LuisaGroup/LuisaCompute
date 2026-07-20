//
// Created by mike on 1/10/26.
//

#include <atomic>
#include <bit>
#include <cstdint>
#include <thread>

#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_device.h"
#include "hip_event.h"

namespace luisa::compute::hip {

namespace {

// HSA signal comparisons use signed 64-bit values. Flipping the sign bit maps
// the complete unsigned fence domain onto the signed ordering without losing
// either zero or UINT64_MAX.
constexpr auto kHsaSignedOrderBit = uint64_t{1u} << 63u;

[[nodiscard]] constexpr auto encode_fence(uint64_t value) noexcept {
    return std::bit_cast<int64_t>(value ^ kHsaSignedOrderBit);
}

[[nodiscard]] constexpr auto decode_fence(int64_t value) noexcept {
    return std::bit_cast<uint64_t>(value) ^ kHsaSignedOrderBit;
}

class HIPEventSignalUpdate {

private:
    int64_t *_semaphore;
    uint64_t _value;

    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<HIPEventSignalUpdate> pool;
        return pool;
    }

public:
    HIPEventSignalUpdate(int64_t *semaphore, uint64_t value) noexcept
        : _semaphore{semaphore}, _value{value} {}

    [[nodiscard]] static auto create(int64_t *semaphore, uint64_t value) noexcept {
        return _pool().create(semaphore, value);
    }

    void apply() noexcept {
        auto semaphore = std::atomic_ref<int64_t>{*_semaphore};
        auto encoded_current = semaphore.load(std::memory_order_acquire);
        for (;;) {
            auto current = decode_fence(encoded_current);
            if (current >= _value) { break; }
            auto encoded_desired = encode_fence(_value);
            if (semaphore.compare_exchange_weak(
                    encoded_current, encoded_desired,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                break;
            }
        }
        recycle();
    }

    void recycle() noexcept {
        _pool().destroy(this);
    }
};

static_assert(std::atomic_ref<int64_t>::is_always_lock_free,
              "HIP timeline events require lock-free host 64-bit atomics.");

void luisa_hip_event_signal_callback(void *data) noexcept {
    static_cast<HIPEventSignalUpdate *>(data)->apply();
}

}// namespace

HIPEvent::HIPEvent(HIPDevice *device) noexcept
    : _semaphore_device_ptr{}, _semaphore_host_ptr{} {
    auto stream_mem_op_support = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(&stream_mem_op_support,
                                          hipDeviceAttributeCanUseStreamWaitValue,
                                          device->device_id()));
    LUISA_ASSERT(stream_mem_op_support,
                 "HIP device does not support stream-ordered memory operations. "
                 "HIPEvent cannot be created.");
    // allocate memory for the semaphore with the hipMallocSignalMemory flag
    LUISA_CHECK_HIP(hipExtMallocWithFlags(
        &_semaphore_device_ptr, sizeof(uint64_t), hipMallocSignalMemory));
    // Signal-memory allocations are fine-grained atomic SVM allocations, so
    // the pointer returned by hipExtMallocWithFlags is directly host-visible.
    // hipHostGetDevicePointer performs the opposite host-to-device mapping and
    // must not be used on this device allocation.
    _semaphore_host_ptr = static_cast<int64_t *>(_semaphore_device_ptr);
    LUISA_ASSERT(_semaphore_host_ptr != nullptr,
                 "Failed to allocate HIP event semaphore memory.");
    LUISA_ASSERT(reinterpret_cast<uintptr_t>(_semaphore_host_ptr) %
                         std::atomic_ref<int64_t>::required_alignment ==
                     0u,
                 "HIP event semaphore memory is not sufficiently aligned for host atomics.");
    // ROCr created the hsa_signal_value_t object represented by this pointer;
    // update that existing signed value atomically rather than starting a new
    // C++ object lifetime over it.
    std::atomic_ref<int64_t>{*_semaphore_host_ptr}.store(
        encode_fence(0u), std::memory_order_release);
    LUISA_VERBOSE_WITH_LOCATION("Created HIPEvent (semaphore ptr = {}).",
                                _semaphore_device_ptr);
}

HIPEvent::~HIPEvent() noexcept {
    if (_semaphore_device_ptr) {
        // hipFree synchronizes pending work, including signal callbacks, before
        // releasing allocations created by hipExtMallocWithFlags.
        LUISA_CHECK_HIP(hipFree(_semaphore_device_ptr));
    }
}

void HIPEvent::signal(hipStream_t stream, uint64_t value) noexcept {
    // A plain stream write can regress a timeline when signals on independent
    // streams complete out of order. Apply an atomic maximum from a host
    // callback instead; the callback remains ordered after preceding work in
    // its stream and blocks following work until the update is visible.
    auto update = HIPEventSignalUpdate::create(_semaphore_host_ptr, value);
    auto result = hipLaunchHostFunc(stream, luisa_hip_event_signal_callback, update);
    if (result != hipSuccess) { update->recycle(); }
    LUISA_CHECK_HIP(result);
}

void HIPEvent::wait(hipStream_t stream, uint64_t value) noexcept {
    LUISA_CHECK_HIP(hipStreamWaitValue64(
        stream, _semaphore_device_ptr,
        static_cast<uint64_t>(encode_fence(value)), hipStreamWaitValueGte));
}

void HIPEvent::synchronize(uint64_t value) const noexcept {
    constexpr auto max_wait_iterations_before_yield = 1024u;
    auto wait_iterations = 0u;
    while (!has_signaled(value)) {
        if (++wait_iterations >= max_wait_iterations_before_yield) {
            wait_iterations = 0u;
            std::this_thread::yield();
        }
    }
}

bool HIPEvent::has_signaled(uint64_t value) const noexcept {
    auto encoded = std::atomic_ref<int64_t>{*_semaphore_host_ptr}.load(
        std::memory_order_acquire);
    return decode_fence(encoded) >= value;
}

}// namespace luisa::compute::hip
