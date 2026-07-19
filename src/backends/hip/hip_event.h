//
// Created by mike on 1/10/26.
//

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPDevice;

// Timeline-semaphore-style event backed by coherent HSA signal memory.
class HIPEvent {

private:
    hipDeviceptr_t _semaphore_device_ptr{};
    int64_t *_semaphore_host_ptr{};

public:
    explicit HIPEvent(HIPDevice *device) noexcept;
    ~HIPEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _semaphore_device_ptr; }
    void signal(hipStream_t stream, uint64_t value) noexcept;
    void wait(hipStream_t stream, uint64_t value) noexcept;
    void synchronize(uint64_t value) const noexcept;
    [[nodiscard]] bool has_signaled(uint64_t value) const noexcept;
};

}// namespace luisa::compute::hip
