#pragma once

#include <condition_variable>
#include <mutex>
#include <luisa/backends/ext/metal4_timing_ext.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalTimingSession final {
private:
    std::mutex _mutex;
    std::condition_variable _feedback_cv;
    uint64_t _pending_feedback{0u};
    uint32_t _max_dispatches;
    NS::SharedPtr<MTL4::CounterHeap> _heap;
    Metal4TimingSample _sample;

public:
    MetalTimingSession(uint64_t sample_id, uint32_t max_dispatches,
                       uint64_t frequency, NS::SharedPtr<MTL4::CounterHeap> heap) noexcept;
    [[nodiscard]] uint64_t begin_command_buffer() noexcept;
    [[nodiscard]] uint32_t begin_dispatch(MTL4::ComputeCommandEncoder *encoder,
                                          uint64_t command_buffer_ordinal, uint64_t shader_checksum,
                                          uint3 dispatch_size, uint3 block_size) noexcept;
    void end_dispatch(MTL4::ComputeCommandEncoder *encoder, uint32_t ordinal) noexcept;
    void unsupported_indirect_dispatch() noexcept;
    void will_commit(uint64_t ordinal, uint64_t dispatch_count, bool non_dispatch_work) noexcept;
    void did_commit(uint64_t ordinal) noexcept;
    void feedback(uint64_t ordinal, MTL4::CommitFeedback *feedback) noexcept;
    void callbacks_finished(uint64_t ordinal) noexcept;
    void completing(uint64_t ordinal) noexcept;
    [[nodiscard]] Metal4TimingSample finish() noexcept;
};

struct MetalTimingSubmission {
    luisa::shared_ptr<MetalTimingSession> session;
    uint64_t ordinal{0u};
    uint64_t dispatch_count{0u};
    bool contains_non_dispatch_work{false};
};

class MetalTimingExt final : public Metal4TimingExt {
private:
    MTL::Device *_device;

public:
    explicit MetalTimingExt(MTL::Device *device) noexcept : _device{device} {}
    [[nodiscard]] Metal4TimingCapabilities capabilities() const noexcept override;
    [[nodiscard]] bool begin_sample(uint64_t stream_handle, uint64_t sample_id,
                                    uint32_t max_dispatches, bool capture_dispatch_timestamps) const noexcept override;
    [[nodiscard]] Metal4TimingSample end_sample(uint64_t stream_handle) const noexcept override;
};

}// namespace luisa::compute::metal
