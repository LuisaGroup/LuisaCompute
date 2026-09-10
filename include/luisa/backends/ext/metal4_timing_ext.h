#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

struct Metal4TimingCapabilities {
    bool command_buffer_feedback{true};
    // Creation/format capability only. Actual nonzero, ordered timestamp
    // results must still be validated on the executing device.
    bool timestamp_heap{false};
    bool legacy_dispatch_sampling{false};
    bool legacy_stage_sampling{false};
    uint64_t timestamp_frequency_hz{0u};
    luisa::string error;
};

struct Metal4CommandBufferTiming {
    uint64_t ordinal{0u};
    uint64_t dispatch_count{0u};
    bool contains_non_dispatch_work{false};
    double gpu_begin_seconds{0.0};
    double gpu_end_seconds{0.0};
    // steady_clock epoch, NOT the GPU/CommitFeedback clock. Only compare
    // these host values with each other; no cross-clock subtraction is valid.
    uint64_t host_commit_begin_nanoseconds{0u};
    uint64_t host_commit_return_nanoseconds{0u};
    uint64_t host_feedback_begin_nanoseconds{0u};
    uint64_t host_callbacks_end_nanoseconds{0u};
    // Observation immediately before publishing CPU completion, after native
    // resource recycling. The recording itself adds a little host overhead.
    uint64_t host_completion_publish_nanoseconds{0u};
    bool valid{false};
};

struct Metal4DispatchTiming {
    uint64_t ordinal{0u};
    uint64_t command_buffer_ordinal{0u};
    uint64_t shader_checksum{0u};
    uint3 dispatch_size{};
    uint3 block_size{};
    uint64_t begin_ticks{0u};
    uint64_t end_ticks{0u};
    double elapsed_nanoseconds{0.0};
    bool valid{false};
};

struct Metal4TimingSample {
    uint64_t sample_id{0u};
    uint64_t timestamp_frequency_hz{0u};
    bool dispatch_timestamps_enabled{false};
    bool overflow{false};
    luisa::vector<Metal4CommandBufferTiming> command_buffers;
    luisa::vector<Metal4DispatchTiming> dispatches;
    luisa::string error;
};

// Opt-in diagnostic instrumentation, borrowed from Device::extension().
// Caller must serialize begin/dispatch/end on one ordinary compute/graphics
// stream belonging to this device. Different streams have independent samples.
// begin drains preexisting work; end disarms before draining the sampled work.
// Neither boundary's own synchronization submission is recorded. Synchronize
// calls inside the sample ARE recorded, including their empty command buffers.
//
// Dispatch timestamps use MTL4 precise counters around each actual direct
// dispatch. These are instrumented dispatch intervals, NOT zero-overhead kernel
// times: timestamps may split encoders and perturb scheduling. Commit-feedback
// intervals are separate command-buffer wall intervals, never kernel times.
// Indirect command ranges fail the sample rather than masquerade as one kernel.
// A false capture_dispatch_timestamps records feedback/dispatch metadata only,
// without counter commands, for an instrumentation control run.
// max_dispatches must be in [1, 1048576]; overflow fails the returned sample.
class Metal4TimingExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "Metal4TimingExt";
    [[nodiscard]] virtual Metal4TimingCapabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual bool begin_sample(uint64_t stream_handle, uint64_t sample_id,
                                            uint32_t max_dispatches = 4096u,
                                            bool capture_dispatch_timestamps = true) const noexcept = 0;
    // Returns an error if no sample is active, any record overflowed/failed,
    // or a requested timestamp is unavailable. Raw evidence is retained.
    [[nodiscard]] virtual Metal4TimingSample end_sample(uint64_t stream_handle) const noexcept = 0;
};

}// namespace luisa::compute
