#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// A proven upper bound on the number of auxiliary work items emitted by one
/// invocation of a named main-coroutine continuation. The scheduler uses
/// these bounds for admission control; they are semantic contracts rather
/// than measured estimates.
struct WavefrontCoroAuxiliaryProducer {
    luisa::string continuation;
    uint max_emitted_per_invocation{0u};
};

/// Returns whether a queue with occupancy `queued` can run every invocation
/// of a producer without exceeding `capacity`, assuming the producer obeys
/// its per-invocation emission bound. Both factors of the product are 32-bit,
/// so the 64-bit multiplication is exact and cannot overflow.
[[nodiscard]] constexpr bool wavefront_auxiliary_queue_can_admit(
    uint capacity, uint queued, uint producer_invocations,
    uint max_emitted_per_invocation) noexcept {
    return queued <= capacity &&
           static_cast<uint64_t>(producer_invocations) *
                   max_emitted_per_invocation <=
               static_cast<uint64_t>(capacity - queued);
}

/// A scheduler-visible side work queue fed by coroutine continuations.
///
/// The producer-side storage and publication operation are deliberately left
/// typed and owned by the client. This interface only exposes the host-side
/// queue protocol needed to co-schedule that work with coroutine
/// continuations. Implementations must obey the following temporal contract:
///
///  1. reset() makes the device and host counts zero before producers run;
///  2. enqueue_count_readback() appends a device-to-host count copy;
///  3. host_count() is read only after the stream has synchronized that copy;
///  4. host_count() is the TOTAL number of live items sharing this capacity;
///     it equals the sum of stage_host_count() over all stages;
///  5. dispatch_stage() consumes exactly the selected stage's observed count.
///     An item may terminate or move to another stage (including itself), but
///     may not duplicate itself. All transitions and count changes precede
///     the next readback on the same stream;
///  6. prepare_for_producer() may reorder storage, without changing live work,
///     to make the admitted number of free slots available to the producer.
///
/// Stage indices are stable for the lifetime of the registration. Each stage
/// competes separately with main continuations by cardinality. If a main
/// producer is blocked by total occupancy, admission_stage() supplies a
/// non-empty stage to advance toward releasing capacity. The default chooses
/// the first non-empty stage, so registration order expresses drain priority.
/// Stage transitions need no additional admission: they retain the same slot.
///
/// Every producer must enforce its declared emission bound independently of
/// scheduling. Given the invariant q <= C, admission requires n * b <= C - q;
/// therefore the next occupancy q' <= q + n * b <= C. The scheduler checks
/// this predicate before every producer dispatch and validates observed queue
/// counts after every synchronization.
///
/// The default stage methods preserve the original single-stage protocol:
/// dispatch() consumes all items and leaves device and host counts zero.
/// Multi-stage implementations override dispatch_stage(); their dispatch()
/// is not called by the scheduler. Storage and payload types remain client
/// owned and independent of any renderer-specific state.
template<typename... Args>
class WavefrontCoroAuxiliaryWork {

public:
    virtual ~WavefrontCoroAuxiliaryWork() noexcept = default;

    [[nodiscard]] virtual luisa::string_view name() const noexcept = 0;
    [[nodiscard]] virtual uint capacity() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const WavefrontCoroAuxiliaryProducer>
    producers() const noexcept = 0;

    virtual void reset(Stream &stream) noexcept = 0;
    virtual void enqueue_count_readback(Stream &stream) noexcept = 0;
    [[nodiscard]] virtual uint host_count() const noexcept = 0;
    [[nodiscard]] virtual uint stage_count() const noexcept { return 1u; }
    [[nodiscard]] virtual luisa::string_view stage_name(uint stage) const noexcept {
        LUISA_ASSERT(stage == 0u, "Invalid single-stage auxiliary index {}.", stage);
        return name();
    }
    [[nodiscard]] virtual uint stage_host_count(uint stage) const noexcept {
        LUISA_ASSERT(stage == 0u, "Invalid single-stage auxiliary index {}.", stage);
        return host_count();
    }
    [[nodiscard]] virtual uint admission_stage() const noexcept {
        for (auto i = 0u; i < stage_count(); ++i) {
            if (stage_host_count(i) != 0u) { return i; }
        }
        return stage_count();
    }
    virtual void prepare_for_producer(Stream &, uint) noexcept {}
    virtual void dispatch_stage(
        uint stage, Stream &stream,
        luisa::compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        LUISA_ASSERT(stage == 0u, "Invalid single-stage auxiliary index {}.", stage);
        dispatch(stream, args...);
    }
    virtual void dispatch(
        Stream &stream,
        luisa::compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept = 0;
};

}// namespace luisa::compute::coro
