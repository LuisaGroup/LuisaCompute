#pragma once

#include <luisa/core/basic_types.h>
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
///  4. dispatch() consumes exactly host_count() items and leaves both counts
///     zero, with the reset ordered after the consumer on the same stream.
///
/// Every producer must enforce its declared emission bound independently of
/// scheduling. Given the invariant q <= C, admission requires n * b <= C - q;
/// therefore the next occupancy q' <= q + n * b <= C. The scheduler checks
/// this predicate before every producer dispatch and validates observed queue
/// counts after every synchronization.
///
/// This is sufficient for single-stage side tasks such as visibility work and
/// is independent of any renderer-specific payload.
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
    virtual void dispatch(
        Stream &stream,
        luisa::compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept = 0;
};

}// namespace luisa::compute::coro
