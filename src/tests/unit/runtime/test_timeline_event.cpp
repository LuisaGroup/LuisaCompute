// Test timeline-event ordering and the full unsigned fence-value domain.
// This test covers:
// - Initial completion state
// - GPU stream waits on host-visible timeline signals
// - Values crossing the signed 64-bit boundary and UINT64_MAX
// - Monotonic completion when independent streams finish signals out of order

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

void test_initial_state(Device &device) {
    auto event = device.create_timeline_event();
    expect(event.is_completed(0u))
        << "A fresh timeline event must have completed fence zero.";
    expect(!event.is_completed(1u))
        << "A fresh timeline event must not report fence one as completed.";
    expect(!event.is_completed(std::numeric_limits<uint64_t>::max()))
        << "A fresh timeline event must not report UINT64_MAX as completed.";
}

void test_unsigned_fence_domain(Device &device) {
    constexpr std::array values{
        uint64_t{1u},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        uint64_t{1u} << 63u,
        std::numeric_limits<uint64_t>::max()};

    auto event = device.create_timeline_event();
    auto stream = device.create_stream();
    for (auto value : values) {
        stream << event.signal(value) << synchronize();
        expect(event.is_completed(value))
            << luisa::format("Timeline event did not complete fence {}.", value);
    }
}

void test_stream_wait_at_uint64_max(Device &device) {
    constexpr uint32_t expected = 0x4c435445u;
    uint32_t actual = 0u;

    auto buffer = device.create_buffer<uint32_t>(1u);
    auto event = device.create_timeline_event();
    auto producer = device.create_stream();
    auto consumer = device.create_stream();

    producer << buffer.copy_from(luisa::span{&expected, 1u})
             << event.signal(std::numeric_limits<uint64_t>::max());
    consumer << event.wait(std::numeric_limits<uint64_t>::max())
             << buffer.copy_to(luisa::span{&actual, 1u})
             << synchronize();
    producer << synchronize();

    expect(actual == expected)
        << "A stream wait at UINT64_MAX must order preceding producer work.";
    expect(event.is_completed(std::numeric_limits<uint64_t>::max()))
        << "UINT64_MAX must remain observable as a completed fence.";
}

void test_host_callbacks_precede_event_completion(Device &device) {
    if (device.backend_name() != "metal" &&
        device.backend_name() != "metal4") { return; }

    auto event = device.create_timeline_event();
    auto stream = device.create_stream();
    std::atomic_bool callback_started{false};
    std::atomic_bool release_callback{false};
    std::atomic_bool callback_completed{false};
    stream << luisa::move_only_function<void()>{[&]() noexcept {
        callback_started.store(true, std::memory_order_release);
        while (!release_callback.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        callback_completed.store(true, std::memory_order_release);
    }}
           << event.signal(1u);
    while (!callback_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{50u});
    auto completed_before_callback = event.is_completed(1u);
    release_callback.store(true, std::memory_order_release);
    event.synchronize(1u);

    expect(!completed_before_callback)
        << "A Metal event must not report host completion while an earlier "
           "stream callback is still pending";
    expect(callback_completed.load(std::memory_order_acquire))
        << "Metal event synchronization must include earlier download and "
           "user callbacks";
}

void test_cross_stream_monotonicity(Device &device) {
    // A late lower-valued signal is the HIP race being tested here. Some
    // native external-timeline APIs reject that execution pattern outright.
    if (device.backend_name() != "hip") { return; }

    auto gate = device.create_timeline_event();
    auto event = device.create_timeline_event();
    auto fast = device.create_stream();
    auto relay = device.create_stream();
    auto slow = device.create_stream();

    // Every wait is submitted after its matching signal, while the dependency
    // chain still forces fence 2 to complete before the late fence-1 signal.
    // This makes the test valid for backends that reject future event waits.
    fast << event.signal(2u);
    relay << event.wait(2u)
          << gate.signal(1u);
    slow << gate.wait(1u)
         << event.signal(1u)
         << synchronize();
    relay << synchronize();
    fast << synchronize();

    expect(event.is_completed(2u))
        << "A late lower-valued signal must not regress timeline completion.";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_initial_state(dc->device);
    test_unsigned_fence_domain(dc->device);
    test_stream_wait_at_uint64_max(dc->device);
    test_host_callbacks_precede_event_completion(dc->device);
    test_cross_stream_monotonicity(dc->device);
}
