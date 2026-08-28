// Test timeline-event ordering and the unsigned fence-value domain.
// This test covers:
// - Initial completion state
// - GPU stream waits on host-visible timeline signals
// - Values crossing the signed 64-bit boundary and the platform maximum
// - Monotonic completion when independent streams finish signals out of order
//
// Vulkan drivers bound the pending timeline-semaphore window
// (maxTimelineSemaphoreValueDifference), so the full unsigned domain is not
// representable everywhere (e.g. many desktop drivers report 2^31-1). The
// huge-value cases below only run when the device reports a matching
// capability; otherwise the test exercises the platform's own maximum instead.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

// The largest timeline value the platform can reach with a single signal from
// the current counter. Backends without such a limit (and unknown queries)
// report the full unsigned domain.
[[nodiscard]] uint64_t timeline_max_reachable_value(const Device &device) noexcept {
    if (device.backend_name() == "vk") {
        auto text = device.query("timeline_semaphore_max_value_difference");
        uint64_t value{};
        auto [end, ec] = std::from_chars(
            text.data(), text.data() + text.size(), value);
        if (ec == std::errc{} && end == text.data() + text.size()) {
            return value;
        }
    }
    return std::numeric_limits<uint64_t>::max();
}

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
    constexpr std::array full_domain_values{
        uint64_t{1u},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        uint64_t{1u} << 63u,
        std::numeric_limits<uint64_t>::max()};

    // Keep every full-domain value the platform can represent, then always
    // exercise the platform's own maximum one-shot jump as a boundary.
    auto max_value = timeline_max_reachable_value(device);
    luisa::vector<uint64_t> values;
    values.reserve(full_domain_values.size() + 1u);
    for (auto value : full_domain_values) {
        if (value <= max_value) { values.emplace_back(value); }
    }
    if (max_value > 1u && (values.empty() || values.back() != max_value)) {
        values.emplace_back(max_value);
    }

    auto event = device.create_timeline_event();
    auto stream = device.create_stream();
    for (auto value : values) {
        stream << event.signal(value) << synchronize();
        // Stream synchronization only waits for the stream's own fence. The
        // event's host-completion watermark is published by the backend's
        // async executor, so wait on it explicitly before querying completion.
        event.synchronize(value);
        expect(event.is_completed(value))
            << luisa::format("Timeline event did not complete fence {}.", value);
    }
}

void test_stream_wait_at_max_value(Device &device) {
    constexpr uint32_t expected = 0x4c435445u;
    uint32_t actual = 0u;
    auto max_value = timeline_max_reachable_value(device);

    auto buffer = device.create_buffer<uint32_t>(1u);
    auto event = device.create_timeline_event();
    auto producer = device.create_stream();
    auto consumer = device.create_stream();

    producer << buffer.copy_from(luisa::span{&expected, 1u})
             << event.signal(max_value);
    consumer << event.wait(max_value)
             << buffer.copy_to(luisa::span{&actual, 1u})
             << synchronize();
    producer << synchronize();
    // Wait for the backend's async executor to publish host completion of the
    // maximum signal before querying it.
    event.synchronize(max_value);

    expect(actual == expected)
        << "A stream wait at the maximum timeline value must order preceding "
           "producer work.";
    expect(event.is_completed(max_value))
        << "The maximum timeline value must remain observable as a completed "
           "fence.";
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
    test_stream_wait_at_max_value(dc->device);
    test_host_callbacks_precede_event_completion(dc->device);
    test_cross_stream_monotonicity(dc->device);
}
