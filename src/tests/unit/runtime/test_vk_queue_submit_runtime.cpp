// Everything in this file is behind the Vulkan submit profiler's private
// compile-time macro LUISA_VULKAN_PROFILE_SUBMIT (default 0), because the
// assertions below read the profiler's `vk_profile_*` counters: with the macro
// off those keys answer the empty string, so the test would assert nothing.
// A default build therefore gets a `main` that only prints that requirement.
// The macro is a source-level opt-in: it is deliberately not an XMake/CMake
// option and not an environment variable.
//
// Runtime test for the capability-adaptive queue submission of the Vulkan backend
// (plan P2.1: vkQueueSubmit2 + VkSemaphoreSubmitInfo stage masks on a
// synchronization2 device, unchanged vkQueueSubmit on the Vulkan 1.2 floor).
// This test covers:
// - which entry point the device reports (vk_queue_submit_path) and that it agrees
//   with the Vulkan API floor the device was created with
// - one dispatch batch: exactly one native submission, and it is the reported
//   variant (submit2 on a sync2 device, legacy on a 1.2 device), with the
//   dispatched values still correct
// - several batches: one native submission each, all of the reported variant, and
//   `queue_submits == queue_submit2 + queue_submit_legacy`
// - a cross-stream event signal/wait: the timeline signal is one native submission
//   of the reported variant, the wait-only submission is *not* counted, and the
//   consumer stream sees the producer's writes (i.e. the wait really carried its
//   timeline value and stage mask)
// - repeated signal/wait rounds over the same timeline: values keep advancing, the
//   ordering keeps holding, and the counters keep matching the submission shape
// Run it twice to cover both native paths in one build:
//   <binary> vk                 -> the device's auto API version (submit2 here)
//   <binary> vk --min-api 1.2   -> forces the legacy vkQueueSubmit path
// The present path is not exercised: it needs a window/swapchain, which this
// environment does not have (see the report); the present submission *shape* is
// covered on the host by test_vk_queue_submit_plan.
// The gate of the instrument (same default as src/backends/vk/submit_profile.h).
#if !defined(LUISA_VULKAN_PROFILE_SUBMIT)
#define LUISA_VULKAN_PROFILE_SUBMIT 0
#endif

#if LUISA_VULKAN_PROFILE_SUBMIT

#include "ut/ut.hpp"
#include "test_device.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto kThreadsPerDispatch = 32u;

// Nothing latches the mode: it is compiled in, so the counters this test reads
// exist exactly when the backend library was built with the same macro.

[[nodiscard]] uint64_t counter(Device &device, const char *key) noexcept {
    auto text = device.query(key);
    return text.empty() ? 0u : std::strtoull(text.c_str(), nullptr, 10);
}

[[nodiscard]] bool reported_submit2(Device &device) noexcept {
    auto path = device.query("vk_queue_submit_path");
    LUISA_ASSERT(!path.empty(),
                 "the Vulkan backend must report the queue-submission path it uses.");
    return path == "submit2";
}

// Answers only the API-version floor, so the rest of the device is created exactly
// as the default configuration creates it. A Vulkan 1.2 floor is the backend's
// documented "Android-compatible" mode and is what makes the legacy submission
// path reachable on a 1.3-capable machine (plan P2.1).
class MinApiConfig final : public VulkanDeviceConfigExt {
    uint32_t _min_api_version;
public:
    explicit MinApiConfig(uint32_t min_api_version) noexcept
        : _min_api_version{min_api_version} {}
    [[nodiscard]] uint32_t min_api_version() const noexcept override {
        return _min_api_version;
    }
};

void check_variant_counters(Device &device, uint64_t expected_submissions,
                            const char *what) noexcept {
    auto submits = counter(device, "vk_profile_queue_submits");
    auto submit2 = counter(device, "vk_profile_queue_submit2");
    auto legacy = counter(device, "vk_profile_queue_submit_legacy");
    LUISA_INFO("{}: submits={} submit2={} legacy={}", what, submits, submit2, legacy);
    expect(submit2 + legacy == submits)
        << "every counted native submission must be exactly one variant";
    auto sync2 = reported_submit2(device);
    if (sync2) {
        expect(submit2 == expected_submissions && legacy == 0u)
            << what << ": a synchronization2 device must submit through "
                       "vkQueueSubmit2 only (plan P2.1)";
    } else {
        expect(legacy == expected_submissions && submit2 == 0u)
            << what << ": a Vulkan 1.2 device must keep the legacy vkQueueSubmit "
                       "path only (plan P2.1)";
    }
}

// Writes `index` into the slice of this dispatch, so a dropped or reordered
// submission shows up as a wrong value and not only as a wrong counter.
Kernel1D stamp_kernel = [](BufferFloat out, UInt index) noexcept {
    out.write(index * kThreadsPerDispatch + dispatch_id().x, cast<float>(index));
};

// Fills every element with `value`: a producer that is cheap enough to run many
// rounds while still being long enough to overlap with a consumer stream.
Kernel1D fill_kernel = [](BufferFloat out, Float value) noexcept {
    out.write(dispatch_id().x, value);
};

// Doubles whatever `input` holds: correct only if the producer's dispatches
// finished before this one ran.
Kernel1D double_kernel = [](BufferFloat input, BufferFloat out) noexcept {
    out.write(dispatch_id().x, input.read(dispatch_id().x) * 2.0f);
};

void check_stamps(const luisa::vector<float> &host, size_t dispatch_count,
                  const char *what) noexcept {
    auto ok = host.size() == dispatch_count * kThreadsPerDispatch;
    if (ok) {
        for (auto i = size_t{0u}; i < dispatch_count && ok; ++i) {
            for (auto t = size_t{0u}; t < kThreadsPerDispatch; ++t) {
                if (host[i * kThreadsPerDispatch + t] != static_cast<float>(i)) {
                    LUISA_WARNING("{}: dispatch {} thread {} produced {}",
                                  what, i, t,
                                  host[i * kThreadsPerDispatch + t]);
                    ok = false;
                    break;
                }
            }
        }
    }
    expect(ok) << what << ": every dispatched thread must have written its slice";
}

void reg_reported_path(Device &device, bool requested_vulkan_1_2) {
    "vk_queue_submit_path_is_reported"_test = [&device, requested_vulkan_1_2] {
        auto path = device.query("vk_queue_submit_path");
        LUISA_INFO("queue-submission path: {} (1.2 floor requested: {})",
                   path, requested_vulkan_1_2);
        expect(path == "submit2" || path == "legacy")
            << "the backend must report one of the two native submission paths";
        if (requested_vulkan_1_2) {
            expect(path == "legacy")
                << "a device created with a Vulkan 1.2 floor must use the legacy "
                   "vkQueueSubmit path (plan P2.1)";
        }
        // A device that is not 1.2-limited may still be 1.2-only hardware, so the
        // auto case is only required to be consistent with the counters below.
        expect(counter(device, "vk_profile_queue_submit2") +
                   counter(device, "vk_profile_queue_submit_legacy") ==
               counter(device, "vk_profile_queue_submits"));
    };
}

void reg_single_batch(Device &device) {
    "vk_queue_submit_variant_single_batch"_test = [&device] {
        auto shader = device.compile(stamp_kernel);
        Buffer<float> buffer = device.create_buffer<float>(kThreadsPerDispatch);
        Stream stream = device.create_stream();
        luisa::vector<float> host(kThreadsPerDispatch, -1.0f);
        auto before_submits = counter(device, "vk_profile_queue_submits");
        auto before_submit2 = counter(device, "vk_profile_queue_submit2");
        auto before_legacy = counter(device, "vk_profile_queue_submit_legacy");
        {
            CommandList list;
            list << shader(buffer, 0u).dispatch(kThreadsPerDispatch);
            stream << list.commit();
        }
        auto after_submits = counter(device, "vk_profile_queue_submits");
        auto after_submit2 = counter(device, "vk_profile_queue_submit2");
        auto after_legacy = counter(device, "vk_profile_queue_submit_legacy");
        auto delta = after_submits - before_submits;
        auto delta_submit2 = after_submit2 - before_submit2;
        auto delta_legacy = after_legacy - before_legacy;
        LUISA_INFO("one batch: submits {} -> {} (submit2 +{}, legacy +{})",
                   before_submits, after_submits, delta_submit2, delta_legacy);
        expect(delta == 1u) << "one command batch is one native submission (F6)";
        expect(delta_submit2 + delta_legacy == delta);
        if (reported_submit2(device)) {
            expect(delta_submit2 == 1u && delta_legacy == 0u)
                << "a one-batch dispatch on a sync2 device must report "
                   "queue_submit2=1 and queue_submit_legacy=0";
        } else {
            expect(delta_legacy == 1u && delta_submit2 == 0u)
                << "a one-batch dispatch on a Vulkan 1.2 device must report the "
                   "reverse: queue_submit_legacy=1 and queue_submit2=0";
        }
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_stamps(host, 1u, "one batch");
    };
}

void reg_multi_batch(Device &device) {
    "vk_queue_submit_variant_multi_batch"_test = [&device] {
        constexpr auto batch_count = size_t{6u};
        auto shader = device.compile(stamp_kernel);
        Buffer<float> buffer =
            device.create_buffer<float>(batch_count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        luisa::vector<float> host(batch_count * kThreadsPerDispatch, -1.0f);
        auto before_submits = counter(device, "vk_profile_queue_submits");
        auto before_submit2 = counter(device, "vk_profile_queue_submit2");
        auto before_legacy = counter(device, "vk_profile_queue_submit_legacy");
        for (auto i = size_t{0u}; i < batch_count; ++i) {
            CommandList list;
            list << shader(buffer, static_cast<uint32_t>(i))
                        .dispatch(kThreadsPerDispatch);
            stream << list.commit();
        }
        auto delta_submits = counter(device, "vk_profile_queue_submits") - before_submits;
        auto delta_submit2 = counter(device, "vk_profile_queue_submit2") - before_submit2;
        auto delta_legacy = counter(device, "vk_profile_queue_submit_legacy") - before_legacy;
        LUISA_INFO("{} batches: submits +{}, submit2 +{}, legacy +{}",
                   batch_count, delta_submits, delta_submit2, delta_legacy);
        expect(delta_submits == batch_count)
            << "one native submission per command batch";
        expect(delta_submit2 + delta_legacy == delta_submits);
        if (reported_submit2(device)) {
            expect(delta_submit2 == batch_count && delta_legacy == 0u)
                << "every batch of a sync2 device submits through vkQueueSubmit2";
        } else {
            expect(delta_legacy == batch_count && delta_submit2 == 0u)
                << "every batch of a Vulkan 1.2 device submits through "
                   "vkQueueSubmit";
        }
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_stamps(host, batch_count, "multi-batch");
    };
}

void reg_cross_stream_event(Device &device) {
    "vk_queue_submit_cross_stream_event"_test = [&device] {
        // Producer stream writes, signals a timeline event; consumer stream waits on
        // that value and reads what the producer wrote. The wait submission carries
        // the event's timeline value and its ALL_COMMANDS stage mask, so the read
        // is ordered after the write.
        constexpr auto count = static_cast<size_t>(kThreadsPerDispatch) * 4u;
        auto shader = device.compile(stamp_kernel);
        auto doubler = device.compile(double_kernel);
        Buffer<float> produced = device.create_buffer<float>(count);
        Buffer<float> consumed = device.create_buffer<float>(count);
        Stream producer = device.create_stream();
        Stream consumer = device.create_stream();
        Event event = device.create_event();
        // Four independent slices on the producer, each a separate batch, all
        // followed by one event signal.
        constexpr auto slice_count = size_t{4u};
        for (auto i = size_t{0u}; i < slice_count; ++i) {
            CommandList list;
            list << shader(produced, static_cast<uint32_t>(i))
                        .dispatch(kThreadsPerDispatch);
            producer << list.commit();
        }
        auto before_submits = counter(device, "vk_profile_queue_submits");
        auto before_submit2 = counter(device, "vk_profile_queue_submit2");
        auto before_legacy = counter(device, "vk_profile_queue_submit_legacy");
        producer << event.signal();
        auto after_signal_submits = counter(device, "vk_profile_queue_submits");
        auto after_signal_submit2 = counter(device, "vk_profile_queue_submit2");
        auto after_signal_legacy = counter(device, "vk_profile_queue_submit_legacy");
        auto signal_delta = after_signal_submits - before_submits;
        auto signal_submit2 = after_signal_submit2 - before_submit2;
        auto signal_legacy = after_signal_legacy - before_legacy;
        LUISA_INFO("event signal: submits +{} (submit2 +{}, legacy +{})",
                   signal_delta, signal_submit2, signal_legacy);
        expect(signal_delta == 1u)
            << "an event signal on an empty command list is one native submission";
        if (reported_submit2(device)) {
            expect(signal_submit2 == 1u && signal_legacy == 0u)
                << "the event signal must use the same native path as a batch";
        } else {
            expect(signal_legacy == 1u && signal_submit2 == 0u)
                << "the event signal must keep the legacy path on a 1.2 device";
        }
        // The wait-only submission is deliberately not counted as a `queue_submits`
        // entry: that counter describes the dispatch path (F6).
        auto before_wait_submits = counter(device, "vk_profile_queue_submits");
        luisa::vector<float> host(count, -1.0f);
        consumer << event.wait()
                 << doubler(produced, consumed).dispatch(count)
                 << consumed.copy_to(luisa::span{host})
                 << synchronize();
        auto wait_delta = counter(device, "vk_profile_queue_submits") - before_wait_submits;
        LUISA_INFO("event wait + consumer batch: submits +{}", wait_delta);
        expect(wait_delta == 1u)
            << "the consumer contributes exactly its own batch; the wait-only "
               "submission is not counted";
        auto ok = true;
        for (auto i = size_t{0u}; i < count; ++i) {
            auto expected = 2.0f * static_cast<float>(i / kThreadsPerDispatch);
            if (host[i] != expected) {
                if (ok) {
                    LUISA_WARNING("cross-stream index {}: got {} expected {}",
                                  i, host[i], expected);
                }
                ok = false;
            }
        }
        expect(ok) << "the consumer stream must observe the producer's writes "
                      "through the event wait";
        producer << synchronize();
    };
}

void reg_repeated_event_rounds(Device &device) {
    "vk_queue_submit_event_rounds"_test = [&device] {
        // Repeated signal/wait rounds on one timeline: the values keep advancing and
        // the ordering keeps holding. A stage mask that is wrong (or a dropped
        // timeline value) turns into a read of a stale value here.
        constexpr auto count = static_cast<size_t>(kThreadsPerDispatch);
        constexpr auto rounds = size_t{8u};
        auto shader = device.compile(fill_kernel);
        auto doubler = device.compile(double_kernel);
        Buffer<float> produced = device.create_buffer<float>(count);
        Buffer<float> consumed = device.create_buffer<float>(count);
        Stream producer = device.create_stream();
        Stream consumer = device.create_stream();
        Event event = device.create_event();
        luisa::vector<float> host(count, -1.0f);
        auto correct = true;
        for (auto round = size_t{0u}; round < rounds; ++round) {
            CommandList produce_list;
            produce_list << shader(produced, static_cast<float>(round))
                                .dispatch(kThreadsPerDispatch);
            producer << produce_list.commit() << event.signal();
            consumer << event.wait()
                     << doubler(produced, consumed).dispatch(count)
                     << consumed.copy_to(luisa::span{host});
            // Host wait each round: the point is the GPU-side ordering of one round,
            // not the pipelining between rounds.
            consumer << synchronize();
            for (auto i = size_t{0u}; i < count; ++i) {
                auto expected = 2.0f * static_cast<float>(round);
                if (host[i] != expected) {
                    if (correct) {
                        LUISA_WARNING("round {} index {}: got {} expected {}",
                                      round, i, host[i], expected);
                    }
                    correct = false;
                    break;
                }
            }
        }
        expect(correct) << "every round must read the values its producer wrote";
        expect(event.last_fence() == rounds)
            << "one timeline value per signal round";
        producer << synchronize();
        // Cumulative: every counted submission of this process is one variant.
        check_variant_counters(device, counter(device, "vk_profile_queue_submits"),
                               "event rounds (cumulative)");
    };
}

}// namespace

int main(int argc, char *argv[]) {
    // `--min-api 1.2` forces the legacy submission path in the same build: it is a
    // device-creation input, so it is parsed here and removed from argv.
    auto requested_1_2 = false;
    {
        auto kept = 0;
        for (auto i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "--min-api") == 0 && i + 1 < argc) {
                auto value = std::string_view{argv[i + 1]};
                requested_1_2 = value == "1.2" || value == "12";
                ++i;// skip the value as well
                continue;
            }
            argv[kept++] = argv[i];
        }
        argv[kept] = nullptr;
        argc = kept;
    }
    DeviceConfig device_config;
    if (requested_1_2) {
        device_config.extension = luisa::make_unique<MinApiConfig>(VK_API_VERSION_1_2);
    }
    auto dc = luisa::test::create_device_from_ut(
        argc, argv, requested_1_2 ? &device_config : nullptr);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "vk") {
        LUISA_INFO("Skipping: the queue-submission paths are Vulkan-only.");
        return 0;
    }
    reg_reported_path(device, requested_1_2);
    reg_single_batch(device);
    reg_multi_batch(device);
    reg_cross_stream_event(device);
    reg_repeated_event_rounds(device);
    return 0;
}


#else// !LUISA_VULKAN_PROFILE_SUBMIT

// The submission counters this test reads are part of the compile-time submit
// profiler (see src/backends/vk/submit_profile.h), so a default build has
// nothing to check and - more importantly - must not pretend that it did: the
// binary prints what it needs and exits successfully.
#include <cstdio>

int main() {
    std::fprintf(stderr,
                 "test_vk_queue_submit_runtime: nothing was tested. This binary "
                 "needs the compile-time macro LUISA_VULKAN_PROFILE_SUBMIT=1 in the "
                 "backend library *and* in this test; it is a private source-level "
                 "opt-in, deliberately not an XMake/CMake option and not an "
                 "environment variable.\n");
    return 0;
}

#endif// LUISA_VULKAN_PROFILE_SUBMIT
