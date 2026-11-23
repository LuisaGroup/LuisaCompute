// Device-level coverage of replayable recording (plan P4, narrow v1), through the
// public `VulkanReplayExt` only: the test never names a `lc::vk::` type and never
// includes a backend header.
//
// What is covered:
// - record() does not submit: a recorded batch leaves the buffers untouched until
//   the first replay(), and `status().submits` stays 0
// - record+replay produces bit-identical results to an ordinary dispatch of the
//   same list (two dispatches plus a device-to-device copy)
// - the one dynamic channel: changed device-buffer contents are observed by the
//   next replay, while the recorded uniform stays what it was
// - update() refuses with UNSUPPORTED_UPDATE and the graph keeps replaying the
//   recorded uniform afterwards
// - several replays may be in flight on one stream; the last one's completion is
//   what `synchronize()` waits for, and `statistics`/`status` count them
// - an ordinary `Event` still orders a replay against another stream
// - a shader destroyed after record() does not invalidate the graph (the
//   recording retains its pipeline and its interned layouts by id)
// - destroying a watched buffer, or the recording stream, turns a later replay()
//   into the matching typed error instead of touching freed memory
// - destroy() refuses BUSY while the latest replay is pending, and the
//   synchronize-then-retry sequence succeeds
// - the whole rejection matrix (upload, download, texture copy, texture
//   argument, indirect, printer, bindless, acceleration build, custom command,
//   callbacks, presents, destructor callbacks) returns the exact typed reason and
//   consumes no native resource
// - handle lifecycle: unknown handle, double destroy, replay after destroy
// - a decode-shaped repeated-dispatch graph (32 dispatches) is compared with an
//   ordinary dispatch of the same list on host submission time. The counters are
//   OFF in this binary on purpose; only the measured host time is reported and no
//   GPU-side claim is made.
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include <luisa/backends/ext/vk_replay_ext.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Plan P4 (parent review): `--min-api 1.2` forces the legacy vkQueueSubmit path in
// this same binary, so the whole record/replay lifecycle is also exercised on the
// Vulkan 1.2 floor. The backend latches its submission entry point once at device
// creation, so one process can only cover one of the two paths.
bool g_min_api_1_2{false};

// Answers only the API-version floor, so the rest of the device is created exactly
// as the default configuration creates it.
class ReplayDeviceConfig final : public VulkanDeviceConfigExt {
public:
    [[nodiscard]] uint32_t min_api_version() const noexcept override {
        return g_min_api_1_2 ? VK_API_VERSION_1_2 : 0u;
    }
};

// Removes `--min-api <version>` from argv in place (before Boost.UT parses it) and
// records what was asked for.
void take_min_api_option(int &argc, char **argv) noexcept {
    auto kept = 0;
    for (auto i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--min-api") == 0 && i + 1 < argc) {
            auto value = std::string_view{argv[i + 1]};
            g_min_api_1_2 = value == "1.2" || value == "12";
            ++i;// skip the value as well
            continue;
        }
        argv[kept++] = argv[i];
    }
    argv[kept] = nullptr;
    argc = kept;
}

constexpr auto kBlockSize = size_t{64u};

// d[i] = a[i] + b[i] + s. Three BUFFER arguments and one UNIFORM: exactly the
// argument contract of v1, and the uniform is what update() refuses to change.
Kernel1D add_kernel = [](BufferFloat a, BufferFloat b, BufferFloat d, Float s) noexcept {
    auto i = dispatch_id().x;
    d.write(i, a.read(i) + b.read(i) + s);
};

// One dispatch writes its own index into its own 64-value slice, so a batch of N
// dispatches has to leave N * 64 identifiable values behind (the decode-shaped
// case uses this shape).
Kernel1D write_dispatch_index = [](BufferFloat out, UInt index) noexcept {
    out.write(index * static_cast<uint32_t>(kBlockSize) + dispatch_id().x,
              cast<float>(index));
};

// The same body with a compile-time block size: only such a shader carries the
// indirect-dispatch metadata binding the backend requires for
// `dispatch(indirect_buffer, ...)`, which the ordinary-path regression exercises.
Kernel1D write_index_fixed_block = [](BufferFloat out) noexcept {
    set_block_size(64u, 1u, 1u);
    out.write(kernel_id() * 64u + dispatch_id().x, cast<float>(kernel_id()));
};

// A GPU-bound graph, so "the latest replay is still pending" is a wide, honest
// window instead of a race with the retirement thread.
Kernel1D spin_kernel = [](BufferFloat out, UInt iterations) noexcept {
    auto i = dispatch_id().x;
    Float acc = cast<float>(i);
    UInt k = 0u;
    $while (k < iterations) {
        acc = sqrt(acc + 1.0f);
        k += 1u;
    };
    out.write(i, acc);
};

// A texture-reading dispatch: rejected in v1 (ARG_TEXTURE).
Kernel2D read_texture = [](ImageFloat image, BufferFloat out) noexcept {
    auto coord = dispatch_id().xy();
    out.write(coord.x + coord.y * 8u, image.read(coord).x);
};

// A print dispatch: rejected in v1 (DISPATCH_PRINTER).
Kernel1D print_kernel = [](BufferFloat out) noexcept {
    out.write(dispatch_id().x, 1.0f);
    device_log("replay printer {}", dispatch_id().x);
};

[[nodiscard]] VulkanReplayExt *replay_ext(Device &device) noexcept {
    auto *ext = device.extension<VulkanReplayExt>();
    expect(ext != nullptr)
        << "the Vulkan backend must expose VulkanReplayExt";
    return ext;
}

[[nodiscard]] const char *reason_name(VulkanReplayRejectReason reason) noexcept {
    // The name is a string literal in the public header, so its address outlives
    // any expectation message.
    auto name = vulkan_replay_reject_reason_name(reason);
    return name.data();
}

[[nodiscard]] const char *status_name(VulkanReplayStatus status) noexcept {
    return vulkan_replay_status_name(status).data();
}

// A no-op custom command: v1 rejects every CustomCommand, so it is refused before
// anything ever executes it. Executing it would be a no-op anyway.
class NoopCustomCmd final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;

public:
    explicit NoopCustomCmd() noexcept = default;
    [[nodiscard]] luisa::span<ResourceUsage> get_resource_usages() noexcept override {
        return _usages;
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }
    void execute(VkPhysicalDevice, VkDevice, VkQueue,
                 VkCommandBuffer, VkDescriptorPool) const noexcept override {}
};

void check_rejected(Device &device, VulkanReplayExt *ext, Stream &stream,
                    CommandList &&list, VulkanReplayRejectReason expected,
                    const char *what) noexcept {
    static_cast<void>(device);
    auto live_before = ext->live_replay_count();
    auto result = ext->record(stream.handle(), std::move(list));
    LUISA_INFO("reject case '{}': status={} reason={} handle={} message='{}'",
               what, status_name(result.status), reason_name(result.reason),
               result.handle, result.message);
    expect(result.status == VulkanReplayStatus::REJECTED)
        << what << ": an unsupported input must be refused with REJECTED";
    expect(result.reason == expected)
        << what << ": the refusal must carry the exact typed reason";
    expect(!result.message.empty())
        << what << ": a refusal always explains itself";
    expect(result.handle == 0u)
        << what << ": a refused record creates no handle";
    expect(ext->live_replay_count() == live_before)
        << what << ": a refusal must not consume a native resource";
}

// ---------------------------------------------------------------------------
// Which native submission entry point a replay uses
// ---------------------------------------------------------------------------
void reg_submit_path(Device &device) {
    "replay_submit_path"_test = [&device] {
        // The backend's diagnostic of the entry point it latched at creation (plan
        // P2.1). `Event::_signal` is the single funnel for both an ordinary batch and
        // a replay, so what this reports is also what a replay uses - and the replay
        // window of the counter tripwire asserts the same fact as counts
        // (`queue_submit2 == replay_submits` or `queue_submit_legacy ==
        // replay_submits`).
        auto path = device.query("vk_queue_submit_path");
        expect(!path.empty())
            << "the backend must report the queue-submission path it uses";
        expect(path == "submit2" || path == "legacy");
        LUISA_INFO("replay submission path: {} (forced 1.2 floor: {})",
                   path, g_min_api_1_2 ? "yes" : "no");
        if (g_min_api_1_2) {
            expect(path == "legacy")
                << "--min-api 1.2 must select the legacy vkQueueSubmit path, so the "
                   "replays in this run go through it";
        }
    };
}

// ---------------------------------------------------------------------------
// record() does not submit
// ---------------------------------------------------------------------------
void reg_record_does_not_submit(Device &device) {
    "replay_record_does_not_submit"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 256u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host_a(n, 1.0f);
        luisa::vector<float> host_b(n, 2.0f);
        luisa::vector<float> zeros(n, 0.0f);
        stream << a.copy_from(luisa::span{host_a})
               << b.copy_from(luisa::span{host_b})
               << d.copy_from(luisa::span{zeros})
               << synchronize();
        CommandList list;
        list << shader(a, b, d, 10.0f).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS)
            << "a direct compute dispatch with buffer/uniform arguments must record";
        expect(recorded.handle != 0u);
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::SUCCESS);
        expect(graph.recorded_dispatches == 1u);
        expect(graph.submits == 0u)
            << "record() must not submit anything";
        expect(graph.stream_handle == stream.handle());
        // The proof that nothing ran: the target buffer still holds its sentinel.
        luisa::vector<float> host_d(n, -1.0f);
        stream << d.copy_to(luisa::span{host_d}) << synchronize();
        auto untouched = true;
        for (auto value : host_d) {
            untouched &= value == 0.0f;
        }
        expect(untouched)
            << "a recorded batch must leave device state untouched until replay()";
        // The first replay is the submission.
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::SUCCESS);
        expect(submitted.invalidated_by == VulkanReplayInvalidationReason::NONE);
        expect(submitted.timeline_value != 0u)
            << "a replay signals a timeline value on the owning stream";
        stream << d.copy_to(luisa::span{host_d}) << synchronize();
        auto values_ok = true;
        for (auto value : host_d) {
            values_ok &= std::abs(value - 13.0f) < 1e-5f;
        }
        expect(values_ok) << "the first replay must run the recorded dispatch";
        expect(ext->status(recorded.handle).submits == 1u);
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        expect(ext->live_replay_count() == 0u);
    };
}

// ---------------------------------------------------------------------------
// record+replay == ordinary dispatch, and results repeat
// ---------------------------------------------------------------------------
void reg_replay_matches_dispatch(Device &device) {
    "replay_matches_ordinary_dispatch"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 256u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Buffer<float> e = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host_a(n);
        luisa::vector<float> host_b(n);
        for (auto i = 0u; i < n; ++i) {
            host_a[i] = static_cast<float>(i) * 0.5f;
            host_b[i] = 1.0f / static_cast<float>(i + 1u);
        }
        stream << a.copy_from(luisa::span{host_a})
               << b.copy_from(luisa::span{host_b})
               << synchronize();
        // Ordinary path: two dispatches and one copy, recorded twice (once into a
        // command list that is dispatched, once into a replay graph).
        {
            CommandList list;
            list << shader(a, b, d, 3.0f).dispatch(n)
                 << shader(d, a, d, -1.0f).dispatch(n)
                 << e.copy_from(d);
            stream << list.commit() << synchronize();
        }
        luisa::vector<float> ordinary(n, 0.0f);
        stream << e.copy_to(luisa::span{ordinary}) << synchronize();
        // Same list, recorded and replayed four times.
        {
            CommandList list;
            list << shader(a, b, d, 3.0f).dispatch(n)
                 << shader(d, a, d, -1.0f).dispatch(n)
                 << e.copy_from(d);
            auto recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS);
            if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
            auto graph = ext->status(recorded.handle);
            expect(graph.recorded_dispatches == 2u)
                << "the graph counts the two recorded dispatches";
            expect(graph.recorded_buffer_copies == 1u)
                << "the graph counts the recorded buffer copy";
            luisa::vector<float> replay_host(n, -1.0f);
            for (auto step = 0u; step < 4u; ++step) {
                auto submitted = ext->replay(recorded.handle);
                expect(submitted.status == VulkanReplayStatus::SUCCESS);
                stream << e.copy_to(luisa::span{replay_host}) << synchronize();
                auto identical = true;
                for (auto i = 0u; i < n; ++i) {
                    identical &= std::abs(replay_host[i] - ordinary[i]) < 1e-5f;
                }
                expect(identical)
                    << "a replay must reproduce the ordinary dispatch bit for bit "
                       "(same inputs, same uniform, same shape)";
            }
            expect(ext->status(recorded.handle).submits == 4u);
            expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        }
    };
}

// ---------------------------------------------------------------------------
// The one dynamic channel: device-buffer contents
// ---------------------------------------------------------------------------
void reg_changed_contents(Device &device) {
    "replay_changed_contents_reflected"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 128u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host(n);
        for (auto i = 0u; i < n; ++i) { host[i] = 1.0f; }
        stream << a.copy_from(luisa::span{host})
               << b.copy_from(luisa::span{host})
               << synchronize();
        CommandList list;
        list << shader(a, b, d, 100.0f).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        luisa::vector<float> first(n, -1.0f);
        stream << d.copy_to(luisa::span{first}) << synchronize();
        auto first_ok = true;
        for (auto value : first) { first_ok &= std::abs(value - 102.0f) < 1e-5f; }
        expect(first_ok) << "1 + 1 + 100";
        // Change the *contents* of a device buffer through an ordinary batch - the
        // only thing a graph re-reads.
        luisa::vector<float> updated(n);
        for (auto i = 0u; i < n; ++i) { updated[i] = 5.0f; }
        stream << a.copy_from(luisa::span{updated}) << synchronize();
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        luisa::vector<float> second(n, -1.0f);
        stream << d.copy_to(luisa::span{second}) << synchronize();
        auto second_ok = true;
        for (auto value : second) { second_ok &= std::abs(value - 106.0f) < 1e-5f; }
        expect(second_ok)
            << "a replay re-reads the device buffer contents (5 + 1 + 100)";
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// update() always refuses and leaves the graph byte-identical
// ---------------------------------------------------------------------------
void reg_update_refused(Device &device) {
    "replay_update_refused_keeps_old_uniforms"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host(n, 1.0f);
        stream << a.copy_from(luisa::span{host})
               << b.copy_from(luisa::span{host})
               << synchronize();
        CommandList list;
        list << shader(a, b, d, 7.0f).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        // Ask for a different uniform: refused, with the graph untouched.
        CommandList update_list;
        update_list << shader(a, b, d, 1000.0f).dispatch(n);
        auto updated = ext->update(recorded.handle, std::move(update_list));
        expect(updated == VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE)
            << "a replay graph encodes its uniform bytes, so update() has no "
               "smaller granularity than a new recording";
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::SUCCESS);
        expect(graph.submits == 0u)
            << "the refusal must not have touched the graph";
        expect(graph.recorded_dispatches == 1u)
            << "the recording is byte-identical after the refusal";
        // And the recorded value is still the one that runs.
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        luisa::vector<float> result(n, -1.0f);
        stream << d.copy_to(luisa::span{result}) << synchronize();
        auto kept_old = true;
        for (auto value : result) { kept_old &= std::abs(value - 9.0f) < 1e-5f; }
        expect(kept_old)
            << "after the refusal the graph still dispatches the recorded uniform "
               "(1 + 1 + 7), never the rejected one (1 + 1 + 1000)";
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// Several replays in flight
// ---------------------------------------------------------------------------
void reg_multiple_in_flight(Device &device) {
    "replay_multiple_in_flight"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 256u;
        constexpr auto replays = 8u;
        Buffer<float> out = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(write_dispatch_index);
        CommandList list;
        list << shader(out, 0u).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        for (auto i = 0u; i < replays; ++i) {
            auto submitted = ext->replay(recorded.handle);
            expect(submitted.status == VulkanReplayStatus::SUCCESS)
                << "independent replays of one graph may be in flight";
        }
        // One synchronize covers all of them: the last replay's notification is
        // what the stream's `sync()` waits for.
        stream << synchronize();
        expect(ext->status(recorded.handle).submits == replays)
            << "every replay is counted";
        luisa::vector<float> host(n, -1.0f);
        stream << out.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto value : host) { ok &= std::abs(value) < 1e-5f; }
        expect(ok) << "the repeated replays must all have written the same slice";
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS)
            << "destroy() after a synchronize must succeed";
    };
}

// ---------------------------------------------------------------------------
// Ordinary Event ordering across streams
// ---------------------------------------------------------------------------
void reg_event_ordering(Device &device) {
    "replay_cross_stream_event_order"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 128u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream producer = device.create_stream();
        Stream consumer = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host(n, 2.0f);
        producer << a.copy_from(luisa::span{host})
                 << b.copy_from(luisa::span{host})
                 << synchronize();
        CommandList list;
        list << shader(a, b, d, 0.5f).dispatch(n);
        auto recorded = ext->record(producer.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto event = device.create_event();
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        // The replay is one submission of the producer stream, so the ordinary
        // event signal after it orders the consumer behind it.
        producer << event.signal();
        luisa::vector<float> consumer_host(n, -1.0f);
        consumer << event.wait()
                 << d.copy_to(luisa::span{consumer_host})
                 << synchronize();
        auto ordered = true;
        for (auto value : consumer_host) {
            ordered &= std::abs(value - 4.5f) < 1e-5f;
        }
        expect(ordered)
            << "a replay must be ordered by the stream's ordinary timeline path, "
               "so an Event signalled after it covers its work";
        producer << synchronize();
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// Shader destruction after record()
// ---------------------------------------------------------------------------
void reg_shader_destroyed(Device &device) {
    "replay_shader_destroyed_after_record"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 128u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        luisa::vector<float> host(n, 1.0f);
        stream << a.copy_from(luisa::span{host})
               << b.copy_from(luisa::span{host})
               << synchronize();
        VulkanReplayRecordResult recorded{};
        {
            auto shader = device.compile(add_kernel);
            CommandList list;
            list << shader(a, b, d, 4.0f).dispatch(n);
            recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS);
        }// the shader is destroyed here, before the first replay
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::SUCCESS)
            << "destroying the shader must not invalidate the recording: the graph "
               "retains its pipeline and its interned layouts by id";
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::SUCCESS)
            << "a replay after the shader's death must still submit";
        luisa::vector<float> result(n, -1.0f);
        stream << d.copy_to(luisa::span{result}) << synchronize();
        auto ok = true;
        for (auto value : result) { ok &= std::abs(value - 6.0f) < 1e-5f; }
        expect(ok) << "and it must still produce the recorded result (1 + 1 + 4)";
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// One graph, several pipeline layouts, every shader destroyed before the replay
// ---------------------------------------------------------------------------
void reg_multiple_pipeline_layouts(Device &device) {
    "replay_multiple_pipeline_layouts_survive_shader_destruction"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 128u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Buffer<float> out = device.create_buffer<float>(kBlockSize);
        Stream stream = device.create_stream();
        luisa::vector<float> host_a(n, 1.0f);
        luisa::vector<float> host_b(n, 2.0f);
        stream << a.copy_from(luisa::span{host_a})
               << b.copy_from(luisa::span{host_b})
               << synchronize();
        VulkanReplayRecordResult recorded{};
        {
            // Two compute shaders with structurally different descriptor-set
            // bindings, hence structurally different interned pipeline layouts:
            // `add_kernel` binds three storage buffers and a uniform, the index
            // writer binds one buffer and an integer. Both handles are destroyed at
            // the end of this scope, i.e. before the first replay.
            auto add_shader = device.compile(add_kernel);
            auto index_shader = device.compile(write_dispatch_index);
            CommandList list;
            list << add_shader(a, b, d, 5.0f).dispatch(n)
                 << index_shader(out, 0u).dispatch(kBlockSize);
            recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS)
                << "a list of two different compute shaders is replayable";
        }// both shader handles are gone here
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::SUCCESS);
        expect(graph.recorded_dispatches == 2u);
        // The observable that makes "every shader's layouts are retained" checkable
        // from outside: a graph that kept only the first dispatch's pipeline layout
        // would report 1 here, and the second shader's destruction would then have
        // destroyed a layout the recorded buffer still binds.
        LUISA_INFO(
            "multi-layout graph: {} interned pipeline layout(s), {} interned "
            "descriptor-set layout(s), {} watched buffer(s)",
            graph.interned_pipeline_layouts, graph.interned_descriptor_set_layouts,
            graph.recorded_dispatches);
        expect(graph.interned_pipeline_layouts >= 2u)
            << "both shaders' interned pipeline layouts must be retained";
        expect(graph.interned_descriptor_set_layouts >= 2u)
            << "and both shaders' descriptor-set layouts";
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::SUCCESS)
            << "a replay must work with every recorded shader already destroyed";
        luisa::vector<float> d_host(n, -1.0f);
        luisa::vector<float> out_host(kBlockSize, -1.0f);
        stream << d.copy_to(luisa::span{d_host})
               << out.copy_to(luisa::span{out_host})
               << synchronize();
        auto add_ok = true;
        for (auto value : d_host) { add_ok &= std::abs(value - 8.0f) < 1e-5f; }
        expect(add_ok) << "the first shader's result (1 + 2 + 5)";
        auto index_ok = true;
        for (auto value : out_host) { index_ok &= std::abs(value) < 1e-5f; }
        expect(index_ok) << "the second shader's result (its dispatch index 0)";
        // Destroying the graph gives every retained layout back; a second graph with
        // the same two structures must then still be creatable, which is the
        // outside-visible sign that the release was symmetric rather than
        // over-releasing a layout other shaders still use.
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        expect(ext->live_replay_count() == 0u);
        {
            auto add_shader = device.compile(add_kernel);
            auto index_shader = device.compile(write_dispatch_index);
            CommandList list;
            list << add_shader(a, b, d, 1.0f).dispatch(n)
                 << index_shader(out, 1u).dispatch(kBlockSize);
            auto again = ext->record(stream.handle(), std::move(list));
            expect(again.status == VulkanReplayStatus::SUCCESS)
                << "the interned layouts are still live after the graph's release";
            if (again.status == VulkanReplayStatus::SUCCESS) {
                expect(ext->replay(again.handle).status == VulkanReplayStatus::SUCCESS);
                stream << d.copy_to(luisa::span{d_host}) << synchronize();
                auto ok = true;
                for (auto value : d_host) { ok &= std::abs(value - 4.0f) < 1e-5f; }
                expect(ok) << "and it still dispatches correctly (1 + 2 + 1)";
                expect(ext->destroy(again.handle) == VulkanReplayStatus::SUCCESS);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// A captured buffer is watched like a runtime one
// ---------------------------------------------------------------------------
void reg_captured_buffer(Device &device) {
    "replay_captured_buffer_destroyed"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> out = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        VulkanReplayRecordResult recorded{};
        auto submits_before_destruction = uint64_t{0u};
        {
            // `src` is closed over by the kernel, so it is a *captured* BUFFER
            // argument (`shader->captured()`), not a runtime one. The dispatch's own
            // argument table contains only `out`.
            Buffer<float> src = device.create_buffer<float>(n);
            luisa::vector<float> host(n, 3.0f);
            stream << src.copy_from(luisa::span{host}) << synchronize();
            Kernel1D capture_kernel = [&](BufferFloat dst) noexcept {
                dst.write(dispatch_id().x, src->read(dispatch_id().x));
            };
            auto shader = device.compile(capture_kernel);
            CommandList list;
            list << shader(out).dispatch(n);
            recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS)
                << "a kernel with a captured buffer argument is replayable";
            if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
            LUISA_INFO(
                "captured-buffer graph: {} watched buffer(s)",
                ext->status(recorded.handle).recorded_dispatches);
            // The captured buffer is still alive, so the recording runs.
            expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
            submits_before_destruction = ext->status(recorded.handle).submits;
            expect(submits_before_destruction == 1u)
                << "exactly one replay was submitted while the captured buffer lived";
            luisa::vector<float> result(n, -1.0f);
            stream << out.copy_to(luisa::span{result}) << synchronize();
            auto ok = true;
            for (auto value : result) { ok &= std::abs(value - 3.0f) < 1e-5f; }
            expect(ok) << "the captured buffer is read, not a stale copy of it";
        }// `src` (and the shader) are destroyed here
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto before = ext->status(recorded.handle);
        expect(before.invalidation == VulkanReplayInvalidationReason::RESOURCE_DESTROYED)
            << "the captured buffer's destruction already invalidated the graph";
        expect(before.submits == submits_before_destruction);
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::RESOURCE_DESTROYED)
            << "destroying a *captured* buffer must invalidate the graph exactly like "
               "destroying a runtime one";
        expect(submitted.invalidated_by ==
               VulkanReplayInvalidationReason::RESOURCE_DESTROYED);
        expect(submitted.timeline_value == 0u)
            << "a refused replay signals no timeline value";
        expect(!submitted.message.empty());
        auto after = ext->status(recorded.handle);
        expect(after.submits == before.submits)
            << "and it submits nothing: the graph's submit count is unchanged";
        expect(after.invalidation == VulkanReplayInvalidationReason::RESOURCE_DESTROYED);
        // The graph still frees its native state without touching the freed buffer.
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        expect(ext->live_replay_count() == 0u);
        stream << synchronize();
    };
}

// ---------------------------------------------------------------------------
// Invalidation by resource / stream destruction
// ---------------------------------------------------------------------------
void reg_buffer_invalidation(Device &device) {
    "replay_invalidated_by_buffer_destruction"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> a = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        luisa::vector<float> host(n, 1.0f);
        stream << a.copy_from(luisa::span{host}) << synchronize();
        VulkanReplayRecordResult recorded{};
        {
            Buffer<float> b = device.create_buffer<float>(n);
            Buffer<float> d = device.create_buffer<float>(n);
            stream << b.copy_from(luisa::span{host}) << synchronize();
            CommandList list;
            list << shader(a, b, d, 1.0f).dispatch(n);
            recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS);
            expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
            stream << synchronize();
        }// `b` and `d` are destroyed here
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::RESOURCE_DESTROYED)
            << "destroying a watched buffer must mark the graph before the native "
               "buffer goes away";
        expect(graph.invalidation == VulkanReplayInvalidationReason::RESOURCE_DESTROYED);
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::RESOURCE_DESTROYED)
            << "a later replay must return the typed error, not submit";
        expect(submitted.invalidated_by ==
               VulkanReplayInvalidationReason::RESOURCE_DESTROYED);
        expect(!submitted.message.empty());
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS)
            << "an invalidated graph is still destroyable (and frees its state)";
        expect(ext->live_replay_count() == 0u);
    };
}

void reg_stream_invalidation(Device &device) {
    "replay_invalidated_by_stream_destruction"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        auto shader = device.compile(add_kernel);
        VulkanReplayRecordResult recorded{};
        {
            Stream stream = device.create_stream();
            luisa::vector<float> host(n, 1.0f);
            stream << a.copy_from(luisa::span{host})
                   << b.copy_from(luisa::span{host})
                   << synchronize();
            CommandList list;
            list << shader(a, b, d, 2.0f).dispatch(n);
            recorded = ext->record(stream.handle(), std::move(list));
            expect(recorded.status == VulkanReplayStatus::SUCCESS);
            expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        }// the recording stream is destroyed here
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto graph = ext->status(recorded.handle);
        expect(graph.status == VulkanReplayStatus::STREAM_DESTROYED)
            << "destroying the recording stream must mark the graphs bound to it";
        auto submitted = ext->replay(recorded.handle);
        expect(submitted.status == VulkanReplayStatus::STREAM_DESTROYED);
        expect(submitted.invalidated_by == VulkanReplayInvalidationReason::STREAM_DESTROYED);
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS)
            << "freeing the native state after the stream is gone must be safe";
        expect(ext->live_replay_count() == 0u);
    };
}

// ---------------------------------------------------------------------------
// destroy() BUSY, then synchronize and retry
// ---------------------------------------------------------------------------
void reg_destroy_busy(Device &device) {
    "replay_destroy_busy_then_sync_succeeds"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> out = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(spin_kernel);
        // A GPU-bound graph, so the pending window is milliseconds wide instead of
        // racing the retirement thread.
        CommandList list;
        for (auto i = 0u; i < 4u; ++i) {
            list << shader(out, 200000u).dispatch(n);
        }
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        auto busy = ext->destroy(recorded.handle);
        expect(busy == VulkanReplayStatus::BUSY)
            << "destroy() must refuse while the latest replay has not retired";
        expect(ext->live_replay_count() == 1u)
            << "a BUSY refusal must leave the graph alive";
        stream << synchronize();
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS)
            << "synchronize + retry must succeed";
        expect(ext->live_replay_count() == 0u);
        luisa::vector<float> host(n, -1.0f);
        stream << out.copy_to(luisa::span{host}) << synchronize();
        auto written = true;
        for (auto value : host) { written &= value > 0.0f; }
        expect(written) << "the graph really executed before it was freed";
    };
}

// ---------------------------------------------------------------------------
// handle lifecycle
// ---------------------------------------------------------------------------
void reg_handle_lifecycle(Device &device) {
    "replay_handle_lifecycle"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        // Unknown handle: a typed answer, never a crash.
        auto unknown = ext->status(0xdeadbeefull);
        expect(unknown.status == VulkanReplayStatus::INVALID_HANDLE);
        expect(ext->replay(0xdeadbeefull).status == VulkanReplayStatus::INVALID_HANDLE);
        expect(ext->destroy(0xdeadbeefull) == VulkanReplayStatus::INVALID_HANDLE);
        // Invalid stream handles.
        constexpr auto n = 64u;
        Buffer<float> out = device.create_buffer<float>(n);
        auto shader = device.compile(write_dispatch_index);
        {
            CommandList list;
            list << shader(out, 0u).dispatch(n);
            auto result = ext->record(0x1234u, std::move(list));
            expect(result.status == VulkanReplayStatus::INVALID_STREAM)
                << "a handle that is not a stream of this device is typed, not "
                   "reinterpret_cast";
        }
        {
            CommandList list;
            list << shader(out, 0u).dispatch(n);
            auto result = ext->record(0u, std::move(list));
            expect(result.status == VulkanReplayStatus::INVALID_ARGUMENT);
        }
        // An empty list is a refusal, not an empty graph.
        {
            CommandList list;
            auto result = ext->record(device.create_stream().handle(), std::move(list));
            expect(result.status == VulkanReplayStatus::REJECTED);
            expect(result.reason == VulkanReplayRejectReason::EMPTY_LIST);
        }
        // Double destroy.
        Stream stream = device.create_stream();
        CommandList list;
        list << shader(out, 0u).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto live_before = ext->live_replay_count();
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        expect(ext->live_replay_count() + 1u == live_before);
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::INVALID_HANDLE)
            << "a second destroy is a typed no-op, never a double free";
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::INVALID_HANDLE)
            << "a destroyed graph is gone";
        expect(ext->status(recorded.handle).status == VulkanReplayStatus::INVALID_HANDLE);
        stream << synchronize();
    };
}

// ---------------------------------------------------------------------------
// The rejection matrix
// ---------------------------------------------------------------------------
void reg_reject_matrix(Device &device) {
    "replay_rejects_unsupported"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 64u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        auto index_shader = device.compile(write_dispatch_index);
        auto tex_shader = device.compile(read_texture);
        auto printer_shader = device.compile(print_kernel);
        Image<float> tex2d = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(8u, 8u));
        Buffer<float4> pixels = device.create_buffer<float4>(64u);
        BindlessArray heap = device.create_bindless_array(4u);
        heap.emplace_on_update(0u, a);
        IndirectDispatchBuffer indirect = device.create_indirect_dispatch_buffer(1u);
        luisa::vector<float> host(n, 1.0f);
        stream << a.copy_from(luisa::span{host})
               << b.copy_from(luisa::span{host})
               << heap.update()
               << synchronize();

        // command-level refusals
        {
            CommandList list;
            list << a.copy_from(luisa::span{host});
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_BUFFER_UPLOAD, "upload");
        }
        {
            CommandList list;
            list << a.copy_to(luisa::span{host});
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_BUFFER_DOWNLOAD, "download");
        }
        {
            CommandList list;
            list << tex2d.copy_from(pixels);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_TEXTURE, "buffer-to-texture copy");
        }
        {
            // Built directly: the classifier decides on the command tag alone and
            // never dereferences the handle, so proving the refusal does not need
            // real ray-tracing geometry (the tag is also covered by the host test
            // for all five build commands).
            CommandList list;
            list << luisa::make_unique<AccelBuildCommand>(
                0x1234u, 0u, AccelBuildRequest::FORCE_BUILD,
                luisa::vector<AccelBuildCommand::Modification>{}, false);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_ACCEL_BUILD, "accel build");
        }
        {
            // A bindless update with nothing pending produces no command at all, so
            // stage one modification first: the update is a real
            // BindlessArrayUpdateCommand and is refused as one.
            heap.emplace_on_update(1u, b);
            CommandList list;
            list << heap.update();
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_BINDLESS_UPDATE,
                           "bindless update");
        }
        {
            CommandList list;
            list << luisa::make_unique<NoopCustomCmd>();
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::COMMAND_CUSTOM, "custom command");
        }
        {
            CommandList list;
            list << index_shader(d, 0u).dispatch(indirect, 0u, 1u);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::DISPATCH_INDIRECT,
                           "indirect dispatch");
        }
        {
            CommandList list;
            list << printer_shader(d).dispatch(n);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::DISPATCH_PRINTER,
                           "printer dispatch");
        }
        // argument-level refusals
        {
            CommandList list;
            list << tex_shader(tex2d, d).dispatch(8u, 8u);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::ARG_TEXTURE, "texture argument");
        }
        {
            // The same rule on the *captured* table: this kernel closes over a
            // bindless array, so the array is not among the dispatch's runtime
            // arguments at all. The refusal happens in the classification pass, i.e.
            // before any native object (retained layout, command buffer, pool) is
            // created - which is what the unchanged `live_replay_count()` and
            // `vk_replay_graphs` prove.
            BindlessArray captured_heap = device.create_bindless_array(4u);
            captured_heap.emplace_on_update(0u, a);
            stream << captured_heap.update() << synchronize();
            Kernel1D captured_bindless_kernel = [&](BufferFloat dst) noexcept {
                dst.write(dispatch_id().x,
                          captured_heap->buffer<float>(0u).read(0u));
            };
            auto captured_shader = device.compile(captured_bindless_kernel);
            CommandList list;
            list << captured_shader(d).dispatch(n);
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::ARG_BINDLESS_ARRAY,
                           "captured bindless argument");
            expect(device.query("vk_replay_graphs") == "0")
                << "a captured-argument refusal creates no graph";
        }
        // list-level refusals
        {
            CommandList list;
            list << shader(a, b, d, 1.0f).dispatch(n);
            list.add_callback([] {});
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::LIST_CALLBACKS, "list callback");
        }
        {
            CommandList list;
            list << shader(a, b, d, 1.0f).dispatch(n);
            // A present is refused at list level, before anything is
            // dereferenced, so a well-formed present without a swapchain is
            // enough to drive the rule.
            list.add_present(SwapchainPresent{
                .chain = nullptr, .frame = tex2d.view()});
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::LIST_PRESENTS, "present");
        }
        auto dtor_ran = false;
        {
            CommandList list;
            list << shader(a, b, d, 1.0f).dispatch(n);
            list.add_dtor_callback([&dtor_ran] { dtor_ran = true; });
            check_rejected(device, ext, stream, std::move(list),
                           VulkanReplayRejectReason::LIST_DTOR_CALLBACKS,
                           "destructor callback");
            expect(dtor_ran)
                << "record() owns the list, so the destructor callbacks run there - "
                   "once, at record time, never during a replay";
        }
        // The device is still fully usable after the whole matrix.
        expect(ext->live_replay_count() == 0u);
        stream << d.copy_from(a) << synchronize();
        luisa::vector<float> check(n, -1.0f);
        stream << d.copy_to(luisa::span{check}) << synchronize();
        expect(check[0] == 1.0f);
    };
}

// ---------------------------------------------------------------------------
// Ordinary-path regression after using replay
// ---------------------------------------------------------------------------
void reg_ordinary_path_after_replay(Device &device) {
    "replay_ordinary_path_regression"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 256u;
        Buffer<float> a = device.create_buffer<float>(n);
        Buffer<float> b = device.create_buffer<float>(n);
        Buffer<float> d = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(add_kernel);
        auto index_shader = device.compile(write_dispatch_index);
        auto indirect_shader = device.compile(write_index_fixed_block);
        luisa::vector<float> host(n, 3.0f);
        stream << a.copy_from(luisa::span{host})
               << b.copy_from(luisa::span{host})
               << synchronize();
        CommandList list;
        list << shader(a, b, d, 1.0f).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        stream << synchronize();
        // An ordinary upload, download, dispatch and indirect dispatch still work,
        // and produce the values they always did (the guard for the record-prefix
        // extraction).
        luisa::vector<float> uploaded(n, 4.0f);
        stream << a.copy_from(luisa::span{uploaded}) << synchronize();
        stream << shader(a, b, d, 2.0f).dispatch(n) << synchronize();
        luisa::vector<float> result(n, -1.0f);
        stream << d.copy_to(luisa::span{result}) << synchronize();
        auto ok = true;
        for (auto value : result) { ok &= std::abs(value - 9.0f) < 1e-5f; }
        expect(ok) << "an ordinary dispatch after a replay: 4 + 3 + 2";
        // The replay still reads the *old* uniform and the new buffer contents.
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        stream << d.copy_to(luisa::span{result}) << synchronize();
        auto replay_ok = true;
        for (auto value : result) { replay_ok &= std::abs(value - 8.0f) < 1e-5f; }
        expect(replay_ok) << "4 + 3 + 1 (the recorded uniform)";
        // Indirect dispatch: a stream of its own kind, and it still works - but
        // only on a backend built with the native XIR-to-SPIR-V codegen, because
        // that is what gives a shader the indirect-dispatch metadata ABI. In an
        // HLSL-fallback build the backend refuses an indirect dispatch outright
        // (vulkan XIR metadata assertion), which is a property of the build, not of
        // the record-prefix extraction this test guards.
        if (device.query("vk_spirv_codegen_route") != "native") {
            LUISA_INFO(
                "ordinary-path indirect dispatch: skipped - this build has no "
                "native XIR-to-SPIR-V codegen (vk_spirv_codegen_route={})",
                device.query("vk_spirv_codegen_route"));
            expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
            return;
        }
        constexpr auto dispatch_count = 2u;
        Kernel1D clear_commands = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(dispatch_count);
        };
        Kernel1D author_commands = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            // One 64-thread block per record, stamped with that record's kernel id,
            // so the expected contents are exactly "slot i*64..i*64+63 holds i".
            // The third field is the record's *logical* size in threads (the backend
            // divides it by the block size to get the group count), which is why it
            // has to equal the block size here and not 1.
            commands.set_kernel(dispatch_id().x,
                                make_uint3(64u, 1u, 1u),
                                make_uint3(64u, 1u, 1u),
                                dispatch_id().x);
        };
        auto clear_shader = device.compile(clear_commands);
        auto author_shader = device.compile(author_commands);
        IndirectDispatchBuffer commands = device.create_indirect_dispatch_buffer(dispatch_count);
        Buffer<float> indirect_out = device.create_buffer<float>(dispatch_count * kBlockSize);
        luisa::vector<float> sentinel(dispatch_count * kBlockSize, -7.0f);
        stream << indirect_out.copy_from(luisa::span{sentinel}) << synchronize();
        stream << clear_shader(commands).dispatch(1u)
               << author_shader(commands).dispatch(dispatch_count);
        for (auto i = 0u; i < dispatch_count; ++i) {
            stream << indirect_shader(indirect_out).dispatch(commands, i, 1u);
        }
        stream << synchronize();
        luisa::vector<float> indirect_host(dispatch_count * kBlockSize, -1.0f);
        stream << indirect_out.copy_to(luisa::span{indirect_host}) << synchronize();
        auto indirect_ok = true;
        auto reported = 0u;
        for (auto i = 0u; i < dispatch_count; ++i) {
            for (auto j = 0u; j < kBlockSize; ++j) {
                auto actual = indirect_host[i * kBlockSize + j];
                auto matches = std::abs(actual - static_cast<float>(i)) < 1e-5f;
                if (!matches && reported < 5u) {
                    LUISA_WARNING(
                        "indirect regression mismatch at record {} slot {}: got {} "
                        "expected {}",
                        i, j, actual, static_cast<float>(i));
                    ++reported;
                }
                indirect_ok &= matches;
            }
        }
        expect(indirect_ok) << "an ordinary indirect dispatch still works";
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// Decode-shaped repeated dispatch: record+replay versus ordinary dispatch
// ---------------------------------------------------------------------------
void reg_decode_shaped(Device &device) {
    "replay_decode_shaped_batch"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        // The gallery's measured launch count of one Qwen decode token (Q), the
        // same shape the submit-profile tripwire uses.
        constexpr auto dispatch_count = size_t{438u};
        constexpr auto replays = 8u;
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer = device.create_buffer<float>(dispatch_count * kBlockSize);
        Stream stream = device.create_stream();
        auto make_list = [&] {
            CommandList list;
            list.reserve(dispatch_count, 0u);
            for (auto i = 0u; i < dispatch_count; ++i) {
                list << shader(buffer, i).dispatch(kBlockSize);
            }
            return list;
        };
        // Ordinary path, host submission time only (the sync is outside the
        // measured window, so no GPU time is folded in).
        luisa::vector<double> ordinary_us;
        ordinary_us.reserve(replays);
        for (auto i = 0u; i < replays; ++i) {
            auto list = make_list();
            auto begin = std::chrono::steady_clock::now();
            stream << list.commit();
            auto end = std::chrono::steady_clock::now();
            ordinary_us.emplace_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
            stream << synchronize();
        }
        // Record once, then replay the same number of times.
        auto recorded = ext->record(stream.handle(), make_list());
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        stream << synchronize();
        luisa::vector<double> replay_us;
        replay_us.reserve(replays);
        for (auto i = 0u; i < replays; ++i) {
            auto begin = std::chrono::steady_clock::now();
            auto submitted = ext->replay(recorded.handle);
            auto end = std::chrono::steady_clock::now();
            expect(submitted.status == VulkanReplayStatus::SUCCESS);
            replay_us.emplace_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
            stream << synchronize();
        }
        std::sort(ordinary_us.begin(), ordinary_us.end());
        std::sort(replay_us.begin(), replay_us.end());
        auto median = [](luisa::vector<double> const &values) noexcept {
            return values.empty() ? 0.0 : values[values.size() / 2u];
        };
        auto ordinary_median = median(ordinary_us);
        auto replay_median = median(replay_us);
        LUISA_INFO(
            "decode-shaped replay: {} dispatches, counters off | ordinary batch "
            "host submission median {:.1f} us (min {:.1f}, max {:.1f}) | replay "
            "host submission median {:.1f} us (min {:.1f}, max {:.1f}) | delta "
            "{:.1f} us of {} dispatches",
            dispatch_count, ordinary_median, ordinary_us.front(), ordinary_us.back(),
            replay_median, replay_us.front(), replay_us.back(),
            ordinary_median - replay_median, dispatch_count);
        expect(ordinary_median > 0.0 && replay_median > 0.0)
            << "both paths must be measured";
        // Only the structural claim is asserted: the results are identical. The
        // timing above is reported, not asserted - a host-side win depends on the
        // machine, and this test is not a benchmark.
        luisa::vector<float> host(dispatch_count * kBlockSize, -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < dispatch_count; ++i) {
            for (auto j = 0u; j < kBlockSize; ++j) {
                ok &= std::abs(host[i * kBlockSize + j] - static_cast<float>(i)) < 1e-5f;
            }
        }
        expect(ok) << "a replayed decode-shaped batch leaves the same values behind";
        expect(ext->status(recorded.handle).submits == replays);
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
    };
}

// ---------------------------------------------------------------------------
// Device teardown drains a live graph
// ---------------------------------------------------------------------------
void reg_device_teardown(Device &device) {
    "replay_device_teardown_drains_graph"_test = [&device] {
        auto *ext = replay_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto n = 128u;
        Buffer<float> out = device.create_buffer<float>(n);
        Stream stream = device.create_stream();
        auto shader = device.compile(write_dispatch_index);
        CommandList list;
        list << shader(out, 0u).dispatch(n);
        auto recorded = ext->record(stream.handle(), std::move(list));
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        expect(ext->replay(recorded.handle).status == VulkanReplayStatus::SUCCESS);
        // The graph is *deliberately* left alive: this test is the only one that
        // exercises `~Device`'s drain, which must synchronize the owning stream,
        // free the recorded command buffer and its state, and release the interned
        // layouts while the logical device is still alive. The evidence is a clean
        // process exit with no "reached registry teardown without being drained"
        // warning - that warning is emitted exactly when the drain did not happen.
        expect(ext->live_replay_count() >= 1u);
        LUISA_INFO(
            "leaving {} live replay graph(s) for device teardown to drain",
            ext->live_replay_count());
        stream << synchronize();
    };
}

}// namespace

int main(int argc, char *argv[]) {
    // Must happen before the device exists: the API-version floor is a
    // device-creation input.
    take_min_api_option(argc, argv);
    DeviceConfig device_config;
    device_config.extension = luisa::make_unique<ReplayDeviceConfig>();
    auto dc = luisa::test::create_device_from_ut(argc, argv, &device_config);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "vk") {
        LUISA_INFO("Skipping: replayable recording is a Vulkan-only extension.");
        return 0;
    }
    log_level_verbose();
    reg_submit_path(device);
    reg_record_does_not_submit(device);
    reg_replay_matches_dispatch(device);
    reg_changed_contents(device);
    reg_update_refused(device);
    reg_multiple_in_flight(device);
    reg_event_ordering(device);
    reg_shader_destroyed(device);
    reg_multiple_pipeline_layouts(device);
    reg_captured_buffer(device);
    reg_buffer_invalidation(device);
    reg_stream_invalidation(device);
    reg_destroy_busy(device);
    reg_handle_lifecycle(device);
    reg_reject_matrix(device);
    reg_ordinary_path_after_replay(device);
    reg_decode_shaped(device);
    // Last on purpose: it leaves a graph alive so `~Device`'s drain runs.
    reg_device_teardown(device);
    return 0;
}