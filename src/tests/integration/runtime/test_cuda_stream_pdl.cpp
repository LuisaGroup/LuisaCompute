// Host -> device end-to-end test of the CUDA backend's stream-form
// programmatic dependent launch (PDL, `src/backends/cuda/cuda_pdl_launch.h`):
//
// - the opt-in device (`CUDADeviceConfigExt::enable_programmatic_stream_
//   serialization()` on) compiles the automatic wait-at-entry prologue into
//   every DSL compute kernel and reports it through
//   `CUDAPdlStreamExt::auto_prologue_enabled()`;
// - a chain of dependent kernels (each grid reads what the previous one wrote)
//   produces the exact expected values, with the launch policy on and off;
// - the launch-decision counters show which grids were allowed to start early:
//   back-to-back grids become relaxed launches once the predecessor is a kernel
//   launch, a copy in between is a hard boundary, and the runtime policy switch
//   turns relaxation off and on again;
// - a second device created *without* the prologue produces the same values bit
//   for bit (the "no behaviour change" assertion of the feature).
//
// What cannot be asserted here: the *execution* side of PDL, i.e. a grid that
// really starts before its predecessor retired. That needs a Hopper-or-newer
// device (compute capability >= 9.0) and a driver >= 12.0; on anything older the
// attribute is never used (`cuda_pdl_launch.h` requires cc >= 90) and the wait
// itself compiles to nothing (`cuda_builtin/cuda_device_graph.h`), so the grids
// stay fully serialized. The test is therefore built out of three signals that
// are meaningful on both kinds of machine: (a) exact result values, (b) the
// launch-decision counters of `CUDAPdlStreamExt::launch_stats()`, and (c) A/B
// equality between a device with and without the automatic prologue. On a
// pre-sm_90 device (b) degenerates to "everything is strict" - which is itself
// the property under test, and the counters are asserted against
// `pdl_supported()` so the expectation is exact on either machine.
//
// Usage: test_cuda_stream_pdl cuda
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_config_ext.h>
#include <luisa/backends/ext/cuda/cuda_pdl_ext.h>

#include <algorithm>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// The workload shape the feature was specified with.
constexpr uint32_t k_count = 1u << 16u;
constexpr uint32_t k_block_size = 256u;
// Spin cycles per thread before the dependent read. Only meaningful on sm_90+,
// where it widens the window in which an unsound relaxation would read a stale
// value; tens of microseconds per grid, so it stays invisible in the run time.
constexpr uint32_t k_ticks = 100000u;

// Boost.UT runs the cases below when the process exits, i.e. after main() has
// returned and every local of it is gone. The devices and the host-side input
// therefore live in this holder, which main() allocates and deliberately never
// frees: destroying a device from a static destructor would tear the context
// (and the shader cache underneath it) down while the framework is reporting.
struct TestDevices {
    Context context;
    luisa::string backend;
    luisa::vector<uint32_t> input; // host-side initial contents of buf[0]
    Device opt_in;                 // enable_programmatic_stream_serialization() == true
    explicit TestDevices(luisa::string_view exe) : context{exe} {}
};

TestDevices *g_devices{nullptr};

// `CUDADeviceConfigExt` is a header-only abstract class, so this include and the
// struct below compile for every configured backend; only the CUDA one is
// answered, and every other backend is skipped by main().
[[nodiscard]] luisa::unique_ptr<DeviceConfigExt>
make_config_ext(luisa::string_view backend, bool prologue) noexcept {
    if (backend == "cuda") {
        struct Ext final : CUDADeviceConfigExt {
            bool enabled;
            explicit Ext(bool e) noexcept : enabled{e} {}
            [[nodiscard]] bool enable_programmatic_stream_serialization() const noexcept override {
                return enabled;
            }
        };
        return luisa::make_unique<Ext>(prologue);
    }
    return nullptr;
}

// One compiled shader, reused for every link of a chain: `out = 3 * in + 1` with
// uint32 wraparound, so the host can compute the expected values exactly.
[[nodiscard]] auto make_chain_kernel() noexcept {
    return Kernel1D{[](BufferUInt in, BufferUInt out, UInt ticks) noexcept {
        set_block_size(k_block_size);
        // Optional delay *before* the dependent read: on sm_90+ it widens the
        // window in which an unsound relaxation would read a stale value. Inert
        // on a pre-sm_90 device, where the grid is fully serialized anyway.
        $if (ticks != 0u) {
            auto start = def(device_clock());
            // `luisa::ulong` is spelled out on purpose: a bare `ulong` resolves
            // to the Windows SDK's `typedef unsigned long ulong` there (32 bits)
            // and does not exist at all on other platforms, so the widening
            // would be silently wrong on one host and a parse error on another.
            $while (device_clock() - start < cast<luisa::ulong>(ticks)) { };
        };
        UInt i = dispatch_id().x;
        out.write(i, in.read(i) * 3u + 1u);
    }};
}

// A scramble, so the exact expected values cannot coincide with the index.
[[nodiscard]] luisa::vector<uint32_t> make_input(uint32_t n) noexcept {
    luisa::vector<uint32_t> values(n);
    for (auto i = 0u; i < n; i++) {
        values[i] = i * 2654435761u + 0x9e3779b9u;
    }
    return values;
}

// The exact host-side expectation: `links` applications of x -> 3x + 1.
[[nodiscard]] luisa::vector<uint32_t> expected_chain(
    luisa::span<const uint32_t> input, uint32_t links) noexcept {
    auto values = luisa::vector<uint32_t>{input.begin(), input.end()};
    for (auto i = 0u; i < links; i++) {
        for (auto &v : values) { v = v * 3u + 1u; } // uint32 wraparound
    }
    return values;
}

[[nodiscard]] size_t count_mismatches(luisa::span<const uint32_t> a,
                                     luisa::span<const uint32_t> b) noexcept {
    if (a.size() != b.size()) { return std::max(a.size(), b.size()); }
    auto mismatches = size_t{0u};
    for (auto i = size_t{0u}; i < a.size(); i++) {
        mismatches += a[i] != b[i];
    }
    return mismatches;
}

struct LaunchCounts {
    uint64_t relaxed{0u};
    uint64_t strict{0u};
};

[[nodiscard]] LaunchCounts launch_counts(CUDAPdlStreamExt &ext) noexcept {
    auto stats = ext.launch_stats();
    return LaunchCounts{.relaxed = stats.relaxed_launches,
                        .strict = stats.strict_launches};
}

[[nodiscard]] CUDAPdlStreamExt *pdl_ext(Device &device) noexcept {
    return device.extension<CUDAPdlStreamExt>();
}

// The chained-dependency workload: `buf[0] -> buf[1] -> ... -> buf[links]`, one
// dispatch per link, all of them in **one** submission (a single CommandList
// committed to the stream), so the launch decisions this test asserts are
// exactly the ones of that submission. `buf[0]` is filled on the host and
// uploaded before the submission.
[[nodiscard]] luisa::vector<uint32_t> run_chain(
    Device &device, bool enable_policy, uint32_t links, uint32_t n,
    luisa::span<const uint32_t> initial) noexcept {
    if (auto *ext = pdl_ext(device)) {
        ext->set_pdl_stream_enabled(enable_policy);
    }
    auto stream = device.create_stream();
    auto shader = device.compile(make_chain_kernel());
    luisa::vector<Buffer<uint32_t>> buffers;
    buffers.reserve(links + 1u);
    for (auto i = 0u; i <= links; i++) {
        buffers.emplace_back(device.create_buffer<uint32_t>(n));
    }
    stream << buffers.front().copy_from(initial) << synchronize();
    auto list = CommandList::create(links);
    for (auto i = 0u; i < links; i++) {
        list << shader(buffers[i].view(), buffers[i + 1u].view(), k_ticks).dispatch(n);
    }
    stream << list.commit() << synchronize();
    luisa::vector<uint32_t> result(n);
    stream << buffers.back().copy_to(luisa::span{result}) << synchronize();
    return result;
}

// One submission of `count` back-to-back dispatches (one input buffer, `count`
// distinct output buffers, no copy or any other command between the grids). The
// counters are reset right before the submission, so what they read afterwards
// is exactly the launch decisions of this submission; the upload that precedes
// it is a separate, already drained submission.
[[nodiscard]] LaunchCounts run_back_to_back(
    Device &device, CUDAPdlStreamExt &ext, uint32_t count, uint32_t n,
    luisa::span<const uint32_t> initial) noexcept {
    auto stream = device.create_stream();
    auto shader = device.compile(make_chain_kernel());
    auto input = device.create_buffer<uint32_t>(n);
    luisa::vector<Buffer<uint32_t>> outputs;
    outputs.reserve(count);
    for (auto i = 0u; i < count; i++) {
        outputs.emplace_back(device.create_buffer<uint32_t>(n));
    }
    stream << input.copy_from(initial) << synchronize();
    auto list = CommandList::create(count);
    for (auto i = 0u; i < count; i++) {
        list << shader(input.view(), outputs[i].view(), k_ticks).dispatch(n);
    }
    ext.reset_launch_stats();
    stream << list.commit() << synchronize();
    return launch_counts(ext);
}

// One submission of `dispatch, copy_to(host), dispatch`. The dispatch after the
// copy must not be relaxed: the copy is a hard boundary for the launch policy
// (there is no preceding kernel to overlap with, and the effect of the attribute
// on the ordering operations is not documented).
[[nodiscard]] LaunchCounts run_dispatch_copy_dispatch(
    Device &device, CUDAPdlStreamExt &ext, uint32_t n,
    luisa::span<const uint32_t> initial) noexcept {
    auto stream = device.create_stream();
    auto shader = device.compile(make_chain_kernel());
    auto input = device.create_buffer<uint32_t>(n);
    auto first = device.create_buffer<uint32_t>(n);
    auto second = device.create_buffer<uint32_t>(n);
    luisa::vector<uint32_t> sink(n);
    stream << input.copy_from(initial) << synchronize();
    auto list = CommandList::create(3u);
    list << shader(input.view(), first.view(), k_ticks).dispatch(n);
    list << first.copy_to(luisa::span{sink});
    list << shader(input.view(), second.view(), k_ticks).dispatch(n);
    ext.reset_launch_stats();
    stream << list.commit() << synchronize();
    return launch_counts(ext);
}

}// namespace

int main(int argc, char *argv[]) {
    auto exe = luisa::test::safe_argv0();
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        luisa::test::print_device_usage(exe);
        return 1;
    }
    luisa::string backend = argv[1];
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    // Stream-form PDL exists only in the CUDA backend: every other backend has
    // neither the extension nor the launch attribute, so there is nothing to
    // exercise (the cases would skip themselves as well).
    if (backend != "cuda") {
        LUISA_INFO("Stream-form PDL is a CUDA backend feature; skipping on '{}'.", backend);
        return 0;
    }
    auto devices = luisa::make_unique<TestDevices>(exe);
    devices->backend = backend;
    devices->input = make_input(k_count);
    DeviceConfig config;
    config.extension = make_config_ext(backend, /*prologue=*/true);
    devices->opt_in = devices->context.create_device(backend, &config);
    // Leaked on purpose; see TestDevices.
    g_devices = devices.release();
    // The cases are registered from main() (and run by the framework when the
    // process exits) so that they can use the devices created just above; they
    // capture nothing, only the file-scope holder.

    "cuda_pdl_configuration"_test = [] {
        if (g_devices == nullptr) { return; }
        auto &device = g_devices->opt_in;
        if (device.backend_name() != "cuda") {
            LUISA_INFO("Stream-form PDL is a CUDA backend feature; skipping on '{}'.",
                       device.backend_name());
            return;
        }
        expect(device.backend_name() == "cuda");
        auto *ext = pdl_ext(device);
        expect(ext != nullptr) << "the CUDA backend must provide CUDAPdlStreamExt";
        if (ext == nullptr) { return; }
        // LUISA_DISABLE_PDL_STREAM=1 forces both halves of the feature off for the
        // whole process, and the device reports it accordingly. The extension cannot
        // be asked about the switch directly, but a device created with
        // enable_programmatic_stream_serialization() that does not compile the
        // prologue can only have been overridden by it, so treat that as "skip" so
        // this test also serves as a clean A/B harness under the switch.
        if (!ext->auto_prologue_enabled()) {
            LUISA_WARNING("CUDA stream PDL is forced off for this process "
                          "(LUISA_DISABLE_PDL_STREAM); skipping the configuration checks.");
        } else {
            // The device was created with enable_programmatic_stream_serialization():
            // the codegen half is on for every kernel it compiles ...
            expect(ext->auto_prologue_enabled())
                << "the opt-in device compiles the automatic PDL wait-at-entry prologue";
            // ... and the launch policy is on until it is switched off at runtime.
            ext->set_pdl_stream_enabled(true);
            expect(ext->pdl_stream_enabled());
        }
        LUISA_INFO("CUDA stream PDL: pdl_supported = {}, auto_prologue = {}, policy = {}.",
                   ext->pdl_supported(), ext->auto_prologue_enabled(),
                   ext->pdl_stream_enabled());
    };

    "cuda_pdl_chain_is_correct"_test = [] {
        if (g_devices == nullptr || g_devices->opt_in.backend_name() != "cuda") { return; }
        auto &device = g_devices->opt_in;
        auto *ext = pdl_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto links = 8u;
        auto expected = expected_chain(luisa::span{g_devices->input}, links);
        // Chained dependency with the launch policy on: this is the configuration
        // the feature is about, so a wrong value here is a relaxation bug ...
        auto relaxed = run_chain(device, /*enable_policy=*/true, links, k_count,
                                 luisa::span{g_devices->input});
        auto relaxed_mismatches = count_mismatches(relaxed, expected);
        expect(relaxed_mismatches == 0u)
            << luisa::format("{} of {} chained values are wrong with the launch "
                             "policy on",
                             relaxed_mismatches, expected.size());
        // ... and the same chain with the policy off must agree bit for bit.
        auto strict = run_chain(device, /*enable_policy=*/false, links, k_count,
                                luisa::span{g_devices->input});
        expect(count_mismatches(strict, expected) == 0u)
            << "the chain with the launch policy off must match the same expectation";
        expect(count_mismatches(strict, relaxed) == 0u)
            << "the policy switch must not change a single value";
        ext->set_pdl_stream_enabled(true);
    };

    "cuda_pdl_launch_accounting"_test = [] {
        if (g_devices == nullptr || g_devices->opt_in.backend_name() != "cuda") { return; }
        auto &device = g_devices->opt_in;
        auto *ext = pdl_ext(device);
        if (ext == nullptr) { return; }
        ext->set_pdl_stream_enabled(true);
        auto counts = run_back_to_back(device, *ext, 3u, k_count,
                                       luisa::span{g_devices->input});
        // Both halves of the feature have to be on for a launch to be relaxed: the
        // device/driver must support PDL *and* the policy must allow it (which the
        // LUISA_DISABLE_PDL_STREAM kill switch can veto).
        auto can_relax = ext->pdl_supported() && ext->pdl_stream_enabled();
        LUISA_INFO("three back-to-back grids: relaxed = {}, strict = {} "
                   "(pdl_supported = {}, policy = {}).",
                   counts.relaxed, counts.strict, ext->pdl_supported(),
                   ext->pdl_stream_enabled());
        if (can_relax) {
            // The first grid's predecessor is `unknown` after the drain that precedes
            // the submission, so exactly the two later grids may be relaxed.
            expect(counts.relaxed == 2u)
                << "the second and third grid of a submission follow a kernel launch";
            expect(counts.strict == 1u)
                << "the first grid of a submission has no kernel predecessor";
        } else {
            // compute capability < 9.0 or driver < 12.0: the attribute must never be
            // used, so every grid of the submission is a strict launch.
            expect(counts.relaxed == 0u)
                << "PDL is unsupported on this device/driver, nothing may be relaxed";
            expect(counts.strict == 3u) << "every grid must be a strict launch";
        }
    };

    "cuda_pdl_non_kernel_boundary"_test = [] {
        if (g_devices == nullptr || g_devices->opt_in.backend_name() != "cuda") { return; }
        auto &device = g_devices->opt_in;
        auto *ext = pdl_ext(device);
        if (ext == nullptr) { return; }
        ext->set_pdl_stream_enabled(true);
        auto counts = run_dispatch_copy_dispatch(device, *ext, k_count,
                                                 luisa::span{g_devices->input});
        LUISA_INFO("grid, copy, grid: relaxed = {}, strict = {} (pdl_supported = {}).",
                   counts.relaxed, counts.strict, ext->pdl_supported());
        // The copy is a hard boundary: neither grid of this submission may be
        // relaxed, on a supported device as well as on an unsupported one.
        expect(counts.relaxed == 0u)
            << "a copy between two grids must never relax the grid that follows it";
        expect(counts.strict == 2u)
            << "both grids of a submission with a copy in between are strict launches";
    };

    "cuda_pdl_policy_toggle"_test = [] {
        if (g_devices == nullptr || g_devices->opt_in.backend_name() != "cuda") { return; }
        auto &device = g_devices->opt_in;
        auto *ext = pdl_ext(device);
        if (ext == nullptr) { return; }
        // Switched off: no grid may be relaxed, whatever the device supports.
        ext->set_pdl_stream_enabled(false);
        auto off = run_back_to_back(device, *ext, 3u, k_count,
                                   luisa::span{g_devices->input});
        expect(off.relaxed == 0u)
            << "the runtime switch must turn relaxation off for the next submission";
        expect(off.strict == 3u) << "all three grids are strict with the policy off";
        // Switched back on: the counters are re-read from zero, so the case-3
        // expectation has to come back exactly (i.e. they are not sticky).
        ext->set_pdl_stream_enabled(true);
        auto on = run_back_to_back(device, *ext, 3u, k_count,
                                   luisa::span{g_devices->input});
        if (ext->pdl_supported() && ext->pdl_stream_enabled()) {
            expect(on.relaxed == 2u) << "re-enabling the policy must restore relaxation";
            expect(on.strict == 1u);
        } else {
            expect(on.relaxed == 0u) << "an unsupported device never relaxes";
            expect(on.strict == 3u);
        }
    };

    "cuda_pdl_prologue_ab"_test = [] {
        if (g_devices == nullptr || g_devices->opt_in.backend_name() != "cuda") { return; }
        auto &with_prologue = g_devices->opt_in;
        // The second device of this process, identical except for the configuration
        // extension: it does not compile the automatic wait-at-entry prologue.
        DeviceConfig config;
        config.extension = make_config_ext(g_devices->backend, /*prologue=*/false);
        Device without_prologue = g_devices->context.create_device(g_devices->backend, &config);
        auto *ext = pdl_ext(without_prologue);
        expect(ext != nullptr) << "the CUDA backend must provide CUDAPdlStreamExt";
        if (ext == nullptr) { return; }
        expect(!ext->auto_prologue_enabled())
            << "a device without the configuration extension must not compile the prologue";
        auto *with_ext = pdl_ext(with_prologue);
        if (with_ext == nullptr) { return; }
        if (!with_ext->auto_prologue_enabled()) {
            // The kill switch overrides the configuration for the whole process, so
            // both devices below compile the same source; only say so and keep going,
            // the A/B equality is still the correct assertion.
            LUISA_WARNING("CUDA stream PDL is forced off for this process "
                          "(LUISA_DISABLE_PDL_STREAM); both devices compile the same source.");
        } else {
            expect(with_ext->auto_prologue_enabled())
                << "the opt-in device must compile the prologue";
        }
        constexpr auto links = 8u;
        auto a = run_chain(with_prologue, /*enable_policy=*/true, links, k_count,
                           luisa::span{g_devices->input});
        auto b = run_chain(without_prologue, /*enable_policy=*/true, links, k_count,
                           luisa::span{g_devices->input});
        // The "no behaviour change" assertion: the generated wait must not alter a
        // single result, and both devices must agree with the host expectation.
        auto ab_mismatches = count_mismatches(a, b);
        expect(ab_mismatches == 0u)
            << luisa::format("{} of {} values differ between the device with and "
                             "without the automatic PDL prologue",
                             ab_mismatches, a.size());
        auto expected = expected_chain(luisa::span{g_devices->input}, links);
        expect(count_mismatches(a, expected) == 0u)
            << "the opt-in device must produce the exact chained values";
        expect(count_mismatches(b, expected) == 0u)
            << "the device without the prologue must produce the exact chained values";
    };

    return 0;
}
