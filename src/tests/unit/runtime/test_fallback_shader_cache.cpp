#include "ut/ut.hpp"
#include "memory_binary_io.h"

#include <array>
#include <algorithm>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cstring>

#include <luisa/core/binary_io.h>
#include <luisa/core/intrin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

using luisa::test::MemoryBinaryIO;

struct TestFloatingPointState {
    uint64_t control{};
    uint64_t status{};

    [[nodiscard]] static TestFloatingPointState read() noexcept {
#if defined(LUISA_ARCH_X86_64)
        return {_mm_getcsr(), 0u};
#else
        TestFloatingPointState state;
        asm volatile("mrs %0, FPCR" : "=r"(state.control)::"memory");
        asm volatile("mrs %0, FPSR" : "=r"(state.status)::"memory");
        return state;
#endif
    }

    void apply() const noexcept {
#if defined(LUISA_ARCH_X86_64)
        _mm_setcsr(static_cast<uint32_t>(control));
#else
        asm volatile("msr FPCR, %0\n\tisb" ::"r"(control) : "memory");
        asm volatile("msr FPSR, %0" ::"r"(status) : "memory");
#endif
    }

    [[nodiscard]] uint64_t mode() const noexcept {
#if defined(LUISA_ARCH_X86_64)
        return control & 0xffc0u;// Sticky exception flags may change in precise code.
#else
        return control;
#endif
    }

    [[nodiscard]] TestFloatingPointState ieee() const noexcept {
        auto state = *this;
#if defined(LUISA_ARCH_X86_64)
        // Round-to-nearest, masked exceptions, gradual underflow, clear status.
        state.control = (control & ~uint64_t{0xe07fu}) | 0x1f80u;
#else
        state.control &= ~((uint64_t{1u} << 24u) | (uint64_t{3u} << 22u) |
                           (uint64_t{0x1fu} << 8u) | uint64_t{3u});
        state.status = 0u;
#endif
        return state;
    }

    [[nodiscard]] TestFloatingPointState flush() const noexcept {
        auto state = *this;
#if defined(LUISA_ARCH_X86_64)
        state.control |= 0x8040u;
#else
        state.control |= uint64_t{1u} << 24u;
#endif
        return state;
    }
};

struct FallbackDebugProbe {
    std::mutex mutex;
    luisa::span<const uint4> inputs;
    std::array<uint, 8u> expected_multiply{};
    luisa::vector<uint> visits;
    std::thread::id submitter;
    uint64_t expected_mode{};
    bool fast_math{};
    bool correct{true};
};

// The DSL macro emits a capture-free host wrapper. This pointer belongs only
// to the synchronized FTZ fixture and is never retained by the callback.
FallbackDebugProbe *active_fallback_debug_probe = nullptr;

void record_fallback_debug_probe(uint3 dispatch_id, uint token,
                                 uint4 operands, uint2 snapshot) noexcept {
    auto *probe = active_fallback_debug_probe;
    if (probe == nullptr) { return; }// The visit-count oracle detects a missing probe.
    const auto mode = TestFloatingPointState::read().mode();
    const auto thread = std::this_thread::get_id();
    std::scoped_lock lock{probe->mutex};
    if (dispatch_id.x >= probe->visits.size()) {
        probe->correct = false;
        return;
    }
    probe->visits[dispatch_id.x]++;
    const auto row = dispatch_id.x % 8u;
    const auto expected_token = dispatch_id.x ^ 0x5a35a53cu;
    const auto expected_multiply = probe->expected_multiply[row];
    const auto multiply_matches = probe->fast_math && expected_multiply == 0u ?
                                      (snapshot.x & 0x7fffffffu) == 0u :
                                      snapshot.x == expected_multiply;
    probe->correct &= mode == probe->expected_mode && thread != probe->submitter &&
                      dispatch_id.y == 0u && dispatch_id.z == 0u &&
                      token == expected_token && snapshot.y == expected_token &&
                      std::memcmp(&operands, &probe->inputs[row], sizeof(uint4)) == 0 &&
                      multiply_matches;
}
void test_fallback_fast_math_environment(const char *program_path) {
    const auto original = TestFloatingPointState::read();
    struct Restore {
        TestFloatingPointState state;
        ~Restore() noexcept { state.apply(); }
    } restore{original};
    const auto ieee = original.ieee();
    ieee.apply();
    MemoryBinaryIO binary_io;
    Context context{program_path};
    DeviceConfig config{.binary_io = &binary_io};
    auto device = context.create_device("fallback", &config);
    expect(TestFloatingPointState::read().mode() == ieee.mode())
        << "creating a Fallback device must not change caller FP control state";
    // Construct the asynchronous dispatcher and its lazily created pool before
    // changing the submitter's environment. Precise uses the worker's mode.
    ieee.apply();
    auto stream = device.create_stream();
    struct Case {
        uint4 input;
        uint2 precise;
        uint2 fast;
    };
    const std::array cases{
        Case{make_uint4(0x00000001u, 0x40000000u, 0x00000001u, 0u), make_uint2(2u, 2u), make_uint2(0u)},
        Case{make_uint4(0x80000001u, 0x40000000u, 0x80000001u, 0u), make_uint2(0x80000002u), make_uint2(0u)},
        Case{make_uint4(0x00800000u, 0x3f000000u, 0x80800000u, 0u), make_uint2(0x00400000u, 0u), make_uint2(0u)},
        Case{make_uint4(0x80800000u, 0x3f000000u, 0x00800000u, 0u), make_uint2(0x80400000u, 0u), make_uint2(0u)},
        Case{make_uint4(0x00800001u, 0x3f800000u, 0x80800000u, 0u), make_uint2(0x00800001u, 1u), make_uint2(0x00800001u, 0u)},
        Case{make_uint4(0x80800001u, 0x3f800000u, 0x00800000u, 0u), make_uint2(0x80800001u, 0x80000001u), make_uint2(0x80800001u, 0u)},
        Case{make_uint4(0x007fffffu, 0x40000000u, 0x00000001u, 0u), make_uint2(0x00fffffeu, 0x00800000u), make_uint2(0u)},
        Case{make_uint4(0x807fffffu, 0x40000000u, 0x80000001u, 0u), make_uint2(0x80fffffeu, 0x80800000u), make_uint2(0u)}};
    constexpr auto block_count = 64u;
    constexpr auto count = block_count * 32u;
    auto input = device.create_buffer<uint4>(cases.size());
    auto output = device.create_buffer<uint2>(count);
    auto probe_output = device.create_buffer<uint2>(count);
    luisa::vector<uint4> host_input;
    for (auto &&item : cases) { host_input.emplace_back(item.input); }
    luisa::vector<uint2> observed(count);
    luisa::vector<uint2> probe_observed(count);
    stream << input.copy_from(luisa::span{host_input}) << synchronize();
    auto make_kernel = [](bool probe) {
        return Kernel1D{[probe](BufferUInt4 input, BufferUInt2 output) noexcept {
            set_block_size(32u);
            if (probe) {
                $if (thread_x() == 0u) { device_log("fallback-ftz-probe"); };
            }
            if (probe) {
                UInt token = dispatch_x() ^ 0x5a35a53cu;
                UInt4 watched = input.read(dispatch_x() % 8u);
                UInt2 snapshot = make_uint2((watched.x.as<float>() * watched.y.as<float>()).as<uint>(), token);
                $if (thread_x() < 8u) {
                    // Custom host callback only: no default debugger trap.
                    $debug_break_on(token, watched, snapshot,
                                    record_fallback_debug_probe(dispatch_id, token, watched, snapshot));
                };
            }
            // Read again after the opaque host callback, then perform observable
            // arithmetic under the restored shader environment.
            auto operands = input.read(dispatch_x() % 8u);
            auto value = operands.x.as<float>();
            auto factor = operands.y.as<float>();
            auto addend = operands.z.as<float>();
            output.write(dispatch_x(), make_uint2((value * factor).as<uint>(), (value + addend).as<uint>()));
        }};
    };
    // Fallback intentionally excludes printing kernels from its object cache.
    // Validate warm execution and host callbacks with separate kernel variants.
    auto kernel = make_kernel(false);
    auto probe_kernel = make_kernel(true);
    auto precise = device.compile(kernel, {.enable_cache = true, .enable_fast_math = false});
    auto cold_fast = device.compile(kernel, {.enable_cache = true, .enable_fast_math = true});
    expect(binary_io.cache_write_count == 4u) << "precise and fast cold objects must have separate cache entries";
    auto reads = binary_io.cache_read_count;
    auto writes = binary_io.cache_write_count;
    auto warm_fast = device.compile(kernel, {.enable_cache = true, .enable_fast_math = true});
    expect(binary_io.cache_read_count == reads + 2u && binary_io.cache_write_count == writes)
        << "fast FTZ object must be loaded from the existing object and metadata";
    reads = binary_io.cache_read_count;
    auto warm_precise = device.compile(kernel, {.enable_cache = true, .enable_fast_math = false});
    expect(binary_io.cache_read_count == reads + 2u && binary_io.cache_write_count == writes)
        << "precise object must retain its own warm-cache entry";
    auto probe_precise = device.compile(probe_kernel, {.enable_cache = false, .enable_fast_math = false});
    auto probe_fast = device.compile(probe_kernel, {.enable_cache = false, .enable_fast_math = true});

    std::mutex mutex;
    luisa::vector<std::thread::id> worker_threads;
    luisa::vector<uint64_t> worker_modes;
    bool valid_messages = true;
    stream.set_log_callback([&](luisa::string_view message) noexcept {
        auto mode = TestFloatingPointState::read().mode();
        auto thread = std::this_thread::get_id();
        std::scoped_lock lock{mutex};
        valid_messages &= message == "fallback-ftz-probe";
        worker_threads.emplace_back(thread);
        worker_modes.emplace_back(mode);
    });
    FallbackDebugProbe debug_probe;
    debug_probe.inputs = luisa::span{host_input};
    debug_probe.visits.resize(count);
    debug_probe.submitter = std::this_thread::get_id();
    debug_probe.expected_mode = ieee.mode();
    active_fallback_debug_probe = &debug_probe;
    struct ResetDebugProbe {
        ~ResetDebugProbe() noexcept { active_fallback_debug_probe = nullptr; }
    } reset_debug_probe;
    for (auto phase = 0u; phase < 5u; phase++) {
        const auto fast_math = phase == 1u || phase == 3u;
        debug_probe.fast_math = fast_math;
        debug_probe.correct = true;
        std::fill(debug_probe.visits.begin(), debug_probe.visits.end(), 0u);
        for (auto row = 0u; row < cases.size(); row++) {
            debug_probe.expected_multiply[row] = fast_math ? cases[row].fast.x : cases[row].precise.x;
        }
        auto &shader = phase == 1u ? cold_fast : phase == 2u ? warm_precise :
                                             phase == 3u     ? warm_fast :
                                                               precise;
        worker_threads.clear();
        worker_modes.clear();
        valid_messages = true;
        std::fill(observed.begin(), observed.end(), make_uint2(~0u));
        std::fill(probe_observed.begin(), probe_observed.end(), make_uint2(~0u));
        // The first precise dispatch initializes the pool with IEEE state.
        // Later submissions deliberately use a different caller environment.
        auto caller_state = phase % 2u == 0u ? ieee : ieee.flush();
        caller_state.apply();
        auto dispatcher_mode = ~uint64_t{0u};
        auto dispatcher_thread = std::thread::id{};
        stream << output.copy_from(luisa::span{observed})
               << shader(input, output).dispatch(count)
               << output.copy_to(luisa::span{observed})
               << probe_output.copy_from(luisa::span{probe_observed})
               << (fast_math ? probe_fast : probe_precise)(input, probe_output).dispatch(count)
               << probe_output.copy_to(luisa::span{probe_observed})
               << [&]() noexcept {
                      dispatcher_mode = TestFloatingPointState::read().mode();
                      dispatcher_thread = std::this_thread::get_id();
                  }
               << synchronize();
        expect(TestFloatingPointState::read().mode() == caller_state.mode())
            << "Fallback asynchronous submission changed the caller's FP mode";
        expect(dispatcher_thread != std::this_thread::get_id() && dispatcher_mode == ieee.mode())
            << "completion callback must see the asynchronous dispatcher's original FP mode";
        expect(valid_messages && worker_modes.size() == block_count)
            << "every Fallback block must emit one host callback";
        expect(std::all_of(worker_modes.begin(), worker_modes.end(), [&](auto mode) { return mode == ieee.mode(); }))
            << "shader log callback must see the original worker FP mode";
        expect(std::all_of(worker_threads.begin(), worker_threads.end(), [&](auto thread) { return thread != std::this_thread::get_id(); }))
            << "Fallback shader callbacks must execute asynchronously, not on the submitter";
        bool debug_visits_match = true;
        for (auto row = 0u; row < count; row++) {
            debug_visits_match &= debug_probe.visits[row] == (row % 32u < 8u ? 1u : 0u);
        }
        expect(debug_probe.correct && debug_visits_match)
            << "custom debug callback must preserve worker FP mode and evaluate every watched snapshot exactly"
            << " phase=" << phase << " fast_math=" << fast_math;
        bool correct = true;
        for (auto row = 0u; row < count; row++) {
            auto expected = fast_math ? cases[row % 8u].fast : cases[row % 8u].precise;
            for (auto component = 0u; component < 2u; component++) {
                auto actual = observed[row][component];
                correct &= fast_math && expected[component] == 0u ? (actual & 0x7fffffffu) == 0u : actual == expected[component];
                auto probe_actual = probe_observed[row][component];
                correct &= fast_math && expected[component] == 0u ? (probe_actual & 0x7fffffffu) == 0u : probe_actual == expected[component];
            }
        }
        expect(correct) << "Fallback cold/warm fast FTZ or later precise result differs";
    }
}


[[nodiscard]] int run_cached_kernel(
    const char *program_path, const BinaryIO *binary_io,
    int value, bool enable_cache) noexcept {
    Context context{program_path};
    DeviceConfig config{.binary_io = binary_io};
    auto device = context.create_device("fallback", &config);
    auto output = device.create_buffer<int>(1u);
    Kernel1D kernel = [](
                          BufferVar<int> result,
                          Int parameter) noexcept {
        result->write(0u, parameter * 3 + 1);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = enable_cache});
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, value).dispatch(1u)
           << output.copy_to(luisa::span{&result, 1})
           << synchronize();
    return result;
}

[[nodiscard]] std::array<uint4, 4u>
run_boolean_comparison_kernel(const char *program_path) noexcept {
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto output = device.create_buffer<uint4>(4u);
    Kernel1D kernel = [](BufferUInt4 result) noexcept {
        const auto index = dispatch_x();
        const auto lhs = (index & 1u) != 0u;
        const auto rhs = index >= 2u;
        const auto scalar_equal = lhs == rhs;
        const auto scalar_not_equal = lhs != rhs;
        const auto vector_equal =
            make_bool2(lhs, !lhs) == make_bool2(rhs, !rhs);
        const auto vector_not_equal =
            make_bool2(lhs, !lhs) != make_bool2(rhs, !rhs);
        const auto pack = [](Bool2 value) noexcept {
            return select(0u, 1u, value.x) |
                   (select(0u, 1u, value.y) << 1u);
        };
        result->write(index,
                      make_uint4(select(0u, 1u, scalar_equal),
                                 select(0u, 1u, scalar_not_equal),
                                 pack(vector_equal),
                                 pack(vector_not_equal)));
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<uint4, 4u> result{};
    stream << shader(output).dispatch(4u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    return result;
}

[[nodiscard]] std::array<uint, 4u>
run_assume_kernel(const char *program_path) noexcept {
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto output = device.create_buffer<uint>(4u);
    Kernel1D kernel = [](BufferUInt result) noexcept {
        const auto index = dispatch_x();
        assume(index < 4u);
        result->write(index, index + 1u);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<uint, 4u> result{};
    stream << shader(output).dispatch(4u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    return result;
}

[[nodiscard]] std::array<float4, 4u>
run_minimal_codegen_vector_kernel(const char *program_path) noexcept {
    constexpr auto lane_count = 1024u;
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto input = device.create_buffer<float>(4u * lane_count);
    auto output = device.create_buffer<float4>(4u);
    Kernel1D kernel = [](BufferFloat values, BufferFloat4 result) noexcept {
        // Keep scalar SSA values live until they are assembled into vectors.
        // This is the reduced form of the large material-dispatch kernel that
        // exposed FastISel folding four-byte-aligned spills into aligned
        // vector loads. The forced optimization limit below exercises the
        // same O0-IR/O1-machine policy without a production-sized shader.
        constexpr auto kernel_lane_count = 1024u;
        const auto index = dispatch_x();
        luisa::vector<Float> lanes;
        lanes.reserve(kernel_lane_count);
        for (auto lane = 0u; lane < kernel_lane_count; ++lane) {
            lanes.emplace_back(values.read(index * kernel_lane_count + lane));
        }
        Float4 sum = make_float4(0.0f);
        for (auto group = 0u; group < kernel_lane_count / 4u; ++group) {
            const auto lane = group * 4u;
            sum += make_float4(lanes[lane], lanes[lane + 1u],
                               lanes[lane + 2u], lanes[lane + 3u]);
        }
        result->write(index, sum);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<float, 4u * lane_count> values{};
    for (auto index = 0u; index < 4u; ++index) {
        for (auto lane = 0u; lane < lane_count; ++lane) {
            values[index * lane_count + lane] =
                static_cast<float>(index * 1000u + lane);
        }
    }
    std::array<float4, 4u> result{};
    stream << input.copy_from(luisa::span{values})
           << shader(input, output).dispatch(4u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    if (argc == 2 && std::strcmp(argv[1], "--ftz-only") == 0) {
        test_fallback_fast_math_environment(program_path);
        return 0;
    }
    // This must be set before the first fallback backend module is loaded.
#if defined(_WIN32)
    _putenv_s("LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT", "1");
#else
    setenv("LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT", "1", 1);
#endif
    MemoryBinaryIO binary_io;

    "fallback object cache reuses code across devices and keeps uniforms dynamic"_test =
        [&] {
            auto cold_result = run_cached_kernel(
                program_path, &binary_io, 7, true);
            expect(cold_result == 22);
            expect(binary_io.cache_write_count == 2u);
            auto cold_read_count = binary_io.cache_read_count;
            auto cold_write_count = binary_io.cache_write_count;

            auto hot_result = run_cached_kernel(
                program_path, &binary_io, 11, true);
            expect(hot_result == 34);
            expect(
                binary_io.cache_read_count ==
                cold_read_count + 2u);
            expect(
                binary_io.cache_write_count ==
                cold_write_count);

            auto reads_before_disabled =
                binary_io.cache_read_count;
            auto writes_before_disabled =
                binary_io.cache_write_count;
            auto uncached_result = run_cached_kernel(
                program_path, &binary_io, 13, false);
            expect(uncached_result == 40);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled);
        };

    "fallback lowers scalar and vector boolean equality"_test = [&] {
        const auto actual = run_boolean_comparison_kernel(program_path);
        constexpr std::array expected{
            make_uint4(1u, 0u, 3u, 0u),
            make_uint4(0u, 1u, 0u, 3u),
            make_uint4(0u, 1u, 0u, 3u),
            make_uint4(1u, 0u, 3u, 0u)};
        expect(std::memcmp(actual.data(), expected.data(), sizeof(expected)) == 0);
    };

    "fallback lowers scalar boolean assumptions to LLVM i1"_test = [&] {
        const auto actual = run_assume_kernel(program_path);
        constexpr std::array expected{1u, 2u, 3u, 4u};
        expect(actual == expected);
    };

    "fallback minimal codegen preserves vector spill alignment"_test = [&] {
        const auto actual = run_minimal_codegen_vector_kernel(program_path);
        constexpr std::array expected{
            make_float4(130560.0f, 130816.0f, 131072.0f, 131328.0f),
            make_float4(386560.0f, 386816.0f, 387072.0f, 387328.0f),
            make_float4(642560.0f, 642816.0f, 643072.0f, 643328.0f),
            make_float4(898560.0f, 898816.0f, 899072.0f, 899328.0f)};
        expect(std::memcmp(actual.data(), expected.data(), sizeof(expected)) == 0);
    };
}
