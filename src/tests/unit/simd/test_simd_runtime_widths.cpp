#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(
    const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;

    explicit ScopedEnvironmentVariable(
        const char *env_name, const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(),
            previous ? previous->c_str() : nullptr);
    }
};

[[nodiscard]] float half_bits_to_float(uint16_t bits) noexcept {
    auto value = luisa::half{};
    std::memcpy(&value, &bits, sizeof(bits));
    return static_cast<float>(value);
}

[[nodiscard]] uint16_t float_to_half_bits(float value) noexcept {
    auto converted = luisa::half{value};
    auto bits = uint16_t{0u};
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

[[nodiscard]] bool invalid_worker_override_fails_closed(
    const char *program) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    auto child = fork();
    if (child < 0) { return false; }
    if (child == 0) {
        const rlimit no_core{0u, 0u};
        static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
        set_environment_variable("LUISA_SIMD_WORKER_COUNT", "invalid");
        Context context{program};
        auto device = context.create_device("simd");
        static_cast<void>(device.compute_warp_size());
        _exit(EXIT_SUCCESS);
    }
    auto child_status = 0;
    while (waitpid(child, &child_status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(child_status) &&
           WTERMSIG(child_status) == SIGABRT;
#else
    static_cast<void>(program);
    return true;
#endif
}

[[nodiscard]] bool active_assertion_fails_closed(
    const char *program) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    auto child = fork();
    if (child < 0) { return false; }
    if (child == 0) {
        const rlimit no_core{0u, 0u};
        static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
        Context context{program};
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(4u, 1u);
        auto device = context.create_device("simd", &config);
        Kernel1D kernel = []() noexcept {
            set_block_size(4u, 1u, 1u);
            set_warp_size(4u);
            device_assert(
                dispatch_x() != 2u,
                "active SIMD assertion regression");
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        stream << shader().dispatch(4u) << synchronize();
        _exit(EXIT_SUCCESS);
    }
    auto child_status = 0;
    while (waitpid(child, &child_status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(child_status) &&
           WTERMSIG(child_status) == SIGABRT;
#else
    static_cast<void>(program);
    return true;
#endif
}

[[nodiscard]] bool mismatched_loop_barrier_fails_closed(
    const char *program) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    auto child = fork();
    if (child < 0) { return false; }
    if (child == 0) {
        const rlimit no_core{0u, 0u};
        static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
        Context context{program};
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(8u, 1u);
        auto device = context.create_device("simd", &config);
        Kernel1D kernel = []() noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(8u);
            UInt iteration = 0u;
            $while (iteration < 2u) {
                // Packet zero reaches this site in iteration zero; the other
                // packets first reach the same static site in iteration one.
                // A static-ID-only wrapper would incorrectly rendezvous them.
                $if ((thread_x() < 8u) == (iteration == 0u)) {
                    sync_block();
                };
                iteration += 1u;
            };
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        stream << shader().dispatch(32u) << synchronize();
        _exit(EXIT_SUCCESS);
    }
    auto child_status = 0;
    while (waitpid(child, &child_status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(child_status);
#else
    static_cast<void>(program);
    return true;
#endif
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    expect(invalid_worker_override_fails_closed(
        argc > 0 ? argv[0] : ""))
        << "invalid SIMD worker override must fail closed";
    expect(active_assertion_fails_closed(
        argc > 0 ? argv[0] : ""))
        << "an active false SIMD device assertion must fail closed";
    expect(mismatched_loop_barrier_fails_closed(
        argc > 0 ? argv[0] : ""))
        << "mismatched dynamic block-barrier instances must fail closed";
    Context context{argc > 0 ? argv[0] : ""};
    ScopedEnvironmentVariable width_override{
        "LUISA_SIMD_WARP_WIDTH", "2"};
    ScopedEnvironmentVariable worker_override{
        "LUISA_SIMD_WORKER_COUNT", "1"};

    {
        auto device = context.create_device("simd");
        expect(device.compute_warp_size() == 2u)
            << "SIMD environment width override was not honored";
    }

    {
        ScopedEnvironmentVariable invalid_worker_override{
            "LUISA_SIMD_WORKER_COUNT", "invalid"};
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(1u, 1u);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == 1u)
            << "explicit SIMD configuration must override the environment";
    }

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == width)
            << "explicit SIMD width must override the environment";

        constexpr auto block_threads = 32u;
        auto output = device.create_buffer<uint>(block_threads);
        Kernel1D kernel = [width](BufferUInt result) noexcept {
            // Block size and logical warp size are separate contracts. Like
            // fallback's scalar thread loop, SIMD partitions this 32-thread
            // block into packets of the configured width.
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto lane = warp_lane_id();
            auto sum = warp_active_sum(lane + 1u);
            result.write(dispatch_x(), sum + lane);
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        luisa::vector<uint> host(block_threads, 0u);
        stream << shader(output).dispatch(block_threads)
               << output.copy_to(luisa::span{host})
               << synchronize();

        auto sum = width * (width + 1u) / 2u;
        for (auto thread = 0u; thread < block_threads; thread++) {
            auto lane = thread % width;
            expect(host[thread] == sum + lane)
                << "SIMD thread-block packet partition mismatch";
        }

        // DSL $outline adds source-comment metadata to its call site. That
        // metadata must not prevent the SIMD compiler's mandatory callable
        // legalization pass from inlining the outlined region.
        Kernel1D outlined_kernel = [width](BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            UInt value = dispatch_x();
            $outline {
                value = value * 3u + 7u;
            };
            result.write(dispatch_x(), value);
        };
        auto outlined_shader = device.compile(outlined_kernel);
        stream << outlined_shader(output).dispatch(block_threads)
               << output.copy_to(luisa::span{host})
               << synchronize();
        for (auto thread = 0u; thread < block_threads; thread++) {
            expect(host[thread] == thread * 3u + 7u)
                << "SIMD outlined callable legalization mismatch";
        }

        // Cooperative blocks keep one coroutine per SIMD packet. The second
        // block below has only three logical invocations, so every wider
        // configuration also contains wholly inactive packets. Those packets
        // must not participate in the barrier, while the active packet still
        // observes the one shared allocation owned by the block. Cross-packet
        // shared-memory visibility is covered separately by test_shared_mem.
        constexpr auto shared_tail_threads = block_threads + 3u;
        auto shared_tail_output = device.create_buffer<uint>(
            shared_tail_threads);
        Kernel1D shared_tail_kernel = [width](
                                          BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            Shared<uint> tile{32u};
            tile.write(thread_x(), dispatch_x() + 17u);
            sync_block();
            result.write(dispatch_x(), tile.read(thread_x()));
        };
        auto shared_tail_shader = device.compile(shared_tail_kernel);
        luisa::vector<uint> shared_tail_host(
            shared_tail_threads, 0u);
        stream << shared_tail_shader(shared_tail_output).dispatch(shared_tail_threads)
               << shared_tail_output.copy_to(
                      luisa::span{shared_tail_host})
               << synchronize();
        for (auto thread = 0u;
             thread < shared_tail_threads; thread++) {
            expect(shared_tail_host[thread] == thread + 17u)
                << "SIMD cooperative inactive-tail mismatch";
        }

        // Repeated static sites are distinct dynamic barrier instances. A
        // uniform runtime trip count must execute at every width, including
        // the wholly inactive packets in the final three-thread block.
        auto loop_barrier_output = device.create_buffer<uint>(
            shared_tail_threads);
        Kernel1D loop_barrier_kernel = [width](
                                           BufferUInt result,
                                           UInt outer_trip_count,
                                           UInt inner_trip_count) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            Shared<uint> values{32u};
            UInt value = dispatch_x() + 1u;
            UInt outer = 0u;
            $while (outer < outer_trip_count) {
                UInt inner = 0u;
                $while (inner < inner_trip_count) {
                    auto phase = outer * inner_trip_count + inner;
                    values.write(thread_x(), value + phase);
                    sync_block();
                    value = values.read(thread_x()) * 3u + 1u;
                    sync_block();
                    inner += 1u;
                };
                outer += 1u;
            };
            result.write(dispatch_x(), value);
        };
        auto loop_barrier_shader = device.compile(
            loop_barrier_kernel);
        luisa::vector<uint> loop_barrier_host(
            shared_tail_threads, 0u);
        constexpr auto loop_barrier_outer_trip_count = 2u;
        constexpr auto loop_barrier_inner_trip_count = 2u;
        stream << loop_barrier_shader(
                      loop_barrier_output,
                      loop_barrier_outer_trip_count,
                      loop_barrier_inner_trip_count)
                      .dispatch(shared_tail_threads)
               << loop_barrier_output.copy_to(
                      luisa::span{loop_barrier_host})
               << synchronize();
        for (auto thread = 0u;
             thread < shared_tail_threads; thread++) {
            auto expected = thread + 1u;
            for (auto outer = 0u;
                 outer < loop_barrier_outer_trip_count; outer++) {
                for (auto inner = 0u;
                     inner < loop_barrier_inner_trip_count; inner++) {
                    auto phase =
                        outer * loop_barrier_inner_trip_count + inner;
                    expected = (expected + phase) * 3u + 1u;
                }
            }
            expect(loop_barrier_host[thread] == expected)
                << "SIMD repeated block-barrier instance mismatch";
        }

        // A partial 2D edge block is not generally a lane-prefix tail. For
        // example, the right edge below has three live lanes followed by five
        // inactive lanes in each row. Cooperative wrappers must retain the
        // full per-dimension dispatch mask rather than relying on the 1D
        // packet-tail narrowing contract.
        constexpr auto shared_2d_size = make_uint2(11u, 5u);
        auto shared_2d_output = device.create_buffer<uint>(
            shared_2d_size.x * shared_2d_size.y);
        Kernel2D shared_2d_kernel = [width](
                                        BufferUInt result) noexcept {
            set_block_size(8u, 4u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            Shared<uint> tile{32u};
            auto local = thread_x() + thread_y() * 8u;
            auto value = dispatch_x() + dispatch_y() * 100u;
            tile.write(local, value);
            sync_block();
            auto output = dispatch_x() + dispatch_y() * 11u;
            result.write(output, tile.read(local));
        };
        auto shared_2d_shader = device.compile(shared_2d_kernel);
        luisa::vector<uint> shared_2d_host(
            shared_2d_size.x * shared_2d_size.y, 0u);
        stream << shared_2d_shader(shared_2d_output).dispatch(shared_2d_size)
               << shared_2d_output.copy_to(
                      luisa::span{shared_2d_host})
               << synchronize();
        for (auto y = 0u; y < shared_2d_size.y; y++) {
            for (auto x = 0u; x < shared_2d_size.x; x++) {
                expect(shared_2d_host[y * shared_2d_size.x + x] ==
                       x + y * 100u)
                    << "SIMD cooperative sparse 2D edge mismatch";
            }
        }

        // Regression: LLVM integer div/rem may lower to trapping scalar
        // instructions even when a zero divisor exists only in an inactive
        // tail lane. Sanitization therefore has to happen before the vector
        // operation; masking the result afterwards is too late.
        auto tail_threads = width == 1u ? 1u : width - 1u;
        auto tail_input = device.create_buffer<uint>(tail_threads);
        auto tail_output = device.create_buffer<uint>(tail_threads);
        Kernel1D tail_kernel = [width](
                                   BufferUInt input,
                                   BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto divisor = input.read(dispatch_x());
            result.write(dispatch_x(), 17u % divisor);
        };
        auto tail_shader = device.compile(tail_kernel);
        luisa::vector<uint> tail_divisors(tail_threads, 3u);
        luisa::vector<uint> tail_host(tail_threads, 0u);
        stream << tail_input.copy_from(luisa::span{tail_divisors})
               << tail_shader(tail_input, tail_output).dispatch(tail_threads)
               << tail_output.copy_to(luisa::span{tail_host})
               << synchronize();
        for (auto value : tail_host) {
            expect(value == 2u)
                << "inactive SIMD tail lane reached integer remainder";
        }

        // Clock is sampled once at each dynamic packet/cohort. Assertion
        // reduction must treat inactive tail lanes as true: their synthesized
        // dispatch IDs intentionally fail the source predicate below.
        auto debug_threads = width == 1u ? 1u : width - 1u;
        auto debug_output = device.create_buffer<uint>(debug_threads);
        Kernel1D debug_kernel = [width, debug_threads](
                                    BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto begin = device_clock();
            device_assert(
                dispatch_x() < debug_threads,
                "inactive SIMD tail lane reached device_assert");
            auto end = device_clock();
            device_assert(
                end >= begin,
                "SIMD device clock moved backwards");
            result.write(dispatch_x(), ite(end >= begin, 1u, 0u));
        };
        auto debug_shader = device.compile(debug_kernel);
        luisa::vector<uint> debug_host(debug_threads, 0u);
        stream << debug_shader(debug_output).dispatch(debug_threads)
               << debug_output.copy_to(luisa::span{debug_host})
               << synchronize();
        for (auto value : debug_host) {
            expect(value == 1u)
                << "SIMD clock/assert lowering mismatch";
        }

        // Exercise the runtime's fixed-width texture packet callback rather
        // than only the generic Schedule-to-LLVM packet ABI. The first
        // 32-thread row is fully contiguous for every supported width, while
        // the final texel exercises an inactive packet tail.
        constexpr auto image_size = make_uint2(33u, 3u);
        Kernel2D image_kernel = [width](ImageFloat target) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto value = target.read(coordinate);
            target.write(
                coordinate,
                value + make_float4(1.0f, 2.0f, 3.0f, 4.0f));
        };
        auto image_shader = device.compile(image_kernel);
        auto check_float_image = [&](const char *failure_message) noexcept {
            auto image = device.create_image<float>(
                PixelStorage::FLOAT4, image_size);
            luisa::vector<float4> image_input(image_size.x * image_size.y);
            luisa::vector<float4> image_output(image_input.size());
            for (auto y = 0u; y < image_size.y; y++) {
                for (auto x = 0u; x < image_size.x; x++) {
                    image_input[y * image_size.x + x] = make_float4(
                        static_cast<float>(x), static_cast<float>(y),
                        static_cast<float>(x + y), 1.0f);
                }
            }
            stream << image.copy_from(luisa::span{image_input})
                   << image_shader(image).dispatch(image_size)
                   << image.copy_to(luisa::span{image_output})
                   << synchronize();
            for (auto i = size_t{0u}; i < image_input.size(); i++) {
                expect(all(image_output[i] ==
                           image_input[i] +
                               make_float4(1.0f, 2.0f, 3.0f, 4.0f)))
                    << failure_message;
            }
        };
        check_float_image("SIMD contiguous FLOAT4 packet mismatch");
        {
            ScopedEnvironmentVariable disable_contiguous_packets{
                "LUISA_SIMD_DISABLE_CONTIGUOUS_TEXTURE_PACKETS", "1"};
            check_float_image("SIMD generic FLOAT4 packet mismatch");
        }
        if (width >= 4u) {
            ScopedEnvironmentVariable disable_direct_native_packets{
                "LUISA_SIMD_DISABLE_DIRECT_NATIVE_TEXTURE_PACKETS", "1"};
            check_float_image(
                "SIMD callback FLOAT4 packet mismatch");
        }
        if (width == 8u) {
            ScopedEnvironmentVariable disable_gathered_native_reads{
                "LUISA_SIMD_DISABLE_GATHERED_NATIVE_TEXTURE_READS", "1"};
            check_float_image(
                "SIMD callback tail FLOAT4 packet mismatch");
        }

        Kernel2D byte4_image_kernel = [width](
                                          ImageFloat target) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto x = (cast<float>(coordinate.x) - 1.0f) / 30.0f;
            auto y = cast<float>(coordinate.y) / 2.0f;
            target.write(
                coordinate,
                make_float4(
                    x, y, 0.5f / 255.0f,
                    127.5f / 255.0f));
        };
        auto byte4_image_shader = device.compile(byte4_image_kernel);
        auto check_byte4_image = [&](const char *failure_message) noexcept {
            auto image = device.create_image<float>(
                PixelStorage::BYTE4, image_size);
            luisa::vector<std::byte> output(
                image_size.x * image_size.y * 4u);
            stream << byte4_image_shader(image).dispatch(image_size)
                   << image.copy_to(luisa::span{output})
                   << synchronize();
            auto encode = [](float value) noexcept {
                auto rounded = std::round(value * 255.0f);
                return static_cast<uint8_t>(
                    std::clamp(rounded, 0.0f, 255.0f));
            };
            for (auto y = 0u; y < image_size.y; y++) {
                for (auto x = 0u; x < image_size.x; x++) {
                    auto index = (y * image_size.x + x) * 4u;
                    auto expected = std::array{
                        encode((static_cast<float>(x) - 1.0f) / 30.0f),
                        encode(static_cast<float>(y) / 2.0f),
                        encode(0.5f / 255.0f),
                        encode(127.5f / 255.0f),
                    };
                    for (auto component = 0u; component < 4u;
                         component++) {
                        expect(std::to_integer<uint8_t>(
                                   output[index + component]) ==
                               expected[component])
                            << failure_message;
                    }
                }
            }
        };
        check_byte4_image("SIMD direct BYTE4 packet mismatch");
        if (width >= 4u) {
            ScopedEnvironmentVariable disable_direct_byte4_packets{
                "LUISA_SIMD_DISABLE_DIRECT_BYTE4_TEXTURE_PACKETS", "1"};
            check_byte4_image("SIMD callback BYTE4 packet mismatch");
        }

        Kernel2D half4_write_kernel = [width](
                                          ImageFloat target,
                                          BufferFloat4 source) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto index = coordinate.x + coordinate.y * 33u;
            target.write(coordinate, source.read(index));
        };
        Kernel2D half4_read_kernel = [width](
                                         ImageFloat source,
                                         BufferFloat4 output) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto index = coordinate.x + coordinate.y * 33u;
            output.write(index, source.read(coordinate));
        };
        auto half4_write_shader = device.compile(half4_write_kernel);
        auto half4_read_shader = device.compile(half4_read_kernel);
        constexpr std::array half4_special_bits{
            0x00000000u,
            0x80000000u,
            0x7f800000u,
            0xff800000u,
            0x7fc00000u,
            0xffc00000u,
            0x7f800001u,
            0xff800001u,
            0x00000001u,
            0x80000001u,
            0x007fffffu,
            0x807fffffu,
            0x00800000u,
            0x80800000u,
            0x32ffffffu,
            0x33000000u,
            0x337fffffu,
            0x33800000u,
            0x33800001u,
            0x387fffffu,
            0x38800000u,
            0x38800001u,
            0x3f7fffffu,
            0x3f800000u,
            0x3f800fffu,
            0x3f801000u,
            0x3f801001u,
            0x477fdfffu,
            0x477fe000u,
            0x477fefffu,
            0x477ff000u,
            0x477ff001u,
            0x7f7fffffu,
            0xff7fffffu,
        };
        luisa::vector<float4> half4_input(
            image_size.x * image_size.y);
        for (auto i = size_t{0u}; i < half4_input.size() * 4u; i++) {
            auto bits = static_cast<uint32_t>(i) * 0x9e3779b9u +
                        0x7f4a7c15u;
            if (i < half4_special_bits.size()) {
                bits = half4_special_bits[i];
            }
            half4_input[i / 4u][i % 4u] = std::bit_cast<float>(bits);
        }
        auto check_half4_image = [&](const char *failure_message) noexcept {
            auto image = device.create_image<float>(
                PixelStorage::HALF4, image_size);
            auto input = device.create_buffer<float4>(half4_input.size());
            auto output = device.create_buffer<float4>(half4_input.size());
            luisa::vector<uint16_t> raw_output(half4_input.size() * 4u);
            luisa::vector<float4> read_output(half4_input.size());
            stream << input.copy_from(luisa::span{half4_input})
                   << half4_write_shader(image, input).dispatch(image_size)
                   << image.copy_to(luisa::span{raw_output})
                   << half4_read_shader(image, output).dispatch(image_size)
                   << output.copy_to(luisa::span{read_output})
                   << synchronize();
            for (auto i = size_t{0u}; i < half4_input.size() * 4u; i++) {
                auto expected_half = float_to_half_bits(
                    half4_input[i / 4u][i % 4u]);
                expect(raw_output[i] == expected_half)
                    << failure_message;
                auto expected_float = half_bits_to_float(expected_half);
                expect(std::bit_cast<uint32_t>(
                           read_output[i / 4u][i % 4u]) ==
                       std::bit_cast<uint32_t>(expected_float))
                    << failure_message;
            }
        };
        check_half4_image("SIMD direct HALF4 packet mismatch");
        if (width >= 4u) {
            ScopedEnvironmentVariable disable_direct_half4_packets{
                "LUISA_SIMD_DISABLE_DIRECT_HALF4_TEXTURE_PACKETS", "1"};
            check_half4_image("SIMD callback HALF4 packet mismatch");
        }

        auto uint_image = device.create_image<uint>(
            PixelStorage::INT4, image_size);
        luisa::vector<uint4> uint_image_input(image_size.x * image_size.y);
        luisa::vector<uint4> uint_image_output(uint_image_input.size());
        for (auto y = 0u; y < image_size.y; y++) {
            for (auto x = 0u; x < image_size.x; x++) {
                uint_image_input[y * image_size.x + x] = make_uint4(
                    x, y, x + y, x * 3u + y * 5u);
            }
        }
        Kernel2D uint_image_kernel = [width](ImageUInt target) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto value = target.read(coordinate);
            target.write(
                coordinate,
                value + make_uint4(5u, 7u, 11u, 13u));
        };
        auto uint_image_shader = device.compile(uint_image_kernel);
        stream << uint_image.copy_from(luisa::span{uint_image_input})
               << uint_image_shader(uint_image).dispatch(image_size)
               << uint_image.copy_to(luisa::span{uint_image_output})
               << synchronize();
        for (auto i = size_t{0u}; i < uint_image_input.size(); i++) {
            expect(all(uint_image_output[i] ==
                       uint_image_input[i] +
                           make_uint4(5u, 7u, 11u, 13u)))
                << "SIMD contiguous INT4 packet mismatch";
        }

        auto check_int1_image = [&](const char *failure_message) noexcept {
            auto image = device.create_image<uint>(
                PixelStorage::INT1, image_size);
            luisa::vector<uint> image_input(image_size.x * image_size.y);
            luisa::vector<uint> image_output(image_input.size());
            for (auto i = size_t{0u}; i < image_input.size(); i++) {
                image_input[i] =
                    0x81234567u + static_cast<uint32_t>(i) * 0x01020305u;
            }
            stream << image.copy_from(luisa::span{image_input})
                   << uint_image_shader(image).dispatch(image_size)
                   << image.copy_to(luisa::span{image_output})
                   << synchronize();
            for (auto i = size_t{0u}; i < image_input.size(); i++) {
                expect(image_output[i] == image_input[i] + 5u)
                    << failure_message;
            }
        };
        check_int1_image("SIMD direct INT1 packet mismatch");
        if (width >= 4u) {
            ScopedEnvironmentVariable disable_direct_int1_packets{
                "LUISA_SIMD_DISABLE_DIRECT_INT1_TEXTURE_PACKETS", "1"};
            check_int1_image("SIMD callback INT1 packet mismatch");
        }

        // Indirect dispatch uses a backend-owned header/record ABI. Author
        // more records than the capacity to exercise masked out-of-range
        // lanes and an inactive packet tail at every supported width. A zero
        // block dimension deliberately invalidates record 2.
        constexpr auto indirect_capacity = 7u;
        auto indirect = device.create_indirect_dispatch_buffer(
            indirect_capacity);
        auto indirect_output = device.create_buffer<uint>(
            indirect_capacity);
        Kernel1D set_indirect_count = [](
                                          Var<IndirectDispatchBuffer> buffer) noexcept {
            set_block_size(32u, 1u, 1u);
            buffer.set_dispatch_count(indirect_capacity + 5u);
        };
        Kernel1D write_indirect_records = [](
                                              Var<IndirectDispatchBuffer> buffer) noexcept {
            set_block_size(32u, 1u, 1u);
            auto index = dispatch_x();
            auto block_size = ite(
                index == 2u,
                make_uint3(0u, 1u, 1u),
                make_uint3(8u, 1u, 1u));
            buffer.set_kernel(
                index, block_size,
                make_uint3(index + 1u, 1u, 1u), index);
        };
        Kernel1D consume_indirect = [](
                                        BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            result.atomic(kernel_id()).fetch_add(dispatch_size().x);
        };
        auto set_indirect_count_shader = device.compile(
            set_indirect_count);
        auto write_indirect_records_shader = device.compile(
            write_indirect_records);
        auto consume_indirect_shader = device.compile(
            consume_indirect);
        luisa::vector<uint> indirect_host(
            indirect_capacity, 0u);
        stream << indirect_output.copy_from(
                      luisa::span{indirect_host})
               << set_indirect_count_shader(indirect).dispatch(width)
               << write_indirect_records_shader(indirect).dispatch(
                      indirect_capacity + 2u)
               << consume_indirect_shader(indirect_output).dispatch(indirect, 1u, 4u)
               << consume_indirect_shader(indirect_output).dispatch(indirect, 6u)
               << consume_indirect_shader(indirect_output).dispatch(indirect)
               << indirect_output.copy_to(
                      luisa::span{indirect_host})
               << synchronize();
        constexpr std::array indirect_expected{
            1u, 8u, 0u, 32u, 50u, 36u, 98u};
        for (auto i = 0u; i < indirect_capacity; i++) {
            expect(indirect_host[i] == indirect_expected[i])
                << "SIMD indirect dispatch mismatch";
        }

        // Batched direct dispatch shares kernel_id() semantics with indirect
        // records: each logical command receives its zero-based batch index.
        std::fill(
            indirect_host.begin(), indirect_host.end(), 0u);
        constexpr std::array direct_batches{
            make_uint3(1u, 1u, 1u),
            make_uint3(2u, 1u, 1u),
            make_uint3(3u, 1u, 1u)};
        stream << indirect_output.copy_from(
                      luisa::span{indirect_host})
               << consume_indirect_shader(indirect_output).dispatch(luisa::span{direct_batches})
               << indirect_output.copy_to(
                      luisa::span{indirect_host})
               << synchronize();
        constexpr std::array direct_expected{
            1u, 4u, 9u, 0u, 0u, 0u, 0u};
        for (auto i = 0u; i < indirect_capacity; i++) {
            expect(indirect_host[i] == direct_expected[i])
                << "SIMD batched dispatch kernel-id mismatch";
        }
    }
}
